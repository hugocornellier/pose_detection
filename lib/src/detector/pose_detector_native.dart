import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

import '../types.dart';
import '../isolate/pose_detector_core.dart';

/// Startup payload transferred to the background isolate via [Isolate.spawn].
class _IsolateStartupData {
  final SendPort sendPort;
  final TransferableTypedData yoloBytes;
  final TransferableTypedData landmarkBytes;
  final String modeName;
  final String landmarkModelName;
  final double detectorConf;
  final double detectorIou;
  final int maxDetections;
  final double minLandmarkScore;
  final int interpreterPoolSize;
  final String performanceModeName;
  final int? numThreads;

  _IsolateStartupData({
    required this.sendPort,
    required this.yoloBytes,
    required this.landmarkBytes,
    required this.modeName,
    required this.landmarkModelName,
    required this.detectorConf,
    required this.detectorIou,
    required this.maxDetections,
    required this.minLandmarkScore,
    required this.interpreterPoolSize,
    required this.performanceModeName,
    required this.numThreads,
  });
}

/// On-device pose detection and landmark estimation using TensorFlow Lite.
///
/// Implements a two-stage pipeline:
/// 1. YOLOv8n person detector to find bounding boxes
/// 2. BlazePose model to extract 33 body keypoints per detected person
///
/// All inference runs in a background isolate, keeping the UI thread free.
///
/// Usage:
/// ```dart
/// final detector = PoseDetector(
///   mode: PoseMode.boxesAndLandmarks,
///   landmarkModel: PoseLandmarkModel.heavy,
/// );
/// await detector.initialize();
/// final poses = await detector.detect(imageBytes);
/// await detector.dispose();
/// ```
class PoseDetector {
  static const String _packageVersion = '3.0.0';
  static const String _pipelineVersion = 'pipeline_v1';

  /// Version key for the default pose detection pipeline.
  static const String modelVersion =
      'pose_detection:$_packageVersion:mode=boxesAndLandmarks:'
      'landmarkModel=heavy:$_pipelineVersion';

  /// Builds a version key for a specific pose detector configuration.
  static String modelVersionFor({
    PoseMode mode = PoseMode.boxesAndLandmarks,
    PoseLandmarkModel landmarkModel = PoseLandmarkModel.heavy,
  }) {
    return 'pose_detection:$_packageVersion:mode=${mode.name}:'
        'landmarkModel=${landmarkModel.name}:$_pipelineVersion';
  }

  /// Detection mode controlling pipeline behavior.
  final PoseMode mode;

  /// BlazePose model variant to use for landmark extraction.
  final PoseLandmarkModel landmarkModel;

  /// Confidence threshold for person detection (0.0 to 1.0).
  final double detectorConf;

  /// IoU threshold for Non-Maximum Suppression (0.0 to 1.0).
  final double detectorIou;

  /// Maximum number of persons to detect per image.
  final int maxDetections;

  /// Minimum confidence score for landmark predictions (0.0 to 1.0).
  final double minLandmarkScore;

  /// Number of TensorFlow Lite interpreter instances in the landmark model pool.
  ///
  /// Forced to 1 when a performance delegate (XNNPACK/auto) is enabled to
  /// prevent thread contention.
  final int interpreterPoolSize;

  /// Performance configuration for TensorFlow Lite inference.
  final PerformanceConfig performanceConfig;

  _PoseDetectorWorker? _worker;

  /// Creates a pose detector with the specified configuration.
  ///
  /// Parameters:
  /// - [mode]: Detection mode. Default: [PoseMode.boxesAndLandmarks]
  /// - [landmarkModel]: BlazePose model variant. Default: [PoseLandmarkModel.heavy]
  /// - [detectorConf]: Person detection confidence threshold (0.0-1.0). Default: 0.5
  /// - [detectorIou]: NMS IoU threshold (0.0-1.0). Default: 0.45
  /// - [maxDetections]: Maximum number of persons to detect. Default: 10
  /// - [minLandmarkScore]: Minimum landmark confidence score (0.0-1.0). Default: 0.5
  /// - [interpreterPoolSize]: Number of landmark model interpreter instances (1-10). Default: 1.
  ///   Forced to 1 when a performance delegate is enabled.
  /// - [performanceConfig]: TensorFlow Lite performance configuration. Default: auto
  PoseDetector({
    this.mode = PoseMode.boxesAndLandmarks,
    this.landmarkModel = PoseLandmarkModel.heavy,
    this.detectorConf = 0.5,
    this.detectorIou = 0.45,
    this.maxDetections = 10,
    this.minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    this.performanceConfig = const PerformanceConfig(),
  }) : interpreterPoolSize = performanceConfig.mode == PerformanceMode.disabled
           ? interpreterPoolSize
           : 1;

  /// Creates and initializes a pose detector in one step.
  ///
  /// Convenience factory that combines [PoseDetector.new] and [initialize].
  /// Accepts the same parameters as the constructor.
  ///
  /// Example:
  /// ```dart
  /// final detector = await PoseDetector.create();
  ///
  /// // Equivalent to:
  /// final detector = PoseDetector();
  /// await detector.initialize();
  /// ```
  static Future<PoseDetector> create({
    PoseMode mode = PoseMode.boxesAndLandmarks,
    PoseLandmarkModel landmarkModel = PoseLandmarkModel.heavy,
    double detectorConf = 0.5,
    double detectorIou = 0.45,
    int maxDetections = 10,
    double minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
  }) async {
    final detector = PoseDetector(
      mode: mode,
      landmarkModel: landmarkModel,
      detectorConf: detectorConf,
      detectorIou: detectorIou,
      maxDetections: maxDetections,
      minLandmarkScore: minLandmarkScore,
      interpreterPoolSize: interpreterPoolSize,
      performanceConfig: performanceConfig,
    );
    await detector.initialize();
    return detector;
  }

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isReady => _worker?.isReady ?? false;

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isInitialized => isReady;

  /// Initializes the pose detector by loading TensorFlow Lite models.
  ///
  /// Must be called before [detect] or [detectFromMat].
  /// Calling [initialize] twice without [dispose] throws [StateError].
  Future<void> initialize() async {
    if (isReady) {
      throw StateError('PoseDetector already initialized');
    }

    const yoloPath =
        'packages/pose_detection/assets/models/yolov8n_float32.tflite';
    final String landmarkPath =
        'packages/pose_detection/assets/models/pose_landmark_${landmarkModel.name}.tflite';

    final results = await Future.wait([
      rootBundle.load(yoloPath),
      rootBundle.load(landmarkPath),
    ]);

    final yoloBytes = results[0].buffer.asUint8List();
    final landmarkBytes = results[1].buffer.asUint8List();

    await initializeFromBuffers(
      yoloBytes: yoloBytes,
      landmarkBytes: landmarkBytes,
    );
  }

  /// Initializes the pose detector from pre-loaded model bytes.
  ///
  /// Used when asset loading from the main isolate is not available, or when
  /// bytes have already been loaded. Spawns the background isolate with the
  /// provided model data.
  ///
  /// Parameters:
  /// - [yoloBytes]: Raw bytes of the YOLOv8n person detection TFLite model
  /// - [landmarkBytes]: Raw bytes of the BlazePose landmark TFLite model
  Future<void> initializeFromBuffers({
    required Uint8List yoloBytes,
    required Uint8List landmarkBytes,
  }) async {
    if (isReady) {
      throw StateError('PoseDetector already initialized');
    }

    final worker = _PoseDetectorWorker();

    try {
      await worker.initialize(
        yoloBytes: yoloBytes,
        landmarkBytes: landmarkBytes,
        mode: mode,
        landmarkModel: landmarkModel,
        detectorConf: detectorConf,
        detectorIou: detectorIou,
        maxDetections: maxDetections,
        minLandmarkScore: minLandmarkScore,
        interpreterPoolSize: interpreterPoolSize,
        performanceConfig: performanceConfig,
      );
    } catch (e) {
      if (worker.isReady) {
        await worker.dispose();
      }
      rethrow;
    }

    _worker = worker;
  }

  /// Detects poses in an image from raw bytes (JPEG, PNG, etc.).
  ///
  /// Decodes the image bytes using OpenCV and performs pose detection in a
  /// background isolate.
  ///
  /// Returns a list of [Pose] objects, one per detected person.
  /// Returns an empty list if image decoding fails or no persons are detected.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Pose>> detect(Uint8List imageBytes) async {
    if (!isReady) {
      throw StateError(
        'PoseDetector not initialized. Call initialize() first.',
      );
    }
    try {
      final List<dynamic> result = await _worker!.sendRequest<List<dynamic>>(
        'detect',
        {
          'bytes': TransferableTypedData.fromList([imageBytes]),
        },
      );
      return _deserializePoses(result);
    } catch (_) {
      return <Pose>[];
    }
  }

  /// Detects poses in an image file at [path].
  ///
  /// Convenience wrapper that reads the file and calls [detect].
  /// Not available on Flutter Web (uses `dart:io`).
  ///
  /// Throws [StateError] if [initialize] has not been called successfully.
  /// Throws [FileSystemException] if the file cannot be read.
  Future<List<Pose>> detectFromFilepath(String path) async {
    final bytes = await File(path).readAsBytes();
    return detect(bytes);
  }

  /// Detects poses in a pre-decoded [cv.Mat] image.
  ///
  /// The Mat's raw pixel data is extracted and transferred to the background
  /// isolate using zero-copy [TransferableTypedData]. The original Mat is NOT
  /// disposed by this method — the caller is responsible for disposal.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Pose>> detectFromMat(cv.Mat image) {
    if (!isReady) {
      throw StateError(
        'PoseDetector not initialized. Call initialize() first.',
      );
    }
    final int rows = image.rows;
    final int cols = image.cols;
    final int type = image.type.value;
    final Uint8List data = image.data;
    return detectFromMatBytes(data, width: cols, height: rows, matType: type);
  }

  /// Detects poses from raw pixel bytes without constructing a [cv.Mat] first.
  ///
  /// Bytes are transferred via zero-copy [TransferableTypedData] and the Mat
  /// is reconstructed inside the background isolate.
  ///
  /// Parameters:
  /// - [bytes]: Raw pixel data (typically BGR format, 3 bytes per pixel)
  /// - [width]: Image width in pixels
  /// - [height]: Image height in pixels
  /// - [matType]: OpenCV MatType value (default: CV_8UC3 = 16 for BGR)
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Pose>> detectFromMatBytes(
    Uint8List bytes, {
    required int width,
    required int height,
    int matType = 16,
  }) async {
    if (!isReady) {
      throw StateError(
        'PoseDetector not initialized. Call initialize() first.',
      );
    }
    final List<dynamic> result = await _worker!.sendRequest<List<dynamic>>(
      'detectMat',
      {
        'bytes': TransferableTypedData.fromList([bytes]),
        'width': width,
        'height': height,
        'matType': matType,
      },
    );
    return _deserializePoses(result);
  }

  /// Releases all resources used by the detector.
  Future<void> dispose() async {
    await _worker?.dispose();
    _worker = null;
  }

  List<Pose> _deserializePoses(List<dynamic> result) => result
      .map((map) => Pose.fromMap(Map<String, dynamic>.from(map as Map)))
      .toList();

  /// Isolate entry point: initializes [PoseDetectorCore] and listens for requests.
  @pragma('vm:entry-point')
  static void _isolateEntry(_IsolateStartupData data) async {
    final SendPort mainSendPort = data.sendPort;
    final ReceivePort workerReceivePort = ReceivePort();

    PoseDetectorCore? core;

    try {
      final yoloBytes = data.yoloBytes.materialize().asUint8List();
      final landmarkBytes = data.landmarkBytes.materialize().asUint8List();

      final mode = PoseMode.values.firstWhere((m) => m.name == data.modeName);
      final landmarkModel = PoseLandmarkModel.values.firstWhere(
        (m) => m.name == data.landmarkModelName,
      );
      final performanceMode = PerformanceMode.values.firstWhere(
        (m) => m.name == data.performanceModeName,
      );

      core = PoseDetectorCore();
      await core.initializeFromBuffers(
        yoloBytes: yoloBytes,
        landmarkBytes: landmarkBytes,
        mode: mode,
        landmarkModel: landmarkModel,
        detectorConf: data.detectorConf,
        detectorIou: data.detectorIou,
        maxDetections: data.maxDetections,
        minLandmarkScore: data.minLandmarkScore,
        interpreterPoolSize: data.interpreterPoolSize,
        performanceConfig: PerformanceConfig(
          mode: performanceMode,
          numThreads: data.numThreads,
        ),
      );

      mainSendPort.send(workerReceivePort.sendPort);
    } catch (e, st) {
      mainSendPort.send({
        'error': 'Pose detection isolate initialization failed: $e\n$st',
      });
      return;
    }

    workerReceivePort.listen((message) async {
      if (message is! Map) return;

      final int? id = message['id'] as int?;
      final String? op = message['op'] as String?;

      if (id == null || op == null) return;

      try {
        switch (op) {
          case 'detect':
            if (core == null) {
              mainSendPort.send({
                'id': id,
                'error': 'PoseDetectorCore not initialized in isolate',
              });
              return;
            }
            final ByteBuffer bb = (message['bytes'] as TransferableTypedData)
                .materialize();
            final Uint8List imageBytes = bb.asUint8List();
            final cv.Mat mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
            if (mat.isEmpty) {
              mat.dispose();
              mainSendPort.send({'id': id, 'result': <dynamic>[]});
              return;
            }
            try {
              final poses = await core!.detectDirect(
                mat,
                imageWidth: mat.cols,
                imageHeight: mat.rows,
              );
              mainSendPort.send({
                'id': id,
                'result': poses.map((p) => p.toMap()).toList(),
              });
            } finally {
              mat.dispose();
            }

          case 'detectMat':
            if (core == null) {
              mainSendPort.send({
                'id': id,
                'error': 'PoseDetectorCore not initialized in isolate',
              });
              return;
            }
            final ByteBuffer bb = (message['bytes'] as TransferableTypedData)
                .materialize();
            final Uint8List matBytes = bb.asUint8List();
            final int width = message['width'] as int;
            final int height = message['height'] as int;
            final int matTypeValue = message['matType'] as int;
            final matType = cv.MatType(matTypeValue);
            final mat = cv.Mat.fromList(height, width, matType, matBytes);
            try {
              final poses = await core!.detectDirect(
                mat,
                imageWidth: width,
                imageHeight: height,
              );
              mainSendPort.send({
                'id': id,
                'result': poses.map((p) => p.toMap()).toList(),
              });
            } finally {
              mat.dispose();
            }

          case 'dispose':
            await core?.dispose();
            core = null;
            workerReceivePort.close();
        }
      } catch (e, st) {
        mainSendPort.send({'id': id, 'error': '$e\n$st'});
      }
    });
  }
}

class _PoseDetectorWorker extends IsolateWorkerBase {
  @override
  String get workerDisposeOp => 'dispose';

  Future<void> initialize({
    required Uint8List yoloBytes,
    required Uint8List landmarkBytes,
    required PoseMode mode,
    required PoseLandmarkModel landmarkModel,
    required double detectorConf,
    required double detectorIou,
    required int maxDetections,
    required double minLandmarkScore,
    required int interpreterPoolSize,
    required PerformanceConfig performanceConfig,
  }) async {
    await initWorker(
      (sendPort) => Isolate.spawn(
        PoseDetector._isolateEntry,
        _IsolateStartupData(
          sendPort: sendPort,
          yoloBytes: TransferableTypedData.fromList([yoloBytes]),
          landmarkBytes: TransferableTypedData.fromList([landmarkBytes]),
          modeName: mode.name,
          landmarkModelName: landmarkModel.name,
          detectorConf: detectorConf,
          detectorIou: detectorIou,
          maxDetections: maxDetections,
          minLandmarkScore: minLandmarkScore,
          interpreterPoolSize: interpreterPoolSize,
          performanceModeName: performanceConfig.mode.name,
          numThreads: performanceConfig.numThreads,
        ),
        debugName: 'PoseDetector',
      ),
      timeout: const Duration(seconds: 30),
      timeoutMessage: 'Pose detection isolate initialization timed out',
    );
  }
}
