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
class _DetectionIsolateStartupData {
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

  _DetectionIsolateStartupData({
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
/// final detector = await PoseDetector.create(
///   mode: PoseMode.boxesAndLandmarks,
///   landmarkModel: PoseLandmarkModel.heavy,
/// );
/// final poses = await detector.detect(imageBytes);
/// await detector.dispose();
/// ```
class PoseDetector {
  static const String _packageVersion = '3.1.0';
  static const String _pipelineVersion = 'pipeline_v1';
  static const String _decodeFailureErrorPrefix =
      'pose_detection_decode_failure:';

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

  _PoseDetectorWorker? _worker;

  /// Creates a pose detector instance.
  ///
  /// The detector is not ready for use until you call [initialize].
  PoseDetector();

  /// Creates and initializes a pose detector in one step.
  ///
  /// Convenience factory that combines [PoseDetector.new] and [initialize].
  /// Accepts the same parameters as [initialize].
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
    // Web-only; accepted here for API parity but ignored on native.
    bool useLiteRt = true,
    String liteRtAccelerator = 'auto',
  }) async {
    final detector = PoseDetector();
    await detector.initialize(
      mode: mode,
      landmarkModel: landmarkModel,
      detectorConf: detectorConf,
      detectorIou: detectorIou,
      maxDetections: maxDetections,
      minLandmarkScore: minLandmarkScore,
      interpreterPoolSize: interpreterPoolSize,
      performanceConfig: performanceConfig,
    );
    return detector;
  }

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isReady => _worker?.isReady ?? false;

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isInitialized => isReady;

  /// Initializes the pose detector by loading TensorFlow Lite models.
  ///
  /// Must be called before [detect] or [detectFromMat].
  /// On native platforms, calling [initialize] twice without [dispose] throws
  /// [StateError]. On web, the existing models are disposed and reinitialized.
  Future<void> initialize({
    PoseMode mode = PoseMode.boxesAndLandmarks,
    PoseLandmarkModel landmarkModel = PoseLandmarkModel.heavy,
    double detectorConf = 0.5,
    double detectorIou = 0.45,
    int maxDetections = 10,
    double minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    // Web-only; accepted here for API parity but ignored on native.
    bool useLiteRt = true,
    String liteRtAccelerator = 'auto',
  }) async {
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
      mode: mode,
      landmarkModel: landmarkModel,
      detectorConf: detectorConf,
      detectorIou: detectorIou,
      maxDetections: maxDetections,
      minLandmarkScore: minLandmarkScore,
      interpreterPoolSize: interpreterPoolSize,
      performanceConfig: performanceConfig,
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
    PoseMode mode = PoseMode.boxesAndLandmarks,
    PoseLandmarkModel landmarkModel = PoseLandmarkModel.heavy,
    double detectorConf = 0.5,
    double detectorIou = 0.45,
    int maxDetections = 10,
    double minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
  }) async {
    if (isReady) {
      throw StateError('PoseDetector already initialized');
    }

    final effectivePoolSize = performanceConfig.mode == PerformanceMode.disabled
        ? interpreterPoolSize
        : 1;

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
        interpreterPoolSize: effectivePoolSize,
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
  ///
  /// Throws [StateError] if called before [initialize].
  /// Throws [FormatException] if the image bytes cannot be decoded.
  Future<List<Pose>> detect(Uint8List imageBytes) async {
    if (!isReady) {
      throw StateError(
        'PoseDetector not initialized. Call initialize() first.',
      );
    }
    final List<dynamic> result;
    try {
      result = await _worker!.sendRequest<List<dynamic>>('detect', {
        'bytes': TransferableTypedData.fromList([imageBytes]),
      });
    } on StateError catch (error) {
      final String message = error.message;
      if (message.startsWith(_decodeFailureErrorPrefix)) {
        final detail = message.substring(_decodeFailureErrorPrefix.length);
        throw FormatException(
          detail.trim().isEmpty
              ? 'Image bytes could not be decoded.'
              : detail.trim(),
          imageBytes,
        );
      }
      rethrow;
    }
    return _deserializePoses(result);
  }

  /// Detects poses in an image file at [path].
  ///
  /// Convenience wrapper that reads the file and calls [detect].
  /// Not available on Flutter Web (uses `dart:io`).
  ///
  /// Throws [StateError] if [initialize] has not been called successfully.
  /// Throws [FileSystemException] if the file cannot be read.
  /// Throws [FormatException] if the file bytes cannot be decoded as an image.
  Future<List<Pose>> detectFromFilepath(String path) async {
    final bytes = await File(path).readAsBytes();
    return detect(bytes);
  }

  /// Detects poses in a pre-decoded [cv.Mat] image.
  ///
  /// The Mat's raw pixel data is extracted and transferred to the background
  /// isolate using zero-copy [TransferableTypedData]. The original Mat is NOT
  /// disposed by this method; the caller is responsible for disposal.
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

  /// Detects poses directly from a [CameraFrame] produced by
  /// [prepareCameraFrame].
  ///
  /// The frame's YUV/BGRA/RGBA to BGR colour conversion and any optional
  /// rotation happen inside the detection isolate, not on the calling thread.
  /// Use this from a `CameraController.startImageStream` callback to keep the
  /// UI thread free of OpenCV work.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Pose>> detectFromCameraFrame(
    CameraFrame frame, {
    int? maxDim,
  }) async {
    if (!isReady) {
      throw StateError(
        'PoseDetector not initialized. Call initialize() first.',
      );
    }
    final List<dynamic> result = await _worker!.sendRequest<List<dynamic>>(
      'detectCameraFrame',
      {
        'bytes': TransferableTypedData.fromList([frame.bytes]),
        'width': frame.width,
        'height': frame.height,
        'strideCols': frame.strideCols,
        'conversion': frame.conversion.index,
        'rotation': frame.rotation?.index,
        'maxDim': maxDim,
      },
    );
    return _deserializePoses(result);
  }

  /// One-call wrapper for live camera streams: takes a `CameraImage`-shaped
  /// object directly (any object exposing `width`, `height`, and `planes` with
  /// `bytes` / `bytesPerRow` / `bytesPerPixel`).
  ///
  /// The frame is packed into a transferable [CameraFrame] on the calling
  /// isolate. OpenCV colour conversion, optional rotation, optional downscale,
  /// and inference run inside the detection isolate.
  ///
  /// Returns an empty list (not an error) when the plane shape can't be
  /// decoded. Throws at runtime if [cameraImage] doesn't expose the expected
  /// shape.
  ///
  /// [isBgra] selects BGRA (macOS, default) vs. RGBA (Linux) for the desktop
  /// single-plane path; ignored for YUV input.
  ///
  /// Throws [StateError] if [initialize] has not been called.
  Future<List<Pose>> detectFromCameraImage(
    Object cameraImage, {
    CameraFrameRotation? rotation,
    bool isBgra = true,
    int? maxDim,
  }) async {
    if (!isReady) {
      throw StateError(
        'PoseDetector not initialized. Call initialize() first.',
      );
    }
    final frame = prepareCameraFrameFromImage(
      cameraImage,
      rotation: rotation,
      isBgra: isBgra,
    );
    if (frame == null) return const <Pose>[];
    return detectFromCameraFrame(frame, maxDim: maxDim);
  }

  /// Releases all resources used by the detector.
  Future<void> dispose() async {
    final worker = _worker;
    _worker = null;
    if (worker == null) return;

    // Graceful shutdown: send 'dispose' as an RPC and await the ack before
    // letting the worker force-kill the isolate. flutter_litert's
    // IsolateWorkerBase calls Isolate.kill(priority: immediate) which races
    // past any queued 'dispose' message, so without this round-trip the
    // isolate dies before it can free its TFLite interpreters. On Android
    // each detector leaks ~10-26MB of native memory; after ~13 creates the
    // low-memory killer reaps the test process.
    try {
      await worker
          .sendRequest<dynamic>('dispose', const <String, dynamic>{})
          .timeout(const Duration(seconds: 5));
    } catch (_) {
      // Best-effort: fall through to the force-kill below.
    }
    await worker.dispose();
  }

  List<Pose> _deserializePoses(List<dynamic> result) => result
      .map((map) => Pose.fromMap(Map<String, dynamic>.from(map as Map)))
      .toList();

  /// Isolate entry point: initializes [PoseDetectorCore] and listens for requests.
  @pragma('vm:entry-point')
  static void _detectionIsolateEntry(_DetectionIsolateStartupData data) async {
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
            cv.Mat? decoded;
            try {
              decoded = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
              if (decoded.isEmpty) {
                throw const FormatException(
                  'Image bytes could not be decoded.',
                );
              }
            } catch (e) {
              decoded?.dispose();
              mainSendPort.send({
                'id': id,
                'error':
                    '$_decodeFailureErrorPrefix Image bytes could not be decoded: $e',
              });
              return;
            }
            final cv.Mat mat = decoded;
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

          case 'detectCameraFrame':
            if (core == null) {
              mainSendPort.send({
                'id': id,
                'error': 'PoseDetectorCore not initialized in isolate',
              });
              return;
            }
            final Uint8List frameBytes =
                (message['bytes'] as TransferableTypedData)
                    .materialize()
                    .asUint8List();
            final frameMat = _matFromCameraFrameMessage(message, frameBytes);
            try {
              final poses = await core!.detectDirect(
                frameMat,
                imageWidth: frameMat.cols,
                imageHeight: frameMat.rows,
              );
              mainSendPort.send({
                'id': id,
                'result': poses.map((p) => p.toMap()).toList(),
              });
            } finally {
              frameMat.dispose();
            }

          case 'dispose':
            await core?.dispose();
            core = null;
            // ACK the dispose so the main side can await it before force-
            // killing the isolate. See PoseDetector.dispose().
            mainSendPort.send({'id': id, 'result': null});
            workerReceivePort.close();
        }
      } catch (e, st) {
        mainSendPort.send({'id': id, 'error': '$e\n$st'});
      }
    });
  }

  /// Decodes a [CameraFrame] isolate message into a 3-channel BGR [cv.Mat],
  /// applying the conversion (YUV→BGR or BGRA/RGBA→BGR, with optional stride
  /// crop) and any requested rotation. Runs inside the detection isolate.
  ///
  /// Op ordering is tuned to keep the big allocations tiny: for BGRA frames we
  /// resize and rotate on the 4-channel buffer and defer `cvtColor` to the end
  /// (so it converts the post-resize ~maxDim buffer, not full-res). For YUV we
  /// must `cvtColor` first because the packed layout isn't resizable, but we
  /// then resize before rotating so the rotate runs on the small BGR buffer.
  /// Output is byte-identical to the rotate→resize→cvtColor order because
  /// `cv.rotate` 90/180/270 is a lossless permutation, `cv.resize`
  /// (`INTER_LINEAR`) interpolates each channel independently, and the
  /// BGRA→BGR conversion is a per-pixel alpha drop.
  static cv.Mat _matFromCameraFrameMessage(Map message, Uint8List bytes) {
    final int width = message['width'] as int;
    final int height = message['height'] as int;
    final int strideCols = message['strideCols'] as int;
    final conversion =
        CameraFrameConversion.values[message['conversion'] as int];
    final int? rotationIndex = message['rotation'] as int?;
    final int? maxDim = message['maxDim'] as int?;

    int? rotateFlag() {
      if (rotationIndex == null) return null;
      return switch (CameraFrameRotation.values[rotationIndex]) {
        CameraFrameRotation.cw90 => cv.ROTATE_90_CLOCKWISE,
        CameraFrameRotation.cw180 => cv.ROTATE_180,
        CameraFrameRotation.cw270 => cv.ROTATE_90_COUNTERCLOCKWISE,
      };
    }

    cv.Mat maybeResize(cv.Mat m) {
      if (maxDim == null || (m.cols <= maxDim && m.rows <= maxDim)) return m;
      final double scale = maxDim / (m.cols > m.rows ? m.cols : m.rows);
      final resized = cv.resize(m, (
        (m.cols * scale).toInt(),
        (m.rows * scale).toInt(),
      ), interpolation: cv.INTER_LINEAR);
      m.dispose();
      return resized;
    }

    cv.Mat maybeRotate(cv.Mat m) {
      final flag = rotateFlag();
      if (flag == null) return m;
      final rotated = cv.rotate(m, flag);
      m.dispose();
      return rotated;
    }

    switch (conversion) {
      case CameraFrameConversion.bgra2bgr:
      case CameraFrameConversion.rgba2bgr:
        final bgraOrRgba = cv.Mat.fromList(
          height,
          strideCols,
          cv.MatType.CV_8UC4,
          bytes,
        );
        cv.Mat current = strideCols != width
            ? bgraOrRgba.region(cv.Rect(0, 0, width, height))
            : bgraOrRgba;

        if (maxDim != null &&
            (current.cols > maxDim || current.rows > maxDim)) {
          final double scale =
              maxDim /
              (current.cols > current.rows ? current.cols : current.rows);
          final resized = cv.resize(current, (
            (current.cols * scale).toInt(),
            (current.rows * scale).toInt(),
          ), interpolation: cv.INTER_LINEAR);
          if (!identical(current, bgraOrRgba)) current.dispose();
          current = resized;
        }

        final flag = rotateFlag();
        if (flag != null) {
          final rotated = cv.rotate(current, flag);
          if (!identical(current, bgraOrRgba)) current.dispose();
          current = rotated;
        }

        final cvtCode = conversion == CameraFrameConversion.bgra2bgr
            ? cv.COLOR_BGRA2BGR
            : cv.COLOR_RGBA2BGR;
        final bgr = cv.cvtColor(current, cvtCode);
        if (!identical(current, bgraOrRgba)) current.dispose();
        bgraOrRgba.dispose();
        return bgr;

      case CameraFrameConversion.yuv2bgrNv12:
      case CameraFrameConversion.yuv2bgrNv21:
      case CameraFrameConversion.yuv2bgrI420:
        final yuvMat = cv.Mat.fromList(
          height + height ~/ 2,
          width,
          cv.MatType.CV_8UC1,
          bytes,
        );
        final cvtCode = switch (conversion) {
          CameraFrameConversion.yuv2bgrNv12 => cv.COLOR_YUV2BGR_NV12,
          CameraFrameConversion.yuv2bgrNv21 => cv.COLOR_YUV2BGR_NV21,
          CameraFrameConversion.yuv2bgrI420 => cv.COLOR_YUV2BGR_I420,
          _ => cv.COLOR_YUV2BGR_NV12,
        };
        cv.Mat current = cv.cvtColor(yuvMat, cvtCode);
        yuvMat.dispose();
        current = maybeResize(current);
        current = maybeRotate(current);
        return current;
    }
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
        PoseDetector._detectionIsolateEntry,
        _DetectionIsolateStartupData(
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
