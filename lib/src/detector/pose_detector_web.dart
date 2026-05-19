// ignore_for_file: implementation_imports

import 'dart:async';
import 'dart:developer' as developer;
import 'dart:js_interop';
import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:web/web.dart' as web;

import '../types.dart';
import '../util/pose_helpers.dart';
import '../models/person_detector_web.dart';
import '../models/pose_landmark_model_web.dart';
import 'package:flutter_litert/src/web/web_detector_utils.dart'
    show decodeBitmap, WebGpuFallback;

/// Per-stage timing accumulator (microseconds). Populated by `detect()` when
/// `PoseDetector.debugTimings` is true; reset per call.
class WebDetectTimings {
  /// Image decode (createImageBitmap) duration.
  int decodeUs = 0;

  /// YOLO pre-processing (canvas + getImageData + RGBA→Float32).
  int yoloPreUs = 0;

  /// YOLO inference (interpreter.run + readback).
  int yoloInferUs = 0;

  /// BlazePose pre-processing summed across detections.
  int lmPreUs = 0;

  /// BlazePose inference summed across detections.
  int lmInferUs = 0;

  /// Post-processing (NMS / Pose construction). Reserved; not currently
  /// populated because postProcessing on the LiteRT path is folded into
  /// the timed inference stage.
  int postUs = 0;

  /// Total wall-clock duration of the `detect()` call.
  int totalUs = 0;

  /// Number of detections produced by YOLO for this call.
  int detections = 0;

  /// Returns the timings as a JSON-friendly map of microsecond values.
  Map<String, int> toJsonUs() => {
    'decode_us': decodeUs,
    'yolo_pre_us': yoloPreUs,
    'yolo_infer_us': yoloInferUs,
    'lm_pre_us': lmPreUs,
    'lm_infer_us': lmInferUs,
    'post_us': postUs,
    'total_us': totalUs,
    'detections': detections,
  };
}

/// Web implementation of the on-device pose detector.
///
/// Implements the same two-stage pipeline as the native version:
/// 1. YOLOv8n person detector to find bounding boxes
/// 2. BlazePose model to extract 33 body keypoints per detected person
///
/// Key differences from native:
/// - No opencv_dart or cv.Mat support
/// - No dart:io
/// - API-parity methods such as [detectFromMat], [detectFromCameraFrame], and
///   [detectFromCameraImage] throw [UnsupportedError]
/// - Image decoding uses browser image decode (`createImageBitmap`)
/// - Person crop uses Canvas drawImage
/// - Landmark extraction uses RGBA from Canvas getImageData
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
class PoseDetector with WebGpuFallback {
  static const String _packageVersion = '3.1.0';
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

  final YoloV8PersonDetector _yolo = YoloV8PersonDetector();
  late final PoseLandmarkModelRunner _lm;

  PoseMode _mode = PoseMode.boxesAndLandmarks;
  PoseLandmarkModel _landmarkModel = PoseLandmarkModel.heavy;
  double _detectorConf = 0.5;
  double _detectorIou = 0.45;
  int _maxDetections = 10;
  double _minLandmarkScore = 0.5;
  PerformanceConfig _performanceConfig = PerformanceConfig.disabled;
  bool _useLiteRt = true;
  String _liteRtAccelerator = 'auto';

  bool _isInitialized = false;

  /// Canvas for person crop/resize to 256x256 for landmark extraction.
  web.HTMLCanvasElement? _cropCanvas;
  web.CanvasRenderingContext2D? _cropCtx;

  /// Last-call per-stage timings (set when [debugTimings] is true).
  WebDetectTimings? lastTimings;

  /// When true, [detect] populates [lastTimings].
  bool debugTimings = false;

  /// Creates a pose detector instance.
  ///
  /// The detector is not ready for use until you call [initialize].
  PoseDetector() {
    // liteRtParallelism: 2 loads two CompiledModel instances on the
    // LiteRT.js path so multiple landmark inferences can be in-flight on
    // distinct interpreters. No-op for the legacy tflite-js path.
    _lm = PoseLandmarkModelRunner(poolSize: 1, liteRtParallelism: 2);
  }

  /// Creates and initializes a pose detector in one step.
  ///
  /// Convenience factory that combines [PoseDetector.new] and [initialize].
  /// Accepts the same parameters as [initialize].
  static Future<PoseDetector> create({
    PoseMode mode = PoseMode.boxesAndLandmarks,
    PoseLandmarkModel landmarkModel = PoseLandmarkModel.heavy,
    double detectorConf = 0.5,
    double detectorIou = 0.45,
    int maxDetections = 10,
    double minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = PerformanceConfig.disabled,
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
      performanceConfig: performanceConfig,
      useLiteRt: useLiteRt,
      liteRtAccelerator: liteRtAccelerator,
    );
    return detector;
  }

  /// Initializes the pose detector by loading TensorFlow Lite models.
  ///
  /// On web, this initializes the selected browser runtime. The LiteRT.js path
  /// is the default; the legacy tflite-js path still uses [initializeWeb].
  /// Must be called before [detect].
  /// If already initialized, will dispose existing models and reinitialize.
  ///
  /// Throws an exception if model loading fails.
  Future<void> initialize({
    PoseMode mode = PoseMode.boxesAndLandmarks,
    PoseLandmarkModel landmarkModel = PoseLandmarkModel.heavy,
    double detectorConf = 0.5,
    double detectorIou = 0.45,
    int maxDetections = 10,
    double minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = PerformanceConfig.disabled,
    bool useLiteRt = true,
    String liteRtAccelerator = 'auto',
  }) async {
    if (_isInitialized) {
      await dispose();
    }

    _mode = mode;
    _landmarkModel = landmarkModel;
    _detectorConf = detectorConf;
    _detectorIou = detectorIou;
    _maxDetections = maxDetections;
    _minLandmarkScore = minLandmarkScore;
    _performanceConfig = performanceConfig;
    _useLiteRt = useLiteRt;
    _liteRtAccelerator = liteRtAccelerator;

    // Initialize TFLite.js WASM runtime (no-op on the LiteRT.js path;
    // kept for the legacy tflite-js path when useLiteRt is false).
    await initializeWeb();

    await _lm.initialize(
      _landmarkModel,
      performanceConfig: _performanceConfig,
      useLiteRt: _useLiteRt,
      liteRtAccelerator: _liteRtAccelerator,
    );
    await _yolo.initialize(
      performanceConfig: _performanceConfig,
      useLiteRt: _useLiteRt,
      liteRtAccelerator: _liteRtAccelerator,
    );

    // Create canvas for person crop/resize
    _cropCanvas = web.HTMLCanvasElement();
    _cropCanvas!.width = 256;
    _cropCanvas!.height = 256;
    _cropCtx = _cropCanvas!.getContext('2d') as web.CanvasRenderingContext2D;

    _isInitialized = true;
  }

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isReady => _isInitialized;

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isInitialized => _isInitialized;

  /// The accelerator currently in use across model runners (`'webgpu'` /
  /// `'wasm'`), or null on the legacy tflite-js path or before initialization
  /// completes. Returns `'webgpu'` if any runner is still on WebGPU so runtime
  /// fallback remains enabled for mixed WebGPU/WASM compile outcomes.
  ///
  /// May change at runtime if a `LiteRtRuntimeError` fires on the WebGPU
  /// path and the detector swaps to WASM.
  @override
  String? get activeAccelerator {
    final String? yolo = _yolo.activeAccelerator;
    final String? landmarks = _lm.activeAccelerator;
    if (yolo == 'webgpu' || landmarks == 'webgpu') return 'webgpu';
    return yolo ?? landmarks;
  }

  @override
  Future<void> swapToWasm() async {
    _liteRtAccelerator = 'wasm';
    try {
      await _yolo.dispose();
      await _lm.dispose();
    } catch (_) {
      // Best-effort: an interpreter that already errored may not dispose
      // cleanly. Continue to re-init regardless.
    }
    await _lm.initialize(
      _landmarkModel,
      performanceConfig: _performanceConfig,
      useLiteRt: true,
      liteRtAccelerator: 'wasm',
    );
    await _yolo.initialize(
      performanceConfig: _performanceConfig,
      useLiteRt: true,
      liteRtAccelerator: 'wasm',
    );
  }

  /// Releases all resources used by the detector.
  ///
  /// Call this when done using the detector to free memory.
  /// After calling dispose, you must call [initialize] again before detection.
  Future<void> dispose() async {
    await _yolo.dispose();
    await _lm.dispose();
    _cropCanvas = null;
    _cropCtx = null;
    _isInitialized = false;
  }

  /// Detects poses from encoded image bytes (JPEG, PNG, etc.).
  ///
  /// On web, the image bytes are decoded with browser image decode
  /// (`createImageBitmap`).
  /// Person crops are generated using Canvas drawImage, and landmark input is
  /// extracted as RGBA data from Canvas getImageData.
  ///
  /// Parameters:
  /// - [imageBytes]: Encoded image bytes (JPEG, PNG, BMP, etc.)
  ///
  /// Returns a list of [Pose] objects, one per detected person.
  /// On web, returns an empty list if the image bytes cannot be decoded
  /// (browser image decode failure does not throw through this API).
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Pose>> detect(Uint8List imageBytes) async {
    if (!_isInitialized) {
      throw StateError(
        'PoseDetector not initialized. Call initialize() first.',
      );
    }
    return withFallback(() => _detectInner(imageBytes));
  }

  Future<List<Pose>> _detectInner(Uint8List imageBytes) async {
    final WebDetectTimings? t = debugTimings ? WebDetectTimings() : null;
    final Stopwatch totalSw = (t != null)
        ? (Stopwatch()..start())
        : Stopwatch();
    final Stopwatch sw = Stopwatch();

    // Decode image to ImageBitmap (off-thread, no load-event roundtrip).
    if (t != null) sw.start();
    final web.ImageBitmap? bitmap = await decodeBitmap(imageBytes);
    if (t != null) {
      sw.stop();
      t.decodeUs = sw.elapsedMicroseconds;
      sw.reset();
    }
    if (bitmap == null) {
      if (t != null) {
        totalSw.stop();
        t.totalUs = totalSw.elapsedMicroseconds;
        lastTimings = t;
      }
      return <Pose>[];
    }

    final int imageWidth = bitmap.width;
    final int imageHeight = bitmap.height;

    // Stage 1: Person detection
    if (t != null) sw.start();
    final List<Detection> dets = await _yolo.detect(
      bitmap,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      confThres: _detectorConf,
      iouThres: _detectorIou,
      maxDet: _maxDetections,
      personOnly: true,
      timingPreUs: t == null ? null : (v) => t.yoloPreUs = v,
      timingInferUs: t == null ? null : (v) => t.yoloInferUs = v,
    );
    if (t != null) {
      sw.stop();
      sw.reset();
    }

    if (_mode == PoseMode.boxes) {
      if (t != null) {
        totalSw.stop();
        t.totalUs = totalSw.elapsedMicroseconds;
        t.detections = dets.length;
        lastTimings = t;
      }
      return buildBoxOnlyPoses(dets, imageWidth, imageHeight);
    }

    // Stage 2: Landmark extraction for each detection.
    //
    // Pipeline pattern: while the GPU is computing landmarks for detection N,
    // we preprocess detection N+1 on the CPU. We hold at most one in-flight
    // future and rotate between two buffer slots so a fresh preprocess does
    // not stomp the in-flight tensor's data.
    final List<Pose> results = <Pose>[];
    final web.CanvasRenderingContext2D ctx = _cropCtx!;

    final List<({Float32List input, Float32List landmarks, Float32List score})>
    slots = (dets.length > 1)
        ? <({Float32List input, Float32List landmarks, Float32List score})>[
            _lm.allocateOwnedBuffers(),
            _lm.allocateOwnedBuffers(),
          ]
        : const <
            ({Float32List input, Float32List landmarks, Float32List score})
          >[];

    Future<PoseLandmarks>? prevFuture;
    Detection? prevDet;
    double prevRatio = 0;
    int prevPadX = 0, prevPadY = 0;
    int prevX1 = 0, prevY1 = 0;
    Stopwatch? prevInferSw;

    for (int i = 0; i < dets.length; i++) {
      final Detection d = dets[i];
      final int x1 = d.bboxXYXY[0].clamp(0.0, imageWidth.toDouble()).toInt();
      final int y1 = d.bboxXYXY[1].clamp(0.0, imageHeight.toDouble()).toInt();
      final int x2 = d.bboxXYXY[2].clamp(0.0, imageWidth.toDouble()).toInt();
      final int y2 = d.bboxXYXY[3].clamp(0.0, imageHeight.toDouble()).toInt();
      final int cropWidth = (x2 - x1).clamp(1, imageWidth);
      final int cropHeight = (y2 - y1).clamp(1, imageHeight);

      // Match the reference web demo: exact bbox crop + letterbox to 256x256.
      final lb = computeLetterboxParams(
        srcWidth: cropWidth,
        srcHeight: cropHeight,
        targetWidth: 256,
        targetHeight: 256,
      );
      final double ratio = lb.scale;
      final int padX = lb.padLeft;
      final int padY = lb.padTop;
      final int resizedWidth = lb.newWidth;
      final int resizedHeight = lb.newHeight;

      if (t != null) sw.start();
      ctx.fillStyle = 'rgb(114,114,114)'.toJS;
      ctx.fillRect(0, 0, 256, 256);
      ctx.drawImage(
        bitmap,
        x1,
        y1,
        cropWidth,
        cropHeight,
        padX,
        padY,
        resizedWidth,
        resizedHeight,
      );

      // Extract RGBA pixel data for landmark model.
      final web.ImageData poseImageData = ctx.getImageData(0, 0, 256, 256);
      final rgbaClamped = poseImageData.data.toDart;
      final Uint8List rgbaBytes = Uint8List.view(rgbaClamped.buffer);
      if (t != null) {
        sw.stop();
        t.lmPreUs += sw.elapsedMicroseconds;
        sw.reset();
      }

      // Fire inference. With slots, runFromRgba's synchronous part copies
      // RGBA → Float32 into the slot's buffer and uploads to the LiteRT.js
      // tensor before yielding, so we can reuse buffers in a 2-slot rotation
      // safely (slot reused at i+2 only after future at i+1 awaited).
      Future<PoseLandmarks> future;
      Stopwatch? inferSw;
      if (t != null) inferSw = Stopwatch()..start();
      if (slots.isEmpty) {
        future = _lm.runFromRgba(rgbaBytes);
      } else {
        final slot = slots[i & 1];
        future = _lm.runFromRgba(
          rgbaBytes,
          ownedInput: slot.input,
          ownedLandmarks: slot.landmarks,
          ownedScore: slot.score,
        );
      }

      // While that runs on GPU, await the previous iteration's result and
      // turn it into a Pose.
      if (prevFuture != null) {
        await _drainPipelined(
          prevFuture,
          prevDet!,
          prevRatio,
          prevPadX,
          prevPadY,
          prevX1,
          prevY1,
          imageWidth,
          imageHeight,
          results,
          t,
          prevInferSw,
        );
      }

      prevFuture = future;
      prevDet = d;
      prevRatio = ratio;
      prevPadX = padX;
      prevPadY = padY;
      prevX1 = x1;
      prevY1 = y1;
      prevInferSw = inferSw;
    }

    if (prevFuture != null) {
      await _drainPipelined(
        prevFuture,
        prevDet!,
        prevRatio,
        prevPadX,
        prevPadY,
        prevX1,
        prevY1,
        imageWidth,
        imageHeight,
        results,
        t,
        prevInferSw,
      );
    }

    bitmap.close();
    if (t != null) {
      totalSw.stop();
      t.totalUs = totalSw.elapsedMicroseconds;
      t.detections = dets.length;
      lastTimings = t;
    }
    return results;
  }

  Future<void> _drainPipelined(
    Future<PoseLandmarks> future,
    Detection d,
    double ratio,
    int padX,
    int padY,
    int cropX,
    int cropY,
    int imageWidth,
    int imageHeight,
    List<Pose> out,
    WebDetectTimings? t,
    Stopwatch? inferSw,
  ) async {
    try {
      final PoseLandmarks landmarks = await future;
      if (t != null && inferSw != null) {
        inferSw.stop();
        t.lmInferUs += inferSw.elapsedMicroseconds;
      }
      if (landmarks.score >= _minLandmarkScore) {
        final List<PoseLandmark> pts = _transformLandmarksLetterbox(
          landmarks.landmarks,
          cropX.toDouble(),
          cropY.toDouble(),
          ratio,
          padX.toDouble(),
          padY.toDouble(),
          imageWidth,
          imageHeight,
        );
        out.add(
          Pose(
            boundingBox: BoundingBox.ltrb(
              d.bboxXYXY[0],
              d.bboxXYXY[1],
              d.bboxXYXY[2],
              d.bboxXYXY[3],
            ),
            score: d.score,
            landmarks: pts,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
          ),
        );
      } else {
        out.add(buildBoxOnlyPose(d, imageWidth, imageHeight));
      }
    } catch (e, stackTrace) {
      assert(() {
        developer.log(
          'Pose landmark extraction failed on web',
          name: 'pose_detection',
          error: e,
          stackTrace: stackTrace,
        );
        return true;
      }());
      out.add(buildBoxOnlyPose(d, imageWidth, imageHeight));
    }
  }

  /// Not supported on web. Use [detect] with encoded image bytes instead.
  Future<List<Pose>> detectFromFilepath(String path) {
    throw UnsupportedError(
      'detectFromFilepath is not supported on web. Use detect(imageBytes) instead.',
    );
  }

  /// Not supported on web. Use [detect] with encoded image bytes instead.
  Future<List<Pose>> detectFromMat(Object mat) {
    throw UnsupportedError(
      'detectFromMat is not supported on web. Use detect(imageBytes) instead.',
    );
  }

  /// Not supported on web. Use [detect] with encoded image bytes instead.
  Future<List<Pose>> detectFromMatBytes(
    Uint8List bytes, {
    required int width,
    required int height,
    int matType = 16,
  }) {
    throw UnsupportedError(
      'detectFromMatBytes is not supported on web. Use detect(imageBytes) instead.',
    );
  }

  /// Detects poses from a [CameraFrame]. Not supported on web.
  Future<List<Pose>> detectFromCameraFrame(CameraFrame frame, {int? maxDim}) {
    throw UnsupportedError(
      'detectFromCameraFrame is not supported on web. Use detect(imageBytes) instead.',
    );
  }

  /// Detects poses from a CameraImage-shaped object. Not supported on web.
  Future<List<Pose>> detectFromCameraImage(
    Object cameraImage, {
    CameraFrameRotation? rotation,
    bool? isBgra,
    int? maxDim,
  }) {
    throw UnsupportedError(
      'detectFromCameraImage is not supported on web. Use detect(imageBytes) instead.',
    );
  }

  /// Transforms letterboxed 256x256 normalized landmarks back to original image space.
  ///
  /// Inverse of the exact-bbox crop + letterbox resize used before the landmark model.
  ///
  /// Parameters:
  /// - [landmarks]: Landmarks with x/y in [0, 1] normalized space
  /// - [cropX]: X origin of the crop region in original image
  /// - [cropY]: Y origin of the crop region in original image
  /// - [ratio]: Resize ratio used to fit the crop into 256x256
  /// - [padX]: Horizontal letterbox padding in the 256x256 input
  /// - [padY]: Vertical letterbox padding in the 256x256 input
  /// - [imageWidth]: Original image width for clamping
  /// - [imageHeight]: Original image height for clamping
  List<PoseLandmark> _transformLandmarksLetterbox(
    List<PoseLandmark> landmarks,
    double cropX,
    double cropY,
    double ratio,
    double padX,
    double padY,
    int imageWidth,
    int imageHeight,
  ) {
    final List<PoseLandmark> pts = <PoseLandmark>[];
    for (final PoseLandmark lm in landmarks) {
      final double xInput = lm.x * 256.0;
      final double yInput = lm.y * 256.0;
      final double xContent = (xInput - padX) / ratio;
      final double yContent = (yInput - padY) / ratio;
      final double xOrig = (cropX + xContent).clamp(0.0, imageWidth.toDouble());
      final double yOrig = (cropY + yContent).clamp(
        0.0,
        imageHeight.toDouble(),
      );
      pts.add(
        PoseLandmark(
          type: lm.type,
          x: xOrig,
          y: yOrig,
          z: lm.z,
          visibility: lm.visibility,
        ),
      );
    }
    return pts;
  }
}
