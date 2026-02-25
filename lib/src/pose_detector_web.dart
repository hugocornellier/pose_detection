import 'dart:async';
import 'dart:developer' as developer;
import 'dart:js_interop';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:web/web.dart' as web;

import 'types.dart';
import 'models/person_detector_web.dart';
import 'models/pose_landmark_model_web.dart';

/// Web implementation of the on-device pose detector.
///
/// Implements the same two-stage pipeline as the native version:
/// 1. YOLOv8n person detector to find bounding boxes
/// 2. BlazePose model to extract 33 body keypoints per detected person
///
/// Key differences from native:
/// - No opencv_dart (no cv.Mat, no detectFromMat method)
/// - No dart:io
/// - Only has [detect] (no detectFromMat)
/// - Image decoding uses HTMLImageElement via Blob/URL
/// - Person crop uses Canvas drawImage
/// - Landmark extraction uses RGBA from Canvas getImageData
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
  final YoloV8PersonDetector _yolo = YoloV8PersonDetector();
  late final PoseLandmarkModelRunner _lm;

  /// Detection mode controlling pipeline behavior.
  ///
  /// - [PoseMode.boxes]: Fast detection returning only bounding boxes (Stage 1 only)
  /// - [PoseMode.boxesAndLandmarks]: Full pipeline returning boxes + 33 landmarks (both stages)
  final PoseMode mode;

  /// BlazePose model variant to use for landmark extraction.
  ///
  /// - [PoseLandmarkModel.lite]: Fastest, good accuracy
  /// - [PoseLandmarkModel.full]: Balanced speed/accuracy
  /// - [PoseLandmarkModel.heavy]: Slowest, best accuracy (default)
  final PoseLandmarkModel landmarkModel;

  /// Confidence threshold for person detection (0.0 to 1.0).
  final double detectorConf;

  /// IoU (Intersection over Union) threshold for Non-Maximum Suppression (0.0 to 1.0).
  final double detectorIou;

  /// Maximum number of persons to detect per image.
  final int maxDetections;

  /// Minimum confidence score for landmark predictions (0.0 to 1.0).
  final double minLandmarkScore;

  /// Always 1 on web (single-threaded).
  final int interpreterPoolSize;

  /// Performance configuration (accepted for API compatibility, ignored on web).
  final PerformanceConfig performanceConfig;

  /// Whether to use native preprocessing (accepted for API compatibility, ignored on web).
  final bool useNativePreprocessing;

  bool _isInitialized = false;

  /// Canvas for person crop/resize to 256x256 for landmark extraction.
  web.HTMLCanvasElement? _cropCanvas;
  web.CanvasRenderingContext2D? _cropCtx;

  /// Creates a pose detector with the specified configuration.
  ///
  /// Parameters:
  /// - [mode]: Detection mode (boxes only or boxes + landmarks). Default: [PoseMode.boxesAndLandmarks]
  /// - [landmarkModel]: BlazePose model variant. Default: [PoseLandmarkModel.heavy]
  /// - [detectorConf]: Person detection confidence threshold (0.0-1.0). Default: 0.5
  /// - [detectorIou]: NMS IoU threshold for duplicate suppression (0.0-1.0). Default: 0.45
  /// - [maxDetections]: Maximum number of persons to detect. Default: 10
  /// - [minLandmarkScore]: Minimum landmark confidence score (0.0-1.0). Default: 0.5
  /// - [interpreterPoolSize]: Ignored on web (always 1).
  /// - [performanceConfig]: Ignored on web (always CPU/WASM).
  /// - [useNativePreprocessing]: Ignored on web (always uses Canvas API).
  PoseDetector({
    this.mode = PoseMode.boxesAndLandmarks,
    this.landmarkModel = PoseLandmarkModel.heavy,
    this.detectorConf = 0.5,
    this.detectorIou = 0.45,
    this.maxDetections = 10,
    this.minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    this.performanceConfig = PerformanceConfig.disabled,
    this.useNativePreprocessing = true,
  }) : interpreterPoolSize = 1 {
    _lm = PoseLandmarkModelRunner(poolSize: 1);
  }

  /// Initializes the pose detector by loading TensorFlow Lite models.
  ///
  /// On web, this also initializes the TFLite.js WASM runtime via [initializeWeb].
  /// Must be called before [detect].
  /// If already initialized, will dispose existing models and reinitialize.
  ///
  /// Throws an exception if model loading fails.
  Future<void> initialize() async {
    if (_isInitialized) {
      await dispose();
    }

    // Initialize TFLite.js WASM runtime
    await initializeWeb();

    await _lm.initialize(landmarkModel, performanceConfig: performanceConfig);
    await _yolo.initialize(performanceConfig: performanceConfig);

    // Create canvas for person crop/resize
    _cropCanvas = web.HTMLCanvasElement();
    _cropCanvas!.width = 256;
    _cropCanvas!.height = 256;
    _cropCtx = _cropCanvas!.getContext('2d') as web.CanvasRenderingContext2D;

    _isInitialized = true;
  }

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isInitialized => _isInitialized;

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
  /// On web, the image bytes are decoded via an HTMLImageElement using a Blob URL.
  /// Person crops are generated using Canvas drawImage, and landmark input is
  /// extracted as RGBA data from Canvas getImageData.
  ///
  /// Parameters:
  /// - [imageBytes]: Encoded image bytes (JPEG, PNG, BMP, etc.)
  ///
  /// Returns a list of [Pose] objects, one per detected person.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Pose>> detect(Uint8List imageBytes) async {
    if (!_isInitialized) {
      throw StateError(
        'PoseDetector not initialized. Call initialize() first.',
      );
    }

    // Decode image to HTMLImageElement using Canvas API
    final web.HTMLImageElement? htmlImage = await _decodeImage(imageBytes);
    if (htmlImage == null) return <Pose>[];

    final int imageWidth = htmlImage.naturalWidth;
    final int imageHeight = htmlImage.naturalHeight;

    // Stage 1: Person detection
    final List<YoloDetection> dets = await _yolo.detect(
      htmlImage,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      confThres: detectorConf,
      iouThres: detectorIou,
      maxDet: maxDetections,
      personOnly: true,
    );

    if (mode == PoseMode.boxes) {
      return _buildBoxOnlyResults(dets, imageWidth, imageHeight);
    }

    // Stage 2: Landmark extraction for each detection
    final List<Pose> results = <Pose>[];
    final web.CanvasRenderingContext2D ctx = _cropCtx!;

    for (final YoloDetection d in dets) {
      final int x1 = d.bboxXYXY[0].clamp(0.0, imageWidth.toDouble()).toInt();
      final int y1 = d.bboxXYXY[1].clamp(0.0, imageHeight.toDouble()).toInt();
      final int x2 = d.bboxXYXY[2].clamp(0.0, imageWidth.toDouble()).toInt();
      final int y2 = d.bboxXYXY[3].clamp(0.0, imageHeight.toDouble()).toInt();
      final int cropWidth = (x2 - x1).clamp(1, imageWidth);
      final int cropHeight = (y2 - y1).clamp(1, imageHeight);

      // Match the reference web demo: exact bbox crop + letterbox to 256x256.
      final double ratio = math.min(256.0 / cropHeight, 256.0 / cropWidth);
      final int resizedWidth = (cropWidth * ratio).round();
      final int resizedHeight = (cropHeight * ratio).round();
      final int padX = (256 - resizedWidth) ~/ 2;
      final int padY = (256 - resizedHeight) ~/ 2;

      ctx.fillStyle = 'rgb(114,114,114)'.toJS;
      ctx.fillRect(0, 0, 256, 256);
      ctx.drawImage(
        htmlImage,
        x1,
        y1,
        cropWidth,
        cropHeight,
        padX,
        padY,
        resizedWidth,
        resizedHeight,
      );

      // Extract RGBA pixel data for landmark model
      final web.ImageData poseImageData = ctx.getImageData(0, 0, 256, 256);
      final rgbaClamped = poseImageData.data.toDart;
      final Uint8List rgbaBytes = Uint8List.view(rgbaClamped.buffer);

      try {
        final PoseLandmarks landmarks = await _lm.runFromRgba(rgbaBytes);

        if (landmarks.score >= minLandmarkScore) {
          final List<PoseLandmark> pts = _transformLandmarksLetterbox(
            landmarks.landmarks,
            x1.toDouble(),
            y1.toDouble(),
            ratio,
            padX.toDouble(),
            padY.toDouble(),
            imageWidth,
            imageHeight,
          );
          results.add(
            Pose(
              boundingBox: BoundingBox(
                left: d.bboxXYXY[0],
                top: d.bboxXYXY[1],
                right: d.bboxXYXY[2],
                bottom: d.bboxXYXY[3],
              ),
              score: d.score,
              landmarks: pts,
              imageWidth: imageWidth,
              imageHeight: imageHeight,
            ),
          );
        } else {
          results.add(
            Pose(
              boundingBox: BoundingBox(
                left: d.bboxXYXY[0],
                top: d.bboxXYXY[1],
                right: d.bboxXYXY[2],
                bottom: d.bboxXYXY[3],
              ),
              score: d.score,
              landmarks: const <PoseLandmark>[],
              imageWidth: imageWidth,
              imageHeight: imageHeight,
            ),
          );
        }
      } catch (e, stackTrace) {
        assert(() {
          // Ignore in release mode, but surface the actual failure during debug.
          developer.log(
            'Pose landmark extraction failed on web',
            name: 'pose_detection_tflite',
            error: e,
            stackTrace: stackTrace,
          );
          return true;
        }());
        results.add(
          Pose(
            boundingBox: BoundingBox(
              left: d.bboxXYXY[0],
              top: d.bboxXYXY[1],
              right: d.bboxXYXY[2],
              bottom: d.bboxXYXY[3],
            ),
            score: d.score,
            landmarks: const <PoseLandmark>[],
            imageWidth: imageWidth,
            imageHeight: imageHeight,
          ),
        );
      }
    }

    return results;
  }

  /// Decodes encoded image bytes (JPEG, PNG, etc.) to an HTMLImageElement.
  ///
  /// Creates a Blob from the bytes, generates an object URL, and loads it
  /// into an HTMLImageElement. Returns null if loading fails.
  Future<web.HTMLImageElement?> _decodeImage(Uint8List bytes) async {
    final web.Blob blob = web.Blob([bytes.toJS].toJS);
    final String url = web.URL.createObjectURL(blob);
    try {
      final web.HTMLImageElement htmlImage = web.HTMLImageElement();
      final Completer<void> loadCompleter = Completer<void>();

      void loadHandler(web.Event _) {
        if (!loadCompleter.isCompleted) {
          loadCompleter.complete();
        }
      }

      void errorHandler(web.Event _) {
        if (!loadCompleter.isCompleted) {
          loadCompleter.completeError('Failed to load image');
        }
      }

      htmlImage.addEventListener('load', loadHandler.toJS);
      htmlImage.addEventListener('error', errorHandler.toJS);
      htmlImage.src = url;

      await loadCompleter.future;

      htmlImage.removeEventListener('load', loadHandler.toJS);
      htmlImage.removeEventListener('error', errorHandler.toJS);

      return htmlImage;
    } catch (_) {
      return null;
    } finally {
      web.URL.revokeObjectURL(url);
    }
  }

  /// Builds box-only results (no landmarks).
  List<Pose> _buildBoxOnlyResults(
    List<YoloDetection> dets,
    int imageWidth,
    int imageHeight,
  ) {
    final List<Pose> out = <Pose>[];
    for (final YoloDetection d in dets) {
      out.add(
        Pose(
          boundingBox: BoundingBox(
            left: d.bboxXYXY[0],
            top: d.bboxXYXY[1],
            right: d.bboxXYXY[2],
            bottom: d.bboxXYXY[3],
          ),
          score: d.score,
          landmarks: const <PoseLandmark>[],
          imageWidth: imageWidth,
          imageHeight: imageHeight,
        ),
      );
    }
    return out;
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
