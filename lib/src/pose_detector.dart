import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:meta/meta.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'types.dart';
import 'image_utils.dart';
import 'native_image_utils.dart';
import 'person_detector.dart';
import 'pose_landmark_model.dart';

/// Helper class to store preprocessing data for each detected person.
///
/// Contains the detection info, preprocessed image, and transformation parameters
/// needed to convert landmark coordinates back to original image space.
class _PersonCropData {
  /// The original YOLO detection result.
  final YoloDetection detection;

  /// The letterboxed 256x256 image ready for landmark extraction (Dart path).
  final img.Image? letterboxed;

  /// The resized/letterboxed 256x256 cv.Mat ready for landmark extraction (Native path).
  final cv.Mat? letterboxedMat;

  /// Scale ratio used in letterbox preprocessing (uniform scale).
  /// For resize mode, this is set to 1.0 and cropWidth/cropHeight are used instead.
  final double scaleRatio;

  /// Left padding added during letterboxing (0 for resize mode).
  final int padLeft;

  /// Top padding added during letterboxing (0 for resize mode).
  final int padTop;

  /// X coordinate of crop origin in original image.
  final int cropX;

  /// Y coordinate of crop origin in original image.
  final int cropY;

  /// Width of crop in original image (for resize mode inverse transform).
  final int cropWidth;

  /// Height of crop in original image (for resize mode inverse transform).
  final int cropHeight;

  /// Whether this crop was preprocessed with resize (true) or letterbox (false).
  final bool useResize;

  _PersonCropData({
    required this.detection,
    this.letterboxed,
    this.letterboxedMat,
    required this.scaleRatio,
    required this.padLeft,
    required this.padTop,
    required this.cropX,
    required this.cropY,
    this.cropWidth = 0,
    this.cropHeight = 0,
    this.useResize = false,
  });

  /// Disposes native resources if using native preprocessing.
  void dispose() {
    letterboxedMat?.dispose();
  }
}

/// On-device pose detection and landmark estimation using TensorFlow Lite.
///
/// Implements a two-stage pipeline:
/// 1. YOLOv8n person detector to find bounding boxes
/// 2. BlazePose model to extract 33 body keypoints per detected person
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

  /// Test-only override for image decoding behavior.
  ///
  /// When set, replaces the default image decoder with a custom function.
  /// Used in tests to simulate decoding failures or unusual image conditions.
  @visibleForTesting
  @Deprecated(
    'Will be removed in 2.0.0. Use detectOnMat with cv.imdecode instead.',
  )
  static img.Image? Function(Uint8List bytes)? imageDecoderOverride;

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
  ///
  /// Only detections with confidence scores above this threshold are returned.
  /// Lower values detect more persons but may include false positives.
  /// Default: 0.5
  final double detectorConf;

  /// IoU (Intersection over Union) threshold for Non-Maximum Suppression (0.0 to 1.0).
  ///
  /// Controls duplicate detection suppression. Lower values are more aggressive
  /// at removing overlapping boxes.
  /// Default: 0.45
  final double detectorIou;

  /// Maximum number of persons to detect per image.
  ///
  /// Limits the number of detections returned, keeping only the highest
  /// confidence detections.
  /// Default: 10
  final int maxDetections;

  /// Minimum confidence score for landmark predictions (0.0 to 1.0).
  ///
  /// Poses with landmark scores below this threshold are filtered out.
  /// Only used when [mode] is [PoseMode.boxesAndLandmarks].
  /// Default: 0.5
  final double minLandmarkScore;

  /// Number of TensorFlow Lite interpreter instances in the landmark model pool.
  ///
  /// **IMPORTANT:** When XNNPACK is enabled (via [performanceConfig]), this is
  /// automatically forced to 1 to prevent thread contention and performance spikes.
  ///
  /// Controls the degree of parallelism for landmark extraction when multiple
  /// people are detected. Only relevant when XNNPACK is disabled.
  ///
  /// **Performance characteristics:**
  /// - With XNNPACK enabled: Always pool=1 (forced for stability)
  /// - With XNNPACK disabled: Can use higher pool sizes for parallelism
  ///
  /// **Memory usage:** ~10MB × poolSize for landmark model instances.
  ///
  /// Default: 1 (optimal for XNNPACK stability)
  final int interpreterPoolSize;

  /// Performance configuration for TensorFlow Lite inference.
  ///
  /// Controls CPU/GPU acceleration via delegates. Default is no acceleration
  /// for backward compatibility.
  ///
  /// Use [PerformanceConfig.xnnpack()] for 2-5x CPU speedup.
  ///
  /// Example:
  /// ```dart
  /// // Default (no acceleration)
  /// final detector = PoseDetector();
  ///
  /// // XNNPACK with auto thread detection (recommended)
  /// final detector = PoseDetector(
  ///   performanceConfig: PerformanceConfig.xnnpack(),
  /// );
  ///
  /// // XNNPACK with custom threads
  /// final detector = PoseDetector(
  ///   performanceConfig: PerformanceConfig.xnnpack(numThreads: 2),
  /// );
  /// ```
  final PerformanceConfig performanceConfig;

  /// Whether to use native OpenCV preprocessing for faster image processing.
  ///
  /// When enabled, uses SIMD-accelerated OpenCV operations for:
  /// - Letterbox preprocessing (5-15x faster)
  /// - Image cropping (10-30x faster)
  /// - Tensor conversion (3-5x faster)
  ///
  /// Default: true (recommended for best performance)
  final bool useNativePreprocessing;

  bool _isInitialized = false;

  /// Creates a pose detector with the specified configuration.
  ///
  /// Parameters:
  /// - [mode]: Detection mode (boxes only or boxes + landmarks). Default: [PoseMode.boxesAndLandmarks]
  /// - [landmarkModel]: BlazePose model variant. Default: [PoseLandmarkModel.heavy]
  /// - [detectorConf]: Person detection confidence threshold (0.0-1.0). Default: 0.5
  /// - [detectorIou]: NMS IoU threshold for duplicate suppression (0.0-1.0). Default: 0.45
  /// - [maxDetections]: Maximum number of persons to detect. Default: 10
  /// - [minLandmarkScore]: Minimum landmark confidence score (0.0-1.0). Default: 0.5
  /// - [interpreterPoolSize]: Number of landmark model interpreter instances (1-10). Default: 5
  /// - [performanceConfig]: TensorFlow Lite performance configuration. Default: no acceleration
  /// - [useNativePreprocessing]: Whether to use OpenCV for faster preprocessing. Default: true
  ///
  /// **Performance Configuration:**
  /// ```dart
  /// // Default (no acceleration, backward compatible)
  /// final detector = PoseDetector();
  ///
  /// // XNNPACK acceleration (2-5x faster, recommended)
  /// final detector = PoseDetector(
  ///   performanceConfig: PerformanceConfig.xnnpack(),
  /// );
  ///
  /// // Custom thread count
  /// final detector = PoseDetector(
  ///   performanceConfig: PerformanceConfig.xnnpack(numThreads: 2),
  /// );
  /// ```
  ///
  /// **Choosing interpreterPoolSize:**
  /// - When XNNPACK is enabled, pool size is always forced to 1 for stable performance
  /// - When XNNPACK is disabled, you can use higher pool sizes for parallelism
  /// - Each interpreter adds ~10MB memory overhead
  ///
  /// **IMPORTANT:** XNNPACK with multiple interpreters causes thread contention and
  /// performance spikes. The pool size is automatically set to 1 when XNNPACK is enabled.
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
  }) : interpreterPoolSize = performanceConfig.mode == PerformanceMode.disabled
            ? interpreterPoolSize
            : 1 {
    _lm = PoseLandmarkModelRunner(poolSize: this.interpreterPoolSize);
  }

  /// Initializes the pose detector by loading TensorFlow Lite models.
  ///
  /// Must be called before [detect] or [detectOnImage].
  /// If already initialized, will dispose existing models and reinitialize.
  ///
  /// Throws an exception if model loading fails.
  Future<void> initialize() async {
    if (_isInitialized) {
      await dispose();
    }
    await _lm.initialize(landmarkModel, performanceConfig: performanceConfig);
    await _yolo.initialize(performanceConfig: performanceConfig);
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
    _isInitialized = false;
  }

  /// Detects poses in an image from raw bytes.
  ///
  /// Decodes the image bytes and performs pose detection.
  ///
  /// Parameters:
  /// - [imageBytes]: Raw image data in a supported format (JPEG, PNG, etc.)
  ///
  /// Returns a list of [Pose] objects, one per detected person.
  /// Returns an empty list if image decoding fails or no persons are detected.
  ///
  /// Throws [StateError] if called before [initialize].
  @Deprecated('Will be removed in 2.0.0. Use detectOnMat instead.')
  Future<List<Pose>> detect(List<int> imageBytes) async {
    if (!_isInitialized) {
      throw StateError(
        'PoseDetector not initialized. Call initialize() first.',
      );
    }
    try {
      final decoder = imageDecoderOverride ?? img.decodeImage;
      final img.Image? image = decoder(Uint8List.fromList(imageBytes));
      if (image == null) return <Pose>[];
      return detectOnImage(image);
    } catch (e) {
      return <Pose>[];
    }
  }

  /// Detects poses directly from a cv.Mat without any image encoding/decoding.
  ///
  /// This is the most efficient method for real-time camera processing as it
  /// bypasses all image format conversions. The caller is responsible for
  /// converting camera frames to cv.Mat in BGR format.
  ///
  /// Parameters:
  /// - [mat]: OpenCV Mat in BGR format (CV_8UC3)
  /// - [imageWidth]: Original image width (for coordinate mapping)
  /// - [imageHeight]: Original image height (for coordinate mapping)
  ///
  /// Returns a list of [Pose] objects, one per detected person.
  /// The caller is responsible for disposing the input Mat.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Pose>> detectOnMat(
    cv.Mat mat, {
    required int imageWidth,
    required int imageHeight,
  }) async {
    if (!_isInitialized) {
      throw StateError(
        'PoseDetector not initialized. Call initialize() first.',
      );
    }

    final List<YoloDetection> dets = await _yolo.detectOnMat(
      mat,
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

    final List<_PersonCropData> cropDataList = <_PersonCropData>[];
    for (final YoloDetection d in dets) {
      final double x1 = d.bboxXYXY[0].clamp(0.0, imageWidth.toDouble());
      final double y1 = d.bboxXYXY[1].clamp(0.0, imageHeight.toDouble());
      final double x2 = d.bboxXYXY[2].clamp(0.0, imageWidth.toDouble());
      final double y2 = d.bboxXYXY[3].clamp(0.0, imageHeight.toDouble());
      final double bw = x2 - x1;
      final double bh = y2 - y1;

      final double cx = (x1 + x2) / 2.0;
      final double cy = (y1 + y2) / 2.0;
      final double side = (bw > bh ? bw : bh) * 1.25;

      final cv.Mat? square = NativeImageUtils.extractAlignedSquare(
        mat,
        cx,
        cy,
        side,
        0.0,
      );

      if (square == null) continue;

      final cv.Mat resized = cv.resize(
          square,
          (
            256,
            256,
          ),
          interpolation: cv.INTER_LINEAR);
      square.dispose();

      final double sqX1 = cx - side / 2.0;
      final double sqY1 = cy - side / 2.0;

      cropDataList.add(
        _PersonCropData(
          detection: d,
          letterboxedMat: resized,
          scaleRatio: 1.0,
          padLeft: 0,
          padTop: 0,
          cropX: sqX1.round(),
          cropY: sqY1.round(),
          cropWidth: side.round(),
          cropHeight: side.round(),
          useResize: true,
        ),
      );
    }

    final List<PoseLandmarks?> allLandmarks = <PoseLandmarks?>[];
    for (final _PersonCropData data in cropDataList) {
      try {
        final PoseLandmarks lms = await _lm.runOnMat(data.letterboxedMat!);
        allLandmarks.add(lms);
      } catch (e) {
        allLandmarks.add(null);
      }
    }

    final List<Pose> results = _buildLandmarkResults(
      cropDataList,
      allLandmarks,
      imageWidth,
      imageHeight,
    );

    for (final data in cropDataList) {
      data.dispose();
    }

    return results;
  }

  /// Detects poses in a decoded image.
  ///
  /// Performs the two-stage detection pipeline:
  /// 1. Detects persons using YOLOv8n
  /// 2. Extracts landmarks using BlazePose in parallel for all detected persons
  ///
  /// Parameters:
  /// - [image]: A decoded image from the `image` package
  ///
  /// Returns a list of [Pose] objects, one per detected person.
  /// Each pose contains:
  /// - A bounding box in original image coordinates
  /// - A confidence score (0.0-1.0)
  /// - 33 body landmarks (if [mode] is [PoseMode.boxesAndLandmarks])
  ///
  /// **Performance:** Landmark extraction runs in parallel using an interpreter pool.
  /// The [interpreterPoolSize] determines the maximum number of concurrent inferences.
  /// For example, with 5 people detected and pool size 5, all landmarks are extracted
  /// simultaneously, providing ~5x speedup compared to sequential processing.
  ///
  /// When [useNativePreprocessing] is true (default), uses SIMD-accelerated OpenCV
  /// operations for 5-15x faster preprocessing.
  ///
  /// Throws [StateError] if called before [initialize].
  @Deprecated('Will be removed in 2.0.0. Use detectOnMat instead.')
  Future<List<Pose>> detectOnImage(img.Image image) async {
    if (!_isInitialized) {
      throw StateError(
        'PoseDetector not initialized. Call initialize() first.',
      );
    }

    if (useNativePreprocessing) {
      return _detectOnImageNative(image);
    }

    return _detectOnImageDart(image);
  }

  /// Native pipeline using OpenCV for preprocessing.
  Future<List<Pose>> _detectOnImageNative(img.Image image) async {
    final cv.Mat mat = NativeImageUtils.imageToMat(image);

    try {
      final List<YoloDetection> dets = await _yolo.detectOnMat(
        mat,
        imageWidth: image.width,
        imageHeight: image.height,
        confThres: detectorConf,
        iouThres: detectorIou,
        maxDet: maxDetections,
        personOnly: true,
      );

      if (mode == PoseMode.boxes) {
        mat.dispose();
        return _buildBoxOnlyResults(dets, image.width, image.height);
      }

      final List<_PersonCropData> cropDataList = <_PersonCropData>[];
      for (final YoloDetection d in dets) {
        final double x1 = d.bboxXYXY[0].clamp(0.0, image.width.toDouble());
        final double y1 = d.bboxXYXY[1].clamp(0.0, image.height.toDouble());
        final double x2 = d.bboxXYXY[2].clamp(0.0, image.width.toDouble());
        final double y2 = d.bboxXYXY[3].clamp(0.0, image.height.toDouble());
        final double bw = x2 - x1;
        final double bh = y2 - y1;

        final double cx = (x1 + x2) / 2.0;
        final double cy = (y1 + y2) / 2.0;
        final double side = (bw > bh ? bw : bh) * 1.25;

        final cv.Mat? square = NativeImageUtils.extractAlignedSquare(
          mat,
          cx,
          cy,
          side,
          0.0,
        );

        if (square == null) continue;

        final cv.Mat resized = cv.resize(
            square,
            (
              256,
              256,
            ),
            interpolation: cv.INTER_LINEAR);
        square.dispose();

        final double sqX1 = cx - side / 2.0;
        final double sqY1 = cy - side / 2.0;

        cropDataList.add(
          _PersonCropData(
            detection: d,
            letterboxedMat: resized,
            scaleRatio: 1.0,
            padLeft: 0,
            padTop: 0,
            cropX: sqX1.round(),
            cropY: sqY1.round(),
            cropWidth: side.round(),
            cropHeight: side.round(),
            useResize: true,
          ),
        );
      }

      final List<PoseLandmarks?> allLandmarks = <PoseLandmarks?>[];
      for (final _PersonCropData data in cropDataList) {
        try {
          final PoseLandmarks lms = await _lm.runOnMat(data.letterboxedMat!);
          allLandmarks.add(lms);
        } catch (e) {
          allLandmarks.add(null);
        }
      }

      final List<Pose> results = _buildLandmarkResults(
        cropDataList,
        allLandmarks,
        image.width,
        image.height,
      );

      for (final data in cropDataList) {
        data.dispose();
      }
      mat.dispose();

      return results;
    } catch (e) {
      mat.dispose();
      rethrow;
    }
  }

  /// Dart pipeline using pure Dart for preprocessing (fallback).
  Future<List<Pose>> _detectOnImageDart(img.Image image) async {
    final List<YoloDetection> dets = await _yolo.detectOnImage(
      image,
      confThres: detectorConf,
      iouThres: detectorIou,
      maxDet: maxDetections,
      personOnly: true,
    );

    if (mode == PoseMode.boxes) {
      return _buildBoxOnlyResults(dets, image.width, image.height);
    }

    final List<Future<_PersonCropData>> cropFutures = dets.map((d) async {
      final int x1 = d.bboxXYXY[0].clamp(0.0, image.width.toDouble()).toInt();
      final int y1 = d.bboxXYXY[1].clamp(0.0, image.height.toDouble()).toInt();
      final int x2 = d.bboxXYXY[2].clamp(0.0, image.width.toDouble()).toInt();
      final int y2 = d.bboxXYXY[3].clamp(0.0, image.height.toDouble()).toInt();
      final int cw = (x2 - x1).clamp(1, image.width);
      final int ch = (y2 - y1).clamp(1, image.height);

      final img.Image crop = img.copyCrop(
        image,
        x: x1,
        y: y1,
        width: cw,
        height: ch,
      );
      final List<double> ratio = <double>[];
      final List<int> dwdh = <int>[];
      final img.Image letter = ImageUtils.letterbox256(crop, ratio, dwdh);

      return _PersonCropData(
        detection: d,
        letterboxed: letter,
        scaleRatio: ratio.first,
        padLeft: dwdh[0],
        padTop: dwdh[1],
        cropX: x1,
        cropY: y1,
      );
    }).toList();

    final List<_PersonCropData> cropDataList = await Future.wait(cropFutures);

    final List<Future<PoseLandmarks?>> futures = cropDataList.map((data) async {
      try {
        return await _lm.run(data.letterboxed!);
      } catch (e) {
        return null;
      }
    }).toList();

    final List<PoseLandmarks?> allLandmarks = await Future.wait(futures);

    return _buildLandmarkResults(
      cropDataList,
      allLandmarks,
      image.width,
      image.height,
    );
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

  /// Builds full results with landmarks from crop data.
  List<Pose> _buildLandmarkResults(
    List<_PersonCropData> cropDataList,
    List<PoseLandmarks?> allLandmarks,
    int imageWidth,
    int imageHeight,
  ) {
    final List<Pose> results = <Pose>[];
    for (int i = 0; i < cropDataList.length; i++) {
      final _PersonCropData data = cropDataList[i];
      final PoseLandmarks? lms = allLandmarks[i];

      if (lms == null || lms.score < minLandmarkScore) continue;

      final List<PoseLandmark> pts = <PoseLandmark>[];
      for (final PoseLandmark lm in lms.landmarks) {
        double xOrig, yOrig;

        if (data.useResize) {
          xOrig = (data.cropX + lm.x * data.cropWidth).clamp(
            0.0,
            imageWidth.toDouble(),
          );
          yOrig = (data.cropY + lm.y * data.cropHeight).clamp(
            0.0,
            imageHeight.toDouble(),
          );
        } else {
          final double xp = lm.x * 256.0;
          final double yp = lm.y * 256.0;
          final double xCrop = (xp - data.padLeft) / data.scaleRatio;
          final double yCrop = (yp - data.padTop) / data.scaleRatio;
          xOrig = (data.cropX.toDouble() + xCrop).clamp(
            0.0,
            imageWidth.toDouble(),
          );
          yOrig = (data.cropY.toDouble() + yCrop).clamp(
            0.0,
            imageHeight.toDouble(),
          );
        }

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

      results.add(
        Pose(
          boundingBox: BoundingBox(
            left: data.detection.bboxXYXY[0],
            top: data.detection.bboxXYXY[1],
            right: data.detection.bboxXYXY[2],
            bottom: data.detection.bboxXYXY[3],
          ),
          score: data.detection.score,
          landmarks: pts,
          imageWidth: imageWidth,
          imageHeight: imageHeight,
        ),
      );
    }
    return results;
  }
}
