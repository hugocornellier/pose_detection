import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:meta/meta.dart';
import 'types.dart';
import 'image_utils.dart';
import 'person_detector.dart';
import 'pose_landmark_model.dart';

/// Helper class to store preprocessing data for each detected person.
///
/// Contains the detection info, preprocessed image, and transformation parameters
/// needed to convert landmark coordinates back to original image space.
class _PersonCropData {
  /// The original YOLO detection result.
  final YoloDetection detection;

  /// The letterboxed 256x256 image ready for landmark extraction.
  final img.Image letterboxed;

  /// Scale ratio used in letterbox preprocessing.
  final double scaleRatio;

  /// Left padding added during letterboxing.
  final int padLeft;

  /// Top padding added during letterboxing.
  final int padTop;

  /// X coordinate of crop origin in original image.
  final int cropX;

  /// Y coordinate of crop origin in original image.
  final int cropY;

  _PersonCropData({
    required this.detection,
    required this.letterboxed,
    required this.scaleRatio,
    required this.padLeft,
    required this.padTop,
    required this.cropX,
    required this.cropY,
  });
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
  Future<List<Pose>> detect(List<int> imageBytes) async {
    if (!_isInitialized) {
      throw StateError(
          'PoseDetector not initialized. Call initialize() first.');
    }
    try {
      final decoder = imageDecoderOverride ?? img.decodeImage;
      final img.Image? image = decoder(Uint8List.fromList(imageBytes));
      if (image == null) return <Pose>[];
      return detectOnImage(image);
    } catch (e) {
      // Return empty list if image decoding fails (invalid/corrupted image data)
      return <Pose>[];
    }
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
  /// Throws [StateError] if called before [initialize].
  Future<List<Pose>> detectOnImage(img.Image image) async {
    if (!_isInitialized) {
      throw StateError(
          'PoseDetector not initialized. Call initialize() first.');
    }

    final List<YoloDetection> dets = await _yolo.detectOnImage(
      image,
      confThres: detectorConf,
      iouThres: detectorIou,
      // Use dynamic topkPreNms (0) - scales based on image size
      maxDet: maxDetections,
      personOnly: true,
    );

    if (mode == PoseMode.boxes) {
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
            imageWidth: image.width,
            imageHeight: image.height,
          ),
        );
      }
      return out;
    }

    // Phase 1: Preprocess all detections (crop and letterbox) in parallel
    final List<Future<_PersonCropData>> cropFutures = dets.map((d) async {
      final int x1 = d.bboxXYXY[0].clamp(0.0, image.width.toDouble()).toInt();
      final int y1 = d.bboxXYXY[1].clamp(0.0, image.height.toDouble()).toInt();
      final int x2 = d.bboxXYXY[2].clamp(0.0, image.width.toDouble()).toInt();
      final int y2 = d.bboxXYXY[3].clamp(0.0, image.height.toDouble()).toInt();
      final int cw = (x2 - x1).clamp(1, image.width);
      final int ch = (y2 - y1).clamp(1, image.height);

      final img.Image crop =
          img.copyCrop(image, x: x1, y: y1, width: cw, height: ch);
      final List<double> ratio = <double>[];
      final List<int> dwdh = <int>[];
      // Don't reuse canvas buffer when processing multiple people in parallel
      // to avoid race conditions where all references point to the same buffer
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

    // Phase 2: Run landmark extraction in parallel for all persons
    // The landmark model runner uses an interpreter pool to enable safe concurrent
    // execution. Each run() call acquires an interpreter, runs inference, and releases
    // it back to the pool. Future.wait() executes all inferences concurrently up to
    // the pool size limit.
    final List<Future<PoseLandmarks?>> futures = cropDataList.map((data) async {
      try {
        return await _lm.run(data.letterboxed);
      } catch (e) {
        // Return null on failure to avoid crashing the entire batch
        return null;
      }
    }).toList();

    final List<PoseLandmarks?> allLandmarks = await Future.wait(futures);

    // Phase 3: Post-process results and transform coordinates
    final List<Pose> results = <Pose>[];
    for (int i = 0; i < cropDataList.length; i++) {
      final _PersonCropData data = cropDataList[i];
      final PoseLandmarks? lms = allLandmarks[i];

      // Skip if landmark extraction failed or score too low
      if (lms == null || lms.score < minLandmarkScore) continue;

      final List<PoseLandmark> pts = <PoseLandmark>[];
      for (final PoseLandmark lm in lms.landmarks) {
        final double xp = lm.x * 256.0;
        final double yp = lm.y * 256.0;
        final double xContent = (xp - data.padLeft) / data.scaleRatio;
        final double yContent = (yp - data.padTop) / data.scaleRatio;
        final double xOrig = (data.cropX.toDouble() + xContent)
            .clamp(0.0, image.width.toDouble());
        final double yOrig = (data.cropY.toDouble() + yContent)
            .clamp(0.0, image.height.toDouble());
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
          imageWidth: image.width,
          imageHeight: image.height,
        ),
      );
    }

    return results;
  }
}
