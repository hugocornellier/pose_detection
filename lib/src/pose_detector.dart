import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'types.dart';
import 'image_utils.dart';
import 'person_detector.dart';
import 'pose_landmark_model.dart';

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
  final PoseLandmarkModelRunner _lm = PoseLandmarkModelRunner();

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

  bool _isInitialized = false;
  img.Image? _canvasBuffer256;

  /// Creates a pose detector with the specified configuration.
  ///
  /// Parameters:
  /// - [mode]: Detection mode (boxes only or boxes + landmarks). Default: [PoseMode.boxesAndLandmarks]
  /// - [landmarkModel]: BlazePose model variant. Default: [PoseLandmarkModel.heavy]
  /// - [detectorConf]: Person detection confidence threshold (0.0-1.0). Default: 0.5
  /// - [detectorIou]: NMS IoU threshold for duplicate suppression (0.0-1.0). Default: 0.45
  /// - [maxDetections]: Maximum number of persons to detect. Default: 10
  /// - [minLandmarkScore]: Minimum landmark confidence score (0.0-1.0). Default: 0.5
  PoseDetector({
    this.mode = PoseMode.boxesAndLandmarks,
    this.landmarkModel = PoseLandmarkModel.heavy,
    this.detectorConf = 0.5,
    this.detectorIou = 0.45,
    this.maxDetections = 10,
    this.minLandmarkScore = 0.5,
  });

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
    await _lm.initialize(landmarkModel);
    await _yolo.initialize();
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
    _canvasBuffer256 = null;
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
      throw StateError('PoseDetector not initialized. Call initialize() first.');
    }
    try {
      final img.Image? image = img.decodeImage(Uint8List.fromList(imageBytes));
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
  /// 2. Extracts landmarks using BlazePose (if [mode] is [PoseMode.boxesAndLandmarks])
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
  /// Throws [StateError] if called before [initialize].
  Future<List<Pose>> detectOnImage(img.Image image) async {
    if (!_isInitialized) {
      throw StateError('PoseDetector not initialized. Call initialize() first.');
    }

    final List<YoloDetection> dets = await _yolo.detectOnImage(
      image,
      confThres: detectorConf,
      iouThres: detectorIou,
      topkPreNms: 100,
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

    final List<Pose> results = <Pose>[];
    for (final YoloDetection d in dets) {
      final int x1 = d.bboxXYXY[0].clamp(0.0, image.width.toDouble()).toInt();
      final int y1 = d.bboxXYXY[1].clamp(0.0, image.height.toDouble()).toInt();
      final int x2 = d.bboxXYXY[2].clamp(0.0, image.width.toDouble()).toInt();
      final int y2 = d.bboxXYXY[3].clamp(0.0, image.height.toDouble()).toInt();
      final int cw = (x2 - x1).clamp(1, image.width);
      final int ch = (y2 - y1).clamp(1, image.height);

      final img.Image crop = img.copyCrop(image, x: x1, y: y1, width: cw, height: ch);
      final List<double> ratio = <double>[];
      final List<int> dwdh = <int>[];
      _canvasBuffer256 ??= img.Image(width: 256, height: 256);
      final img.Image letter = ImageUtils.letterbox256(
        crop,
        ratio,
        dwdh,
        reuseCanvas: _canvasBuffer256
      );
      final double r = ratio.first;
      final int dw = dwdh[0];
      final int dh = dwdh[1];

      final PoseLandmarks lms = await _lm.run(letter);
      if (lms.score < minLandmarkScore) continue;

      final List<PoseLandmark> pts = <PoseLandmark>[];
      for (final PoseLandmark lm in lms.landmarks) {
        final double xp = lm.x * 256.0;
        final double yp = lm.y * 256.0;
        final double xContent = (xp - dw) / r;
        final double yContent = (yp - dh) / r;
        final double xOrig = (x1.toDouble() + xContent)
            .clamp(0.0, image.width.toDouble());
        final double yOrig = (y1.toDouble() + yContent)
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
            left: d.bboxXYXY[0],
            top: d.bboxXYXY[1],
            right: d.bboxXYXY[2],
            bottom: d.bboxXYXY[3],
          ),
          score: d.score,
          landmarks: pts,
          imageWidth: image.width,
          imageHeight: image.height,
        ),
      );
    }

    return results;
  }
}
