/// BlazePose model variant for landmark extraction.
///
/// Controls the accuracy/performance trade-off:
/// - [lite]: Fastest, good accuracy
/// - [full]: Balanced speed/accuracy
/// - [heavy]: Slowest, best accuracy
enum PoseLandmarkModel { lite, full, heavy }

/// Detection mode controlling the two-stage pipeline behavior.
///
/// - [boxes]: Fast detection returning only bounding boxes (Stage 1 only)
/// - [boxesAndLandmarks]: Full pipeline returning boxes + 33 landmarks (both stages)
enum PoseMode { boxes, boxesAndLandmarks }

/// Collection of pose landmarks with confidence score (internal use).
class PoseLandmarks {
  final List<PoseLandmark> landmarks;
  final double score;

  PoseLandmarks({
    required this.landmarks,
    required this.score,
  });
}

/// A single body keypoint with 3D coordinates and visibility score.
///
/// Coordinates are in the original image space (pixels).
/// The [z] coordinate represents depth relative to the body center (not absolute depth).
class PoseLandmark {
  /// The body part this landmark represents (nose, leftShoulder, etc.)
  final PoseLandmarkType type;

  /// X coordinate in pixels (original image space)
  final double x;

  /// Y coordinate in pixels (original image space)
  final double y;

  /// Z coordinate representing depth relative to hips midpoint (not absolute depth)
  final double z;

  /// Visibility/confidence score (0.0 to 1.0). Higher means more confident the landmark is visible.
  final double visibility;

  PoseLandmark({
    required this.type,
    required this.x,
    required this.y,
    required this.z,
    required this.visibility,
  });

  /// Converts x coordinate to normalized range (0.0 to 1.0)
  double xNorm(int imageWidth) => (x / imageWidth).clamp(0.0, 1.0);

  /// Converts y coordinate to normalized range (0.0 to 1.0)
  double yNorm(int imageHeight) => (y / imageHeight).clamp(0.0, 1.0);

  /// Converts landmark coordinates to integer pixel point
  Point toPixel(int imageWidth, int imageHeight) {
    return Point(x.toInt(), y.toInt());
  }
}

/// Body part types for the 33 BlazePose landmarks.
///
/// Follows MediaPipe BlazePose topology with landmarks for face, torso, arms, and legs.
///
/// Available landmarks:
/// - **Face**: [nose], [leftEyeInner], [leftEye], [leftEyeOuter], [rightEyeInner],
///   [rightEye], [rightEyeOuter], [leftEar], [rightEar], [mouthLeft], [mouthRight]
/// - **Torso**: [leftShoulder], [rightShoulder], [leftHip], [rightHip]
/// - **Arms**: [leftElbow], [rightElbow], [leftWrist], [rightWrist]
/// - **Hands**: [leftPinky], [rightPinky], [leftIndex], [rightIndex], [leftThumb], [rightThumb]
/// - **Legs**: [leftKnee], [rightKnee], [leftAnkle], [rightAnkle]
/// - **Feet**: [leftHeel], [rightHeel], [leftFootIndex], [rightFootIndex]
///
/// Example:
/// ```dart
/// final pose = poses.first;
/// final nose = pose.getLandmark(PoseLandmarkType.nose);
/// final leftWrist = pose.getLandmark(PoseLandmarkType.leftWrist);
/// final rightAnkle = pose.getLandmark(PoseLandmarkType.rightAnkle);
///
/// if (nose != null) {
///   print('Nose at (${nose.x}, ${nose.y}) with visibility ${nose.visibility}');
/// }
/// ```
enum PoseLandmarkType {
  nose,
  leftEyeInner,
  leftEye,
  leftEyeOuter,
  rightEyeInner,
  rightEye,
  rightEyeOuter,
  leftEar,
  rightEar,
  mouthLeft,
  mouthRight,
  leftShoulder,
  rightShoulder,
  leftElbow,
  rightElbow,
  leftWrist,
  rightWrist,
  leftPinky,
  rightPinky,
  leftIndex,
  rightIndex,
  leftThumb,
  rightThumb,
  leftHip,
  rightHip,
  leftKnee,
  rightKnee,
  leftAnkle,
  rightAnkle,
  leftHeel,
  rightHeel,
  leftFootIndex,
  rightFootIndex,
}

/// 2D integer pixel coordinate.
class Point {
  final int x;
  final int y;

  Point(this.x, this.y);
}

/// Axis-aligned bounding box in pixel coordinates.
///
/// Coordinates are in the original image space (not normalized).
class BoundingBox {
  /// Left edge x-coordinate in pixels
  final double left;

  /// Top edge y-coordinate in pixels
  final double top;

  /// Right edge x-coordinate in pixels
  final double right;

  /// Bottom edge y-coordinate in pixels
  final double bottom;

  const BoundingBox({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });
}

/// Defines the standard skeleton connections between BlazePose landmarks.
///
/// Each connection is a pair of [PoseLandmarkType] values representing
/// the start and end points of a line segment in the body skeleton.
///
/// Use this constant to draw skeleton overlays on detected poses:
/// ```dart
/// for (final connection in poseLandmarkConnections) {
///   final start = pose.getLandmark(connection[0]);
///   final end = pose.getLandmark(connection[1]);
///   if (start != null && end != null && start.visibility > 0.5 && end.visibility > 0.5) {
///     // Draw line from start to end
///     canvas.drawLine(
///       Offset(start.x, start.y),
///       Offset(end.x, end.y),
///       paint,
///     );
///   }
/// }
/// ```
const List<List<PoseLandmarkType>> poseLandmarkConnections = [
  // Face
  [PoseLandmarkType.leftEye, PoseLandmarkType.nose],
  [PoseLandmarkType.rightEye, PoseLandmarkType.nose],
  [PoseLandmarkType.leftEye, PoseLandmarkType.leftEar],
  [PoseLandmarkType.rightEye, PoseLandmarkType.rightEar],
  [PoseLandmarkType.mouthLeft, PoseLandmarkType.mouthRight],
  // Torso
  [PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder],
  [PoseLandmarkType.leftShoulder, PoseLandmarkType.leftHip],
  [PoseLandmarkType.rightShoulder, PoseLandmarkType.rightHip],
  [PoseLandmarkType.leftHip, PoseLandmarkType.rightHip],
  // Left arm
  [PoseLandmarkType.leftShoulder, PoseLandmarkType.leftElbow],
  [PoseLandmarkType.leftElbow, PoseLandmarkType.leftWrist],
  [PoseLandmarkType.leftWrist, PoseLandmarkType.leftPinky],
  [PoseLandmarkType.leftWrist, PoseLandmarkType.leftIndex],
  [PoseLandmarkType.leftWrist, PoseLandmarkType.leftThumb],
  // Right arm
  [PoseLandmarkType.rightShoulder, PoseLandmarkType.rightElbow],
  [PoseLandmarkType.rightElbow, PoseLandmarkType.rightWrist],
  [PoseLandmarkType.rightWrist, PoseLandmarkType.rightPinky],
  [PoseLandmarkType.rightWrist, PoseLandmarkType.rightIndex],
  [PoseLandmarkType.rightWrist, PoseLandmarkType.rightThumb],
  // Left leg
  [PoseLandmarkType.leftHip, PoseLandmarkType.leftKnee],
  [PoseLandmarkType.leftKnee, PoseLandmarkType.leftAnkle],
  [PoseLandmarkType.leftAnkle, PoseLandmarkType.leftHeel],
  [PoseLandmarkType.leftAnkle, PoseLandmarkType.leftFootIndex],
  // Right leg
  [PoseLandmarkType.rightHip, PoseLandmarkType.rightKnee],
  [PoseLandmarkType.rightKnee, PoseLandmarkType.rightAnkle],
  [PoseLandmarkType.rightAnkle, PoseLandmarkType.rightHeel],
  [PoseLandmarkType.rightAnkle, PoseLandmarkType.rightFootIndex],
];

/// Detected person with bounding box and optional body landmarks.
///
/// This is the main result returned by [PoseDetector.detect()].
///
/// Contains:
/// - [boundingBox]: Location of the detected person in the image
/// - [score]: Confidence score from the person detector (0.0 to 1.0)
/// - [landmarks]: List of 33 body keypoints (empty if [PoseMode.boxes])
/// - [imageWidth] and [imageHeight]: Original image dimensions for coordinate reference
///
/// Example:
/// ```dart
/// final poses = await detector.detect(imageBytes);
/// for (final pose in poses) {
///   print('Person detected with confidence ${pose.score}');
///   if (pose.hasLandmarks) {
///     final nose = pose.getLandmark(PoseLandmarkType.nose);
///     print('Nose at (${nose?.x}, ${nose?.y})');
///   }
/// }
/// ```
class Pose {
  /// Bounding box of the detected person in pixel coordinates
  final BoundingBox boundingBox;

  /// Confidence score from person detector (0.0 to 1.0)
  final double score;

  /// List of 33 body landmarks. Empty if using [PoseMode.boxes].
  final List<PoseLandmark> landmarks;

  /// Width of the original image in pixels
  final int imageWidth;

  /// Height of the original image in pixels
  final int imageHeight;

  const Pose({
    required this.boundingBox,
    required this.score,
    required this.landmarks,
    required this.imageWidth,
    required this.imageHeight,
  });

  /// Gets a specific landmark by type, or null if not found
  PoseLandmark? getLandmark(PoseLandmarkType type) {
    try {
      return landmarks.firstWhere((l) => l.type == type);
    } catch (_) {
      return null;
    }
  }

  /// Returns true if this pose has landmarks
  bool get hasLandmarks => landmarks.isNotEmpty;

  @override
  String toString() {
    final String landmarksInfo = landmarks
        .map((l) =>
            '${l.type.name}: (${l.x.toStringAsFixed(2)}, ${l.y.toStringAsFixed(2)}) vis=${l.visibility.toStringAsFixed(2)}')
        .join('\n');
    return 'Pose(\n'
        '  score=${score.toStringAsFixed(3)},\n'
        '  landmarks=${landmarks.length},\n'
        '  coords:\n$landmarksInfo\n)';
  }
}
