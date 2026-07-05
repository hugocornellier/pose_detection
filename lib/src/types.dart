import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart'
    show BoundingBox, LandmarkMixin;

/// BlazePose model variant for landmark extraction.
///
/// Controls the accuracy/performance trade-off:
/// - [lite]: Fastest, good accuracy
/// - [full]: Balanced speed/accuracy
/// - [heavy]: Slowest, best accuracy
enum PoseLandmarkModel {
  /// Fastest model with good accuracy.
  lite,

  /// Balanced model with medium speed and accuracy.
  full,

  /// Slowest model with best accuracy.
  heavy,
}

/// Detection mode controlling the two-stage pipeline behavior.
///
/// - [boxes]: Fast detection returning only bounding boxes (Stage 1 only)
/// - [boxesAndLandmarks]: Full pipeline returning boxes + 33 landmarks (both stages)
enum PoseMode {
  /// Fast detection mode returning only bounding boxes (Stage 1 only).
  boxes,

  /// Full pipeline mode returning bounding boxes and 33 landmarks per person.
  boxesAndLandmarks,
}

/// Collection of pose landmarks with a confidence score.
class PoseLandmarks {
  /// List of 33 body landmarks extracted from the BlazePose model.
  final List<PoseLandmark> landmarks;

  /// Confidence score for the landmark extraction (0.0 to 1.0).
  final double score;

  /// Raw person-segmentation buffer at model resolution (256x256), quantized to
  /// 0-255, or null when segmentation was not requested. Row-major.
  ///
  /// This is the model-space buffer; the detector core maps it into image space
  /// as a [SegmentationMask] before it reaches the public [Pose] API.
  final Uint8List? segmentationMask;

  /// Creates a collection of pose landmarks with a confidence score.
  PoseLandmarks({
    required this.landmarks,
    required this.score,
    this.segmentationMask,
  });
}

/// A single body keypoint with 3D coordinates and visibility score.
///
/// Coordinates are in the original image space (pixels).
/// The [z] coordinate represents depth relative to the hips midpoint (not absolute depth).
class PoseLandmark with LandmarkMixin {
  /// The body part this landmark represents (nose, leftShoulder, etc.)
  final PoseLandmarkType type;

  /// X coordinate in pixels (original image space)
  @override
  final double x;

  /// Y coordinate in pixels (original image space)
  @override
  final double y;

  /// Z coordinate representing depth relative to hips midpoint (not absolute depth)
  final double z;

  /// Visibility/confidence score (0.0 to 1.0). Higher means more confident the landmark is visible.
  final double visibility;

  /// Creates a pose landmark with 3D coordinates and visibility score.
  PoseLandmark({
    required this.type,
    required this.x,
    required this.y,
    required this.z,
    required this.visibility,
  });

  /// Serializes this landmark to a map for isolate message passing.
  Map<String, dynamic> toMap() => {
    'type': type.index,
    'x': x,
    'y': y,
    'z': z,
    'visibility': visibility,
  };

  /// Deserializes a [PoseLandmark] from a map produced by [toMap].
  static PoseLandmark fromMap(Map<String, dynamic> map) => PoseLandmark(
    type: PoseLandmarkType.values[map['type'] as int],
    x: (map['x'] as num).toDouble(),
    y: (map['y'] as num).toDouble(),
    z: (map['z'] as num).toDouble(),
    visibility: (map['visibility'] as num).toDouble(),
  );
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
  /// Nose tip landmark.
  nose,

  /// Left eye inner corner landmark.
  leftEyeInner,

  /// Left eye center landmark.
  leftEye,

  /// Left eye outer corner landmark.
  leftEyeOuter,

  /// Right eye inner corner landmark.
  rightEyeInner,

  /// Right eye center landmark.
  rightEye,

  /// Right eye outer corner landmark.
  rightEyeOuter,

  /// Left ear landmark.
  leftEar,

  /// Right ear landmark.
  rightEar,

  /// Left mouth corner landmark.
  mouthLeft,

  /// Right mouth corner landmark.
  mouthRight,

  /// Left shoulder landmark.
  leftShoulder,

  /// Right shoulder landmark.
  rightShoulder,

  /// Left elbow landmark.
  leftElbow,

  /// Right elbow landmark.
  rightElbow,

  /// Left wrist landmark.
  leftWrist,

  /// Right wrist landmark.
  rightWrist,

  /// Left pinky finger base landmark.
  leftPinky,

  /// Right pinky finger base landmark.
  rightPinky,

  /// Left index finger base landmark.
  leftIndex,

  /// Right index finger base landmark.
  rightIndex,

  /// Left thumb base landmark.
  leftThumb,

  /// Right thumb base landmark.
  rightThumb,

  /// Left hip landmark.
  leftHip,

  /// Right hip landmark.
  rightHip,

  /// Left knee landmark.
  leftKnee,

  /// Right knee landmark.
  rightKnee,

  /// Left ankle landmark.
  leftAnkle,

  /// Right ankle landmark.
  rightAnkle,

  /// Left heel landmark.
  leftHeel,

  /// Right heel landmark.
  rightHeel,

  /// Left foot index toe landmark.
  leftFootIndex,

  /// Right foot index toe landmark.
  rightFootIndex,
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
  [PoseLandmarkType.leftEye, PoseLandmarkType.nose],
  [PoseLandmarkType.rightEye, PoseLandmarkType.nose],
  [PoseLandmarkType.leftEye, PoseLandmarkType.leftEar],
  [PoseLandmarkType.rightEye, PoseLandmarkType.rightEar],
  [PoseLandmarkType.mouthLeft, PoseLandmarkType.mouthRight],
  [PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder],
  [PoseLandmarkType.leftShoulder, PoseLandmarkType.leftHip],
  [PoseLandmarkType.rightShoulder, PoseLandmarkType.rightHip],
  [PoseLandmarkType.leftHip, PoseLandmarkType.rightHip],
  [PoseLandmarkType.leftShoulder, PoseLandmarkType.leftElbow],
  [PoseLandmarkType.leftElbow, PoseLandmarkType.leftWrist],
  [PoseLandmarkType.leftWrist, PoseLandmarkType.leftPinky],
  [PoseLandmarkType.leftWrist, PoseLandmarkType.leftIndex],
  [PoseLandmarkType.leftWrist, PoseLandmarkType.leftThumb],
  [PoseLandmarkType.rightShoulder, PoseLandmarkType.rightElbow],
  [PoseLandmarkType.rightElbow, PoseLandmarkType.rightWrist],
  [PoseLandmarkType.rightWrist, PoseLandmarkType.rightPinky],
  [PoseLandmarkType.rightWrist, PoseLandmarkType.rightIndex],
  [PoseLandmarkType.rightWrist, PoseLandmarkType.rightThumb],
  [PoseLandmarkType.leftHip, PoseLandmarkType.leftKnee],
  [PoseLandmarkType.leftKnee, PoseLandmarkType.leftAnkle],
  [PoseLandmarkType.leftAnkle, PoseLandmarkType.leftHeel],
  [PoseLandmarkType.leftAnkle, PoseLandmarkType.leftFootIndex],
  [PoseLandmarkType.rightHip, PoseLandmarkType.rightKnee],
  [PoseLandmarkType.rightKnee, PoseLandmarkType.rightAnkle],
  [PoseLandmarkType.rightAnkle, PoseLandmarkType.rightHeel],
  [PoseLandmarkType.rightAnkle, PoseLandmarkType.rightFootIndex],
];

/// A per-person segmentation mask produced by the BlazePose landmark model.
///
/// BlazePose emits a coarse person-vs-background probability map alongside the
/// 33 landmarks. This is only populated when the detector is initialized with
/// `enableSegmentation: true` and [PoseMode.boxesAndLandmarks]; otherwise the
/// owning [Pose.segmentationMask] is null.
///
/// The mask buffer is at model resolution ([width] x [height], typically
/// 256x256) and covers the square image region described by [imageLeft],
/// [imageTop], [imageWidth], and [imageHeight] in original-image pixel
/// coordinates. That region is the padded person crop and may extend past the
/// image edges.
///
/// Example (render as a tinted overlay with `dart:ui`):
/// ```dart
/// final mask = pose.segmentationMask;
/// if (mask != null) {
///   ui.decodeImageFromPixels(
///     mask.toRgbaBytes(r: 0, g: 200, b: 255),
///     mask.width,
///     mask.height,
///     ui.PixelFormat.rgba8888,
///     (ui.Image image) {
///       // canvas.drawImageRect(image, src, dstRectInImageSpace, paint);
///     },
///   );
/// }
/// ```
class SegmentationMask {
  /// Mask buffer width in pixels (model resolution, e.g. 256).
  final int width;

  /// Mask buffer height in pixels (model resolution, e.g. 256).
  final int height;

  /// Row-major person probability per pixel, quantized to 0-255
  /// (0 = background, 255 = person). Length is [width] * [height].
  final Uint8List confidences;

  /// Left edge (original-image pixels) of the region the mask covers.
  final double imageLeft;

  /// Top edge (original-image pixels) of the region the mask covers.
  final double imageTop;

  /// Width (original-image pixels) of the region the mask covers.
  final double imageWidth;

  /// Height (original-image pixels) of the region the mask covers.
  final double imageHeight;

  /// Creates a segmentation mask.
  const SegmentationMask({
    required this.width,
    required this.height,
    required this.confidences,
    required this.imageLeft,
    required this.imageTop,
    required this.imageWidth,
    required this.imageHeight,
  });

  /// Person probability in [0, 1] at original-image pixel ([x], [y]).
  ///
  /// Returns 0 for points outside the mask region. Uses nearest-neighbour
  /// sampling.
  double confidenceAt(num x, num y) {
    if (imageWidth <= 0 || imageHeight <= 0) return 0.0;
    final double u = (x - imageLeft) / imageWidth;
    final double v = (y - imageTop) / imageHeight;
    if (u < 0 || u >= 1 || v < 0 || v >= 1) return 0.0;
    final int px = (u * width).floor().clamp(0, width - 1);
    final int py = (v * height).floor().clamp(0, height - 1);
    return confidences[py * width + px] / 255.0;
  }

  /// Expands the mask into an RGBA byte buffer ([width] * [height] * 4) tinted
  /// with ([r], [g], [b]) and per-pixel alpha equal to the person probability.
  ///
  /// Feed the result to `dart:ui`'s `decodeImageFromPixels(bytes, width,
  /// height, PixelFormat.rgba8888, ...)` to obtain a drawable image, then blit
  /// it into the mask region with `Canvas.drawImageRect`.
  Uint8List toRgbaBytes({int r = 0, int g = 255, int b = 0}) {
    final Uint8List out = Uint8List(width * height * 4);
    for (int i = 0; i < confidences.length; i++) {
      final int o = i * 4;
      out[o] = r;
      out[o + 1] = g;
      out[o + 2] = b;
      out[o + 3] = confidences[i];
    }
    return out;
  }

  /// Serializes this mask to a map for isolate message passing.
  Map<String, dynamic> toMap() => {
    'width': width,
    'height': height,
    'confidences': confidences,
    'imageLeft': imageLeft,
    'imageTop': imageTop,
    'imageWidth': imageWidth,
    'imageHeight': imageHeight,
  };

  /// Deserializes a [SegmentationMask] from a map produced by [toMap].
  static SegmentationMask fromMap(Map<String, dynamic> map) => SegmentationMask(
    width: map['width'] as int,
    height: map['height'] as int,
    confidences: map['confidences'] as Uint8List,
    imageLeft: (map['imageLeft'] as num).toDouble(),
    imageTop: (map['imageTop'] as num).toDouble(),
    imageWidth: (map['imageWidth'] as num).toDouble(),
    imageHeight: (map['imageHeight'] as num).toDouble(),
  );
}

/// Detected person with bounding box and optional body landmarks.
///
/// This is the main result returned by [PoseDetector.detect].
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

  /// Coarse person-segmentation mask for this pose, or null.
  ///
  /// Only populated when the detector was initialized with
  /// `enableSegmentation: true` under [PoseMode.boxesAndLandmarks], the pose has
  /// landmarks, and the platform supports segmentation (currently native, not
  /// web). See [SegmentationMask].
  final SegmentationMask? segmentationMask;

  /// Creates a detected pose with bounding box, landmarks, and image dimensions.
  const Pose({
    required this.boundingBox,
    required this.score,
    required this.landmarks,
    required this.imageWidth,
    required this.imageHeight,
    this.segmentationMask,
  });

  /// Serializes this pose to a map for isolate message passing.
  Map<String, dynamic> toMap() => {
    'boundingBox': boundingBox.toMap(),
    'score': score,
    'landmarks': landmarks.map((l) => l.toMap()).toList(),
    'imageWidth': imageWidth,
    'imageHeight': imageHeight,
    if (segmentationMask != null) 'segmentationMask': segmentationMask!.toMap(),
  };

  /// Deserializes a [Pose] from a map produced by [toMap].
  static Pose fromMap(Map<String, dynamic> map) => Pose(
    boundingBox: BoundingBox.fromMap(
      map['boundingBox'] as Map<String, dynamic>,
    ),
    score: (map['score'] as num).toDouble(),
    landmarks: (map['landmarks'] as List<dynamic>)
        .map((l) => PoseLandmark.fromMap(Map<String, dynamic>.from(l as Map)))
        .toList(),
    imageWidth: map['imageWidth'] as int,
    imageHeight: map['imageHeight'] as int,
    segmentationMask: map['segmentationMask'] == null
        ? null
        : SegmentationMask.fromMap(
            Map<String, dynamic>.from(map['segmentationMask'] as Map),
          ),
  );

  /// Gets a specific landmark by type, or null if not found
  PoseLandmark? getLandmark(PoseLandmarkType type) {
    for (final l in landmarks) {
      if (l.type == type) return l;
    }
    return null;
  }

  /// Returns true if this pose has landmarks
  bool get hasLandmarks => landmarks.isNotEmpty;

  @override
  String toString() {
    final String landmarksInfo = landmarks
        .map(
          (l) =>
              '${l.type.name}: (${l.x.toStringAsFixed(2)}, ${l.y.toStringAsFixed(2)}) vis=${l.visibility.toStringAsFixed(2)}',
        )
        .join('\n');
    return 'Pose(\n'
        '  score=${score.toStringAsFixed(3)},\n'
        '  landmarks=${landmarks.length},\n'
        '  coords:\n$landmarksInfo\n)';
  }
}
