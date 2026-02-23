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

/// Performance modes for TensorFlow Lite delegate selection.
///
/// Determines which hardware acceleration delegates are used for inference.
///
/// ## Platform Support
///
/// | Mode | macOS | Linux | Windows | iOS | Android |
/// |------|-------|-------|---------|-----|---------|
/// | [disabled] | CPU | CPU | CPU | CPU | CPU |
/// | [xnnpack] | XNNPACK | XNNPACK | CPU* | CPU* | CPU* |
/// | [gpu] | CPU | CPU | CPU | Metal | OpenGL** |
/// | [auto] | XNNPACK | XNNPACK | CPU | Metal | CPU |
///
/// \* Falls back to CPU (XNNPACK not supported on this platform)
/// \** Experimental, may crash on some devices
enum PerformanceMode {
  /// No acceleration delegates (CPU-only, backward compatible).
  ///
  /// - Most compatible (works on all platforms)
  /// - No additional memory overhead
  /// - Baseline performance
  disabled,

  /// XNNPACK delegate for CPU optimization.
  ///
  /// - **Desktop only**: macOS, Linux (crashes on Windows, Android, iOS)
  /// - 2-5x faster than disabled mode
  /// - Uses SIMD vectorization (NEON on ARM, AVX on x86)
  /// - Minimal memory overhead (+2-3MB per interpreter)
  ///
  /// On unsupported platforms, automatically falls back to CPU-only execution.
  xnnpack,

  /// GPU delegate for hardware acceleration.
  ///
  /// - **iOS**: Uses Metal (reliable, recommended)
  /// - **Android**: Uses OpenGL/OpenCL (experimental, may crash on some devices)
  /// - **Desktop**: Not supported (falls back to CPU)
  ///
  /// ## Android GPU Delegate Warning
  ///
  /// The Android GPU delegate has known compatibility issues:
  /// - OpenCL unavailable on many devices (Pixel 6, Android 12+)
  /// - OpenGL ES 3.1+ required for fallback
  /// - Memory issues on some Samsung devices
  /// - Partial op support can cause slower performance than CPU
  ///
  /// Only use on Android if you've tested on your target devices.
  gpu,

  /// Automatically choose best delegate for current platform.
  ///
  /// Current behavior:
  /// - **macOS/Linux**: XNNPACK (2-5x speedup)
  /// - **Windows**: CPU-only (XNNPACK crashes)
  /// - **iOS**: Metal GPU delegate
  /// - **Android**: CPU-only (GPU/XNNPACK unreliable)
  ///
  /// This is the recommended default for cross-platform apps.
  auto,
}

/// Configuration for TensorFlow Lite interpreter performance.
///
/// Controls delegate usage and threading for CPU/GPU acceleration.
///
/// ## Recommended Usage
///
/// For cross-platform apps, use `PerformanceConfig.auto()` (the default):
///
/// ```dart
/// // Auto mode - optimal settings per platform (recommended)
/// final detector = PoseDetector();
/// await detector.initialize(); // Uses PerformanceConfig.auto() by default
/// ```
///
/// ## Platform-Specific Examples
///
/// ```dart
/// // Desktop (macOS/Linux): XNNPACK for 2-5x speedup
/// final detector = PoseDetector(
///   performanceConfig: PerformanceConfig.xnnpack(numThreads: 4),
/// );
///
/// // iOS: GPU delegate via Metal (fast and reliable)
/// final detector = PoseDetector(
///   performanceConfig: PerformanceConfig.gpu(),
/// );
///
/// // Android: CPU-only recommended (GPU is experimental)
/// final detector = PoseDetector(
///   performanceConfig: PerformanceConfig.disabled,
/// );
///
/// // Android: GPU delegate (experimental - test on target devices first!)
/// final detector = PoseDetector(
///   performanceConfig: PerformanceConfig.gpu(),
/// );
/// ```
class PerformanceConfig {
  /// Performance mode controlling delegate selection.
  final PerformanceMode mode;

  /// Number of threads for CPU execution.
  ///
  /// - null: Auto-detect optimal count (min(4, Platform.numberOfProcessors))
  /// - 0: No thread pool (single-threaded, good for tiny models)
  /// - 1-8: Explicit thread count
  ///
  /// Diminishing returns after 4 threads for typical models.
  /// Applies to XNNPACK delegate and CPU-only execution.
  final int? numThreads;

  /// Creates a performance configuration.
  ///
  /// Parameters:
  /// - [mode]: Performance mode. Default: [PerformanceMode.auto]
  /// - [numThreads]: Number of threads (null for auto-detection)
  const PerformanceConfig({this.mode = PerformanceMode.auto, this.numThreads});

  /// Creates config with XNNPACK enabled (desktop only).
  ///
  /// XNNPACK provides 2-5x speedup on macOS and Linux.
  /// On unsupported platforms (Windows, Android, iOS), falls back to CPU-only.
  const PerformanceConfig.xnnpack({this.numThreads})
      : mode = PerformanceMode.xnnpack;

  /// Creates config with GPU delegate enabled.
  ///
  /// - **iOS**: Uses Metal (reliable, recommended)
  /// - **Android**: Uses OpenGL/OpenCL (experimental, may crash)
  /// - **Desktop**: Falls back to CPU-only
  const PerformanceConfig.gpu({this.numThreads}) : mode = PerformanceMode.gpu;

  /// Creates config with auto mode (recommended for cross-platform apps).
  ///
  /// Automatically selects the best delegate for each platform:
  /// - macOS/Linux: XNNPACK
  /// - Windows: CPU-only
  /// - iOS: Metal GPU
  /// - Android: CPU-only
  const PerformanceConfig.auto({this.numThreads}) : mode = PerformanceMode.auto;

  /// CPU-only configuration (no delegates, maximum compatibility).
  static const PerformanceConfig disabled = PerformanceConfig(
    mode: PerformanceMode.disabled,
  );

  /// Gets the effective number of threads to use.
  ///
  /// Returns null if mode is disabled.
  int? getEffectiveThreadCount() {
    if (mode == PerformanceMode.disabled) return null;

    if (numThreads != null) {
      return numThreads!.clamp(0, 8);
    }

    return null;
  }
}

/// Collection of pose landmarks with confidence score (internal use).
class PoseLandmarks {
  /// List of 33 body landmarks extracted from the BlazePose model.
  final List<PoseLandmark> landmarks;

  /// Confidence score for the landmark extraction (0.0 to 1.0).
  final double score;

  /// Creates a collection of pose landmarks with a confidence score.
  PoseLandmarks({required this.landmarks, required this.score});
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

  /// Creates a pose landmark with 3D coordinates and visibility score.
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

/// 2D integer pixel coordinate.
class Point {
  /// X coordinate in pixels
  final int x;

  /// Y coordinate in pixels
  final int y;

  /// Creates a 2D pixel coordinate at position ([x], [y]).
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

  /// Creates an axis-aligned bounding box with the specified edges.
  ///
  /// All coordinates are in pixels in the original image space.
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

  /// Creates a detected pose with bounding box, landmarks, and image dimensions.
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
