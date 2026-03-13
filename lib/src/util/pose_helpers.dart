import 'package:flutter_litert/flutter_litert.dart';
import '../types.dart';

/// Returns the asset path for the given [PoseLandmarkModel] variant.
String poseLandmarkModelPath(PoseLandmarkModel model) =>
    'packages/pose_detection/assets/models/pose_landmark_${model.name}.tflite';

/// Parses raw BlazePose model outputs into structured [PoseLandmarks].
///
/// Applies sigmoid activation to score, visibility, and presence values.
/// Normalizes x/y coordinates from 256x256 pixel space to [0, 1] range.
PoseLandmarks parsePoseLandmarks(
  List<dynamic> landmarksData,
  List<dynamic> scoreData,
) {
  final double score = sigmoid((scoreData[0][0] as num).toDouble());
  final List<dynamic> raw = landmarksData[0] as List<dynamic>;
  final List<PoseLandmark> lm = <PoseLandmark>[];

  for (int i = 0; i < 33; i++) {
    final int base = i * 5;
    final double x = clamp01((raw[base + 0] as num).toDouble() / 256.0);
    final double y = clamp01((raw[base + 1] as num).toDouble() / 256.0);
    final double z = (raw[base + 2] as num).toDouble();
    final double visibility = sigmoid((raw[base + 3] as num).toDouble());
    final double presence = sigmoid((raw[base + 4] as num).toDouble());
    final double vis = (visibility * presence).clamp(0.0, 1.0);

    lm.add(
      PoseLandmark(
        type: PoseLandmarkType.values[i],
        x: x,
        y: y,
        z: z,
        visibility: vis,
      ),
    );
  }

  return PoseLandmarks(landmarks: lm, score: score);
}

/// Builds box-only [Pose] results from detections (no landmarks).
List<Pose> buildBoxOnlyPoses(
  List<Detection> dets,
  int imageWidth,
  int imageHeight,
) {
  return [for (final d in dets) buildBoxOnlyPose(d, imageWidth, imageHeight)];
}

/// Builds a box-only [Pose] from a single [Detection] (no landmarks).
Pose buildBoxOnlyPose(Detection d, int imageWidth, int imageHeight) {
  return Pose(
    boundingBox: BoundingBox.ltrb(
      d.bboxXYXY[0],
      d.bboxXYXY[1],
      d.bboxXYXY[2],
      d.bboxXYXY[3],
    ),
    score: d.score,
    landmarks: const <PoseLandmark>[],
    imageWidth: imageWidth,
    imageHeight: imageHeight,
  );
}
