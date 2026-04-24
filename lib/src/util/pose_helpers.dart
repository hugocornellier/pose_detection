import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
import '../types.dart';

/// Returns the asset path for the given [PoseLandmarkModel] variant.
String poseLandmarkModelPath(PoseLandmarkModel model) =>
    'packages/pose_detection/assets/models/pose_landmark_${model.name}.tflite';

/// Parses raw BlazePose model outputs from flat Float32List buffers into
/// structured [PoseLandmarks].
///
/// Applies sigmoid activation to score, visibility, and presence values.
/// Normalizes x/y coordinates from 256x256 pixel space to [0, 1] range.
///
/// Expected buffer layouts:
/// - [landmarksData]: 195 floats = 33 landmarks × (x, y, z, visibility, presence)
/// - [scoreData]: at least 1 float (reads index 0)
PoseLandmarks parsePoseLandmarksFlat(
  Float32List landmarksData,
  Float32List scoreData,
) {
  final double score = sigmoid(scoreData[0]);
  final List<PoseLandmark> lm = <PoseLandmark>[];

  for (int i = 0; i < 33; i++) {
    final int base = i * 5;
    final double x = clamp01(landmarksData[base + 0] / 256.0);
    final double y = clamp01(landmarksData[base + 1] / 256.0);
    final double z = landmarksData[base + 2];
    final double visibility = sigmoid(landmarksData[base + 3]);
    final double presence = sigmoid(landmarksData[base + 4]);
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

/// Parses raw BlazePose model outputs from dynamic nested lists into
/// structured [PoseLandmarks]. Used by the web path where the JS
/// interpreter returns nested `List<dynamic>` outputs.
///
/// Delegates to [parsePoseLandmarksFlat] after copying values into a
/// [Float32List]; the copy cost is negligible for 195 floats.
PoseLandmarks parsePoseLandmarks(
  List<dynamic> landmarksData,
  List<dynamic> scoreData,
) {
  final List<dynamic> raw = landmarksData[0] as List<dynamic>;
  final Float32List flatLandmarks = Float32List(raw.length);
  for (int i = 0; i < raw.length; i++) {
    flatLandmarks[i] = (raw[i] as num).toDouble();
  }
  final Float32List flatScore = Float32List(1);
  flatScore[0] = (scoreData[0][0] as num).toDouble();
  return parsePoseLandmarksFlat(flatLandmarks, flatScore);
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
