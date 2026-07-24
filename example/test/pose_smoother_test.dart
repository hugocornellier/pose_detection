import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:pose_detection/pose_detection.dart';
import 'package:pose_detection_example/main.dart';

/// Guards [PoseSmoother] against dropping fields it does not filter.
///
/// Smoothing only touches landmark x/y, but it rebuilds the whole [Pose] to do
/// it, so any field left off that constructor call disappears without an error.
/// `segmentationMask` was dropped this way. The example app never enables
/// segmentation, so nothing surfaced it.
void main() {
  SegmentationMask buildMask() => SegmentationMask(
    width: 4,
    height: 4,
    confidences: Uint8List.fromList(List<int>.filled(16, 200)),
    imageLeft: 10.0,
    imageTop: 20.0,
    imageWidth: 100.0,
    imageHeight: 100.0,
  );

  Pose buildPose({SegmentationMask? mask}) => Pose(
    boundingBox: BoundingBox.ltrb(10.0, 20.0, 110.0, 120.0),
    score: 0.9,
    landmarks: <PoseLandmark>[
      PoseLandmark(
        type: PoseLandmarkType.nose,
        x: 50.0,
        y: 60.0,
        z: 0.1,
        visibility: 0.8,
      ),
    ],
    imageWidth: 640,
    imageHeight: 480,
    segmentationMask: mask,
  );

  group('PoseSmoother', () {
    test('preserves segmentationMask through smoothing', () {
      final smoother = PoseSmoother();
      final mask = buildMask();

      // Two frames: the first seeds the track, the second exercises the
      // filtered path where the Pose is actually rebuilt.
      smoother.apply(<Pose>[buildPose(mask: mask)], 0.0);
      final out = smoother.apply(<Pose>[buildPose(mask: mask)], 1 / 30);

      expect(out, hasLength(1));
      expect(
        out.first.segmentationMask,
        isNotNull,
        reason: 'smoothing rebuilt the Pose and dropped segmentationMask',
      );
      expect(out.first.segmentationMask!.width, mask.width);
      expect(out.first.segmentationMask!.confidences, mask.confidences);
      expect(out.first.segmentationMask!.imageLeft, mask.imageLeft);
    });

    test('preserves a null mask as null', () {
      final smoother = PoseSmoother();
      smoother.apply(<Pose>[buildPose()], 0.0);
      final out = smoother.apply(<Pose>[buildPose()], 1 / 30);

      expect(out, hasLength(1));
      expect(out.first.segmentationMask, isNull);
    });

    test('still carries the unfiltered scalar fields', () {
      final smoother = PoseSmoother();
      final pose = buildPose(mask: buildMask());
      smoother.apply(<Pose>[pose], 0.0);
      final out = smoother.apply(<Pose>[pose], 1 / 30).first;

      expect(out.score, pose.score);
      expect(out.imageWidth, pose.imageWidth);
      expect(out.imageHeight, pose.imageHeight);
      expect(out.landmarks.single.type, PoseLandmarkType.nose);
      expect(out.landmarks.single.z, pose.landmarks.single.z);
      expect(out.landmarks.single.visibility, pose.landmarks.single.visibility);
    });

    test('is a no-op when disabled', () {
      final smoother = PoseSmoother(enabled: false);
      final pose = buildPose(mask: buildMask());
      final out = smoother.apply(<Pose>[pose], 0.0);

      expect(identical(out.first, pose), isTrue);
    });
  });
}
