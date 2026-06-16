// ignore_for_file: avoid_print

// Smoke test for the accelerator / precision control params added to PoseDetector.
//
// Inits the detector with useCompiledModel: true and accelerators: {Accelerator.cpu},
// runs one detect call, and asserts no throw.
//
//   flutter test integration_test/accelerator_control_test.dart -d macos

import 'dart:io' show Platform;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:pose_detection/pose_detection.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test(
    'PoseDetector init with useCompiledModel + cpu accelerator does not throw',
    timeout: const Timeout(Duration(minutes: 5)),
    skip: Platform.isMacOS ? false : 'macOS-only (LiteRT CompiledModel)',
    () async {
      final detector = PoseDetector();
      await detector.initialize(
        mode: PoseMode.boxesAndLandmarks,
        landmarkModel: PoseLandmarkModel.lite,
        useCompiledModel: true,
        accelerators: {Accelerator.cpu},
        precision: Precision.fp32,
      );

      try {
        final imageData = await rootBundle.load('assets/samples/pose1.jpg');
        final bytes = imageData.buffer.asUint8List();
        final poses = await detector.detect(bytes);
        print('accelerator_control_test: detected ${poses.length} pose(s)');
        // Just assert we got a list back (no throw is the main assertion).
        expect(poses, isA<List<Pose>>());
      } finally {
        await detector.dispose();
      }
    },
  );

  test(
    'PoseDetector.create with accelerators + precision params does not throw',
    timeout: const Timeout(Duration(minutes: 5)),
    skip: Platform.isMacOS ? false : 'macOS-only (LiteRT CompiledModel)',
    () async {
      final detector = await PoseDetector.create(
        mode: PoseMode.boxes,
        landmarkModel: PoseLandmarkModel.lite,
        useCompiledModel: true,
        accelerators: {Accelerator.cpu},
        precision: Precision.fp16,
      );

      try {
        final imageData = await rootBundle.load('assets/samples/pose1.jpg');
        final bytes = imageData.buffer.asUint8List();
        final poses = await detector.detect(bytes);
        print(
          'accelerator_control_test (create): detected ${poses.length} pose(s)',
        );
        expect(poses, isA<List<Pose>>());
      } finally {
        await detector.dispose();
      }
    },
  );
}
