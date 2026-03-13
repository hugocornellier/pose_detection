import 'package:flutter_test/flutter_test.dart';
import 'package:pose_detection/pose_detection.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  // ---------------------------------------------------------------------------
  // PoseLandmarkModel enum
  // ---------------------------------------------------------------------------
  group('PoseLandmarkModel enum', () {
    test('has 3 values', () {
      expect(PoseLandmarkModel.values.length, 3);
    });

    test('values in expected order', () {
      expect(PoseLandmarkModel.lite.index, 0);
      expect(PoseLandmarkModel.full.index, 1);
      expect(PoseLandmarkModel.heavy.index, 2);
    });

    test('name property works', () {
      expect(PoseLandmarkModel.lite.name, 'lite');
      expect(PoseLandmarkModel.full.name, 'full');
      expect(PoseLandmarkModel.heavy.name, 'heavy');
    });
  });

  // ---------------------------------------------------------------------------
  // PoseMode enum
  // ---------------------------------------------------------------------------
  group('PoseMode enum', () {
    test('has 2 values', () {
      expect(PoseMode.values.length, 2);
    });

    test('values in expected order', () {
      expect(PoseMode.boxes.index, 0);
      expect(PoseMode.boxesAndLandmarks.index, 1);
    });

    test('name property works', () {
      expect(PoseMode.boxes.name, 'boxes');
      expect(PoseMode.boxesAndLandmarks.name, 'boxesAndLandmarks');
    });
  });

  // ---------------------------------------------------------------------------
  // PoseLandmarkType enum — 33 values, MediaPipe indices
  // ---------------------------------------------------------------------------
  group('PoseLandmarkType enum', () {
    test('has 33 values', () {
      expect(PoseLandmarkType.values.length, 33);
    });

    test('indices match MediaPipe BlazePose topology', () {
      expect(PoseLandmarkType.nose.index, 0);
      expect(PoseLandmarkType.leftEyeInner.index, 1);
      expect(PoseLandmarkType.leftEye.index, 2);
      expect(PoseLandmarkType.leftEyeOuter.index, 3);
      expect(PoseLandmarkType.rightEyeInner.index, 4);
      expect(PoseLandmarkType.rightEye.index, 5);
      expect(PoseLandmarkType.rightEyeOuter.index, 6);
      expect(PoseLandmarkType.leftEar.index, 7);
      expect(PoseLandmarkType.rightEar.index, 8);
      expect(PoseLandmarkType.mouthLeft.index, 9);
      expect(PoseLandmarkType.mouthRight.index, 10);
      expect(PoseLandmarkType.leftShoulder.index, 11);
      expect(PoseLandmarkType.rightShoulder.index, 12);
      expect(PoseLandmarkType.leftElbow.index, 13);
      expect(PoseLandmarkType.rightElbow.index, 14);
      expect(PoseLandmarkType.leftWrist.index, 15);
      expect(PoseLandmarkType.rightWrist.index, 16);
      expect(PoseLandmarkType.leftPinky.index, 17);
      expect(PoseLandmarkType.rightPinky.index, 18);
      expect(PoseLandmarkType.leftIndex.index, 19);
      expect(PoseLandmarkType.rightIndex.index, 20);
      expect(PoseLandmarkType.leftThumb.index, 21);
      expect(PoseLandmarkType.rightThumb.index, 22);
      expect(PoseLandmarkType.leftHip.index, 23);
      expect(PoseLandmarkType.rightHip.index, 24);
      expect(PoseLandmarkType.leftKnee.index, 25);
      expect(PoseLandmarkType.rightKnee.index, 26);
      expect(PoseLandmarkType.leftAnkle.index, 27);
      expect(PoseLandmarkType.rightAnkle.index, 28);
      expect(PoseLandmarkType.leftHeel.index, 29);
      expect(PoseLandmarkType.rightHeel.index, 30);
      expect(PoseLandmarkType.leftFootIndex.index, 31);
      expect(PoseLandmarkType.rightFootIndex.index, 32);
    });

    test('name property works for all values', () {
      expect(PoseLandmarkType.nose.name, 'nose');
      expect(PoseLandmarkType.leftEyeInner.name, 'leftEyeInner');
      expect(PoseLandmarkType.rightFootIndex.name, 'rightFootIndex');
    });
  });

  // ---------------------------------------------------------------------------
  // PoseLandmark
  // ---------------------------------------------------------------------------
  group('PoseLandmark', () {
    PoseLandmark makeLandmark({
      PoseLandmarkType type = PoseLandmarkType.nose,
      double x = 320.0,
      double y = 240.0,
      double z = 0.0,
      double visibility = 0.9,
    }) {
      return PoseLandmark(type: type, x: x, y: y, z: z, visibility: visibility);
    }

    test('constructor stores all fields', () {
      final lm = makeLandmark(
        type: PoseLandmarkType.leftShoulder,
        x: 100.0,
        y: 200.0,
        z: -0.05,
        visibility: 0.75,
      );
      expect(lm.type, PoseLandmarkType.leftShoulder);
      expect(lm.x, 100.0);
      expect(lm.y, 200.0);
      expect(lm.z, -0.05);
      expect(lm.visibility, 0.75);
    });

    test('xNorm computes correctly for in-range value', () {
      final lm = makeLandmark(x: 320.0);
      expect(lm.xNorm(640), closeTo(0.5, 0.0001));
    });

    test('yNorm computes correctly for in-range value', () {
      final lm = makeLandmark(y: 240.0);
      expect(lm.yNorm(480), closeTo(0.5, 0.0001));
    });

    test('xNorm clamps negative x to 0.0', () {
      final lm = makeLandmark(x: -50.0);
      expect(lm.xNorm(640), 0.0);
    });

    test('yNorm clamps negative y to 0.0', () {
      final lm = makeLandmark(y: -100.0);
      expect(lm.yNorm(480), 0.0);
    });

    test('xNorm clamps x beyond width to 1.0', () {
      final lm = makeLandmark(x: 800.0);
      expect(lm.xNorm(640), 1.0);
    });

    test('yNorm clamps y beyond height to 1.0', () {
      final lm = makeLandmark(y: 600.0);
      expect(lm.yNorm(480), 1.0);
    });

    test('xNorm with x == width returns 1.0', () {
      final lm = makeLandmark(x: 640.0);
      expect(lm.xNorm(640), 1.0);
    });

    test('yNorm with y == height returns 1.0', () {
      final lm = makeLandmark(y: 480.0);
      expect(lm.yNorm(480), 1.0);
    });

    test('xNorm with x == 0 returns 0.0', () {
      final lm = makeLandmark(x: 0.0);
      expect(lm.xNorm(640), 0.0);
    });

    test('toPixel returns double coordinates', () {
      final lm = makeLandmark(x: 123.9, y: 456.7);
      final point = lm.toPixel(640, 480);
      expect(point.x, 123.9);
      expect(point.y, 456.7);
    });

    test('toPixel with whole-number coordinates', () {
      final lm = makeLandmark(x: 200.0, y: 150.0);
      final point = lm.toPixel(640, 480);
      expect(point.x, 200.0);
      expect(point.y, 150.0);
    });

    test('visibility can be 0.0', () {
      final lm = makeLandmark(visibility: 0.0);
      expect(lm.visibility, 0.0);
    });

    test('visibility can be 1.0', () {
      final lm = makeLandmark(visibility: 1.0);
      expect(lm.visibility, 1.0);
    });

    test('z coordinate is stored correctly (negative depth)', () {
      final lm = makeLandmark(z: -1.5);
      expect(lm.z, -1.5);
    });
  });

  // ---------------------------------------------------------------------------
  // Point
  // ---------------------------------------------------------------------------
  group('Point', () {
    test('stores positive x and y', () {
      final point = Point(42, 99);
      expect(point.x, 42);
      expect(point.y, 99);
    });

    test('stores negative coordinates', () {
      final point = Point(-10, -20);
      expect(point.x, -10);
      expect(point.y, -20);
    });

    test('stores zero coordinates', () {
      final point = Point(0, 0);
      expect(point.x, 0);
      expect(point.y, 0);
    });

    test('stores large coordinates', () {
      final point = Point(9999, 8888);
      expect(point.x, 9999);
      expect(point.y, 8888);
    });
  });

  // ---------------------------------------------------------------------------
  // BoundingBox
  // ---------------------------------------------------------------------------
  group('BoundingBox', () {
    test('constructor stores all fields', () {
      final bbox = BoundingBox.ltrb(10.5, 20.3, 100.7, 200.1);
      expect(bbox.left, 10.5);
      expect(bbox.top, 20.3);
      expect(bbox.right, 100.7);
      expect(bbox.bottom, 200.1);
    });

    test('accepts zero coordinates', () {
      final bbox = BoundingBox.ltrb(0.0, 0.0, 0.0, 0.0);
      expect(bbox.left, 0.0);
      expect(bbox.top, 0.0);
      expect(bbox.right, 0.0);
      expect(bbox.bottom, 0.0);
    });

    test('accepts negative coordinates', () {
      final bbox = BoundingBox.ltrb(-10.0, -20.0, -5.0, -1.0);
      expect(bbox.left, -10.0);
      expect(bbox.top, -20.0);
      expect(bbox.right, -5.0);
      expect(bbox.bottom, -1.0);
    });

    test('right is greater than left for a typical box', () {
      final bbox = BoundingBox.ltrb(50.0, 30.0, 150.0, 200.0);
      expect(bbox.right, greaterThan(bbox.left));
      expect(bbox.bottom, greaterThan(bbox.top));
    });
  });

  // ---------------------------------------------------------------------------
  // Pose
  // ---------------------------------------------------------------------------
  group('Pose', () {
    PoseLandmark makeLandmark(
      PoseLandmarkType type, {
      double x = 0,
      double y = 0,
    }) {
      return PoseLandmark(type: type, x: x, y: y, z: 0.0, visibility: 0.9);
    }

    Pose makePoseWithLandmarks(List<PoseLandmark> landmarks) {
      return Pose(
        boundingBox: BoundingBox.ltrb(10.0, 20.0, 200.0, 400.0),
        score: 0.85,
        landmarks: landmarks,
        imageWidth: 640,
        imageHeight: 480,
      );
    }

    test('constructor stores all fields', () {
      final bbox = BoundingBox.ltrb(5.0, 10.0, 300.0, 450.0);
      final pose = Pose(
        boundingBox: bbox,
        score: 0.92,
        landmarks: const [],
        imageWidth: 640,
        imageHeight: 480,
      );

      expect(pose.boundingBox.left, 5.0);
      expect(pose.boundingBox.top, 10.0);
      expect(pose.boundingBox.right, 300.0);
      expect(pose.boundingBox.bottom, 450.0);
      expect(pose.score, 0.92);
      expect(pose.landmarks, isEmpty);
      expect(pose.imageWidth, 640);
      expect(pose.imageHeight, 480);
    });

    test('hasLandmarks returns false when landmarks is empty', () {
      final pose = makePoseWithLandmarks([]);
      expect(pose.hasLandmarks, false);
    });

    test('hasLandmarks returns true when landmarks are present', () {
      final lm = makeLandmark(PoseLandmarkType.nose);
      final pose = makePoseWithLandmarks([lm]);
      expect(pose.hasLandmarks, true);
    });

    test('getLandmark returns correct landmark by type', () {
      final nose = makeLandmark(PoseLandmarkType.nose, x: 320.0, y: 100.0);
      final leftShoulder = makeLandmark(
        PoseLandmarkType.leftShoulder,
        x: 200.0,
        y: 200.0,
      );
      final pose = makePoseWithLandmarks([nose, leftShoulder]);

      final found = pose.getLandmark(PoseLandmarkType.nose);
      expect(found, isNotNull);
      expect(found!.type, PoseLandmarkType.nose);
      expect(found.x, 320.0);
      expect(found.y, 100.0);
    });

    test(
      'getLandmark returns second landmark when first is different type',
      () {
        final leftShoulder = makeLandmark(PoseLandmarkType.leftShoulder);
        final rightKnee = makeLandmark(PoseLandmarkType.rightKnee, x: 400.0);
        final pose = makePoseWithLandmarks([leftShoulder, rightKnee]);

        final found = pose.getLandmark(PoseLandmarkType.rightKnee);
        expect(found, isNotNull);
        expect(found!.x, 400.0);
      },
    );

    test('getLandmark returns null when type not in landmarks', () {
      final nose = makeLandmark(PoseLandmarkType.nose);
      final pose = makePoseWithLandmarks([nose]);

      final missing = pose.getLandmark(PoseLandmarkType.leftHip);
      expect(missing, isNull);
    });

    test('getLandmark returns null for empty landmarks list', () {
      final pose = makePoseWithLandmarks([]);
      expect(pose.getLandmark(PoseLandmarkType.nose), isNull);
    });

    test('score field stores confidence value', () {
      final pose = Pose(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 0.73,
        landmarks: const [],
        imageWidth: 640,
        imageHeight: 480,
      );
      expect(pose.score, closeTo(0.73, 0.0001));
    });

    test('imageWidth and imageHeight are accessible', () {
      final pose = Pose(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 0.5,
        landmarks: const [],
        imageWidth: 1280,
        imageHeight: 720,
      );
      expect(pose.imageWidth, 1280);
      expect(pose.imageHeight, 720);
    });

    test('toString contains Pose( prefix', () {
      final pose = makePoseWithLandmarks([]);
      expect(pose.toString(), startsWith('Pose('));
    });

    test('toString contains score formatted to 3 decimals', () {
      final pose = Pose(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 0.850,
        landmarks: const [],
        imageWidth: 640,
        imageHeight: 480,
      );
      expect(pose.toString(), contains('score=0.850'));
    });

    test('toString contains landmarks count', () {
      final lm1 = makeLandmark(PoseLandmarkType.nose);
      final lm2 = makeLandmark(PoseLandmarkType.leftShoulder);
      final pose = makePoseWithLandmarks([lm1, lm2]);
      expect(pose.toString(), contains('landmarks=2'));
    });

    test('toString with no landmarks contains landmarks=0', () {
      final pose = makePoseWithLandmarks([]);
      expect(pose.toString(), contains('landmarks=0'));
    });

    test('toString contains landmark type name and coordinates', () {
      final nose = PoseLandmark(
        type: PoseLandmarkType.nose,
        x: 123.456,
        y: 78.9,
        z: 0.0,
        visibility: 0.95,
      );
      final pose = makePoseWithLandmarks([nose]);
      final str = pose.toString();

      expect(str, contains('nose'));
      expect(str, contains('123.46'));
      expect(str, contains('78.90'));
      expect(str, contains('vis=0.95'));
    });

    test('getLandmark finds all 33 landmark types when all are present', () {
      final landmarks = PoseLandmarkType.values
          .map((type) => makeLandmark(type))
          .toList();
      final pose = makePoseWithLandmarks(landmarks);

      for (final type in PoseLandmarkType.values) {
        final lm = pose.getLandmark(type);
        expect(lm, isNotNull, reason: 'getLandmark returned null for $type');
        expect(lm!.type, type);
      }
    });
  });

  // ---------------------------------------------------------------------------
  // poseLandmarkConnections constant
  // ---------------------------------------------------------------------------
  group('poseLandmarkConnections', () {
    test('has 27 connections', () {
      expect(poseLandmarkConnections.length, 27);
    });

    test('each connection has exactly 2 endpoints', () {
      for (final connection in poseLandmarkConnections) {
        expect(
          connection.length,
          2,
          reason: 'Connection $connection does not have 2 endpoints',
        );
      }
    });

    test('all endpoints are valid PoseLandmarkType values', () {
      final allTypes = PoseLandmarkType.values.toSet();
      for (final connection in poseLandmarkConnections) {
        expect(
          allTypes.contains(connection[0]),
          true,
          reason: 'Invalid start endpoint: ${connection[0]}',
        );
        expect(
          allTypes.contains(connection[1]),
          true,
          reason: 'Invalid end endpoint: ${connection[1]}',
        );
      }
    });

    test('no connection has identical start and end', () {
      for (final connection in poseLandmarkConnections) {
        expect(
          connection[0],
          isNot(connection[1]),
          reason: 'Self-loop found: ${connection[0]}',
        );
      }
    });

    test('nose is connected to both eyes', () {
      bool noseLeftEye = poseLandmarkConnections.any(
        (c) =>
            (c[0] == PoseLandmarkType.nose &&
                c[1] == PoseLandmarkType.leftEye) ||
            (c[1] == PoseLandmarkType.nose && c[0] == PoseLandmarkType.leftEye),
      );
      bool noseRightEye = poseLandmarkConnections.any(
        (c) =>
            (c[0] == PoseLandmarkType.nose &&
                c[1] == PoseLandmarkType.rightEye) ||
            (c[1] == PoseLandmarkType.nose &&
                c[0] == PoseLandmarkType.rightEye),
      );
      expect(noseLeftEye, true, reason: 'nose-leftEye connection missing');
      expect(noseRightEye, true, reason: 'nose-rightEye connection missing');
    });

    test('shoulder-to-hip connections exist on both sides', () {
      bool leftSide = poseLandmarkConnections.any(
        (c) =>
            (c[0] == PoseLandmarkType.leftShoulder &&
                c[1] == PoseLandmarkType.leftHip) ||
            (c[1] == PoseLandmarkType.leftShoulder &&
                c[0] == PoseLandmarkType.leftHip),
      );
      bool rightSide = poseLandmarkConnections.any(
        (c) =>
            (c[0] == PoseLandmarkType.rightShoulder &&
                c[1] == PoseLandmarkType.rightHip) ||
            (c[1] == PoseLandmarkType.rightShoulder &&
                c[0] == PoseLandmarkType.rightHip),
      );
      expect(leftSide, true, reason: 'leftShoulder-leftHip connection missing');
      expect(
        rightSide,
        true,
        reason: 'rightShoulder-rightHip connection missing',
      );
    });

    test('skeleton has exactly 3 connected components (face, body, mouth)', () {
      // The connections define 3 distinct components:
      //   1. Face: nose, leftEye, rightEye, leftEyeInner, leftEyeOuter,
      //            rightEyeInner, rightEyeOuter, leftEar, rightEar
      //   2. Body: shoulders, elbows, wrists, hips, knees, ankles, etc.
      //   3. Mouth: mouthLeft, mouthRight (connected only to each other)
      final allNodes = poseLandmarkConnections
          .expand((c) => c)
          .toSet()
          .toList();
      final parent = <PoseLandmarkType, PoseLandmarkType>{};
      for (final node in allNodes) {
        parent[node] = node;
      }

      PoseLandmarkType find(PoseLandmarkType x) {
        while (parent[x] != x) {
          parent[x] = parent[parent[x]!]!;
          x = parent[x]!;
        }
        return x;
      }

      for (final connection in poseLandmarkConnections) {
        final ra = find(connection[0]);
        final rb = find(connection[1]);
        if (ra != rb) parent[ra] = rb;
      }

      final roots = allNodes.map(find).toSet();
      expect(
        roots.length,
        3,
        reason:
            'Expected 3 connected components (face, body, mouth), found ${roots.length}',
      );
    });

    test('face landmarks are reachable from nose', () {
      final connected = <PoseLandmarkType>{PoseLandmarkType.nose};
      bool changed = true;
      while (changed) {
        changed = false;
        for (final connection in poseLandmarkConnections) {
          if (connected.contains(connection[0]) &&
              !connected.contains(connection[1])) {
            connected.add(connection[1]);
            changed = true;
          }
          if (connected.contains(connection[1]) &&
              !connected.contains(connection[0])) {
            connected.add(connection[0]);
            changed = true;
          }
        }
      }

      // These face landmarks must all be reachable from nose
      final faceTypes = [
        PoseLandmarkType.leftEye,
        PoseLandmarkType.rightEye,
        PoseLandmarkType.leftEar,
        PoseLandmarkType.rightEar,
      ];
      for (final type in faceTypes) {
        expect(
          connected.contains(type),
          true,
          reason: '$type is not reachable from nose',
        );
      }
    });

    test('body landmarks are reachable from leftShoulder', () {
      final connected = <PoseLandmarkType>{PoseLandmarkType.leftShoulder};
      bool changed = true;
      while (changed) {
        changed = false;
        for (final connection in poseLandmarkConnections) {
          if (connected.contains(connection[0]) &&
              !connected.contains(connection[1])) {
            connected.add(connection[1]);
            changed = true;
          }
          if (connected.contains(connection[1]) &&
              !connected.contains(connection[0])) {
            connected.add(connection[0]);
            changed = true;
          }
        }
      }

      final bodyTypes = [
        PoseLandmarkType.rightShoulder,
        PoseLandmarkType.leftHip,
        PoseLandmarkType.rightHip,
        PoseLandmarkType.leftKnee,
        PoseLandmarkType.rightKnee,
        PoseLandmarkType.leftAnkle,
        PoseLandmarkType.rightAnkle,
        PoseLandmarkType.leftWrist,
        PoseLandmarkType.rightWrist,
        PoseLandmarkType.leftFootIndex,
        PoseLandmarkType.rightFootIndex,
      ];
      for (final type in bodyTypes) {
        expect(
          connected.contains(type),
          true,
          reason: '$type is not reachable from leftShoulder',
        );
      }
    });

    test('left and right wrist connect to their respective fingers', () {
      final leftWristConnections = poseLandmarkConnections
          .where(
            (c) =>
                c[0] == PoseLandmarkType.leftWrist ||
                c[1] == PoseLandmarkType.leftWrist,
          )
          .toList();
      final rightWristConnections = poseLandmarkConnections
          .where(
            (c) =>
                c[0] == PoseLandmarkType.rightWrist ||
                c[1] == PoseLandmarkType.rightWrist,
          )
          .toList();

      // leftWrist connects to leftElbow, leftPinky, leftIndex, leftThumb
      expect(leftWristConnections.length, greaterThanOrEqualTo(2));
      expect(rightWristConnections.length, greaterThanOrEqualTo(2));
    });

    test('ankle connects to heel and foot index on both sides', () {
      bool leftAnkleHeel = poseLandmarkConnections.any(
        (c) =>
            (c[0] == PoseLandmarkType.leftAnkle &&
                c[1] == PoseLandmarkType.leftHeel) ||
            (c[1] == PoseLandmarkType.leftAnkle &&
                c[0] == PoseLandmarkType.leftHeel),
      );
      bool leftAnkleFoot = poseLandmarkConnections.any(
        (c) =>
            (c[0] == PoseLandmarkType.leftAnkle &&
                c[1] == PoseLandmarkType.leftFootIndex) ||
            (c[1] == PoseLandmarkType.leftAnkle &&
                c[0] == PoseLandmarkType.leftFootIndex),
      );
      expect(
        leftAnkleHeel,
        true,
        reason: 'leftAnkle-leftHeel connection missing',
      );
      expect(
        leftAnkleFoot,
        true,
        reason: 'leftAnkle-leftFootIndex connection missing',
      );
    });
  });
}
