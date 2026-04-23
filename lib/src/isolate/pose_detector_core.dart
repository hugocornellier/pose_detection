import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart';
import 'package:meta/meta.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

import '../models/person_detector_native.dart';
import '../models/pose_landmark_model_native.dart';
import '../types.dart';
import '../util/native_image_utils.dart';
import '../util/pose_helpers.dart';

class _PersonCropData {
  final Detection detection;
  final cv.Mat? letterboxedMat;
  final int cropX;
  final int cropY;
  final int cropWidth;
  final int cropHeight;

  _PersonCropData({
    required this.detection,
    this.letterboxedMat,
    required this.cropX,
    required this.cropY,
    this.cropWidth = 0,
    this.cropHeight = 0,
  });

  void dispose() {
    letterboxedMat?.dispose();
  }
}

/// Direct-mode inference core used inside the pose detection background isolate.
///
/// Holds both TFLite models and runs the full pose detection pipeline on the
/// calling thread. Created inside [PoseDetector]'s background isolate.
///
/// Not part of the public API.
@internal
class PoseDetectorCore {
  YoloV8PersonDetector? _yolo;
  PoseLandmarkModelRunner? _lm;

  PoseMode _mode = PoseMode.boxesAndLandmarks;
  int _maxDetections = 10;
  double _minLandmarkScore = 0.5;
  double _detectorConf = 0.5;
  double _detectorIou = 0.45;

  bool get isReady => _yolo != null;

  Future<void> initializeFromBuffers({
    required Uint8List yoloBytes,
    required Uint8List landmarkBytes,
    required PoseMode mode,
    required PoseLandmarkModel landmarkModel,
    required double detectorConf,
    required double detectorIou,
    required int maxDetections,
    required double minLandmarkScore,
    required int interpreterPoolSize,
    required PerformanceConfig performanceConfig,
  }) async {
    _mode = mode;
    _maxDetections = maxDetections;
    _minLandmarkScore = minLandmarkScore;
    _detectorConf = detectorConf;
    _detectorIou = detectorIou;

    // On iOS, use XNNPACK for YOLO to avoid Metal floating-point precision
    // inconsistencies that cause variable detection counts with YOLOv8n.
    // The landmark model continues to use whatever the caller specifies.
    final yoloConfig =
        (Platform.isIOS && performanceConfig.mode == PerformanceMode.auto)
        ? PerformanceConfig.xnnpack()
        : performanceConfig;

    _yolo = YoloV8PersonDetector();
    await _yolo!.initializeFromBuffer(yoloBytes, performanceConfig: yoloConfig);

    _lm = PoseLandmarkModelRunner(poolSize: interpreterPoolSize);
    await _lm!.initializeFromBuffer(
      landmarkBytes,
      performanceConfig: performanceConfig,
    );
  }

  Future<List<Pose>> detectDirect(
    cv.Mat image, {
    required int imageWidth,
    required int imageHeight,
  }) async {
    if (_yolo == null || _lm == null) {
      throw StateError('PoseDetectorCore not initialized.');
    }

    final List<Detection> dets = await _yolo!.detect(
      image,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      confThres: _detectorConf,
      iouThres: _detectorIou,
      maxDet: _maxDetections,
      personOnly: true,
    );

    if (_mode == PoseMode.boxes) {
      return buildBoxOnlyPoses(dets, imageWidth, imageHeight);
    }

    final List<_PersonCropData> cropDataList = <_PersonCropData>[];
    for (final Detection d in dets) {
      final double x1 = d.bboxXYXY[0].clamp(0.0, imageWidth.toDouble());
      final double y1 = d.bboxXYXY[1].clamp(0.0, imageHeight.toDouble());
      final double x2 = d.bboxXYXY[2].clamp(0.0, imageWidth.toDouble());
      final double y2 = d.bboxXYXY[3].clamp(0.0, imageHeight.toDouble());
      final double bw = x2 - x1;
      final double bh = y2 - y1;

      final double cx = (x1 + x2) / 2.0;
      final double cy = (y1 + y2) / 2.0;
      final double side = (bw > bh ? bw : bh) * 1.25;

      final cv.Mat? square = NativeImageUtils.extractAlignedSquare(
        image,
        cx,
        cy,
        side,
        0.0,
      );

      if (square == null) continue;

      final cv.Mat resized = cv.resize(square, (
        256,
        256,
      ), interpolation: cv.INTER_LINEAR);
      square.dispose();

      final double sqX1 = cx - side / 2.0;
      final double sqY1 = cy - side / 2.0;

      cropDataList.add(
        _PersonCropData(
          detection: d,
          letterboxedMat: resized,
          cropX: sqX1.round(),
          cropY: sqY1.round(),
          cropWidth: side.round(),
          cropHeight: side.round(),
        ),
      );
    }

    final List<PoseLandmarks?> allLandmarks = <PoseLandmarks?>[];
    for (final _PersonCropData data in cropDataList) {
      try {
        final PoseLandmarks lms = await _lm!.run(data.letterboxedMat!);
        allLandmarks.add(lms);
      } catch (_) {
        allLandmarks.add(null);
      }
    }

    final List<Pose> results = _buildLandmarkResults(
      cropDataList,
      allLandmarks,
      imageWidth,
      imageHeight,
    );

    for (final data in cropDataList) {
      data.dispose();
    }

    return results;
  }

  List<Pose> _buildLandmarkResults(
    List<_PersonCropData> cropDataList,
    List<PoseLandmarks?> allLandmarks,
    int imageWidth,
    int imageHeight,
  ) {
    final List<Pose> results = <Pose>[];
    for (int i = 0; i < cropDataList.length; i++) {
      final _PersonCropData data = cropDataList[i];
      final PoseLandmarks? lms = allLandmarks[i];

      if (lms == null || lms.score < _minLandmarkScore) {
        results.add(buildBoxOnlyPose(data.detection, imageWidth, imageHeight));
        continue;
      }

      final List<PoseLandmark> pts = <PoseLandmark>[];
      for (final PoseLandmark lm in lms.landmarks) {
        final double xOrig = (data.cropX + lm.x * data.cropWidth).clamp(
          0.0,
          imageWidth.toDouble(),
        );
        final double yOrig = (data.cropY + lm.y * data.cropHeight).clamp(
          0.0,
          imageHeight.toDouble(),
        );

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
          boundingBox: BoundingBox.ltrb(
            data.detection.bboxXYXY[0],
            data.detection.bboxXYXY[1],
            data.detection.bboxXYXY[2],
            data.detection.bboxXYXY[3],
          ),
          score: data.detection.score,
          landmarks: pts,
          imageWidth: imageWidth,
          imageHeight: imageHeight,
        ),
      );
    }
    return results;
  }

  Future<void> dispose() async {
    await _yolo?.dispose();
    await _lm?.dispose();
    _yolo = null;
    _lm = null;
  }
}
