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
    bool useCompiledModel = false,
    Set<Accelerator> accelerators = const {Accelerator.gpu, Accelerator.cpu},
    Precision precision = Precision.fp16,
    bool enableSegmentation = false,
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

    // Unlike the interpreter path above, the CompiledModel path does NOT pin
    // YOLO to CPU on iOS. The detection-count variance that motivated the pin
    // (10 vs 2 detections on the same image, v2.0.6) was a defect of the old
    // TFLite Metal GPU delegate; the LiteRT Next Metal accelerator used here
    // produces stable counts on-device at fp16 while cutting live latency
    // roughly 4x (measured iPhone, 2026-07-17: ~100ms pinned vs ~22ms GPU).
    _yolo = YoloV8PersonDetector();
    await _yolo!.initializeFromBuffer(
      yoloBytes,
      performanceConfig: yoloConfig,
      useCompiledModel: useCompiledModel,
      accelerators: accelerators,
      precision: precision,
    );

    _lm = PoseLandmarkModelRunner(poolSize: interpreterPoolSize);
    await _lm!.initializeFromBuffer(
      landmarkBytes,
      performanceConfig: performanceConfig,
      useCompiledModel: useCompiledModel,
      accelerators: accelerators,
      precision: precision,
      enableSegmentation: enableSegmentation,
    );
  }

  Future<List<Pose>> detectDirect(
    cv.Mat image, {
    required int imageWidth,
    required int imageHeight,
  }) async {
    if (_yolo == null || _lm == null) {
      throw StateError(
        'PoseDetectorCore not initialized. Call initializeFromBuffers() first.',
      );
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

      // Fused crop + resize: warp the aligned square straight to the 256x256
      // landmark input in one resample instead of cropping at native size and
      // then calling cv.resize.
      final cv.Mat? resized = NativeImageUtils.extractAlignedSquare(
        image,
        cx,
        cy,
        side,
        0.0,
        outSize: 256,
      );

      if (resized == null) continue;

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

    // Run landmark inference for all people concurrently. The runner's pool
    // (interpreter or CompiledModel) round-robins across slots and serializes
    // per slot, so this overlaps inference across people when the pool has more
    // than one slot and is safe (back-to-back) when it has one.
    final List<PoseLandmarks?> allLandmarks = await Future.wait(
      cropDataList.map<Future<PoseLandmarks?>>((data) async {
        try {
          return await _lm!.run(data.letterboxedMat!);
        } catch (_) {
          return null;
        }
      }),
    );

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

      // The raw 256x256 mask (when present) covers the padded square crop the
      // landmark model saw, so it maps back to image space with the same
      // cropX/cropY/side geometry used for the landmarks above.
      final Uint8List? rawMask = lms.segmentationMask;
      final SegmentationMask? mask = rawMask == null
          ? null
          : SegmentationMask(
              width: 256,
              height: 256,
              confidences: rawMask,
              imageLeft: data.cropX.toDouble(),
              imageTop: data.cropY.toDouble(),
              imageWidth: data.cropWidth.toDouble(),
              imageHeight: data.cropHeight.toDouble(),
            );

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
          segmentationMask: mask,
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
