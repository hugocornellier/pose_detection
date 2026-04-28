// ignore_for_file: public_member_api_docs

import 'dart:typed_data';
import 'package:flutter_litert/flutter_litert.dart'
    show PerformanceConfig, CameraFrame, CameraFrameRotation;
import '../types.dart';

class PoseDetector {
  static const String _packageVersion = '3.0.0';
  static const String _pipelineVersion = 'pipeline_v1';

  /// Version key for the default pose detection pipeline.
  static const String modelVersion =
      'pose_detection:$_packageVersion:mode=boxesAndLandmarks:'
      'landmarkModel=heavy:$_pipelineVersion';

  /// Builds a version key for a specific pose detector configuration.
  static String modelVersionFor({
    PoseMode mode = PoseMode.boxesAndLandmarks,
    PoseLandmarkModel landmarkModel = PoseLandmarkModel.heavy,
  }) {
    return 'pose_detection:$_packageVersion:mode=${mode.name}:'
        'landmarkModel=${landmarkModel.name}:$_pipelineVersion';
  }

  PoseDetector() {
    throw UnsupportedError('PoseDetector is not supported on this platform.');
  }

  static Future<PoseDetector> create({
    PoseMode mode = PoseMode.boxesAndLandmarks,
    PoseLandmarkModel landmarkModel = PoseLandmarkModel.heavy,
    double detectorConf = 0.5,
    double detectorIou = 0.45,
    int maxDetections = 10,
    double minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = PerformanceConfig.disabled,
    bool useLiteRt = false,
    String liteRtAccelerator = 'wasm',
  }) => throw UnsupportedError('Not supported');

  Future<void> initialize({
    PoseMode mode = PoseMode.boxesAndLandmarks,
    PoseLandmarkModel landmarkModel = PoseLandmarkModel.heavy,
    double detectorConf = 0.5,
    double detectorIou = 0.45,
    int maxDetections = 10,
    double minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = PerformanceConfig.disabled,
    bool useLiteRt = false,
    String liteRtAccelerator = 'wasm',
  }) => throw UnsupportedError('Not supported');

  Future<void> initializeFromBuffers({
    required Uint8List yoloBytes,
    required Uint8List landmarkBytes,
    PoseMode mode = PoseMode.boxesAndLandmarks,
    PoseLandmarkModel landmarkModel = PoseLandmarkModel.heavy,
    double detectorConf = 0.5,
    double detectorIou = 0.45,
    int maxDetections = 10,
    double minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = PerformanceConfig.disabled,
  }) => throw UnsupportedError('Not supported');

  bool get isReady => false;
  bool get isInitialized => false;
  Future<void> dispose() => throw UnsupportedError('Not supported');
  Future<List<Pose>> detect(Uint8List imageBytes) =>
      throw UnsupportedError('Not supported');
  Future<List<Pose>> detectFromFilepath(String path) =>
      throw UnsupportedError('Not supported');
  Future<List<Pose>> detectFromMat(Object mat) =>
      throw UnsupportedError('Not supported');
  Future<List<Pose>> detectFromMatBytes(
    Uint8List bytes, {
    required int width,
    required int height,
    int matType = 16,
  }) => throw UnsupportedError('Not supported');
  Future<List<Pose>> detectFromCameraFrame(CameraFrame frame, {int? maxDim}) =>
      throw UnsupportedError('Not supported');
  Future<List<Pose>> detectFromCameraImage(
    Object cameraImage, {
    CameraFrameRotation? rotation,
    bool isBgra = true,
    int? maxDim,
  }) => throw UnsupportedError('Not supported');
}
