// ignore_for_file: public_member_api_docs

import 'dart:typed_data';
import 'types.dart';

class PoseDetector {
  final PoseMode mode;
  final PoseLandmarkModel landmarkModel;
  final double detectorConf;
  final double detectorIou;
  final int maxDetections;
  final double minLandmarkScore;
  final int interpreterPoolSize;
  final PerformanceConfig performanceConfig;
  final bool useNativePreprocessing;

  PoseDetector({
    this.mode = PoseMode.boxesAndLandmarks,
    this.landmarkModel = PoseLandmarkModel.heavy,
    this.detectorConf = 0.5,
    this.detectorIou = 0.45,
    this.maxDetections = 10,
    this.minLandmarkScore = 0.5,
    this.interpreterPoolSize = 1,
    this.performanceConfig = PerformanceConfig.disabled,
    this.useNativePreprocessing = true,
  }) {
    throw UnsupportedError('PoseDetector is not supported on this platform.');
  }

  Future<void> initialize() => throw UnsupportedError('Not supported');
  bool get isInitialized => false;
  Future<void> dispose() => throw UnsupportedError('Not supported');
  Future<List<Pose>> detect(Uint8List imageBytes) =>
      throw UnsupportedError('Not supported');
  Future<List<Pose>> detectFromMat(
    Object mat, {
    required int imageWidth,
    required int imageHeight,
  }) => throw UnsupportedError('Not supported');
}
