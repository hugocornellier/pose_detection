import 'dart:typed_data';
import 'types.dart';

class PoseDetector {
  PoseDetector({
    PoseMode mode = PoseMode.boxesAndLandmarks,
    PoseLandmarkModel landmarkModel = PoseLandmarkModel.heavy,
    double detectorConf = 0.5,
    double detectorIou = 0.45,
    int maxDetections = 10,
    double minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = PerformanceConfig.disabled,
    bool useNativePreprocessing = true,
  }) {
    throw UnsupportedError('PoseDetector is not supported on this platform.');
  }

  Future<void> initialize() => throw UnsupportedError('Not supported');
  bool get isInitialized => false;
  Future<void> dispose() => throw UnsupportedError('Not supported');
  Future<List<Pose>> detect(Uint8List imageBytes) => throw UnsupportedError('Not supported');
}
