// ignore_for_file: public_member_api_docs

import '../types.dart';

class YoloDetection {
  final int cls;
  final double score;
  final List<double> bboxXYXY;

  YoloDetection({
    required this.cls,
    required this.score,
    required this.bboxXYXY,
  });
}

class YoloV8PersonDetector {
  static const int cocoPersonClassId = 0;

  YoloV8PersonDetector() {
    throw UnsupportedError(
      'YoloV8PersonDetector is not supported on this platform.',
    );
  }

  bool get isInitialized => false;

  Future<void> initialize({PerformanceConfig? performanceConfig}) =>
      throw UnsupportedError('Not supported');

  Future<void> dispose() => throw UnsupportedError('Not supported');

  Future<List<YoloDetection>> detect(
    Object imageOrMat, {
    required int imageWidth,
    required int imageHeight,
    double confThres = 0.35,
    double iouThres = 0.4,
    int maxDet = 10,
    bool personOnly = true,
  }) => throw UnsupportedError('Not supported');

  List<Map<String, dynamic>> decodeOutputsForTest(List<dynamic> outputs) =>
      throw UnsupportedError('Not supported');
}
