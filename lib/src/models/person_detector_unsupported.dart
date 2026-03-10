// ignore_for_file: public_member_api_docs

import 'package:flutter_litert/flutter_litert.dart';

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

  Future<List<Detection>> detect(
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
