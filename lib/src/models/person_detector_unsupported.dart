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
  YoloV8PersonDetector() {
    throw UnsupportedError('YoloV8PersonDetector is not supported on this platform.');
  }
}
