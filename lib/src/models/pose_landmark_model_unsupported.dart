// ignore_for_file: public_member_api_docs

import 'dart:typed_data';

import '../types.dart';

class PoseLandmarkModelRunner {
  PoseLandmarkModelRunner({int poolSize = 1}) {
    throw UnsupportedError(
      'PoseLandmarkModelRunner is not supported on this platform.',
    );
  }

  bool get isInitialized => false;

  int get poolSize => 1;

  Future<void> initialize(
    PoseLandmarkModel model, {
    PerformanceConfig? performanceConfig,
  }) => throw UnsupportedError('Not supported');

  Future<PoseLandmarks> run(Object mat) =>
      throw UnsupportedError('Not supported');

  Future<PoseLandmarks> runFromRgba(Uint8List rgbaBytes) =>
      throw UnsupportedError('Not supported');

  Future<void> dispose() => throw UnsupportedError('Not supported');
}
