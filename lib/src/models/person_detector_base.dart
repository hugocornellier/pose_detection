// ignore_for_file: public_member_api_docs

import 'dart:typed_data';
import 'package:meta/meta.dart';
import 'package:flutter_litert/flutter_litert.dart';

abstract class PersonDetectorBase {
  @protected
  Interpreter? interpreter;
  @protected
  bool isInitializedFlag = false;
  @protected
  late int inW;
  @protected
  late int inH;
  @protected
  final outShapes = <List<int>>[];
  @protected
  Float32List? inputBuffer;
  @protected
  Map<int, Object>? cachedOutputs;

  bool get isInitialized => isInitializedFlag;

  @visibleForTesting
  List<Map<String, dynamic>> decodeOutputsForTest(List<dynamic> outputs) {
    return decodeAndSplitOutputs(outputs);
  }

  @protected
  void disposeBase() {
    interpreter?.close();
    interpreter = null;
    cachedOutputs = null;
    isInitializedFlag = false;
  }
}
