// ignore_for_file: avoid_print, implementation_imports

// Validates the person-only short-circuit path in decodeYoloFlat
// (useFastSingleClass) against the reference full-argmax path.
//
//   flutter test integration_test/pose_yolo_decode_test.dart -d macos
//
// Three gates:
//   1. EQUIVALENCE: on captured real YOLO output buffers (every sample image),
//      the fast path must return Detections bit-for-bit identical to the
//      reference path in normal scenes.
//   2. RECALL (cluttered): a synthetic buffer with > top-k high-confidence
//      non-person detections plus a lower-ranked person documents the intended
//      divergence: the reference path evicts the person via the cross-class
//      top-k cap; the fast path keeps it (recall-preserving, never worse).
//   3. A/B BENCHMARK: interleaved fast-vs-reference timing on the same buffer,
//      reporting p50/p95 so the speedup is proven, not assumed.

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

import 'package:pose_detection/src/util/native_image_utils.dart';

const List<String> kSamples = [
  'pose1.jpg',
  'pose2.jpg',
  'pose3.jpg',
  'pose4.jpg',
  'pose5.jpg',
  'pose6.jpg',
  'pose7.jpg',
];

class _Layout {
  final int inW, inH, channels, anchors;
  final bool channelMajor;
  _Layout(this.inW, this.inH, this.channels, this.anchors, this.channelMajor);
}

_Layout _resolveLayout(Uint8List yoloBytes) {
  final probe = Interpreter.fromBuffer(yoloBytes)..allocateTensors();
  final inShape = probe.getInputTensor(0).shape;
  final outShape = probe.getOutputTensor(0).shape;
  probe.close();
  final int d1 = outShape[outShape.length - 2];
  final int d2 = outShape[outShape.length - 1];
  final bool channelMajor = d1 < d2 && (d1 == 84 || d1 == 85);
  return _Layout(
    inShape[2],
    inShape[1],
    channelMajor ? d1 : d2,
    channelMajor ? d2 : d1,
    channelMajor,
  );
}

double _p(List<int> us, double q) {
  final s = List<int>.from(us)..sort();
  return s[(s.length * q).clamp(0, s.length - 1).toInt()] / 1000.0;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Guards two things at once on real captured buffers:
  //  (1) the PRODUCTION default (no useFastSingleClass arg) is exactly the safe
  //      reference path; nobody has flipped the default;
  //  (2) the experimental fast path's KNOWN precision regression is pinned: on
  //      single-/few-person frames it emits >= as many detections as the
  //      reference (the extra ones are near-threshold false positives the
  //      cross-class top-k suppresses). This test documents WHY the fast path is
  //      off by default so the trade-off can't be silently "fixed".
  test(
    'decodeYoloFlat: production default is the safe path; fast path regresses precision',
    () async {
      final yoloBytes = (await rootBundle.load(
        'packages/pose_detection/assets/models/yolov8n_float32.tflite',
      )).buffer.asUint8List();
      final lay = _resolveLayout(yoloBytes);
      final compiled = CompiledModel.fromBufferWithGpuFallback(
        yoloBytes,
        forceCpu: false,
      );
      final inputBuf = Float32List(lay.inH * lay.inW * 3);

      for (final name in kSamples) {
        final mat = cv.imdecode(
          (await rootBundle.load('assets/samples/$name')).buffer.asUint8List(),
          cv.IMREAD_COLOR,
        );
        final iw = mat.cols, ih = mat.rows;
        final (letter, r, dw, dh) = NativeImageUtils.letterbox(
          mat,
          lay.inW,
          lay.inH,
        );
        NativeImageUtils.matToTensorYolo(letter, buffer: inputBuf);
        letter.dispose();

        // Capture ONE raw output buffer; feed every decode the identical bytes so
        // GPU run-to-run nondeterminism can't cause a false mismatch.
        final out0 = (await compiled.runAsync([inputBuf]))[0];

        List<Detection> decode({bool? fast}) => postProcessDetectionsFlat(
          out0,
          channels: lay.channels,
          anchors: lay.anchors,
          channelMajor: lay.channelMajor,
          inputWidth: lay.inW,
          inputHeight: lay.inH,
          r: r,
          dw: dw,
          dh: dh,
          imageWidth: iw,
          imageHeight: ih,
          confThres: 0.5,
          iouThres: 0.45,
          maxDet: 10,
          filterClassId: 0,
          scoresAreProbabilities: true,
          // omit the flag entirely when fast == null to exercise the default.
          useFastSingleClass: fast ?? false,
        );

        final prod = decode(); // production default (flag omitted)
        final ref = decode(fast: false);
        final fast = decode(fast: true);

        // (1) production default == explicit reference path, box-for-box.
        expect(
          prod.length,
          ref.length,
          reason: '$name: default vs reference count',
        );
        for (int i = 0; i < prod.length; i++) {
          expect(
            prod[i].score,
            ref[i].score,
            reason: '$name[$i]: default score',
          );
          for (int k = 0; k < 4; k++) {
            expect(
              prod[i].bboxXYXY[k],
              ref[i].bboxXYXY[k],
              reason: '$name[$i]: default box[$k]',
            );
          }
        }

        // (2) pin the direction of the fast-path regression (>= ref detections).
        expect(
          fast.length,
          greaterThanOrEqualTo(ref.length),
          reason: '$name: fast path should not drop detections vs reference',
        );

        String fmt(List<Detection> ds) => ds
            .map(
              (d) =>
                  '${d.score.toStringAsFixed(2)}@[${d.bboxXYXY.map((v) => v.toStringAsFixed(0)).join(",")}]',
            )
            .join(' ');
        print(
          '$name (${iw}x$ih): ref=${ref.length} {${fmt(ref)}} | '
          'fast=${fast.length} {${fmt(fast)}}',
        );
        mat.dispose();
      }
      compiled.close();
    },
    timeout: const Timeout(Duration(minutes: 10)),
  );

  test('decodeYoloFlat: cross-class top-k mechanism (why the fast path differs)', () {
    // Synthetic channel-major [1, 84, anchors] buffer. effectiveTopk for a
    // 640x640 frame is clamp(round(100*1),20,200)=100. Put 120 high-confidence
    // NON-person (class 1) detections + 1 person (class 0) at a lower score, so
    // the cross-class top-k (100) evicts the person before the person filter.
    const int channels = 84, anchors = 8400;
    const int classStart = 4; // YOLOv8, no objectness
    final out = Float32List(channels * anchors);
    // Helper to set a class logit at (channel, anchor) in channel-major layout.
    void setLogit(int ch, int a, double v) => out[ch * anchors + a] = v;
    void setBox(int a, double cx, double cy, double w, double h) {
      out[0 * anchors + a] = cx;
      out[1 * anchors + a] = cy;
      out[2 * anchors + a] = w;
      out[3 * anchors + a] = h;
    }

    // Initialize all class logits very negative.
    for (int c = classStart; c < channels; c++) {
      for (int a = 0; a < anchors; a++) {
        out[c * anchors + a] = -30.0;
      }
    }
    // 120 strong class-1 (non-person) boxes, well spaced so NMS keeps them.
    for (int i = 0; i < 120; i++) {
      final a = i;
      setLogit(classStart + 1, a, 6.0); // sigmoid(6)=~0.9975
      setBox(a, 20.0 + i * 4.0, 20.0, 10.0, 10.0);
    }
    // 1 person at a lower (but above-threshold) score, distinct location.
    const int pa = 5000;
    setLogit(classStart + 0, pa, 1.0); // sigmoid(1)=~0.73
    setBox(pa, 600.0, 600.0, 30.0, 30.0);

    List<Detection> decode(bool fast) => postProcessDetectionsFlat(
      out,
      channels: channels,
      anchors: anchors,
      channelMajor: true,
      inputWidth: 640,
      inputHeight: 640,
      r: 1.0,
      dw: 0,
      dh: 0,
      imageWidth: 640,
      imageHeight: 640,
      confThres: 0.5,
      iouThres: 0.45,
      maxDet: 10,
      filterClassId: 0,
      // This fixture intentionally contains logits to exercise the legacy
      // decoder contract, unlike the bundled YOLO model used above/below.
      useFastSingleClass: fast,
    );

    final ref = decode(false);
    final fast = decode(true);
    print(
      'cluttered: reference persons=${ref.length}, fast persons=${fast.length}',
    );
    // Reference drops the person (evicted by the 100-cap top-k of non-persons).
    expect(ref.length, 0, reason: 'reference cross-class top-k evicts person');
    // Fast path keeps it.
    expect(fast.length, 1, reason: 'fast path preserves person recall');
  });

  test(
    'decodeYoloFlat: A/B benchmark fast vs reference (p50/p95)',
    () async {
      final yoloBytes = (await rootBundle.load(
        'packages/pose_detection/assets/models/yolov8n_float32.tflite',
      )).buffer.asUint8List();
      final lay = _resolveLayout(yoloBytes);
      final compiled = CompiledModel.fromBufferWithGpuFallback(
        yoloBytes,
        forceCpu: false,
      );
      final inputBuf = Float32List(lay.inH * lay.inW * 3);

      final mat = cv.imdecode(
        (await rootBundle.load(
          'assets/samples/pose1.jpg',
        )).buffer.asUint8List(),
        cv.IMREAD_COLOR,
      );
      final iw = mat.cols, ih = mat.rows;
      final (letter, r, dw, dh) = NativeImageUtils.letterbox(
        mat,
        lay.inW,
        lay.inH,
      );
      NativeImageUtils.matToTensorYolo(letter, buffer: inputBuf);
      letter.dispose();
      final out0 = (await compiled.runAsync([inputBuf]))[0];
      compiled.close();
      mat.dispose();

      List<Detection> decode(bool fast) => postProcessDetectionsFlat(
        out0,
        channels: lay.channels,
        anchors: lay.anchors,
        channelMajor: lay.channelMajor,
        inputWidth: lay.inW,
        inputHeight: lay.inH,
        r: r,
        dw: dw,
        dh: dh,
        imageWidth: iw,
        imageHeight: ih,
        confThres: 0.5,
        iouThres: 0.45,
        maxDet: 10,
        filterClassId: 0,
        scoresAreProbabilities: true,
        useFastSingleClass: fast,
      );

      // Interleave the two so thermal/scheduler drift cancels.
      const int n = 400, warmup = 40;
      for (int i = 0; i < warmup; i++) {
        decode(true);
        decode(false);
      }
      final fastUs = <int>[], refUs = <int>[];
      for (int i = 0; i < n; i++) {
        final s1 = Stopwatch()..start();
        decode(true);
        s1.stop();
        fastUs.add(s1.elapsedMicroseconds);
        final s2 = Stopwatch()..start();
        decode(false);
        s2.stop();
        refUs.add(s2.elapsedMicroseconds);
      }
      final fp50 = _p(fastUs, 0.5), fp95 = _p(fastUs, 0.95);
      final rp50 = _p(refUs, 0.5), rp95 = _p(refUs, 0.95);
      print('\nDECODE A/B (interleaved, n=$n, pose1)');
      print(
        'reference  p50=${rp50.toStringAsFixed(3)}ms p95=${rp95.toStringAsFixed(3)}ms',
      );
      print(
        'fast       p50=${fp50.toStringAsFixed(3)}ms p95=${fp95.toStringAsFixed(3)}ms',
      );
      print(
        'delta p50  ${(rp50 - fp50).toStringAsFixed(3)}ms '
        '(${(100 * (rp50 - fp50) / rp50).toStringAsFixed(1)}% faster)',
      );
      expect(fp50, lessThan(rp50), reason: 'fast path must be faster at p50');
    },
    timeout: const Timeout(Duration(minutes: 10)),
  );
}
