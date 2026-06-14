// ignore_for_file: public_member_api_docs

import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:math' as math;
import 'dart:ui_web' as ui_web;

import 'package:pose_detection/pose_detection.dart';
import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

/// View types registered once per platform-view, mapped to the canvas that
/// should currently back them. Storing the live canvas (rather than capturing
/// it inside the factory closure) keeps revisited screens from rendering into
/// a disposed canvas.
final Map<String, web.HTMLCanvasElement> _activeCanvases = {};
final Set<String> _registeredViewTypes = {};

/// Shared pose-detection + canvas-overlay machinery for the web example
/// screens that draw on top of an `<video>` element.
///
/// Both [LiveCameraScreen] and the video-file screen pull frames from an
/// [web.HTMLVideoElement] and run [PoseDetector.detectFromVideo] on each
/// frame, drawing the results onto a 2D canvas. Everything that does not
/// depend on *where* the video frames come from lives here; subclasses supply
/// the frame source (camera stream vs. file) and the playback controls.
mixin DetectionOverlayMixin<T extends StatefulWidget> on State<T> {
  // ---- Detector --------------------------------------------------------
  PoseDetector? detector;
  bool isModelReady = false;

  // ---- Video + canvas --------------------------------------------------
  web.HTMLVideoElement? video;
  web.HTMLCanvasElement? displayCanvas;
  bool supportsRvfc = false;

  // ---- Detection settings ---------------------------------------------
  PoseMode mode = PoseMode.boxesAndLandmarks;
  PoseLandmarkModel model = PoseLandmarkModel.lite;

  // ---- Display toggles -------------------------------------------------
  bool showBoundingBoxes = true;
  bool showSkeleton = true;
  bool showKeypoints = true;

  // ---- Colors / sizes --------------------------------------------------
  final Color boundingBoxColor = const Color(0xFF00FFCC);
  final Color skeletonColor = const Color(0xFFF4C2C2);
  final Color keypointColor = const Color(0xFF89CFF0);
  final double bboxThickness = 2.0;
  final double keypointSize = 3.0;
  final double skeletonThickness = 2.0;
  final double minVisibility = 0.5;

  // ---- LiteRT ----------------------------------------------------------
  bool useLiteRt = true;
  String liteRtAccelerator = 'auto';

  // ---- Loop / FPS ------------------------------------------------------
  bool busy = false;
  bool disposed = false;
  int rafHandle = 0;
  final FpsCounter fpsCounter = FpsCounter();
  double fps = 0;
  int lastDetectionMs = 0;
  int detectedPoses = 0;

  // ---- Hooks the host screen must provide ------------------------------

  /// Unique platform-view type for this screen's canvas.
  String get viewType;

  /// Whether the detection loop should keep scheduling frames. Camera: the
  /// stream is running. Video file: the element is playing.
  bool get loopActive;

  /// Re-create the detector after a setting that requires a fresh instance
  /// (mode / model / LiteRT) changes, preserving playback state.
  Future<void> restartDetector();

  /// Called once the detector finishes loading. [backend] is the active
  /// accelerator string (e.g. `webgpu`, `wasm`, `tflite-js`).
  void onDetectorReady(String backend);

  /// Called if detector creation throws.
  void onDetectorError(Object error);

  // ---- Setup -----------------------------------------------------------

  /// Creates the backing canvas and registers it as a platform view. Call
  /// from the host screen's `initState`.
  void registerCanvas() {
    displayCanvas = web.HTMLCanvasElement()
      ..style.width = '100%'
      ..style.height = '100%'
      ..style.objectFit = 'contain'
      ..style.backgroundColor = '#000';
    _activeCanvases[viewType] = displayCanvas!;
    if (_registeredViewTypes.add(viewType)) {
      ui_web.platformViewRegistry.registerViewFactory(
        viewType,
        (int _) => _activeCanvases[viewType]!,
      );
    }
  }

  bool videoPrototypeHasRvfc() {
    try {
      final dynamic ctor = globalContext['HTMLVideoElement'];
      if (ctor == null) return false;
      final proto = (ctor as JSObject)['prototype'];
      if (proto == null) return false;
      return (proto as JSObject).has('requestVideoFrameCallback');
    } catch (_) {
      return false;
    }
  }

  Future<void> waitForMetadata(web.HTMLVideoElement v) async {
    if (v.videoWidth > 0 && v.videoHeight > 0) return;
    final completer = Completer<void>();
    void handler(web.Event _) {
      if (!completer.isCompleted) completer.complete();
    }

    final handlerJs = handler.toJS;
    v.addEventListener('loadedmetadata', handlerJs);
    try {
      await completer.future.timeout(const Duration(seconds: 5));
    } finally {
      v.removeEventListener('loadedmetadata', handlerJs);
    }
  }

  String get activeAccelerator =>
      (detector as dynamic)?.activeAccelerator as String? ?? '?';

  // ---- Detector lifecycle ---------------------------------------------

  Future<void> initializeDetector() async {
    try {
      detector = await PoseDetector.create(
        landmarkModel: model,
        mode: mode,
        useLiteRt: useLiteRt,
        liteRtAccelerator: liteRtAccelerator,
      );
      if (!mounted) {
        await detector?.dispose();
        detector = null;
        return;
      }
      isModelReady = true;
      onDetectorReady(
        activeAccelerator == '?' ? 'tflite-js' : activeAccelerator,
      );
    } catch (e) {
      if (!mounted) return;
      onDetectorError(e);
    }
  }

  Future<void> disposeDetector() async {
    final d = detector;
    detector = null;
    isModelReady = false;
    await d?.dispose();
  }

  // ---- Detection loop --------------------------------------------------

  void scheduleFrame() {
    if (video == null || disposed || !loopActive) return;
    if (supportsRvfc) {
      try {
        // ignore: avoid_dynamic_calls
        (video! as dynamic).requestVideoFrameCallback(
          ((double now, JSObject metadata) {
            onFrame();
          }).toJS,
        );
        return;
      } catch (_) {
        // Fall through to rAF.
      }
    }
    rafHandle = web.window.requestAnimationFrame(
      ((double _) => onFrame()).toJS,
    );
  }

  void onFrame() {
    if (disposed || !loopActive) return;
    runDetection().whenComplete(() {
      if (loopActive) scheduleFrame();
    });
  }

  Future<void> runDetection() async {
    if (busy || detector == null || video == null) return;
    if (video!.videoWidth == 0 || video!.videoHeight == 0) return;
    busy = true;
    try {
      final sw = Stopwatch()..start();
      final poses =
          await (detector! as dynamic).detectFromVideo(video!) as List<Pose>;
      sw.stop();
      draw(poses);
      fpsCounter.tick();
      if (mounted) {
        setState(() {
          fps = fpsCounter.fps.toDouble();
          lastDetectionMs = sw.elapsedMilliseconds;
          detectedPoses = poses.length;
        });
      }
    } catch (_) {
      // Swallow per-frame errors (e.g. video not ready) to avoid stopping the
      // loop.
    } finally {
      busy = false;
    }
  }

  void stopLoop() {
    if (rafHandle != 0) {
      web.window.cancelAnimationFrame(rafHandle);
      rafHandle = 0;
    }
  }

  // ---- Drawing ---------------------------------------------------------

  void draw(List<Pose> poses) {
    final canvas = displayCanvas;
    final v = video;
    if (canvas == null || v == null) return;
    final ctx = canvas.getContext('2d') as web.CanvasRenderingContext2D;
    final int w = v.videoWidth;
    final int h = v.videoHeight;
    if (w == 0 || h == 0) return;

    ctx.drawImage(v, 0, 0, w, h);

    for (final pose in poses) {
      drawPose(ctx, pose);
    }
  }

  void drawPose(web.CanvasRenderingContext2D ctx, Pose pose) {
    if (showBoundingBoxes) {
      ctx.strokeStyle = cssColor(boundingBoxColor).toJS;
      ctx.lineWidth = bboxThickness;
      final box = pose.boundingBox;
      ctx.strokeRect(
        box.left,
        box.top,
        box.right - box.left,
        box.bottom - box.top,
      );
    }
    if (showSkeleton && pose.hasLandmarks) {
      ctx.strokeStyle = cssColor(skeletonColor).toJS;
      ctx.lineWidth = skeletonThickness;
      for (final connection in poseLandmarkConnections) {
        final a = pose.getLandmark(connection[0]);
        final b = pose.getLandmark(connection[1]);
        if (a == null || b == null) continue;
        if (a.visibility < minVisibility || b.visibility < minVisibility) {
          continue;
        }
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }
    if (showKeypoints) {
      ctx.fillStyle = cssColor(keypointColor).toJS;
      for (final p in pose.landmarks) {
        if (p.visibility < minVisibility) continue;
        ctx.beginPath();
        ctx.arc(p.x, p.y, keypointSize, 0, 2 * math.pi);
        ctx.fill();
      }
    }
  }

  String cssColor(Color c) {
    final r = (c.r * 255).round();
    final g = (c.g * 255).round();
    final b = (c.b * 255).round();
    return 'rgb($r,$g,$b)';
  }

  // ---- Shared settings sheet ------------------------------------------

  void showDetectionSettings() {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setSheet) {
          void both(VoidCallback fn) {
            fn();
            setState(() {});
            setSheet(() {});
          }

          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              const Text(
                'Detection',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
              ),
              Wrap(
                spacing: 6,
                children: [
                  const Text('Mode:'),
                  for (final m in PoseMode.values)
                    ChoiceChip(
                      label: Text(m.name),
                      selected: mode == m,
                      onSelected: (_) async {
                        both(() => mode = m);
                        await restartDetector();
                      },
                    ),
                ],
              ),
              Wrap(
                spacing: 6,
                children: [
                  const Text('Model:'),
                  for (final m in PoseLandmarkModel.values)
                    ChoiceChip(
                      label: Text(m.name),
                      selected: model == m,
                      onSelected: (_) async {
                        both(() => model = m);
                        await restartDetector();
                      },
                    ),
                ],
              ),
              const Divider(),
              const Text(
                'Display',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              CheckboxListTile(
                dense: true,
                title: const Text('Bounding boxes'),
                value: showBoundingBoxes,
                onChanged: (v) => both(() => showBoundingBoxes = v ?? true),
              ),
              CheckboxListTile(
                dense: true,
                title: const Text('Skeleton'),
                value: showSkeleton,
                onChanged: (v) => both(() => showSkeleton = v ?? true),
              ),
              CheckboxListTile(
                dense: true,
                title: const Text('Keypoints'),
                value: showKeypoints,
                onChanged: (v) => both(() => showKeypoints = v ?? true),
              ),
              const Divider(),
              const Text(
                'LiteRT.js',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              SwitchListTile(
                dense: true,
                title: const Text('Use LiteRT.js'),
                subtitle: const Text('Auto WebGPU / WASM fallback'),
                value: useLiteRt,
                onChanged: (v) async {
                  both(() => useLiteRt = v);
                  await restartDetector();
                },
              ),
              Wrap(
                spacing: 6,
                children: [
                  const Text('Accelerator:'),
                  for (final a in const <String>['auto', 'webgpu', 'wasm'])
                    ChoiceChip(
                      label: Text(a),
                      selected: liteRtAccelerator == a,
                      onSelected: (_) async {
                        both(() => liteRtAccelerator = a);
                        await restartDetector();
                      },
                    ),
                ],
              ),
              const SizedBox(height: 24),
            ],
          );
        },
      ),
    );
  }
}
