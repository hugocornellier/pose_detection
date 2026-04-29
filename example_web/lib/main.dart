// ignore_for_file: public_member_api_docs

import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:js_interop';
import 'dart:ui_web' as ui_web;

import 'package:flutter/material.dart';
import 'package:pose_detection/pose_detection.dart';
import 'package:web/web.dart' as web;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pose Detection Web',
      home: Scaffold(
        appBar: AppBar(title: const Text('Pose Detection - Web')),
        body: const PoseDetectionWidget(),
      ),
    );
  }
}

class PoseDetectionWidget extends StatefulWidget {
  const PoseDetectionWidget({super.key});

  @override
  State<PoseDetectionWidget> createState() => _PoseDetectionWidgetState();
}

class _PoseDetectionWidgetState extends State<PoseDetectionWidget> {
  String _status = 'Initializing models...';
  Uint8List? _pickedBytes;
  ImageProvider? _preview;
  PoseDetector? _detector;
  bool _isModelReady = false;
  web.HTMLCanvasElement? _displayCanvas;
  bool _hasAnnotation = false;
  double _confidenceThreshold = 0.70;
  final TextEditingController _confidenceController = TextEditingController(
    text: '0.70',
  );
  Timer? _confidenceDebounce;

  static bool _viewFactoryRegistered = false;

  @override
  void initState() {
    super.initState();
    _displayCanvas = web.HTMLCanvasElement()
      ..style.width = '100%'
      ..style.height = '100%'
      ..style.objectFit = 'contain';
    if (!_viewFactoryRegistered) {
      ui_web.platformViewRegistry.registerViewFactory(
        'annotation-canvas',
        (int viewId) => _displayCanvas!,
      );
      _viewFactoryRegistered = true;
    }
    _initializeModel();
  }

  Future<void> _initializeModel() async {
    try {
      setState(() => _status = 'Loading pose detection models...');

      _detector = await PoseDetector.create(
        mode: PoseMode.boxesAndLandmarks,
        landmarkModel: PoseLandmarkModel.heavy,
        detectorConf: _confidenceThreshold,
        useLiteRt: true,
      );

      final String backend =
          (_detector as dynamic).activeAccelerator as String? ?? 'tflite-js';
      setState(() {
        _status =
            'Ready (LiteRT.js, $backend). Select an image to detect poses.';
        _isModelReady = true;
      });
    } catch (e) {
      setState(() {
        _status = 'Failed to initialize: $e';
        _isModelReady = false;
      });
    }
  }

  @override
  void dispose() {
    _confidenceDebounce?.cancel();
    _detector?.dispose();
    _confidenceController.dispose();
    _displayCanvas = null;
    super.dispose();
  }

  void _scheduleConfidenceRerun() {
    _confidenceDebounce?.cancel();
    if (_pickedBytes == null || !_isModelReady) return;
    _confidenceDebounce = Timer(const Duration(milliseconds: 400), () {
      _runDetection();
    });
  }

  Future<void> _pickImageWeb() async {
    final input = web.HTMLInputElement();
    input.accept = 'image/*';
    input.type = 'file';

    final changeCompleter = Completer<void>();
    void changeHandler(web.Event _) {
      changeCompleter.complete();
      input.removeEventListener('change', changeHandler.toJS);
    }

    input.addEventListener('change', changeHandler.toJS);
    input.click();
    await changeCompleter.future;

    final files = input.files;
    if (files == null || files.length == 0) return;
    final file = files.item(0)!;

    final reader = web.FileReader();
    final loadCompleter = Completer<void>();
    void loadHandler(web.Event _) {
      loadCompleter.complete();
      reader.removeEventListener('load', loadHandler.toJS);
    }

    reader.addEventListener('load', loadHandler.toJS);
    reader.readAsArrayBuffer(file);
    await loadCompleter.future;

    final jsBuffer = reader.result as JSArrayBuffer;
    final byteBuffer = jsBuffer.toDart;
    final bytes = Uint8List.view(byteBuffer);

    setState(() {
      _pickedBytes = bytes;
      _preview = MemoryImage(bytes);
      _hasAnnotation = false;
      _status =
          'Loaded ${file.name} (${bytes.lengthInBytes} bytes) - Detecting...';
    });

    await _runDetection();
  }

  Future<void> _runDetection() async {
    if (_pickedBytes == null) {
      setState(() => _status = 'Please select an image first!');
      return;
    }

    if (!_isModelReady || _detector == null) {
      setState(() => _status = 'Models not ready yet, please wait...');
      return;
    }

    final confThres = double.tryParse(_confidenceController.text) ?? 0.70;
    if (confThres < 0.0 || confThres > 1.0) {
      setState(
        () => _status =
            'Invalid confidence threshold! Must be between 0.0 and 1.0',
      );
      return;
    }

    setState(() {
      _status = 'Detecting poses...';
      _hasAnnotation = false;
    });

    try {
      if (_confidenceThreshold != confThres) {
        _confidenceThreshold = confThres;
        await _detector!.dispose();
        _detector = await PoseDetector.create(
          mode: PoseMode.boxesAndLandmarks,
          landmarkModel: PoseLandmarkModel.heavy,
          detectorConf: confThres,
          useLiteRt: true,
        );
      }

      final totalStart = DateTime.now();
      final poses = await _detector!.detect(_pickedBytes!);
      final totalElapsed = DateTime.now().difference(totalStart).inMilliseconds;

      if (poses.isNotEmpty) {
        await _drawAnnotations(poses);
      }

      setState(() {
        _status =
            'Complete! ${poses.length} person(s) detected in ${totalElapsed}ms';
      });
    } catch (e) {
      setState(() => _status = 'Error: $e');
    }
  }

  Future<void> _drawAnnotations(List<Pose> poses) async {
    if (_pickedBytes == null) return;

    // Decode image to get dimensions for canvas
    final blob = web.Blob([_pickedBytes!.toJS].toJS);
    final url = web.URL.createObjectURL(blob);

    try {
      final htmlImage = web.HTMLImageElement();
      final loadCompleter = Completer<void>();
      htmlImage.addEventListener(
        'load',
        ((web.Event _) => loadCompleter.complete()).toJS,
      );
      htmlImage.addEventListener(
        'error',
        ((web.Event _) => loadCompleter.completeError('Failed to load')).toJS,
      );
      htmlImage.src = url;
      await loadCompleter.future;

      final imageWidth = htmlImage.naturalWidth;
      final imageHeight = htmlImage.naturalHeight;

      final canvas = _displayCanvas!;
      canvas.width = imageWidth;
      canvas.height = imageHeight;
      final ctx = canvas.getContext('2d') as web.CanvasRenderingContext2D;

      ctx.drawImage(htmlImage, 0, 0);

      for (final pose in poses) {
        final box = pose.boundingBox;

        final colorStr = pose.score > 0.6 ? 'rgb(0,255,0)' : 'rgb(255,255,0)';

        // Bounding box
        ctx.strokeStyle = colorStr.toJS;
        ctx.lineWidth = 3;
        ctx.strokeRect(
          box.left,
          box.top,
          box.right - box.left,
          box.bottom - box.top,
        );

        // Label
        final label = 'Person ${(pose.score * 100).toStringAsFixed(0)}%';
        ctx.font = '14px Arial';
        final labelWidth = ctx.measureText(label).width + 6;
        ctx.fillStyle = colorStr.toJS;
        ctx.fillRect(box.left, box.top - 20, labelWidth, 20);
        ctx.fillStyle = 'rgb(0,0,0)'.toJS;
        ctx.fillText(label, box.left + 2, box.top - 4);

        if (pose.landmarks.isNotEmpty) {
          _drawSkeleton(ctx, pose.landmarks);
          _drawLandmarks(ctx, pose.landmarks);
        }
      }

      setState(() => _hasAnnotation = true);
    } finally {
      web.URL.revokeObjectURL(url);
    }
  }

  void _drawLandmarks(
    web.CanvasRenderingContext2D ctx,
    List<PoseLandmark> landmarks,
  ) {
    ctx.fillStyle = 'rgb(255,0,0)'.toJS;
    for (final lm in landmarks) {
      if (lm.visibility < 0.5) continue;
      ctx.beginPath();
      ctx.arc(lm.x, lm.y, 4, 0, 2 * math.pi);
      ctx.fill();
    }
  }

  void _drawSkeleton(
    web.CanvasRenderingContext2D ctx,
    List<PoseLandmark> landmarks,
  ) {
    ctx.strokeStyle = 'rgb(0,255,255)'.toJS;
    ctx.lineWidth = 2;
    for (final pair in poseLandmarkConnections) {
      final lm1 = landmarks[pair[0].index];
      final lm2 = landmarks[pair[1].index];

      if (lm1.visibility < 0.5 || lm2.visibility < 0.5) continue;

      ctx.beginPath();
      ctx.moveTo(lm1.x, lm1.y);
      ctx.lineTo(lm2.x, lm2.y);
      ctx.stroke();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              ElevatedButton(
                onPressed: _isModelReady ? _pickImageWeb : null,
                child: const Text('Select Image'),
              ),
              const SizedBox(width: 20),
              const Text(
                'Confidence:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(width: 8),
              SizedBox(
                width: 80,
                child: TextField(
                  controller: _confidenceController,
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(
                    isDense: true,
                    contentPadding: EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 8,
                    ),
                    border: OutlineInputBorder(),
                  ),
                  onChanged: (value) {
                    final val = double.tryParse(value);
                    if (val != null && val >= 0.0 && val <= 1.0) {
                      setState(() => _confidenceThreshold = val);
                      _scheduleConfidenceRerun();
                    }
                  },
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Slider(
                  value: _confidenceThreshold,
                  min: 0.3,
                  max: 0.95,
                  divisions: 65,
                  label: _confidenceThreshold.toStringAsFixed(2),
                  onChanged: _isModelReady
                      ? (value) {
                          setState(() {
                            _confidenceThreshold = value;
                            _confidenceController.text = value.toStringAsFixed(
                              2,
                            );
                          });
                          _scheduleConfidenceRerun();
                        }
                      : null,
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: _isModelReady ? Colors.green.shade50 : Colors.blue.shade50,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              children: [
                Icon(
                  _isModelReady ? Icons.check_circle : Icons.hourglass_empty,
                  color: _isModelReady ? Colors.green : Colors.blue,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    _status,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),
          if (_preview != null)
            Expanded(
              child: Container(
                width: double.infinity,
                clipBehavior: Clip.hardEdge,
                decoration: BoxDecoration(
                  color: Colors.grey.shade100,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.grey.shade300),
                ),
                child: _hasAnnotation
                    ? HtmlElementView(viewType: 'annotation-canvas')
                    : FittedBox(
                        fit: BoxFit.contain,
                        child: Image(image: _preview!),
                      ),
              ),
            ),
        ],
      ),
    );
  }
}
