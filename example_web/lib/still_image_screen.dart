// ignore_for_file: public_member_api_docs

import 'dart:async';
import 'dart:js_interop';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui_web' as ui_web;

import 'package:pose_detection/pose_detection.dart';
import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

class StillImageScreen extends StatefulWidget {
  const StillImageScreen({super.key});

  @override
  State<StillImageScreen> createState() => _StillImageScreenState();
}

class _StillImageScreenState extends State<StillImageScreen> {
  // ---- Detector lifecycle -----------------------------------------------
  String _status = 'Initializing models...';
  Uint8List? _pickedBytes;
  ImageProvider? _preview;
  PoseDetector? _detector;
  bool _isModelReady = false;
  web.HTMLCanvasElement? _displayCanvas;
  bool _hasAnnotation = false;

  // ---- Detection mode + model ------------------------------------------
  PoseMode _mode = PoseMode.boxesAndLandmarks;
  PoseLandmarkModel _model = PoseLandmarkModel.heavy;

  // ---- Display toggles -------------------------------------------------
  bool _showBoundingBoxes = true;
  bool _showSkeleton = true;
  bool _showKeypoints = true;

  // ---- Colors ----------------------------------------------------------
  Color _boundingBoxColor = const Color(0xFF00FFCC);
  Color _keypointColor = const Color(0xFF89CFF0);
  Color _skeletonColor = const Color(0xFFF4C2C2);

  // ---- Sizes -----------------------------------------------------------
  double _boundingBoxThickness = 2.0;
  double _keypointSize = 3.0;
  double _skeletonThickness = 2.0;

  // ---- LiteRT settings -------------------------------------------------
  bool _useLiteRt = true;
  String _liteRtAccelerator = 'auto';

  static bool _viewFactoryRegistered = false;
  Timer? _rerunDebounce;

  @override
  void initState() {
    super.initState();
    _displayCanvas = web.HTMLCanvasElement()
      ..style.width = '100%'
      ..style.height = '100%'
      ..style.objectFit = 'contain';
    if (!_viewFactoryRegistered) {
      ui_web.platformViewRegistry.registerViewFactory(
        'pose-annotation-canvas',
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
        landmarkModel: _model,
        mode: _mode,
        useLiteRt: _useLiteRt,
        liteRtAccelerator: _liteRtAccelerator,
      );
      setState(() {
        final backend =
            (_detector as dynamic).activeAccelerator as String? ?? 'tflite-js';
        _status = 'Ready (LiteRT.js, $backend). Pick an image.';
        _isModelReady = true;
      });
    } catch (e) {
      setState(() {
        _status = 'Failed to initialize: $e';
        _isModelReady = false;
      });
    }
  }

  Future<void> _reinitialize() async {
    setState(() {
      _status = 'Reloading models...';
      _isModelReady = false;
    });
    try {
      await _detector?.dispose();
    } catch (_) {}
    _detector = null;
    await _initializeModel();
    if (_pickedBytes != null) {
      await _runDetection();
    }
  }

  @override
  void dispose() {
    _rerunDebounce?.cancel();
    _detector?.dispose();
    _displayCanvas = null;
    super.dispose();
  }

  void _scheduleRerun() {
    _rerunDebounce?.cancel();
    if (_pickedBytes == null || !_isModelReady) return;
    _rerunDebounce = Timer(const Duration(milliseconds: 250), _runDetection);
  }

  Future<void> _pickImage() async {
    final input = web.HTMLInputElement();
    input.accept = 'image/*';
    input.type = 'file';
    final completer = Completer<void>();
    void changeHandler(web.Event _) {
      completer.complete();
      input.removeEventListener('change', changeHandler.toJS);
    }

    input.addEventListener('change', changeHandler.toJS);
    input.click();
    await completer.future;
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
    final bytes = Uint8List.view(jsBuffer.toDart);

    setState(() {
      _pickedBytes = bytes;
      _preview = MemoryImage(bytes);
      _hasAnnotation = false;
      _status =
          'Loaded ${file.name} (${bytes.lengthInBytes} bytes); detecting...';
    });
    await _runDetection();
  }

  Future<void> _pickSample(String assetPath) async {
    final data = await DefaultAssetBundle.of(context).load(assetPath);
    final bytes = data.buffer.asUint8List();
    setState(() {
      _pickedBytes = Uint8List.fromList(bytes);
      _preview = MemoryImage(_pickedBytes!);
      _hasAnnotation = false;
      _status = 'Loaded $assetPath; detecting...';
    });
    await _runDetection();
  }

  Future<void> _runDetection() async {
    if (_pickedBytes == null) return;
    if (!_isModelReady || _detector == null) return;

    setState(() {
      _status = 'Detecting...';
      _hasAnnotation = false;
    });
    try {
      final sw = Stopwatch()..start();
      final poses = await _detector!.detect(_pickedBytes!);
      sw.stop();
      await _drawAnnotations(poses);
      setState(() {
        _status =
            'Detected ${poses.length} pose(s) in ${sw.elapsedMilliseconds}ms';
      });
    } catch (e) {
      setState(() => _status = 'Error: $e');
    }
  }

  Future<void> _drawAnnotations(List<Pose> poses) async {
    if (_pickedBytes == null) return;
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
        ((web.Event _) => loadCompleter.completeError('decode failed')).toJS,
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
        _drawPose(ctx, pose);
      }
      setState(() => _hasAnnotation = true);
    } finally {
      web.URL.revokeObjectURL(url);
    }
  }

  void _drawPose(web.CanvasRenderingContext2D ctx, Pose pose) {
    if (_showBoundingBoxes) {
      ctx.strokeStyle = _cssColor(_boundingBoxColor).toJS;
      ctx.lineWidth = _boundingBoxThickness;
      final box = pose.boundingBox;
      ctx.strokeRect(
        box.left,
        box.top,
        box.right - box.left,
        box.bottom - box.top,
      );
    }

    if (_showSkeleton && pose.hasLandmarks) {
      ctx.strokeStyle = _cssColor(_skeletonColor).toJS;
      ctx.lineWidth = _skeletonThickness;
      for (final connection in poseLandmarkConnections) {
        final a = pose.getLandmark(connection[0]);
        final b = pose.getLandmark(connection[1]);
        if (a == null || b == null) continue;
        if (a.visibility < 0.5 || b.visibility < 0.5) continue;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }

    if (_showKeypoints) {
      ctx.fillStyle = _cssColor(_keypointColor).toJS;
      for (final p in pose.landmarks) {
        if (p.visibility < 0.5) continue;
        ctx.beginPath();
        ctx.arc(p.x, p.y, _keypointSize, 0, 2 * math.pi);
        ctx.fill();
      }
    }
  }

  String _cssColor(Color c) {
    final r = (c.r * 255).round();
    final g = (c.g * 255).round();
    final b = (c.b * 255).round();
    return 'rgb($r,$g,$b)';
  }

  void _showSettings() {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (ctx, setSheet) {
            void setBoth(VoidCallback fn) {
              fn();
              setState(() {});
              setSheet(() {});
            }

            return ListView(
              padding: const EdgeInsets.all(16),
              children: [
                const Text(
                  'Detection settings',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
                _modeSelector(setBoth),
                _modelSelector(setBoth),
                const Divider(),
                const Text(
                  'Display options',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                CheckboxListTile(
                  dense: true,
                  title: const Text('Show bounding boxes'),
                  value: _showBoundingBoxes,
                  onChanged: (v) =>
                      setBoth(() => _showBoundingBoxes = v ?? true),
                ),
                CheckboxListTile(
                  dense: true,
                  title: const Text('Show skeleton'),
                  value: _showSkeleton,
                  onChanged: (v) => setBoth(() => _showSkeleton = v ?? true),
                ),
                CheckboxListTile(
                  dense: true,
                  title: const Text('Show keypoints'),
                  value: _showKeypoints,
                  onChanged: (v) => setBoth(() => _showKeypoints = v ?? true),
                ),
                const Divider(),
                const Text(
                  'Sizes',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                _slider(
                  'BBox thickness',
                  _boundingBoxThickness,
                  0.5,
                  10.0,
                  (v) => setBoth(() => _boundingBoxThickness = v),
                ),
                _slider(
                  'Keypoint size',
                  _keypointSize,
                  0.5,
                  15.0,
                  (v) => setBoth(() => _keypointSize = v),
                ),
                _slider(
                  'Skeleton thickness',
                  _skeletonThickness,
                  0.5,
                  10.0,
                  (v) => setBoth(() => _skeletonThickness = v),
                ),
                const Divider(),
                const Text(
                  'Colors',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                _colorPicker(
                  'Bounding box',
                  _boundingBoxColor,
                  (c) => setBoth(() => _boundingBoxColor = c),
                ),
                _colorPicker(
                  'Keypoints',
                  _keypointColor,
                  (c) => setBoth(() => _keypointColor = c),
                ),
                _colorPicker(
                  'Skeleton',
                  _skeletonColor,
                  (c) => setBoth(() => _skeletonColor = c),
                ),
                const Divider(),
                _liteRtSection(setBoth),
                const SizedBox(height: 24),
              ],
            );
          },
        );
      },
    ).whenComplete(_scheduleRerun);
  }

  Widget _modeSelector(void Function(VoidCallback) setBoth) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          const Text('Mode'),
          const SizedBox(width: 12),
          for (final m in PoseMode.values) ...[
            ChoiceChip(
              label: Text(m.name),
              selected: _mode == m,
              onSelected: (_) async {
                setBoth(() => _mode = m);
                await _reinitialize();
              },
            ),
            const SizedBox(width: 4),
          ],
        ],
      ),
    );
  }

  Widget _modelSelector(void Function(VoidCallback) setBoth) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Wrap(
        crossAxisAlignment: WrapCrossAlignment.center,
        spacing: 6,
        children: [
          const Text('Model'),
          for (final m in PoseLandmarkModel.values)
            ChoiceChip(
              label: Text(m.name),
              selected: _model == m,
              onSelected: (_) async {
                setBoth(() => _model = m);
                await _reinitialize();
              },
            ),
        ],
      ),
    );
  }

  Widget _slider(
    String label,
    double value,
    double min,
    double max,
    void Function(double) onChanged,
  ) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(width: 130, child: Text(label)),
          Expanded(
            child: Slider(
              value: value.clamp(min, max),
              min: min,
              max: max,
              onChanged: (v) {
                onChanged(v);
                _scheduleRerun();
              },
            ),
          ),
          SizedBox(width: 50, child: Text(value.toStringAsFixed(2))),
        ],
      ),
    );
  }

  Widget _colorPicker(String label, Color current, void Function(Color) on) {
    const palette = <Color>[
      Color(0xFF00FFCC),
      Color(0xFF89CFF0),
      Color(0xFFF4C2C2),
      Color(0xFF22AAFF),
      Color(0xFFFFAA22),
      Color(0xFFFF3355),
      Color(0xFF66FF66),
      Color(0xFFFFFF66),
      Color(0xFFFF66FF),
      Color(0xFFFFFFFF),
    ];
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(width: 130, child: Text(label)),
          for (final c in palette)
            GestureDetector(
              onTap: () {
                on(c);
                _scheduleRerun();
              },
              child: Container(
                margin: const EdgeInsets.only(right: 4),
                width: 22,
                height: 22,
                decoration: BoxDecoration(
                  color: c,
                  border: Border.all(
                    color: current.toARGB32() == c.toARGB32()
                        ? Colors.black
                        : Colors.transparent,
                    width: 2,
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _liteRtSection(void Function(VoidCallback) setBoth) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'LiteRT (web runtime)',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        SwitchListTile(
          dense: true,
          title: const Text('Use LiteRT.js'),
          subtitle: const Text(
            'Auto WebGPU / WASM. Disable to use the legacy tflite-js path.',
          ),
          value: _useLiteRt,
          onChanged: (v) async {
            setBoth(() => _useLiteRt = v);
            await _reinitialize();
          },
        ),
        Wrap(
          spacing: 6,
          children: [
            const Text('Accelerator:'),
            for (final a in const <String>['auto', 'webgpu', 'wasm'])
              ChoiceChip(
                label: Text(a),
                selected: _liteRtAccelerator == a,
                onSelected: (_) async {
                  setBoth(() => _liteRtAccelerator = a);
                  await _reinitialize();
                },
              ),
          ],
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Still Image')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: _buildContent(),
      ),
    );
  }

  Widget _buildContent() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            ElevatedButton.icon(
              onPressed: _isModelReady ? _pickImage : null,
              icon: const Icon(Icons.image),
              label: const Text('Select image'),
            ),
            const SizedBox(width: 8),
            for (final s in const [
              'assets/samples/pose1.jpg',
              'assets/samples/pose2.jpg',
              'assets/samples/pose3.jpg',
              'assets/samples/pose4.jpg',
              'assets/samples/pose5.jpg',
              'assets/samples/pose6.jpg',
              'assets/samples/pose7.jpg',
            ])
              Padding(
                padding: const EdgeInsets.only(right: 4),
                child: OutlinedButton(
                  onPressed: _isModelReady ? () => _pickSample(s) : null,
                  child: Text(s.split('/').last.split('.').first),
                ),
              ),
            const Spacer(),
            IconButton(
              icon: const Icon(Icons.tune),
              tooltip: 'Settings',
              onPressed: _showSettings,
            ),
          ],
        ),
        const SizedBox(height: 12),
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
              Expanded(child: Text(_status)),
            ],
          ),
        ),
        const SizedBox(height: 12),
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
                  ? const HtmlElementView(viewType: 'pose-annotation-canvas')
                  : FittedBox(
                      fit: BoxFit.contain,
                      child: Image(image: _preview!),
                    ),
            ),
          ),
      ],
    );
  }
}
