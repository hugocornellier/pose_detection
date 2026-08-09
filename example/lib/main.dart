import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_colorpicker/flutter_colorpicker.dart';
import 'package:camera/camera.dart';
import 'package:pose_detection/pose_detection.dart';
import 'package:flutter_litert/flutter_litert.dart'
    show FrameThrottle, iouLTRB, OneEuroFilter;
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:path_provider/path_provider.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:video_player/video_player.dart';

/// Centers [child] in the available space; when the viewport is too small to
/// fit it, the content scrolls vertically instead of overflowing.
class _ScrollableCentered extends StatelessWidget {
  final Widget child;
  const _ScrollableCentered({required this.child});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) => SingleChildScrollView(
        child: ConstrainedBox(
          constraints: BoxConstraints(minHeight: constraints.maxHeight),
          child: Center(child: child),
        ),
      ),
    );
  }
}

/// Compact labeled swatch that opens a color picker dialog on tap.
class _ColorPickerButton extends StatelessWidget {
  final String label;
  final Color color;
  final ValueChanged<Color> onColorChanged;

  const _ColorPickerButton({
    required this.label,
    required this.color,
    required this.onColorChanged,
  });

  void _pick(BuildContext context) {
    Color tempColor = color;
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Pick $label Color'),
        content: SingleChildScrollView(
          child: ColorPicker(
            pickerColor: color,
            onColorChanged: (c) => tempColor = c,
            pickerAreaHeightPercent: 0.8,
            displayThumbColor: true,
            enableAlpha: true,
            labelTypes: const [ColorLabelType.hex],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              onColorChanged(tempColor);
              Navigator.of(context).pop();
            },
            child: const Text('Select'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () => _pick(context),
      borderRadius: BorderRadius.circular(6),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade300),
          borderRadius: BorderRadius.circular(6),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 18,
              height: 18,
              decoration: BoxDecoration(
                color: color,
                border: Border.all(color: Colors.grey.shade400),
                borderRadius: BorderRadius.circular(3),
              ),
            ),
            const SizedBox(width: 6),
            Text(label, style: const TextStyle(fontSize: 12)),
            const SizedBox(width: 2),
            const Icon(Icons.arrow_drop_down, size: 16),
          ],
        ),
      ),
    );
  }
}

/// Widget shown for the selected dropdown item (inherits parent text style).
class _DropdownSelected extends StatelessWidget {
  final String text;
  const _DropdownSelected(this.text);
  @override
  Widget build(BuildContext context) =>
      Align(alignment: Alignment.centerLeft, child: Text(text));
}

DropdownMenuItem<T> _whiteItem<T>(T value, String label) => DropdownMenuItem<T>(
  value: value,
  child: Text(label, style: const TextStyle(color: Colors.white)),
);

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(
    MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Pose Detection Demo',
      theme: ThemeData(colorSchemeSeed: Colors.blue, useMaterial3: true),
      home: const HomeScreen(),
    ),
  );
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Pose Detection Demo')),
      body: _ScrollableCentered(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 720),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Text(
                    'Choose a Demo',
                    style: Theme.of(context).textTheme.headlineMedium,
                  ),
                ),
                const SizedBox(height: 28),
                _buildSection(context, 'Pose Detection', [
                  _buildModeCard(
                    context,
                    icon: Icons.videocam,
                    title: 'Live Camera',
                    description: 'Real-time pose skeletons from camera feed',
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const LiveCameraScreen(),
                        ),
                      );
                    },
                  ),
                  _buildModeCard(
                    context,
                    icon: Icons.image,
                    title: 'Still Image',
                    description:
                        'Detect person landmarks in photos from gallery',
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const Example(),
                        ),
                      );
                    },
                  ),
                  _buildModeCard(
                    context,
                    icon: Icons.movie_creation_outlined,
                    title: 'Video File',
                    description:
                        'Process an MP4 frame-by-frame with smoothed '
                        'pose skeletons',
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const VideoFileScreen(),
                        ),
                      );
                    },
                  ),
                ]),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSection(BuildContext context, String title, List<Widget> cards) {
    final List<Widget> row = [];
    for (int i = 0; i < cards.length; i++) {
      if (i > 0) row.add(const SizedBox(width: 12));
      row.add(cards[i]);
    }
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Text(
            title,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
              color: Colors.grey[700],
            ),
          ),
        ),
        const SizedBox(height: 12),
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: IntrinsicHeight(
            child: Row(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: row,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildModeCard(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String description,
    required VoidCallback onTap,
  }) {
    return SizedBox(
      width: 190,
      child: Card(
        elevation: 4,
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(12),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(icon, size: 40, color: Colors.blue),
                const SizedBox(height: 12),
                Text(
                  title,
                  style: Theme.of(context).textTheme.titleMedium,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 6),
                Text(
                  description,
                  style: Theme.of(
                    context,
                  ).textTheme.bodySmall?.copyWith(color: Colors.grey[600]),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class Example extends StatefulWidget {
  const Example({super.key});
  @override
  State<Example> createState() => _ExampleState();
}

class _ExampleState extends State<Example> {
  PoseDetector? _poseDetector;
  Uint8List? _imageBytes;
  List<Pose> _poses = [];
  Size? _originalSize;

  bool _isLoading = false;

  int? _detectionTimeMs;
  int? _totalTimeMs;

  PoseMode _detectionMode = PoseMode.boxesAndLandmarks;
  PoseLandmarkModel _detectionModel = PoseLandmarkModel.heavy;

  @override
  void initState() {
    super.initState();
    _initPoseDetector();
  }

  Future<void> _initPoseDetector() async {
    try {
      await _poseDetector?.dispose();
      _poseDetector = PoseDetector();
      await _poseDetector!.initialize(
        mode: _detectionMode,
        landmarkModel: _detectionModel,
        useCompiledModel: true,
      );
    } catch (_) {}
    setState(() {});
  }

  @override
  void dispose() {
    _poseDetector?.dispose();
    super.dispose();
  }

  Future<void> _pickAndRun() async {
    final ImagePicker picker = ImagePicker();
    final XFile? picked = await picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 100,
    );
    if (picked == null) return;

    setState(() {
      _imageBytes = null;
      _poses = [];
      _originalSize = null;
      _isLoading = true;
      _detectionTimeMs = null;
      _totalTimeMs = null;
    });

    final Uint8List bytes = await picked.readAsBytes();

    if (_poseDetector == null || !_poseDetector!.isReady) {
      setState(() => _isLoading = false);
      return;
    }

    await _processImage(bytes);
  }

  Future<void> _processImage(Uint8List bytes) async {
    setState(() => _isLoading = true);

    final DateTime totalStart = DateTime.now();

    final DateTime detectionStart = DateTime.now();
    final List<Pose> poses = await _poseDetector!.detect(bytes);
    final DateTime detectionEnd = DateTime.now();

    Size decodedSize;
    if (poses.isNotEmpty) {
      decodedSize = Size(
        poses.first.imageWidth.toDouble(),
        poses.first.imageHeight.toDouble(),
      );
    } else {
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      decodedSize = Size(
        frame.image.width.toDouble(),
        frame.image.height.toDouble(),
      );
      frame.image.dispose();
    }

    if (!mounted) return;

    final DateTime totalEnd = DateTime.now();
    final int totalTime = totalEnd.difference(totalStart).inMilliseconds;
    final int detectionTime = detectionEnd
        .difference(detectionStart)
        .inMilliseconds;

    setState(() {
      _imageBytes = bytes;
      _originalSize = decodedSize;
      _poses = poses;
      _isLoading = false;
      _detectionTimeMs = detectionTime;
      _totalTimeMs = totalTime;
    });
  }

  @override
  Widget build(BuildContext context) {
    final bool hasImage = _imageBytes != null && _originalSize != null;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Still Image Detection'),
        actions: [
          IconButton(
            onPressed: _pickAndRun,
            icon: const Icon(Icons.add_photo_alternate),
            tooltip: 'Pick Image',
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4.0),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'MODE',
                  style: TextStyle(
                    color: Colors.grey[500],
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 1.0,
                  ),
                ),
                const SizedBox(width: 4),
                DropdownButton<PoseMode>(
                  value: _detectionMode,
                  dropdownColor: Colors.blue[800],
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.onSurface,
                    fontSize: 14,
                  ),
                  underline: const SizedBox(),
                  icon: Icon(
                    Icons.arrow_drop_down,
                    color: Theme.of(context).colorScheme.onSurface,
                  ),
                  selectedItemBuilder: (context) => const [
                    _DropdownSelected('Boxes'),
                    _DropdownSelected('Landmarks'),
                  ],
                  items: [
                    _whiteItem(PoseMode.boxes, 'Boxes'),
                    _whiteItem(PoseMode.boxesAndLandmarks, 'Landmarks'),
                  ],
                  onChanged: (value) async {
                    if (value != null && value != _detectionMode) {
                      setState(() => _detectionMode = value);
                      await _initPoseDetector();
                      if (_imageBytes != null) {
                        await _processImage(_imageBytes!);
                      }
                    }
                  },
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4.0),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'MODEL',
                  style: TextStyle(
                    color: Colors.grey[500],
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 1.0,
                  ),
                ),
                const SizedBox(width: 4),
                DropdownButton<PoseLandmarkModel>(
                  value: _detectionModel,
                  dropdownColor: Colors.blue[800],
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.onSurface,
                    fontSize: 14,
                  ),
                  underline: const SizedBox(),
                  icon: Icon(
                    Icons.arrow_drop_down,
                    color: Theme.of(context).colorScheme.onSurface,
                  ),
                  selectedItemBuilder: (context) => const [
                    _DropdownSelected('Lite'),
                    _DropdownSelected('Full'),
                    _DropdownSelected('Heavy'),
                  ],
                  items: [
                    _whiteItem(PoseLandmarkModel.lite, 'Lite'),
                    _whiteItem(PoseLandmarkModel.full, 'Full'),
                    _whiteItem(PoseLandmarkModel.heavy, 'Heavy'),
                  ],
                  onChanged: (value) async {
                    if (value != null && value != _detectionModel) {
                      setState(() => _detectionModel = value);
                      await _initPoseDetector();
                      if (_imageBytes != null) {
                        await _processImage(_imageBytes!);
                      }
                    }
                  },
                ),
              ],
            ),
          ),
        ],
      ),
      body: Stack(
        children: [
          Center(
            child: hasImage
                ? LayoutBuilder(
                    builder: (context, constraints) {
                      final fitted = applyBoxFit(
                        BoxFit.contain,
                        _originalSize!,
                        Size(constraints.maxWidth, constraints.maxHeight),
                      );
                      final Size renderSize = fitted.destination;
                      final Rect imageRect = Alignment.center.inscribe(
                        renderSize,
                        Offset.zero &
                            Size(constraints.maxWidth, constraints.maxHeight),
                      );

                      return Stack(
                        children: [
                          Positioned.fromRect(
                            rect: imageRect,
                            child: SizedBox.fromSize(
                              size: renderSize,
                              child: Image.memory(
                                _imageBytes!,
                                fit: BoxFit.fill,
                              ),
                            ),
                          ),
                          Positioned(
                            left: imageRect.left,
                            top: imageRect.top,
                            width: imageRect.width,
                            height: imageRect.height,
                            child: CustomPaint(
                              size: Size(imageRect.width, imageRect.height),
                              painter: MultiOverlayPainter(results: _poses),
                            ),
                          ),
                        ],
                      );
                    },
                  )
                : _ScrollableCentered(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.add_photo_alternate,
                          size: 80,
                          color: Colors.grey[300],
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'No image selected',
                          style: TextStyle(
                            fontSize: 18,
                            color: Colors.grey[600],
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Tap the + icon to pick an image',
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.grey[500],
                          ),
                        ),
                      ],
                    ),
                  ),
          ),
          if (hasImage && _totalTimeMs != null)
            Positioned(
              top: 12,
              left: 12,
              child: TimingBadge(
                totalMs: _totalTimeMs!,
                detectionMs: _detectionTimeMs,
                landmarksEnabled: _detectionMode == PoseMode.boxesAndLandmarks,
              ),
            ),
          if (_isLoading)
            Container(
              color: Colors.black54,
              child: const Center(child: CircularProgressIndicator()),
            ),
        ],
      ),
    );
  }
}

String formatInferenceMilliseconds(num microseconds) {
  final milliseconds = microseconds / 1000;
  if (milliseconds < 10) return milliseconds.toStringAsFixed(3);
  if (milliseconds < 100) return milliseconds.toStringAsFixed(2);
  if (milliseconds < 1000) return milliseconds.toStringAsFixed(1);
  return milliseconds.toStringAsFixed(0);
}

class LiveInferenceStats {
  int? _latestUs;
  int _totalUs = 0;
  int _sampleCount = 0;
  int _generation = 0;

  int? get latestUs => _latestUs;
  int get sampleCount => _sampleCount;
  double? get averageUs => _sampleCount == 0 ? null : _totalUs / _sampleCount;

  int beginSample() => _generation;

  bool record(int sampleGeneration, int elapsedUs) {
    if (sampleGeneration != _generation) return false;
    _latestUs = elapsedUs;
    _totalUs += elapsedUs;
    _sampleCount++;
    return true;
  }

  void reset() {
    _latestUs = null;
    _totalUs = 0;
    _sampleCount = 0;
    _generation++;
  }
}

class LiveCameraMetrics extends StatelessWidget {
  final int fps;
  final int? latestInferenceUs;
  final double? averageInferenceUs;

  const LiveCameraMetrics({
    super.key,
    required this.fps,
    required this.latestInferenceUs,
    required this.averageInferenceUs,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        _FpsMetric(fps: fps),
        const _MetricDivider(),
        _InferenceMetric(label: 'LAST', microseconds: latestInferenceUs),
        const _MetricDivider(),
        _InferenceMetric(label: 'AVERAGE', microseconds: averageInferenceUs),
      ],
    );
  }
}

class _FpsMetric extends StatelessWidget {
  final int fps;

  const _FpsMetric({required this.fps});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 44,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text(
            'FPS',
            style: TextStyle(
              color: Colors.white60,
              fontSize: 9,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.7,
            ),
          ),
          Text(
            '$fps',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.w600,
              fontFeatures: [FontFeature.tabularFigures()],
            ),
          ),
        ],
      ),
    );
  }
}

class _InferenceMetric extends StatelessWidget {
  final String label;
  final num? microseconds;

  const _InferenceMetric({required this.label, required this.microseconds});

  @override
  Widget build(BuildContext context) {
    final value = microseconds == null
        ? '—'
        : formatInferenceMilliseconds(microseconds!);
    return Semantics(
      label: microseconds == null
          ? '$label inference time unavailable'
          : '$label inference time $value milliseconds',
      child: SizedBox(
        width: 74,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              label,
              style: const TextStyle(
                color: Colors.white60,
                fontSize: 9,
                fontWeight: FontWeight.w600,
                letterSpacing: 0.7,
              ),
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(
                  width: 46,
                  child: FittedBox(
                    fit: BoxFit.scaleDown,
                    alignment: Alignment.centerRight,
                    child: Text(
                      value,
                      maxLines: 1,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                        fontFeatures: [FontFeature.tabularFigures()],
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 3),
                const SizedBox(
                  width: 18,
                  child: Text(
                    'ms',
                    style: TextStyle(color: Colors.white70, fontSize: 12),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _MetricDivider extends StatelessWidget {
  const _MetricDivider();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 1,
      height: 24,
      margin: const EdgeInsets.symmetric(horizontal: 5),
      color: Colors.white24,
    );
  }
}

class LiveCameraScreen extends StatefulWidget {
  const LiveCameraScreen({super.key});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen> {
  static const double _mobileTopBarExtent = 84;

  CameraController? _cameraController;
  List<CameraDescription> _availableCameras = const [];
  PoseDetector? _poseDetector;
  List<Pose> _poses = [];
  Size? _imageSize;
  int? _sensorOrientation;
  bool _isFrontCamera = false;
  bool _isSwitchingCamera = false;
  final FrameThrottle _throttle = FrameThrottle();
  bool _isInitialized = false;
  DeviceOrientation _deviceOrientation = DeviceOrientation.portraitUp;
  StreamSubscription<AccelerometerEvent>? _accelerometerSub;
  final LiveInferenceStats _inferenceStats = LiveInferenceStats();
  final FpsCounter _fpsCounter = FpsCounter();
  int _fps = 0;
  bool _isImageStreamStarted = false;

  PoseMode _detectionMode = PoseMode.boxesAndLandmarks;
  PoseLandmarkModel _detectionModel = PoseLandmarkModel.heavy;

  // Live backend benchmarking: default to CompiledModel, with a one-tap
  // XNNPACK fallback for immediate A/B checks in the camera view.
  bool _useCompiledModel = true;
  Precision _precision = Precision.fp16;
  PerformanceConfig get _perfConfig => const PerformanceConfig.xnnpack();
  final List<int> _recentInferenceUs = [];
  int _detThisSec = 0;

  void _resetInferenceStats() {
    _inferenceStats.reset();
    _recentInferenceUs.clear();
    _detThisSec = 0;
  }

  /// One-shot-per-orientation probe for iOS rotation debugging.
  /// See `_rotationFlagForFrame`. Used to settle whether iOS buffers are
  /// sensor-native (same as Android) or already rotated by the plugin.
  DeviceOrientation? _iosProbeOrientation;

  @override
  void initState() {
    super.initState();
    _initCamera();

    if (!kIsWeb && (Platform.isAndroid || Platform.isIOS)) {
      _accelerometerSub = accelerometerEventStream().listen((event) {
        final next = event.x.abs() > event.y.abs()
            ? (event.x > 0
                  ? DeviceOrientation.landscapeLeft
                  : DeviceOrientation.landscapeRight)
            : (event.y > 0
                  ? DeviceOrientation.portraitUp
                  : DeviceOrientation.portraitDown);
        if (next == DeviceOrientation.portraitDown &&
            (_deviceOrientation == DeviceOrientation.landscapeLeft ||
                _deviceOrientation == DeviceOrientation.landscapeRight)) {
          return;
        }
        if (next != _deviceOrientation && mounted) {
          setState(() => _deviceOrientation = next);
        }
      });
    }
  }

  Future<void> _reinitDetectorIsolate() async {
    final old = _poseDetector;
    _poseDetector = null;
    await old?.dispose();
    _poseDetector = PoseDetector();
    try {
      await _poseDetector!.initialize(
        mode: _detectionMode,
        landmarkModel: _detectionModel,
        performanceConfig: _perfConfig,
        useCompiledModel: _useCompiledModel,
        precision: _precision,
      );
    } catch (e) {
      if (!_useCompiledModel) rethrow;
      debugPrint(
        'Live camera CompiledModel init failed; falling back to XNNPACK: $e',
      );
      if (mounted) {
        setState(() => _useCompiledModel = false);
      } else {
        _useCompiledModel = false;
      }
      await _poseDetector?.dispose();
      _poseDetector = PoseDetector();
      await _poseDetector!.initialize(
        mode: _detectionMode,
        landmarkModel: _detectionModel,
        performanceConfig: _perfConfig,
        useCompiledModel: false,
      );
    }
  }

  Future<List<Pose>> _detectForLiveCamera(
    CameraImage image, {
    required CameraFrameRotation? rotation,
    required int maxDim,
  }) async {
    final poses = await _poseDetector!.detectFromCameraImage(
      image,
      rotation: rotation,
      maxDim: maxDim,
    );
    return poses;
  }

  Future<void> _toggleAccelerator() async {
    setState(() {
      _useCompiledModel = !_useCompiledModel;
      _resetInferenceStats();
    });
    // ignore: avoid_print
    print(
      '[live-bench] switching backend -> '
      '${_useCompiledModel ? 'compiledmodel-${_precision.name}' : 'xnnpack'}',
    );
    await _reinitDetectorIsolate();
  }

  /// Builds the live-camera top bar as a plain [Material]-backed [Row] rather
  /// than an [AppBar]. AppBar doesn't play well inside a [RotatedBox] (applies
  /// rotated [MediaQuery] safe-area padding, depends on [Scaffold.appBar]-slot
  /// theming) so its `actions` would render invisibly when the phone is in
  /// landscape; a plain Row sidesteps all of that.
  Widget _buildCameraTopBar() {
    final canPop = Navigator.of(context).canPop();
    final isMobile = !kIsWeb && (Platform.isAndroid || Platform.isIOS);
    final metrics = LiveCameraMetrics(
      fps: _fps,
      latestInferenceUs: _inferenceStats.latestUs,
      averageInferenceUs: _inferenceStats.averageUs,
    );

    final controls = <Widget>[
      if (_canSwitchCamera)
        IconButton(
          tooltip: _isFrontCamera
              ? 'Switch to back camera'
              : 'Switch to front camera',
          color: Colors.white,
          icon: Icon(
            Platform.isIOS ? Icons.flip_camera_ios : Icons.flip_camera_android,
          ),
          onPressed: _isSwitchingCamera ? null : _switchCamera,
        ),
      TextButton(
        onPressed: _toggleAccelerator,
        style: TextButton.styleFrom(
          minimumSize: const Size(92, 36),
          padding: const EdgeInsets.symmetric(horizontal: 4),
        ),
        child: Text(
          _useCompiledModel ? 'CM' : 'Interpreter',
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(
            color: Colors.amberAccent,
            fontWeight: FontWeight.bold,
            fontSize: 14,
          ),
        ),
      ),
      PopupMenuButton<void>(
        tooltip: 'Settings',
        icon: const Icon(Icons.settings, color: Colors.white),
        color: Colors.blueGrey[900],
        padding: EdgeInsets.zero,
        itemBuilder: (context) => [
          PopupMenuItem<void>(
            enabled: false,
            padding: EdgeInsets.zero,
            child: StatefulBuilder(
              builder: (context, setMenuState) {
                return _buildSettingsMenuContent(setMenuState);
              },
            ),
          ),
        ],
      ),
    ];

    return Material(
      color: Colors.black.withAlpha(179),
      elevation: 4,
      child: SizedBox(
        height: isMobile ? _mobileTopBarExtent : kToolbarHeight,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 4),
          child: isMobile
              ? Column(
                  children: [
                    SizedBox(
                      height: 48,
                      child: Row(
                        children: [
                          if (canPop)
                            IconButton(
                              tooltip: 'Back',
                              color: Colors.white,
                              icon: const Icon(Icons.arrow_back),
                              onPressed: () => Navigator.of(context).maybePop(),
                            ),
                          const Spacer(),
                          ...controls,
                        ],
                      ),
                    ),
                    const Divider(height: 1, color: Colors.white12),
                    SizedBox(height: 35, child: Center(child: metrics)),
                  ],
                )
              : Row(
                  children: [
                    if (canPop)
                      IconButton(
                        tooltip: 'Back',
                        color: Colors.white,
                        icon: const Icon(Icons.arrow_back),
                        onPressed: () => Navigator.of(context).maybePop(),
                      ),
                    const Expanded(
                      child: Padding(
                        padding: EdgeInsets.symmetric(horizontal: 8),
                        child: Text(
                          'Live Camera Detection',
                          style: TextStyle(color: Colors.white, fontSize: 18),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ),
                    ...controls,
                    const SizedBox(width: 8),
                    metrics,
                  ],
                ),
        ),
      ),
    );
  }

  Widget _buildSettingsMenuContent(StateSetter setMenuState) {
    void update(VoidCallback fn) {
      setState(() {
        fn();
        _resetInferenceStats();
      });
      setMenuState(() {});
    }

    Widget chip<T>({
      required T value,
      required T current,
      required String label,
      required VoidCallback onTap,
    }) {
      final isSelected = current == value;
      return GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: isSelected ? Colors.blue : Colors.white12,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Text(
            label,
            style: TextStyle(
              color: isSelected ? Colors.white : Colors.white70,
              fontSize: 12,
              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
            ),
          ),
        ),
      );
    }

    const sectionLabelStyle = TextStyle(
      color: Colors.white60,
      fontSize: 10,
      fontWeight: FontWeight.w600,
      letterSpacing: 1.2,
    );

    return SizedBox(
      width: 260,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('MODE', style: sectionLabelStyle),
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                for (final (v, label) in const [
                  (PoseMode.boxes, 'Boxes'),
                  (PoseMode.boxesAndLandmarks, 'Landmarks'),
                ])
                  chip<PoseMode>(
                    value: v,
                    current: _detectionMode,
                    label: label,
                    onTap: () async {
                      if (v == _detectionMode) return;
                      update(() => _detectionMode = v);
                      await _reinitDetectorIsolate();
                      if (mounted) setMenuState(() {});
                    },
                  ),
              ],
            ),
            const Divider(color: Colors.white24, height: 24),
            const Text('MODEL', style: sectionLabelStyle),
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                for (final (v, label) in const [
                  (PoseLandmarkModel.lite, 'Lite'),
                  (PoseLandmarkModel.full, 'Full'),
                  (PoseLandmarkModel.heavy, 'Heavy'),
                ])
                  chip<PoseLandmarkModel>(
                    value: v,
                    current: _detectionModel,
                    label: label,
                    onTap: () async {
                      if (v == _detectionModel) return;
                      update(() => _detectionModel = v);
                      await _reinitDetectorIsolate();
                      if (mounted) setMenuState(() {});
                    },
                  ),
              ],
            ),
            const Divider(color: Colors.white24, height: 24),
            const Text('PRECISION · COMPILEDMODEL', style: sectionLabelStyle),
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                for (final (v, label) in const [
                  (Precision.fp16, 'FP16'),
                  (Precision.fp32, 'FP32'),
                ])
                  chip<Precision>(
                    value: v,
                    current: _precision,
                    label: label,
                    onTap: () async {
                      if (v == _precision) return;
                      update(() {
                        _precision = v;
                      });
                      // ignore: avoid_print
                      print('[live-bench] switching precision -> ${v.name}');
                      if (_useCompiledModel) {
                        await _reinitDetectorIsolate();
                      }
                      if (mounted) setMenuState(() {});
                    },
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _initCamera() async {
    try {
      try {
        await _reinitDetectorIsolate();
      } catch (e) {
        debugPrint('Detector isolate init failed: $e');
        _poseDetector = PoseDetector();
        await _poseDetector!.initialize(
          mode: _detectionMode,
          landmarkModel: _detectionModel,
          performanceConfig: _perfConfig,
          useCompiledModel: false,
        );
      }

      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        if (mounted) {
          ScaffoldMessenger.of(
            context,
          ).showSnackBar(const SnackBar(content: Text('No cameras available')));
        }
        return;
      }
      _availableCameras = cameras;

      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      await _startControllerFor(camera, markInitialized: true);
    } catch (e, st) {
      debugPrint('Camera init failed: $e');
      debugPrint('$st');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error initializing camera: $e')),
        );
      }
    }
  }

  Future<void> _startControllerFor(
    CameraDescription camera, {
    required bool markInitialized,
  }) async {
    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup
          .yuv420, // prevents JPEG fallback on Android; ignored on desktop
    );

    await controller.initialize();

    if (!mounted) {
      await controller.dispose();
      return;
    }

    setState(() {
      _cameraController = controller;
      if (markInitialized) _isInitialized = true;
      _sensorOrientation = controller.description.sensorOrientation;
      _isFrontCamera =
          controller.description.lensDirection == CameraLensDirection.front;
    });

    await controller.startImageStream(_processCameraImage);
    _isImageStreamStarted = true;
  }

  bool get _canSwitchCamera {
    if (kIsWeb) return false;
    if (!(Platform.isAndroid || Platform.isIOS)) return false;
    final hasFront = _availableCameras.any(
      (c) => c.lensDirection == CameraLensDirection.front,
    );
    final hasBack = _availableCameras.any(
      (c) => c.lensDirection == CameraLensDirection.back,
    );
    return hasFront && hasBack;
  }

  Future<void> _switchCamera() async {
    if (_isSwitchingCamera) return;
    if (!_canSwitchCamera) return;

    final target = _isFrontCamera
        ? CameraLensDirection.back
        : CameraLensDirection.front;
    final next = _availableCameras.firstWhere(
      (c) => c.lensDirection == target,
      orElse: () => _availableCameras.first,
    );

    final prev = _cameraController;
    setState(() {
      _isSwitchingCamera = true;
      _cameraController = null;
      _poses = [];
      _imageSize = null;
      _resetInferenceStats();
    });
    try {
      if (prev != null) {
        if (_isImageStreamStarted) {
          try {
            await prev.stopImageStream();
          } catch (_) {}
          _isImageStreamStarted = false;
        }
        await prev.dispose();
      }

      await _startControllerFor(next, markInitialized: false);
    } catch (e, st) {
      debugPrint('Camera switch failed: $e');
      debugPrint('$st');
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error switching camera: $e')));
      }
    } finally {
      if (mounted) setState(() => _isSwitchingCamera = false);
    }
  }

  DeviceOrientation _effectiveDeviceOrientation(BuildContext context) {
    final controller = _cameraController;
    if (controller != null) {
      return controller.value.deviceOrientation;
    }

    return MediaQuery.of(context).orientation == Orientation.portrait
        ? DeviceOrientation.portraitUp
        : DeviceOrientation.landscapeLeft;
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_fpsCounter.tick() && mounted) {
      setState(() => _fps = _fpsCounter.fps);
      final n = _recentInferenceUs.length;
      final meanUs = n == 0
          ? 0.0
          : _recentInferenceUs.reduce((a, b) => a + b) / n;
      final backend = _useCompiledModel
          ? 'compiledmodel-${_precision.name}'
          : 'xnnpack';
      final lastMs = _inferenceStats.latestUs == null
          ? '-'
          : (_inferenceStats.latestUs! / 1000).toStringAsFixed(3);
      // ignore: avoid_print
      print(
        '[live-bench] backend=$backend '
        'cameraFps=$_fps detPerSec=$_detThisSec '
        'meanInferMs=${(meanUs / 1000).toStringAsFixed(3)} '
        'lastMs=$lastMs poses=${_poses.length} '
        'mode=${_detectionMode.name}',
      );
      _recentInferenceUs.clear();
      _detThisSec = 0;
    }

    if (Platform.isIOS) {
      final DeviceOrientation d = _effectiveDeviceOrientation(context);
      if (d != _iosProbeOrientation) {
        _iosProbeOrientation = d;
        debugPrint(
          '[ios-probe] orient=${d.name} '
          'sensor=$_sensorOrientation front=$_isFrontCamera '
          'raw=${image.width}x${image.height} '
          'planes=${image.planes.length}',
        );
      }
    }

    await _throttle.run(() async {
      try {
        final stopwatch = Stopwatch()..start();
        final statsGeneration = _inferenceStats.beginSample();

        if (_poseDetector == null || !mounted) return;
        final sensor = _sensorOrientation;
        final CameraFrameRotation? rotation = sensor == null
            ? null
            : rotationForFrame(
                width: image.width,
                height: image.height,
                sensorOrientation: sensor,
                isFrontCamera: _isFrontCamera,
                deviceOrientation: _effectiveDeviceOrientation(context),
              );

        const int maxDim = 640;
        final Size size = detectionSize(
          width: image.width,
          height: image.height,
          rotation: rotation,
          maxDim: maxDim,
        );

        final poses = await _detectForLiveCamera(
          image,
          rotation: rotation,
          maxDim: maxDim,
        );

        stopwatch.stop();
        final detectionTimeUs = stopwatch.elapsedMicroseconds;
        final shouldRecordTiming = _inferenceStats.record(
          statsGeneration,
          detectionTimeUs,
        );
        if (shouldRecordTiming) {
          _recentInferenceUs.add(detectionTimeUs);
          _detThisSec++;
        }

        if (mounted) {
          setState(() {
            _poses = poses;
            _imageSize = size;
          });
        }
      } catch (_) {
        /// Silently handle errors during processing to keep the stream alive.
      }
    });
  }

  @override
  void dispose() {
    _accelerometerSub?.cancel();
    if (_isImageStreamStarted) {
      _cameraController?.stopImageStream();
    }
    _cameraController?.dispose();
    _poseDetector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized || _cameraController == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Live Camera Detection')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    final cameraAspectRatio = _cameraController!.value.aspectRatio;
    final deviceOrientation = MediaQuery.of(context).orientation;
    final effectiveOrientation = _effectiveDeviceOrientation(context);
    final bool isPortrait =
        effectiveOrientation == DeviceOrientation.portraitUp ||
        effectiveOrientation == DeviceOrientation.portraitDown;

    final double displayAspectRatio = isPortrait
        ? 1.0 / cameraAspectRatio
        : cameraAspectRatio;

    final int turns = barQuarterTurns(_deviceOrientation);
    final bool mirrorOverlayHorizontally =
        (Platform.isAndroid && _isFrontCamera) || Platform.isWindows;

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPoseOverlay(
            cameraPreview: CameraPreview(_cameraController!),
            cameraAspectRatio: cameraAspectRatio,
            displayAspectRatio: displayAspectRatio,
            mirrorHorizontally: mirrorOverlayHorizontally,
            sensorOrientation: _sensorOrientation ?? 0,
            deviceOrientation: deviceOrientation,
            isFrontCamera: _isFrontCamera,
            poses: _poses,
            imageSize: _imageSize,
          ),
          _positionedTopBar(turns),
        ],
      ),
    );
  }

  Widget _positionedTopBar(int turns) {
    final bar = _buildCameraTopBar();
    final padding = MediaQuery.of(context).padding;
    final isMobile = !kIsWeb && (Platform.isAndroid || Platform.isIOS);
    final barExtent = isMobile ? _mobileTopBarExtent : kToolbarHeight;
    if (turns == 0) {
      return Positioned(
        top: padding.top,
        left: padding.left,
        right: padding.right,
        child: bar,
      );
    }
    return Positioned(
      top: padding.top,
      bottom: padding.bottom,
      left: turns == 3 ? padding.left : null,
      right: turns == 1 ? padding.right : null,
      width: barExtent,
      child: RotatedBox(quarterTurns: turns, child: bar),
    );
  }
}

// ─────────────────────────── Video File Screen ────────────────────────────

class VideoFileScreen extends StatefulWidget {
  const VideoFileScreen({super.key});

  @override
  State<VideoFileScreen> createState() => _VideoFileScreenState();
}

class _VideoFileScreenState extends State<VideoFileScreen> {
  PoseDetector? _detector;
  bool _isInitialized = false;
  bool _isProcessing = false;
  bool _cancelRequested = false;
  String? _errorMessage;
  String? _statusMessage;

  String? _inputPath;
  String? _outputPath;
  int _totalFrames = 0;
  int _processedFrames = 0;
  double _videoFps = 0;
  int _videoWidth = 0;
  int _videoHeight = 0;
  Duration _elapsed = Duration.zero;
  final Stopwatch _wallClock = Stopwatch();

  VideoPlayerController? _playerController;
  bool _playerReady = false;
  String? _playerError;

  bool _smoothingEnabled = true;
  final PoseSmoother _smoother = PoseSmoother(enabled: true);

  // Paint options, mirroring the Still Image screen. Style options (colors,
  // sizes, toggles) are read per frame; detection mode and model are captured
  // when a run starts.
  bool _showBoundingBoxes = true;
  bool _showSkeleton = true;
  bool _showLandmarks = true;

  Color _boundingBoxColor = const Color(0xFF00FFCC);
  Color _landmarkColor = const Color(0xFF89CFF0);
  Color _skeletonColor = const Color(0xFF66FF66);

  double _boundingBoxThickness = 2.0;
  double _landmarkSize = 3.0;
  double _skeletonThickness = 3.0;

  PoseMode _detectionMode = PoseMode.boxesAndLandmarks;
  PoseLandmarkModel _detectionModel = PoseLandmarkModel.heavy;
  PoseMode? _readyMode;
  PoseLandmarkModel? _readyModel;

  bool get _supportsInAppPlayer {
    if (kIsWeb) return true;
    return Platform.isAndroid || Platform.isIOS || Platform.isMacOS;
  }

  @override
  void initState() {
    super.initState();
    _initDetector();
  }

  Future<void> _initDetector() async {
    try {
      final detector = await PoseDetector.create(
        mode: _detectionMode,
        landmarkModel: _detectionModel,
        performanceConfig: const PerformanceConfig.xnnpack(),
      );
      if (!mounted) {
        await detector.dispose();
        return;
      }
      _readyMode = _detectionMode;
      _readyModel = _detectionModel;
      setState(() {
        _detector = detector;
        _isInitialized = true;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _errorMessage = 'Failed to initialize detector: $e');
    }
  }

  /// Re-creates the detector when the mode or model changed, since the pose
  /// pipeline fixes both at init time and cannot swap them in place.
  Future<void> _ensureDetector() async {
    if (_detector != null &&
        _detector!.isReady &&
        _readyMode == _detectionMode &&
        _readyModel == _detectionModel) {
      return;
    }
    final old = _detector;
    _detector = null;
    await old?.dispose();
    _detector = await PoseDetector.create(
      mode: _detectionMode,
      landmarkModel: _detectionModel,
      performanceConfig: const PerformanceConfig.xnnpack(),
    );
    _readyMode = _detectionMode;
    _readyModel = _detectionModel;
  }

  @override
  void dispose() {
    _cancelRequested = true;
    _detector?.dispose();
    _playerController?.dispose();
    super.dispose();
  }

  Future<void> _disposePlayer() async {
    final c = _playerController;
    _playerController = null;
    _playerReady = false;
    _playerError = null;
    await c?.dispose();
  }

  Future<void> _initPlayerForOutput(String path) async {
    await _disposePlayer();
    if (!_supportsInAppPlayer) return;
    final controller = VideoPlayerController.file(File(path));
    _playerController = controller;
    try {
      await controller.initialize();
      await controller.setLooping(true);
      if (!mounted) {
        await controller.dispose();
        _playerController = null;
        return;
      }
      setState(() => _playerReady = true);
      await controller.play();
    } catch (e) {
      if (!mounted) return;
      setState(() => _playerError = 'Could not load video: $e');
    }
  }

  Future<void> _pickVideo() async {
    const typeGroup = XTypeGroup(
      label: 'Videos',
      extensions: ['mp4', 'mov', 'm4v'],
    );
    final XFile? file = await openFile(acceptedTypeGroups: [typeGroup]);
    if (file == null) return;
    await _processVideo(file.path);
  }

  Future<void> _processVideo(String path) async {
    final inputFile = File(path);
    if (!await inputFile.exists()) {
      setState(() => _errorMessage = 'File does not exist: $path');
      return;
    }

    try {
      await _ensureDetector();
    } catch (e) {
      setState(() => _errorMessage = 'Failed to initialize detector: $e');
      return;
    }

    final cap = cv.VideoCapture.fromFile(path);
    if (!cap.isOpened) {
      cap.release();
      String hint = '';
      if (Platform.isLinux) {
        hint =
            '\n\nLinux requires GStreamer plugins. Try:\n'
            '  sudo apt install gstreamer1.0-libav '
            'gstreamer1.0-plugins-good gstreamer1.0-plugins-bad';
      }
      setState(
        () => _errorMessage =
            'Could not open video.\nFormat may not be supported by the OS '
            'video backend.$hint',
      );
      return;
    }

    final fps = cap.get(cv.CAP_PROP_FPS);
    final width = cap.get(cv.CAP_PROP_FRAME_WIDTH).toInt();
    final height = cap.get(cv.CAP_PROP_FRAME_HEIGHT).toInt();
    final total = cap.get(cv.CAP_PROP_FRAME_COUNT).toInt();

    final docs = await getApplicationDocumentsDirectory();
    final outName = 'pose_${DateTime.now().millisecondsSinceEpoch}.mp4';
    final outPath = '${docs.path}/$outName';

    final writer = cv.VideoWriter.fromFile(outPath, 'avc1', fps, (
      width,
      height,
    ));
    if (!writer.isOpened) {
      cap.release();
      setState(
        () => _errorMessage =
            'Could not open writer for $outPath. The "avc1" (H.264) codec '
            'may not be available on this OS backend.',
      );
      return;
    }

    if (!mounted) {
      cap.release();
      writer.release();
      return;
    }
    await _disposePlayer();
    setState(() {
      _inputPath = path;
      _outputPath = outPath;
      _videoFps = fps;
      _videoWidth = width;
      _videoHeight = height;
      _totalFrames = total;
      _processedFrames = 0;
      _isProcessing = true;
      _cancelRequested = false;
      _errorMessage = null;
      _statusMessage = 'Processing...';
      _elapsed = Duration.zero;
    });
    _wallClock
      ..reset()
      ..start();

    cv.Mat? frame;
    _smoother.reset();
    try {
      int idx = 0;
      while (mounted && !_cancelRequested) {
        final result = cap.read(m: frame);
        final ok = result.$1;
        frame = result.$2;
        if (!ok || frame.isEmpty) break;

        final List<Pose> raw = await _detector!.detectFromMat(frame);
        final double tSec = fps > 0 ? idx / fps : idx / 30.0;
        final List<Pose> poses = _smoother.apply(raw, tSec);
        _drawPosesOnMat(frame, poses);
        writer.write(frame);

        idx++;
        if (idx % 4 == 0) {
          if (!mounted) break;
          setState(() {
            _processedFrames = idx;
            _elapsed = _wallClock.elapsed;
          });
          await Future<void>.delayed(Duration.zero);
        }
      }
      if (mounted) {
        setState(() {
          _processedFrames = idx;
          _elapsed = _wallClock.elapsed;
          _statusMessage = _cancelRequested
              ? 'Cancelled after $idx frames.'
              : 'Done. Wrote $idx frames to:\n$outPath';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _errorMessage = 'Error during processing: $e');
      }
    } finally {
      _wallClock.stop();
      cap.release();
      writer.release();
      frame?.dispose();
      if (mounted) setState(() => _isProcessing = false);
      if (mounted && !_cancelRequested && _outputPath != null) {
        await _initPlayerForOutput(_outputPath!);
      }
    }
  }

  /// Converts a Flutter [Color] to an OpenCV BGR scalar (alpha ignored).
  cv.Scalar _bgr(Color c) => cv.Scalar(
    (c.b * 255).roundToDouble(),
    (c.g * 255).roundToDouble(),
    (c.r * 255).roundToDouble(),
  );

  /// Draws the enabled overlays onto [mat] with OpenCV, mirroring what
  /// MultiOverlayPainter draws on screen for the Still Image mode.
  void _drawPosesOnMat(cv.Mat mat, List<Pose> poses) {
    if (poses.isEmpty) return;
    final black = cv.Scalar(0, 0, 0);

    final w = mat.cols;
    final h = mat.rows;

    for (final pose in poses) {
      if (_showBoundingBoxes) {
        final bboxColor = _bgr(_boundingBoxColor);
        final bb = pose.boundingBox;
        final l = bb.left.toInt().clamp(0, w - 1);
        final t = bb.top.toInt().clamp(0, h - 1);
        final r = bb.right.toInt().clamp(0, w - 1);
        final b = bb.bottom.toInt().clamp(0, h - 1);
        cv.rectangle(
          mat,
          cv.Rect(l, t, (r - l).clamp(1, w), (b - t).clamp(1, h)),
          bboxColor,
          thickness: math.max(1, _boundingBoxThickness.round()),
        );

        final label = 'Person ${(pose.score * 100).toStringAsFixed(0)}%';
        final (sz, _) = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2);
        final labelTop = (t - sz.height - 8).clamp(0, h - 1);
        final labelW = (sz.width + 8).clamp(1, w - l);
        final labelH = (sz.height + 8).clamp(1, h - labelTop);
        cv.rectangle(
          mat,
          cv.Rect(l, labelTop, labelW, labelH),
          bboxColor,
          thickness: -1,
        );
        cv.putText(
          mat,
          label,
          cv.Point(l + 4, labelTop + sz.height + 2),
          cv.FONT_HERSHEY_SIMPLEX,
          0.6,
          black,
          thickness: 2,
        );
      }

      if (!pose.hasLandmarks) continue;

      if (_showSkeleton) {
        final skeletonColor = _bgr(_skeletonColor);
        for (final connection in poseLandmarkConnections) {
          final start = pose.getLandmark(connection[0]);
          final end = pose.getLandmark(connection[1]);
          if (start != null &&
              end != null &&
              start.visibility > 0.5 &&
              end.visibility > 0.5) {
            cv.line(
              mat,
              cv.Point(start.x.toInt(), start.y.toInt()),
              cv.Point(end.x.toInt(), end.y.toInt()),
              skeletonColor,
              thickness: math.max(1, _skeletonThickness.round()),
            );
          }
        }
      }

      if (_showLandmarks) {
        final landmarkColor = _bgr(_landmarkColor);
        for (final l in pose.landmarks) {
          if (l.visibility <= 0.5) continue;
          cv.circle(
            mat,
            cv.Point(l.x.toInt(), l.y.toInt()),
            math.max(1, _landmarkSize.round()),
            landmarkColor,
            thickness: -1,
          );
        }
      }
    }
  }

  Future<void> _openOutputFile() async {
    final path = _outputPath;
    if (path == null) return;
    try {
      if (Platform.isMacOS) {
        await Process.run('open', [path]);
      } else if (Platform.isLinux) {
        await Process.run('xdg-open', [path]);
      } else if (Platform.isWindows) {
        await Process.run('cmd', ['/c', 'start', '', path]);
      } else if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Saved to: $path')));
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Could not open: $e')));
      }
    }
  }

  void _showVideoSettings() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.6,
        minChildSize: 0.3,
        maxChildSize: 0.9,
        builder: (context, scrollController) => StatefulBuilder(
          builder: (context, setSheetState) {
            void updateState(VoidCallback fn) {
              fn();
              setSheetState(() {});
              setState(() {});
            }

            Widget cb(String label, bool v, void Function(bool) set) =>
                CompactCheckbox(
                  label: label,
                  value: v,
                  onChanged: (x) => updateState(() => set(x ?? false)),
                );
            Widget col(String label, Color c, void Function(Color) set) =>
                _ColorPickerButton(
                  label: label,
                  color: c,
                  onColorChanged: (x) => updateState(() => set(x)),
                );
            Widget sl(
              String label,
              double v,
              double mn,
              double mx,
              void Function(double) set,
            ) => CompactSlider(
              label: label,
              value: v,
              min: mn,
              max: mx,
              onChanged: (x) => updateState(() => set(x)),
            );

            return Container(
              decoration: const BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
              ),
              child: Column(
                children: [
                  Container(
                    margin: const EdgeInsets.symmetric(vertical: 8),
                    width: 40,
                    height: 4,
                    decoration: BoxDecoration(
                      color: Colors.grey[300],
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                  Expanded(
                    child: ListView(
                      controller: scrollController,
                      padding: const EdgeInsets.symmetric(horizontal: 16),
                      children: [
                        const Padding(
                          padding: EdgeInsets.symmetric(vertical: 4),
                          child: Text(
                            'Mode and model apply when processing starts; '
                            'styles apply to the remaining frames.',
                            style: TextStyle(fontSize: 12, color: Colors.grey),
                          ),
                        ),
                        ExpansionTile(
                          title: const Text(
                            'Detection',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          initiallyExpanded: true,
                          children: [
                            Align(
                              alignment: Alignment.centerLeft,
                              child: Wrap(
                                spacing: 8,
                                children: [
                                  for (final (m, label) in const [
                                    (PoseMode.boxes, 'Boxes'),
                                    (
                                      PoseMode.boxesAndLandmarks,
                                      'Boxes + Landmarks',
                                    ),
                                  ])
                                    ChoiceChip(
                                      label: Text(label),
                                      selected: _detectionMode == m,
                                      onSelected: (sel) {
                                        if (sel) {
                                          updateState(() => _detectionMode = m);
                                        }
                                      },
                                    ),
                                ],
                              ),
                            ),
                            const SizedBox(height: 8),
                            Align(
                              alignment: Alignment.centerLeft,
                              child: Wrap(
                                spacing: 8,
                                children: [
                                  for (final (m, label) in const [
                                    (PoseLandmarkModel.lite, 'Lite'),
                                    (PoseLandmarkModel.full, 'Full'),
                                    (PoseLandmarkModel.heavy, 'Heavy'),
                                  ])
                                    ChoiceChip(
                                      label: Text(label),
                                      selected: _detectionModel == m,
                                      onSelected: (sel) {
                                        if (sel) {
                                          updateState(
                                            () => _detectionModel = m,
                                          );
                                        }
                                      },
                                    ),
                                ],
                              ),
                            ),
                            const SizedBox(height: 8),
                          ],
                        ),
                        ExpansionTile(
                          title: const Text(
                            'Display Options',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          initiallyExpanded: true,
                          children: [
                            Wrap(
                              spacing: 8,
                              runSpacing: 4,
                              children: [
                                cb(
                                  'Bounding Boxes',
                                  _showBoundingBoxes,
                                  (v) => _showBoundingBoxes = v,
                                ),
                                cb(
                                  'Skeleton',
                                  _showSkeleton,
                                  (v) => _showSkeleton = v,
                                ),
                                cb(
                                  'Landmarks',
                                  _showLandmarks,
                                  (v) => _showLandmarks = v,
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                          ],
                        ),
                        ExpansionTile(
                          title: const Text(
                            'Colors',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          children: [
                            Wrap(
                              spacing: 6,
                              runSpacing: 6,
                              children: [
                                col(
                                  'BBox',
                                  _boundingBoxColor,
                                  (c) => _boundingBoxColor = c,
                                ),
                                col(
                                  'Landmarks',
                                  _landmarkColor,
                                  (c) => _landmarkColor = c,
                                ),
                                col(
                                  'Skeleton',
                                  _skeletonColor,
                                  (c) => _skeletonColor = c,
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                          ],
                        ),
                        ExpansionTile(
                          title: const Text(
                            'Sizes',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          children: [
                            sl(
                              'BBox',
                              _boundingBoxThickness,
                              0.5,
                              10.0,
                              (v) => _boundingBoxThickness = v,
                            ),
                            sl(
                              'Landmark',
                              _landmarkSize,
                              0.5,
                              15.0,
                              (v) => _landmarkSize = v,
                            ),
                            sl(
                              'Skeleton',
                              _skeletonThickness,
                              0.5,
                              10.0,
                              (v) => _skeletonThickness = v,
                            ),
                            const SizedBox(height: 8),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Video File - Pose Detection'),
        actions: [
          IconButton(
            onPressed: _showVideoSettings,
            icon: const Icon(Icons.tune),
            tooltip: 'Settings',
          ),
        ],
      ),
      body: _buildBody(),
      floatingActionButton: _isInitialized && !_isProcessing
          ? FloatingActionButton.extended(
              onPressed: _pickVideo,
              icon: const Icon(Icons.video_file),
              label: const Text('Pick Video'),
            )
          : (_isProcessing
                ? FloatingActionButton.extended(
                    onPressed: () => setState(() => _cancelRequested = true),
                    icon: const Icon(Icons.cancel),
                    label: const Text('Cancel'),
                    backgroundColor: Colors.red,
                  )
                : null),
    );
  }

  Widget _buildBody() {
    if (!_isInitialized && _errorMessage == null) {
      return const _ScrollableCentered(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Initializing detector...'),
          ],
        ),
      );
    }

    final progress = (_totalFrames > 0)
        ? (_processedFrames / _totalFrames).clamp(0.0, 1.0)
        : 0.0;
    final processedFps = (_elapsed.inMilliseconds > 0)
        ? _processedFrames * 1000.0 / _elapsed.inMilliseconds
        : 0.0;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (_errorMessage != null)
            Card(
              color: Colors.red[50],
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    const Icon(Icons.error_outline, color: Colors.red),
                    const SizedBox(width: 12),
                    Expanded(child: Text(_errorMessage!)),
                  ],
                ),
              ),
            ),
          if (_inputPath != null) ...[
            const SizedBox(height: 8),
            _infoRow('Input', _inputPath!),
            if (_videoWidth > 0)
              _infoRow(
                'Source',
                '$_videoWidth×$_videoHeight @ '
                    '${_videoFps.toStringAsFixed(2)} fps · '
                    '$_totalFrames frames',
              ),
          ],
          if (!_isProcessing)
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              dense: true,
              title: const Text('Smoothing (One-Euro filter)'),
              subtitle: Text(
                _smoothingEnabled
                    ? 'On: landmarks filtered across frames'
                    : 'Off: raw per-frame detections',
              ),
              value: _smoothingEnabled,
              onChanged: (v) {
                setState(() {
                  _smoothingEnabled = v;
                  _smoother.enabled = v;
                  _smoother.reset();
                });
              },
            ),
          if (!_isProcessing && _inputPath != null) ...[
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerLeft,
              child: OutlinedButton.icon(
                onPressed: () => _processVideo(_inputPath!),
                icon: const Icon(Icons.refresh),
                label: const Text('Re-run with current settings'),
              ),
            ),
          ],
          const SizedBox(height: 16),
          if (_isProcessing) ...[
            LinearProgressIndicator(value: _totalFrames > 0 ? progress : null),
            const SizedBox(height: 8),
            Text(
              'Frame $_processedFrames / $_totalFrames · '
              '${(progress * 100).toStringAsFixed(1)}% · '
              '${processedFps.toStringAsFixed(1)} fps · '
              'elapsed ${_formatDuration(_elapsed)}',
              style: const TextStyle(
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ] else if (_outputPath != null && _statusMessage != null)
            VideoResultCard(
              statusMessage: _statusMessage!,
              summary:
                  'Total time: ${_formatDuration(_elapsed)} '
                  '(${processedFps.toStringAsFixed(1)} fps avg)',
              preview: _buildOutputPreview(),
              onOpenOutput:
                  (Platform.isMacOS || Platform.isLinux || Platform.isWindows)
                  ? _openOutputFile
                  : null,
            )
          else
            Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const SizedBox(height: 32),
                  Icon(
                    Icons.movie_creation_outlined,
                    size: 96,
                    color: Colors.grey[400],
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Pick an MP4 to run pose detection on every frame.\n'
                    'Output is written to the app documents directory.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey[700]),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _infoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 70,
            child: Text(
              '$label:',
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
          ),
          Expanded(child: SelectableText(value)),
        ],
      ),
    );
  }

  String _formatDuration(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    if (d.inHours > 0) {
      return '${d.inHours}:$m:$s';
    }
    return '$m:$s';
  }

  Widget _buildOutputPreview() {
    if (!_supportsInAppPlayer) return const SizedBox.shrink();
    if (_playerError != null) {
      return Text(_playerError!, style: const TextStyle(color: Colors.red));
    }
    final controller = _playerController;
    if (controller == null || !_playerReady) {
      return const SizedBox(
        height: 64,
        child: Center(
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
              SizedBox(width: 12),
              Flexible(child: Text('Loading preview...')),
            ],
          ),
        ),
      );
    }
    return _OutputVideoPlayer(controller: controller);
  }
}

// ─────────────────────────── Video Result Card ────────────────────────────

/// Result card shown after a video finishes processing.
///
/// Built so it cannot overflow regardless of viewport: it contains no
/// fixed-height content, every row child that can vary in width is flexible,
/// and the preview passed in is expected to size itself to the incoming
/// constraints (see [VideoPlayerChrome]). The card relies on the scrollable
/// body it sits in for vertical space.
class VideoResultCard extends StatelessWidget {
  final String statusMessage;
  final String summary;
  final Widget preview;
  final VoidCallback? onOpenOutput;

  const VideoResultCard({
    super.key,
    required this.statusMessage,
    required this.summary,
    required this.preview,
    this.onOpenOutput,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.check_circle, color: Colors.green),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    statusMessage,
                    style: const TextStyle(fontWeight: FontWeight.w500),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(summary),
            const SizedBox(height: 12),
            preview,
            if (onOpenOutput != null) ...[
              const SizedBox(height: 12),
              Align(
                alignment: Alignment.centerLeft,
                child: ElevatedButton.icon(
                  onPressed: onOpenOutput,
                  icon: const Icon(Icons.play_circle_outline),
                  label: const Text('Open output video'),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// Layout chrome for the output video preview: the video is aspect-fitted
/// within the available width and capped to a fraction of the screen height
/// (so portrait videos cannot blow the layout up), and the controls row
/// drops the time label instead of overflowing when width gets tight.
class VideoPlayerChrome extends StatelessWidget {
  final double aspectRatio;
  final Widget video;
  final Widget progress;
  final bool isPlaying;
  final String positionLabel;
  final VoidCallback onTogglePlay;

  const VideoPlayerChrome({
    super.key,
    required this.aspectRatio,
    required this.video,
    required this.progress,
    required this.isPlaying,
    required this.positionLabel,
    required this.onTogglePlay,
  });

  @override
  Widget build(BuildContext context) {
    final double maxPreviewHeight = math.max(
      120.0,
      MediaQuery.sizeOf(context).height * 0.45,
    );
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Align(
          alignment: Alignment.centerLeft,
          child: ConstrainedBox(
            constraints: BoxConstraints(maxHeight: maxPreviewHeight),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: AspectRatio(
                aspectRatio: aspectRatio,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Container(color: Colors.black),
                    video,
                  ],
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 8),
        LayoutBuilder(
          builder: (context, constraints) {
            final bool showTime = constraints.maxWidth >= 180;
            return Row(
              children: [
                IconButton(
                  icon: Icon(isPlaying ? Icons.pause : Icons.play_arrow),
                  onPressed: onTogglePlay,
                ),
                Expanded(child: progress),
                if (showTime) ...[
                  const SizedBox(width: 8),
                  Text(
                    positionLabel,
                    style: const TextStyle(
                      fontFeatures: [FontFeature.tabularFigures()],
                    ),
                  ),
                ],
              ],
            );
          },
        ),
      ],
    );
  }
}

// ─────────────────────────── Output Video Player ──────────────────────────

class _OutputVideoPlayer extends StatefulWidget {
  final VideoPlayerController controller;
  const _OutputVideoPlayer({required this.controller});

  @override
  State<_OutputVideoPlayer> createState() => _OutputVideoPlayerState();
}

class _OutputVideoPlayerState extends State<_OutputVideoPlayer> {
  void _onTick() {
    if (mounted) setState(() {});
  }

  @override
  void initState() {
    super.initState();
    widget.controller.addListener(_onTick);
  }

  @override
  void didUpdateWidget(covariant _OutputVideoPlayer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.controller != widget.controller) {
      oldWidget.controller.removeListener(_onTick);
      widget.controller.addListener(_onTick);
    }
  }

  @override
  void dispose() {
    widget.controller.removeListener(_onTick);
    super.dispose();
  }

  String _fmt(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$m:$s';
  }

  @override
  Widget build(BuildContext context) {
    final c = widget.controller;
    final value = c.value;
    return VideoPlayerChrome(
      aspectRatio: value.aspectRatio == 0 ? 16 / 9 : value.aspectRatio,
      video: VideoPlayer(c),
      progress: VideoProgressIndicator(
        c,
        allowScrubbing: true,
        padding: const EdgeInsets.symmetric(vertical: 12),
      ),
      isPlaying: value.isPlaying,
      positionLabel: '${_fmt(value.position)} / ${_fmt(value.duration)}',
      onTogglePlay: () {
        if (value.isPlaying) {
          c.pause();
        } else {
          c.play();
        }
      },
    );
  }
}

// ─────────────────────────── Pose Smoother ────────────────────────────────

class _PoseTrack {
  final Map<int, List<OneEuroFilter>> filters = {};
  double lastLeft = 0, lastTop = 0, lastRight = 0, lastBottom = 0;
  bool hasBox = false;
  int missedFrames = 0;
}

/// Tracks detected persons across frames and applies a One-Euro filter to each
/// landmark's coordinates, reducing jitter in the rendered skeletons.
class PoseSmoother {
  bool enabled;
  static const int _maxMissed = 5;
  static const double _minIou = 0.2;
  final List<_PoseTrack> _tracks = [];

  PoseSmoother({this.enabled = true});

  void reset() {
    _tracks.clear();
  }

  List<Pose> apply(List<Pose> poses, double tSec) {
    if (!enabled || poses.isEmpty) {
      if (!enabled) _tracks.clear();
      return poses;
    }

    final unmatched = List<int>.generate(_tracks.length, (i) => i);
    final matchedTrack = List<int?>.filled(poses.length, null);

    for (int p = 0; p < poses.length; p++) {
      double bestIou = _minIou;
      int bestT = -1;
      for (final t in unmatched) {
        if (!_tracks[t].hasBox) continue;
        final iou = _iou(poses[p], _tracks[t]);
        if (iou > bestIou) {
          bestIou = iou;
          bestT = t;
        }
      }
      if (bestT >= 0) {
        matchedTrack[p] = bestT;
        unmatched.remove(bestT);
      }
    }

    final out = <Pose>[];
    for (int p = 0; p < poses.length; p++) {
      _PoseTrack track;
      if (matchedTrack[p] != null) {
        track = _tracks[matchedTrack[p]!];
        track.missedFrames = 0;
      } else {
        track = _PoseTrack();
        _tracks.add(track);
      }
      final bb = poses[p].boundingBox;
      track.lastLeft = bb.left;
      track.lastTop = bb.top;
      track.lastRight = bb.right;
      track.lastBottom = bb.bottom;
      track.hasBox = true;
      out.add(_smoothPose(poses[p], track, tSec));
    }

    for (final t in unmatched) {
      _tracks[t].missedFrames++;
    }
    _tracks.removeWhere((t) => t.missedFrames > _maxMissed);

    return out;
  }

  Pose _smoothPose(Pose pose, _PoseTrack track, double tSec) {
    if (!pose.hasLandmarks) return pose;

    final smoothedLandmarks = <PoseLandmark>[];
    for (int i = 0; i < pose.landmarks.length; i++) {
      final l = pose.landmarks[i];
      var fs = track.filters[i];
      if (fs == null) {
        fs = [
          OneEuroFilter(minCutoff: 1.0, beta: 0.1, dCutoff: 1.0),
          OneEuroFilter(minCutoff: 1.0, beta: 0.1, dCutoff: 1.0),
        ];
        track.filters[i] = fs;
      }
      smoothedLandmarks.add(
        PoseLandmark(
          type: l.type,
          x: fs[0].filter(l.x, tSec),
          y: fs[1].filter(l.y, tSec),
          z: l.z,
          visibility: l.visibility,
        ),
      );
    }

    return Pose(
      boundingBox: pose.boundingBox,
      score: pose.score,
      landmarks: smoothedLandmarks,
      imageWidth: pose.imageWidth,
      imageHeight: pose.imageHeight,
      // Smoothing only filters landmark coordinates. Every other field has to
      // be carried across or it is silently dropped from the smoothed result.
      segmentationMask: pose.segmentationMask,
    );
  }

  double _iou(Pose a, _PoseTrack b) {
    final box = a.boundingBox;
    return iouLTRB(
      box.left,
      box.top,
      box.right,
      box.bottom,
      b.lastLeft,
      b.lastTop,
      b.lastRight,
      b.lastBottom,
    );
  }
}
