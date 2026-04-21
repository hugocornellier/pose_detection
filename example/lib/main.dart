import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:pose_detection/pose_detection.dart';
import 'package:camera/camera.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:sensors_plus/sensors_plus.dart';

void main() {
  runApp(const PoseDetectionApp());
}

class PoseDetectionApp extends StatelessWidget {
  const PoseDetectionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pose Detection Demo',
      theme: ThemeData(colorSchemeSeed: Colors.blue, useMaterial3: true),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Pose Detection Demo')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.accessibility_new, size: 100, color: Colors.blue[300]),
            const SizedBox(height: 48),
            Text(
              'Choose Detection Mode',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            const SizedBox(height: 48),
            _buildModeCard(
              context,
              icon: Icons.image,
              title: 'Still Image',
              description: 'Detect poses in photos from gallery or camera',
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const StillImageScreen(),
                  ),
                );
              },
            ),
            const SizedBox(height: 24),
            _buildModeCard(
              context,
              icon: Icons.videocam,
              title: 'Live Camera',
              description: 'Real-time pose detection from camera feed',
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const CameraScreen()),
                );
              },
            ),
          ],
        ),
      ),
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
      width: 400,
      child: Card(
        elevation: 4,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(12),
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Row(
              children: [
                Icon(icon, size: 64, color: Colors.blue),
                const SizedBox(width: 24),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        title,
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        description,
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          color: Colors.grey[600],
                        ),
                      ),
                    ],
                  ),
                ),
                const Icon(Icons.arrow_forward_ios),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class StillImageScreen extends StatefulWidget {
  const StillImageScreen({super.key});

  @override
  State<StillImageScreen> createState() => _StillImageScreenState();
}

class _StillImageScreenState extends State<StillImageScreen> {
  final PoseDetector _poseDetector = PoseDetector(
    mode: PoseMode.boxesAndLandmarks,
    landmarkModel: PoseLandmarkModel.heavy,
    detectorConf: 0.6,
    detectorIou: 0.4,
    maxDetections: 10,
    minLandmarkScore: 0.5,
    performanceConfig:
        const PerformanceConfig.xnnpack(), // Enable XNNPACK for 2-5x speedup
  );
  final ImagePicker _picker = ImagePicker();

  bool _isInitialized = false;
  bool _isProcessing = false;
  File? _imageFile;
  List<Pose> _results = [];
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _initializeDetectors();
  }

  Future<void> _initializeDetectors() async {
    setState(() {
      _isProcessing = true;
      _errorMessage = null;
    });

    try {
      await _poseDetector.initialize();
      setState(() {
        _isInitialized = true;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Failed to initialize: $e';
      });
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(source: source);
      if (pickedFile == null) return;

      setState(() {
        _imageFile = File(pickedFile.path);
        _results = [];
        _isProcessing = true;
        _errorMessage = null;
      });

      final Uint8List bytes = await _imageFile!.readAsBytes();
      final List<Pose> results = await _poseDetector.detect(bytes);

      setState(() {
        _results = results;
        _isProcessing = false;
        if (results.isEmpty) _errorMessage = 'No people detected in image';
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Error: $e';
      });
    }
  }

  void _showImageSourceDialog() {
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Select Image Source'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.photo_library),
                title: const Text('Gallery'),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.gallery);
                },
              ),
              ListTile(
                leading: const Icon(Icons.camera_alt),
                title: const Text('Camera'),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.camera);
                },
              ),
            ],
          ),
        );
      },
    );
  }

  @override
  void dispose() {
    _poseDetector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Pose Detection Demo'),
        actions: [
          if (_isInitialized && _imageFile != null)
            IconButton(
              icon: const Icon(Icons.info_outline),
              onPressed: _showPoseInfo,
            ),
        ],
      ),
      body: _buildBody(),
      floatingActionButton: _isInitialized && !_isProcessing
          ? FloatingActionButton.extended(
              onPressed: _showImageSourceDialog,
              icon: const Icon(Icons.add_photo_alternate),
              label: const Text('Select Image'),
            )
          : null,
    );
  }

  Widget _buildBody() {
    if (!_isInitialized && _isProcessing) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Initializing pose detector...'),
          ],
        ),
      );
    }

    if (_errorMessage != null && _imageFile == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, size: 64, color: Colors.red),
            const SizedBox(height: 16),
            Text(
              _errorMessage!,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.red),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _initializeDetectors,
              child: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    if (_imageFile == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.person_outline, size: 100, color: Colors.grey[400]),
            const SizedBox(height: 24),
            Text(
              'Select an image to detect pose',
              style: TextStyle(fontSize: 18, color: Colors.grey[600]),
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _showImageSourceDialog,
              icon: const Icon(Icons.add_photo_alternate),
              label: const Text('Select Image'),
            ),
          ],
        ),
      );
    }

    return SingleChildScrollView(
      child: Column(
        children: [
          PoseVisualizerWidget(imageFile: _imageFile!, results: _results),
          if (_isProcessing)
            const Padding(
              padding: EdgeInsets.all(16),
              child: Column(
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 8),
                  Text('Detecting pose...'),
                ],
              ),
            ),
          if (_errorMessage != null && !_isProcessing)
            Padding(
              padding: const EdgeInsets.all(16),
              child: Card(
                color: Colors.red[50],
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    children: [
                      const Icon(Icons.error_outline, color: Colors.red),
                      const SizedBox(width: 8),
                      Expanded(child: Text(_errorMessage!)),
                    ],
                  ),
                ),
              ),
            ),
          if (_results.isNotEmpty)
            Padding(
              padding: const EdgeInsets.all(16),
              child: Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Detections: ${_results.length} ✓',
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          color: Colors.green,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  void _showPoseInfo() {
    if (_results.isEmpty) return;
    final Pose first = _results.first;

    showModalBottomSheet(
      context: context,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.7,
        minChildSize: 0.5,
        maxChildSize: 0.95,
        expand: false,
        builder: (context, scrollController) => ListView(
          controller: scrollController,
          padding: const EdgeInsets.all(16),
          children: [
            Text(
              'Landmark Details (first pose)',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 16),
            ..._buildLandmarkListFor(first),
          ],
        ),
      ),
    );
  }

  List<Widget> _buildLandmarkListFor(Pose result) {
    final List<PoseLandmark> lm = result.landmarks;
    return lm.map((landmark) {
      final Point pixel = landmark.toPixel(
        result.imageWidth,
        result.imageHeight,
      );
      return Card(
        margin: const EdgeInsets.only(bottom: 8),
        child: ListTile(
          leading: CircleAvatar(
            backgroundColor: landmark.visibility > 0.5
                ? Colors.green
                : Colors.orange,
            child: Text(
              landmark.type.index.toString(),
              style: const TextStyle(fontSize: 12),
            ),
          ),
          title: Text(
            _landmarkName(landmark.type),
            style: const TextStyle(fontWeight: FontWeight.w500),
          ),
          subtitle: Text(
            ''
            'Position: (${pixel.x}, ${pixel.y})\n'
            'Visibility: ${(landmark.visibility * 100).toStringAsFixed(0)}%',
          ),
          isThreeLine: true,
        ),
      );
    }).toList();
  }

  String _landmarkName(PoseLandmarkType type) {
    return type
        .toString()
        .split('.')
        .last
        .replaceAllMapped(RegExp(r'[A-Z]'), (match) => ' ${match.group(0)}')
        .trim();
  }
}

class PoseVisualizerWidget extends StatelessWidget {
  final File imageFile;
  final List<Pose> results;

  const PoseVisualizerWidget({
    super.key,
    required this.imageFile,
    required this.results,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        return Stack(
          children: [
            Image.file(imageFile, fit: BoxFit.contain),
            Positioned.fill(
              child: CustomPaint(
                painter: MultiOverlayPainter(results: results),
              ),
            ),
          ],
        );
      },
    );
  }
}

class MultiOverlayPainter extends CustomPainter {
  final List<Pose> results;

  MultiOverlayPainter({required this.results});

  @override
  void paint(Canvas canvas, Size size) {
    if (results.isEmpty) return;

    final int iw = results.first.imageWidth;
    final int ih = results.first.imageHeight;

    final double imageAspect = iw / ih;
    final double canvasAspect = size.width / size.height;
    double scaleX, scaleY;
    double offsetX = 0, offsetY = 0;

    if (canvasAspect > imageAspect) {
      scaleY = size.height / ih;
      scaleX = scaleY;
      offsetX = (size.width - iw * scaleX) / 2;
    } else {
      scaleX = size.width / iw;
      scaleY = scaleX;
      offsetY = (size.height - ih * scaleY) / 2;
    }

    for (final r in results) {
      _drawBbox(canvas, r, scaleX, scaleY, offsetX, offsetY);
      if (r.hasLandmarks) {
        _drawConnections(canvas, r, scaleX, scaleY, offsetX, offsetY);
        _drawLandmarks(canvas, r, scaleX, scaleY, offsetX, offsetY);
      }
    }
  }

  void _drawConnections(
    Canvas canvas,
    Pose result,
    double scaleX,
    double scaleY,
    double offsetX,
    double offsetY,
  ) {
    final Paint paint = Paint()
      ..color = Colors.green.withValues(alpha: 0.8)
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;

    // Use the predefined skeleton connections from the package
    for (final List<PoseLandmarkType> c in poseLandmarkConnections) {
      final PoseLandmark? start = result.getLandmark(c[0]);
      final PoseLandmark? end = result.getLandmark(c[1]);
      if (start != null &&
          end != null &&
          start.visibility > 0.5 &&
          end.visibility > 0.5) {
        canvas.drawLine(
          Offset(start.x * scaleX + offsetX, start.y * scaleY + offsetY),
          Offset(end.x * scaleX + offsetX, end.y * scaleY + offsetY),
          paint,
        );
      }
    }
  }

  void _drawLandmarks(
    Canvas canvas,
    Pose result,
    double scaleX,
    double scaleY,
    double offsetX,
    double offsetY,
  ) {
    for (final PoseLandmark l in result.landmarks) {
      if (l.visibility > 0.5) {
        final Offset center = Offset(
          l.x * scaleX + offsetX,
          l.y * scaleY + offsetY,
        );
        final Paint glow = Paint()..color = Colors.blue.withValues(alpha: 0.3);
        final Paint point = Paint()..color = Colors.red;
        final Paint centerDot = Paint()..color = Colors.white;
        canvas.drawCircle(center, 8, glow);
        canvas.drawCircle(center, 5, point);
        canvas.drawCircle(center, 2, centerDot);
      }
    }
  }

  void _drawBbox(
    Canvas canvas,
    Pose r,
    double scaleX,
    double scaleY,
    double offsetX,
    double offsetY,
  ) {
    final Paint boxPaint = Paint()
      ..color = Colors.orangeAccent.withValues(alpha: 0.9)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final Paint fillPaint = Paint()
      ..color = Colors.orangeAccent.withValues(alpha: 0.08)
      ..style = PaintingStyle.fill;

    final double x1 = r.boundingBox.left * scaleX + offsetX;
    final double y1 = r.boundingBox.top * scaleY + offsetY;
    final double x2 = r.boundingBox.right * scaleX + offsetX;
    final double y2 = r.boundingBox.bottom * scaleY + offsetY;
    final Rect rect = Rect.fromLTRB(x1, y1, x2, y2);
    canvas.drawRect(rect, fillPaint);
    canvas.drawRect(rect, boxPaint);
  }

  @override
  bool shouldRepaint(MultiOverlayPainter oldDelegate) => true;
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _cameraController;
  bool _isImageStreamStarted = false;
  PoseDetector _poseDetector = PoseDetector(
    mode: PoseMode.boxesAndLandmarks,
    landmarkModel:
        PoseLandmarkModel.lite, // Use lite for better real-time performance
    detectorConf: 0.7,
    detectorIou: 0.4,
    maxDetections: 5,
    minLandmarkScore: 0.5,
    performanceConfig:
        const PerformanceConfig.xnnpack(), // Enable XNNPACK for real-time performance
  );

  bool _isInitialized = false;
  bool _isProcessing = false;
  bool _isDisposed = false;
  String? _errorMessage;
  Size? _cameraSize;
  int? _sensorOrientation;
  bool _isFrontCamera = false;

  List<CameraDescription> _availableCameras = const [];
  bool _isSwitchingCamera = false;
  String _deviceOrientation = 'Portrait Up';
  int _fps = 0;
  DateTime? _lastFpsUpdate;
  int _framesSinceLastUpdate = 0;
  int _detectionTimeMs = 0;
  StreamSubscription<AccelerometerEvent>? _accelerometerSub;

  // Performance: Use ValueNotifier for efficient pose overlay updates
  // This avoids full widget rebuilds - only the CustomPaint repaints
  final ValueNotifier<List<Pose>> _poseNotifier = ValueNotifier<List<Pose>>([]);

  // Performance: Dynamic frame throttling based on processing time
  int _lastProcessedTime = 0;
  static const int _minProcessingIntervalMs = 50; // Max ~20 detection FPS

  // Performance metrics (debug)
  int _detectionCount = 0;
  double _avgProcessingTimeMs = 0;

  // Pose-specific settings
  PoseLandmarkModel _landmarkModel = PoseLandmarkModel.lite;

  @override
  void initState() {
    super.initState();
    _initializePoseDetector();
    _initCamera();

    if (!kIsWeb && (Platform.isAndroid || Platform.isIOS)) {
      _accelerometerSub = accelerometerEventStream().listen((event) {
        final next = event.x.abs() > event.y.abs()
            ? (event.x > 0 ? 'Landscape Left' : 'Landscape Right')
            : (event.y > 0 ? 'Portrait Up' : 'Portrait Down');
        if (next == 'Portrait Down' &&
            (_deviceOrientation == 'Landscape Left' ||
                _deviceOrientation == 'Landscape Right')) {
          return;
        }
        if (next != _deviceOrientation && mounted) {
          setState(() => _deviceOrientation = next);
        }
      });
    }
  }

  Future<void> _initializePoseDetector() async {
    try {
      if (_isDisposed) return;

      await _poseDetector.initialize();

      if (_isDisposed) return;

      setState(() {
        _isInitialized = true;
      });
    } catch (e) {
      if (!_isDisposed && mounted) {
        setState(() {
          _errorMessage = 'Failed to initialize pose detector: $e';
        });
      }
    }
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        if (mounted) {
          setState(() => _errorMessage = 'No cameras available');
        }
        return;
      }

      _availableCameras = cameras;

      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      await _startControllerFor(camera, markInitialized: false);
    } catch (e) {
      if (mounted) {
        setState(() => _errorMessage = 'Camera init failed: $e');
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
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await controller.initialize();

    if (!mounted) {
      await controller.dispose();
      return;
    }

    _cameraController = controller;

    setState(() {
      _sensorOrientation = controller.description.sensorOrientation;
      _isFrontCamera =
          controller.description.lensDirection == CameraLensDirection.front;
      _cameraSize = Size(
        controller.value.previewSize!.height,
        controller.value.previewSize!.width,
      );
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

    setState(() => _isSwitchingCamera = true);
    try {
      final prev = _cameraController;
      if (prev != null) {
        if (_isImageStreamStarted) {
          try {
            await prev.stopImageStream();
          } catch (_) {}
          _isImageStreamStarted = false;
        }
        await prev.dispose();
      }
      _poseNotifier.value = [];

      await _startControllerFor(next, markInitialized: false);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error switching camera: $e')));
      }
    } finally {
      if (mounted) setState(() => _isSwitchingCamera = false);
    }
  }

  void _updateFps() {
    _framesSinceLastUpdate++;
    final now = DateTime.now();
    if (_lastFpsUpdate != null) {
      final diff = now.difference(_lastFpsUpdate!).inMilliseconds;
      if (diff >= 1000 && mounted) {
        setState(() {
          _fps = (_framesSinceLastUpdate * 1000 / diff).round();
          _framesSinceLastUpdate = 0;
          _lastFpsUpdate = now;
        });
      }
    } else {
      _lastFpsUpdate = now;
    }
  }

  int get _barQuarterTurns {
    switch (_deviceOrientation) {
      case 'Landscape Left':
        return 1;
      case 'Landscape Right':
        return 3;
      default:
        return 0;
    }
  }

  Widget _buildCameraTopBar() {
    final canPop = Navigator.of(context).canPop();
    final isMobile = !kIsWeb && (Platform.isAndroid || Platform.isIOS);

    final fpsText = SizedBox(
      width: 70,
      child: Text(
        'FPS: $_fps',
        style: const TextStyle(color: Colors.white, fontSize: 14),
        textAlign: isMobile ? TextAlign.left : TextAlign.right,
      ),
    );
    const separator = Text(
      ' | ',
      style: TextStyle(color: Colors.white, fontSize: 14),
    );
    final msText = SizedBox(
      width: 70,
      child: Text(
        '${_detectionTimeMs}ms',
        style: const TextStyle(color: Colors.white, fontSize: 14),
      ),
    );

    return Material(
      color: Colors.black.withAlpha(179),
      elevation: 4,
      child: SizedBox(
        height: kToolbarHeight,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 4),
          child: Row(
            children: [
              if (canPop)
                IconButton(
                  tooltip: 'Back',
                  color: Colors.white,
                  icon: const Icon(Icons.arrow_back),
                  onPressed: () => Navigator.of(context).maybePop(),
                ),
              if (isMobile) ...[
                const SizedBox(width: 8),
                fpsText,
                separator,
                msText,
                const Spacer(),
              ] else
                const Expanded(
                  child: Padding(
                    padding: EdgeInsets.symmetric(horizontal: 8),
                    child: Text(
                      'Live Pose Detection',
                      style: TextStyle(color: Colors.white, fontSize: 18),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ),
              if (_canSwitchCamera)
                IconButton(
                  tooltip: _isFrontCamera
                      ? 'Switch to back camera'
                      : 'Switch to front camera',
                  color: Colors.white,
                  icon: Icon(
                    Platform.isIOS
                        ? Icons.flip_camera_ios
                        : Icons.flip_camera_android,
                  ),
                  onPressed: _isSwitchingCamera ? null : _switchCamera,
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
              if (!isMobile) ...[
                const SizedBox(width: 8),
                fpsText,
                separator,
                msText,
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSettingsMenuContent(StateSetter setMenuState) {
    void update(VoidCallback fn) {
      setState(fn);
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
            const Text('LANDMARK MODEL', style: sectionLabelStyle),
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
                    current: _landmarkModel,
                    label: label,
                    onTap: () async {
                      if (v == _landmarkModel) return;
                      update(() => _landmarkModel = v);
                      await _reinitPoseDetector();
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

  Widget _positionedTopBar(int turns) {
    final bar = _buildCameraTopBar();
    final padding = MediaQuery.of(context).padding;
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
      width: kToolbarHeight,
      child: RotatedBox(quarterTurns: turns, child: bar),
    );
  }

  Future<void> _reinitPoseDetector() async {
    final old = _poseDetector;
    await old.dispose();
    _poseDetector = PoseDetector(
      mode: PoseMode.boxesAndLandmarks,
      landmarkModel: _landmarkModel,
      detectorConf: 0.7,
      detectorIou: 0.4,
      maxDetections: 5,
      minLandmarkScore: 0.5,
      performanceConfig: const PerformanceConfig.xnnpack(),
    );
    await _poseDetector.initialize();
  }

  int? _rotationFlagForFrame({required int width, required int height}) {
    final int? sensor = _sensorOrientation;
    if (sensor == null) return null;

    // iOS: the camera plugin pre-rotates the image stream per
    // AVCaptureConnection.videoOrientation, so the historical portrait-only
    // rotation path still applies.
    if (Platform.isIOS) {
      final bool isPortrait =
          MediaQuery.of(context).orientation == Orientation.portrait;
      if (!isPortrait) return null;
      if (height >= width) return null;
      if (sensor == 90) return cv.ROTATE_90_CLOCKWISE;
      if (sensor == 270) return cv.ROTATE_90_COUNTERCLOCKWISE;
      return null;
    }

    // Android: combined formula covering all four device orientations.
    // `sensorOrientation` is the clockwise rotation needed to display the
    // raw sensor buffer upright in the device's natural orientation;
    // `deviceRotation` is how far the device is rotated clockwise from
    // natural (portraitUp=0, landscapeLeft=90, portraitDown=180,
    // landscapeRight=270; per Flutter's DeviceOrientation enum).
    if (Platform.isAndroid) {
      final DeviceOrientation d =
          _cameraController?.value.deviceOrientation ??
          DeviceOrientation.portraitUp;
      final int deviceRotation = switch (d) {
        DeviceOrientation.portraitUp => 0,
        DeviceOrientation.landscapeLeft => 90,
        DeviceOrientation.portraitDown => 180,
        DeviceOrientation.landscapeRight => 270,
      };

      final int total = _isFrontCamera
          ? (sensor + deviceRotation) % 360
          : (sensor - deviceRotation + 360) % 360;

      return switch (total) {
        90 => cv.ROTATE_90_CLOCKWISE,
        180 => cv.ROTATE_180,
        270 => cv.ROTATE_90_COUNTERCLOCKWISE,
        _ => null,
      };
    }

    // Desktop / web: camera_desktop delivers already-upright frames.
    return null;
  }

  cv.Mat? _convertCameraImageToMat(CameraImage image) {
    try {
      final int width = image.width;
      final int height = image.height;

      // Desktop: camera_desktop provides single-plane 4-channel packed format
      if (image.planes.length == 1 &&
          (image.planes[0].bytesPerPixel ?? 1) >= 4) {
        final bytes = image.planes[0].bytes;
        final stride = image.planes[0].bytesPerRow;

        // Create a 4-channel Mat directly from camera bytes (handles stride)
        final matCols = stride ~/ 4;
        final bgraOrRgba = cv.Mat.fromList(
          height,
          matCols,
          cv.MatType.CV_8UC4,
          bytes,
        );
        // Crop out stride padding if present
        final cropped = matCols != width
            ? bgraOrRgba.region(cv.Rect(0, 0, width, height))
            : bgraOrRgba;

        // Native SIMD-accelerated color conversion
        final colorCode = Platform.isMacOS
            ? cv.COLOR_BGRA2BGR
            : cv.COLOR_RGBA2BGR;
        cv.Mat mat = cv.cvtColor(cropped, colorCode);

        if (!identical(cropped, bgraOrRgba)) cropped.dispose();
        bgraOrRgba.dispose();

        final rotationFlag = _rotationFlagForFrame(
          width: width,
          height: height,
        );
        if (rotationFlag != null) {
          final rotated = cv.rotate(mat, rotationFlag);
          mat.dispose();
          return rotated;
        }
        return mat;
      }

      // Mobile: YUV420. Pack Y+UV into a contiguous buffer via flutter_litert's
      // shared `packYuv420`, then hand to OpenCV for native cvtColor.
      final p0 = image.planes[0];
      final p1 = image.planes.length > 1 ? image.planes[1] : null;
      final p2 = image.planes.length > 2 ? image.planes[2] : null;
      if (p1 == null) return null;

      final packed = packYuv420(
        width: width,
        height: height,
        y: (
          bytes: p0.bytes,
          rowStride: p0.bytesPerRow,
          pixelStride: p0.bytesPerPixel ?? 1,
        ),
        u: (
          bytes: p1.bytes,
          rowStride: p1.bytesPerRow,
          pixelStride: p1.bytesPerPixel ?? 1,
        ),
        v: p2 == null
            ? null
            : (
                bytes: p2.bytes,
                rowStride: p2.bytesPerRow,
                pixelStride: p2.bytesPerPixel ?? 1,
              ),
      );
      if (packed == null) return null;

      final int cvtCode = switch (packed.layout) {
        YuvLayout.nv12 => cv.COLOR_YUV2BGR_NV12,
        YuvLayout.nv21 => cv.COLOR_YUV2BGR_NV21,
        YuvLayout.i420 => cv.COLOR_YUV2BGR_I420,
      };
      final cv.Mat yuvMat = cv.Mat.fromList(
        packed.height + packed.height ~/ 2,
        packed.width,
        cv.MatType.CV_8UC1,
        packed.bytes,
      );
      cv.Mat mat = cv.cvtColor(yuvMat, cvtCode);
      yuvMat.dispose();

      // Rotate image for portrait mode so pose detector sees upright people.
      final rotationFlag = _rotationFlagForFrame(width: width, height: height);
      if (rotationFlag != null) {
        final rotated = cv.rotate(mat, rotationFlag);
        mat.dispose();
        return rotated;
      }

      return mat;
    } catch (_) {
      return null;
    }
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_isDisposed || !_isInitialized || _isProcessing) return;

    // Dynamic throttling: Skip if not enough time has passed
    final int now = DateTime.now().millisecondsSinceEpoch;
    if (now - _lastProcessedTime < _minProcessingIntervalMs) {
      return;
    }

    _isProcessing = true;
    final int startTime = now;

    try {
      cv.Mat? mat = _convertCameraImageToMat(image);
      if (mat == null) {
        _isProcessing = false;
        return;
      }

      if (_isDisposed) {
        mat.dispose();
        return;
      }

      // Track size based on actual post-rotation Mat dims so the overlay
      // coordinate space matches what the detector sees.
      final Size matSize = Size(mat.cols.toDouble(), mat.rows.toDouble());
      if (_cameraSize != matSize && mounted && !_isDisposed) {
        setState(() {
          _cameraSize = matSize;
        });
      }

      // Downscale for performance, the detection model internally resizes
      // to 256px, so full-res frames just waste IPC bandwidth.
      const int maxDim = 640;
      if (mat.cols > maxDim || mat.rows > maxDim) {
        final double scale =
            maxDim / (mat.cols > mat.rows ? mat.cols : mat.rows);
        final cv.Mat resized = cv.resize(mat, (
          (mat.cols * scale).toInt(),
          (mat.rows * scale).toInt(),
        ), interpolation: cv.INTER_LINEAR);
        mat.dispose();
        mat = resized;
      }

      // Use detectFromMat for direct cv.Mat input - no image decoding needed
      final List<Pose> poses = await _poseDetector.detectFromMat(
        mat,
        imageWidth: mat.cols,
        imageHeight: mat.rows,
      );

      // Clean up native Mat resource
      mat.dispose();

      // Update via ValueNotifier instead of setState
      if (!_isDisposed) {
        _poseNotifier.value = poses;
      }

      // Update performance metrics
      _lastProcessedTime = DateTime.now().millisecondsSinceEpoch;
      final int processingTime = _lastProcessedTime - startTime;
      _detectionCount++;
      _avgProcessingTimeMs =
          (_avgProcessingTimeMs * (_detectionCount - 1) + processingTime) /
          _detectionCount;
      if (mounted && !_isDisposed) {
        setState(() => _detectionTimeMs = processingTime);
      }
      _updateFps();
    } catch (e) {
      // Silently ignore errors to maintain camera feed
    } finally {
      _isProcessing = false;
    }
  }

  @override
  void dispose() {
    _isDisposed = true;
    _accelerometerSub?.cancel();
    _poseNotifier.dispose();

    if (_isImageStreamStarted) {
      try {
        _cameraController?.stopImageStream();
      } catch (_) {}
    }
    _cameraController?.dispose();

    _poseDetector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final turns = _barQuarterTurns;
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [_buildBody(), _positionedTopBar(turns)],
      ),
    );
  }

  Widget _buildBody() {
    if (_errorMessage != null && !_isInitialized) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, size: 64, color: Colors.red),
            const SizedBox(height: 16),
            Text(
              _errorMessage!,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.red),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: () {
                setState(() {
                  _errorMessage = null;
                });
                _initializePoseDetector();
              },
              child: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    if (!_isInitialized) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Initializing pose detector...'),
          ],
        ),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        if (_cameraController != null && _cameraController!.value.isInitialized)
          Center(
            child: AspectRatio(
              aspectRatio: _cameraSize != null
                  ? _cameraSize!.width / _cameraSize!.height
                  : _cameraController!.value.aspectRatio,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  CameraPreview(_cameraController!),
                  ValueListenableBuilder<List<Pose>>(
                    valueListenable: _poseNotifier,
                    builder: (context, poses, _) {
                      if (poses.isEmpty) return const SizedBox.shrink();
                      return CustomPaint(
                        painter: CameraPoseOverlayPainter(
                          poses: poses,
                          cameraSize: _cameraSize!,
                          mirrorHorizontally:
                              Platform.isAndroid && _isFrontCamera,
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),
          )
        else
          const Center(child: CircularProgressIndicator()),

        // Status indicator with performance metrics
        Positioned(
          bottom: 16,
          left: 0,
          right: 0,
          child: Center(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                _isProcessing
                    ? 'Processing...'
                    : 'Ready (${_avgProcessingTimeMs.toStringAsFixed(0)}ms avg)',
                style: const TextStyle(color: Colors.white),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class CameraPoseOverlayPainter extends CustomPainter {
  final List<Pose> poses;
  final Size cameraSize;
  final bool mirrorHorizontally;

  CameraPoseOverlayPainter({
    required this.poses,
    required this.cameraSize,
    required this.mirrorHorizontally,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (poses.isEmpty) return;

    // Get image dimensions from first pose
    final int imageWidth = poses.first.imageWidth;
    final int imageHeight = poses.first.imageHeight;

    // Direct scaling: overlay is now wrapped in AspectRatio matching camera preview,
    // so canvas size has the same aspect ratio as the image. No letterbox offsets needed.
    final double scaleX = size.width / imageWidth;
    final double scaleY = size.height / imageHeight;

    for (final pose in poses) {
      _drawBbox(canvas, pose, scaleX, scaleY, 0, 0, size);
      if (pose.hasLandmarks) {
        _drawConnections(canvas, pose, scaleX, scaleY, 0, 0, size);
        _drawLandmarks(canvas, pose, scaleX, scaleY, 0, 0, size);
      }
    }
  }

  double _mirrorX(double x, double scaleX, double offsetX, Size size) {
    final mapped = x * scaleX + offsetX;
    return mirrorHorizontally ? size.width - mapped : mapped;
  }

  void _drawConnections(
    Canvas canvas,
    Pose pose,
    double scaleX,
    double scaleY,
    double offsetX,
    double offsetY,
    Size size,
  ) {
    final Paint paint = Paint()
      ..color = Colors.green.withValues(alpha: 0.8)
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;

    // Use the predefined skeleton connections from the package
    for (final List<PoseLandmarkType> c in poseLandmarkConnections) {
      final PoseLandmark? start = pose.getLandmark(c[0]);
      final PoseLandmark? end = pose.getLandmark(c[1]);
      if (start != null &&
          end != null &&
          start.visibility > 0.5 &&
          end.visibility > 0.5) {
        canvas.drawLine(
          Offset(
            _mirrorX(start.x, scaleX, offsetX, size),
            start.y * scaleY + offsetY,
          ),
          Offset(
            _mirrorX(end.x, scaleX, offsetX, size),
            end.y * scaleY + offsetY,
          ),
          paint,
        );
      }
    }
  }

  void _drawLandmarks(
    Canvas canvas,
    Pose pose,
    double scaleX,
    double scaleY,
    double offsetX,
    double offsetY,
    Size size,
  ) {
    for (final PoseLandmark l in pose.landmarks) {
      if (l.visibility > 0.5) {
        final Offset center = Offset(
          _mirrorX(l.x, scaleX, offsetX, size),
          l.y * scaleY + offsetY,
        );
        final Paint glow = Paint()..color = Colors.blue.withValues(alpha: 0.3);
        final Paint point = Paint()..color = Colors.red;
        final Paint centerDot = Paint()..color = Colors.white;
        canvas.drawCircle(center, 8, glow);
        canvas.drawCircle(center, 5, point);
        canvas.drawCircle(center, 2, centerDot);
      }
    }
  }

  void _drawBbox(
    Canvas canvas,
    Pose pose,
    double scaleX,
    double scaleY,
    double offsetX,
    double offsetY,
    Size size,
  ) {
    final Paint boxPaint = Paint()
      ..color = Colors.orangeAccent.withValues(alpha: 0.9)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final Paint fillPaint = Paint()
      ..color = Colors.orangeAccent.withValues(alpha: 0.08)
      ..style = PaintingStyle.fill;

    final double x1 = _mirrorX(pose.boundingBox.left, scaleX, offsetX, size);
    final double y1 = pose.boundingBox.top * scaleY + offsetY;
    final double x2 = _mirrorX(pose.boundingBox.right, scaleX, offsetX, size);
    final double y2 = pose.boundingBox.bottom * scaleY + offsetY;
    final Rect rect = Rect.fromLTRB(
      x1 < x2 ? x1 : x2,
      y1,
      x1 < x2 ? x2 : x1,
      y2,
    );
    canvas.drawRect(rect, fillPaint);
    canvas.drawRect(rect, boxPaint);
  }

  @override
  bool shouldRepaint(CameraPoseOverlayPainter oldDelegate) {
    if (mirrorHorizontally != oldDelegate.mirrorHorizontally) return true;
    // Only repaint if poses actually changed
    if (poses.length != oldDelegate.poses.length) return true;
    if (poses.isEmpty) return false;

    // Quick check: compare first pose bounding box
    final Pose current = poses.first;
    final Pose old = oldDelegate.poses.first;
    return current.boundingBox.left != old.boundingBox.left ||
        current.boundingBox.top != old.boundingBox.top ||
        current.boundingBox.right != old.boundingBox.right ||
        current.boundingBox.bottom != old.boundingBox.bottom;
  }
}
