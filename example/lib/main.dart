import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:file_selector/file_selector.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:path_provider/path_provider.dart';
import 'package:pose_detection/pose_detection.dart';
import 'package:camera/camera.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:video_player/video_player.dart';

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
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(vertical: 32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.accessibility_new, size: 100, color: Colors.blue[300]),
            const SizedBox(height: 32),
            Text(
              'Choose Detection Mode',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            const SizedBox(height: 32),
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
            const SizedBox(height: 24),
            _buildModeCard(
              context,
              icon: Icons.movie_creation_outlined,
              title: 'Video File',
              description: 'Run pose detection over every frame of an MP4',
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const VideoFileScreen(),
                  ),
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
  PoseDetector? _poseDetector;
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
      _poseDetector = await PoseDetector.create(
        mode: PoseMode.boxesAndLandmarks,
        landmarkModel: PoseLandmarkModel.heavy,
        detectorConf: 0.6,
        detectorIou: 0.4,
        maxDetections: 10,
        minLandmarkScore: 0.5,
        performanceConfig: const PerformanceConfig.xnnpack(),
      );
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
      final List<Pose> results = await _poseDetector!.detect(bytes);

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
    _poseDetector?.dispose();
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

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _cameraController;
  bool _isImageStreamStarted = false;
  PoseDetector? _poseDetector;

  bool _isInitialized = false;
  bool _isProcessing = false;
  bool _isDisposed = false;
  String? _errorMessage;
  Size? _cameraSize;
  int? _sensorOrientation;
  bool _isFrontCamera = false;

  List<CameraDescription> _availableCameras = const [];
  bool _isSwitchingCamera = false;
  DeviceOrientation _deviceOrientation = DeviceOrientation.portraitUp;
  final FpsCounter _fpsCounter = FpsCounter();
  int _fps = 0;
  int _detectionTimeMs = 0;
  StreamSubscription<AccelerometerEvent>? _accelerometerSub;

  final ValueNotifier<List<Pose>> _poseNotifier = ValueNotifier<List<Pose>>([]);

  int _lastProcessedTime = 0;
  static const int _minProcessingIntervalMs = 50;

  int _detectionCount = 0;
  double _avgProcessingTimeMs = 0;

  PoseLandmarkModel _landmarkModel = PoseLandmarkModel.lite;
  bool _smoothingEnabled = true;
  final PoseSmoother _smoother = PoseSmoother(enabled: true);

  @override
  void initState() {
    super.initState();
    _initializePoseDetector();
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

  Future<void> _initializePoseDetector() async {
    try {
      if (_isDisposed) return;

      _poseDetector = await PoseDetector.create(
        mode: PoseMode.boxesAndLandmarks,
        landmarkModel: PoseLandmarkModel.lite,
        detectorConf: 0.7,
        detectorIou: 0.4,
        maxDetections: 5,
        minLandmarkScore: 0.5,
        performanceConfig: const PerformanceConfig.xnnpack(),
      );

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
      _smoother.reset();

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

  DeviceOrientation _effectiveDeviceOrientation(BuildContext context) {
    final controller = _cameraController;
    if (controller != null) {
      return controller.value.deviceOrientation;
    }
    return MediaQuery.of(context).orientation == Orientation.portrait
        ? DeviceOrientation.portraitUp
        : DeviceOrientation.landscapeLeft;
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
            const SizedBox(height: 14),
            const Text('SMOOTHING', style: sectionLabelStyle),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: Text(
                    _smoothingEnabled
                        ? 'On (One-Euro filter)'
                        : 'Off (raw landmarks)',
                    style: const TextStyle(color: Colors.white, fontSize: 12),
                  ),
                ),
                Switch(
                  value: _smoothingEnabled,
                  onChanged: (v) {
                    update(() {
                      _smoothingEnabled = v;
                      _smoother.enabled = v;
                      _smoother.reset();
                    });
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
    await _poseDetector?.dispose();
    _poseDetector = await PoseDetector.create(
      mode: PoseMode.boxesAndLandmarks,
      landmarkModel: _landmarkModel,
      detectorConf: 0.7,
      detectorIou: 0.4,
      maxDetections: 5,
      minLandmarkScore: 0.5,
      performanceConfig: const PerformanceConfig.xnnpack(),
    );
  }

  /// Post-rotation (pre-downscale) frame size, used for overlay coord mapping.
  Size _postRotationSize(int width, int height, CameraFrameRotation? rotation) {
    final bool swap =
        rotation == CameraFrameRotation.cw90 ||
        rotation == CameraFrameRotation.cw270;
    return swap
        ? Size(height.toDouble(), width.toDouble())
        : Size(width.toDouble(), height.toDouble());
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_isDisposed || !_isInitialized || _isProcessing) return;

    final int now = DateTime.now().millisecondsSinceEpoch;
    if (now - _lastProcessedTime < _minProcessingIntervalMs) {
      return;
    }

    _isProcessing = true;
    final int startTime = now;

    try {
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

      if (_isDisposed) return;

      final Size matSize = _postRotationSize(
        image.width,
        image.height,
        rotation,
      );
      if (_cameraSize != matSize && mounted && !_isDisposed) {
        setState(() {
          _cameraSize = matSize;
        });
      }

      const int maxDim = 640;
      final List<Pose> raw = await _poseDetector!.detectFromCameraImage(
        image,
        rotation: rotation,
        isBgra: Platform.isMacOS,
        maxDim: maxDim,
      );

      final List<Pose> poses = _smoother.apply(
        raw,
        DateTime.now().millisecondsSinceEpoch / 1000.0,
      );

      if (!_isDisposed) {
        _poseNotifier.value = poses;
      }

      _lastProcessedTime = DateTime.now().millisecondsSinceEpoch;
      final int processingTime = _lastProcessedTime - startTime;
      _detectionCount++;
      _avgProcessingTimeMs =
          (_avgProcessingTimeMs * (_detectionCount - 1) + processingTime) /
          _detectionCount;
      if (mounted && !_isDisposed) {
        setState(() => _detectionTimeMs = processingTime);
      }
      if (_fpsCounter.tick() && mounted) {
        setState(() => _fps = _fpsCounter.fps);
      }
    } catch (_) {
      /// Silently handle errors during processing to keep the stream alive.
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

    _poseDetector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final turns = barQuarterTurns(_deviceOrientation);
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
        mode: PoseMode.boxesAndLandmarks,
        landmarkModel: PoseLandmarkModel.heavy,
        detectorConf: 0.6,
        detectorIou: 0.4,
        maxDetections: 10,
        minLandmarkScore: 0.5,
        performanceConfig: const PerformanceConfig.xnnpack(),
      );
      if (!mounted) {
        await detector.dispose();
        return;
      }
      setState(() {
        _detector = detector;
        _isInitialized = true;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _errorMessage = 'Failed to initialize detector: $e');
    }
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

  void _drawPosesOnMat(cv.Mat mat, List<Pose> poses) {
    if (poses.isEmpty) return;
    final orange = cv.Scalar(0, 165, 255);
    final green = cv.Scalar(0, 255, 0);
    final red = cv.Scalar(0, 0, 255);
    final black = cv.Scalar(0, 0, 0);

    final w = mat.cols;
    final h = mat.rows;

    for (final pose in poses) {
      final bb = pose.boundingBox;
      final l = bb.left.toInt().clamp(0, w - 1);
      final t = bb.top.toInt().clamp(0, h - 1);
      final r = bb.right.toInt().clamp(0, w - 1);
      final b = bb.bottom.toInt().clamp(0, h - 1);
      cv.rectangle(
        mat,
        cv.Rect(l, t, (r - l).clamp(1, w), (b - t).clamp(1, h)),
        orange,
        thickness: 3,
      );

      final label = 'Person ${(pose.score * 100).toStringAsFixed(0)}%';
      final (sz, _) = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2);
      final labelTop = (t - sz.height - 8).clamp(0, h - 1);
      final labelW = (sz.width + 8).clamp(1, w - l);
      final labelH = (sz.height + 8).clamp(1, h - labelTop);
      cv.rectangle(
        mat,
        cv.Rect(l, labelTop, labelW, labelH),
        orange,
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

      if (!pose.hasLandmarks) continue;

      for (final c in poseLandmarkConnections) {
        final s = pose.getLandmark(c[0]);
        final e = pose.getLandmark(c[1]);
        if (s == null ||
            e == null ||
            s.visibility <= 0.5 ||
            e.visibility <= 0.5) {
          continue;
        }
        cv.line(
          mat,
          cv.Point(s.x.toInt(), s.y.toInt()),
          cv.Point(e.x.toInt(), e.y.toInt()),
          green,
          thickness: 3,
        );
      }
      for (final lm in pose.landmarks) {
        if (lm.visibility <= 0.5) continue;
        cv.circle(
          mat,
          cv.Point(lm.x.toInt(), lm.y.toInt()),
          4,
          red,
          thickness: -1,
        );
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Video File - Pose Detection')),
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
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
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

    return Padding(
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
                    : 'Off: raw per-frame landmarks',
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
            Card(
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
                            _statusMessage!,
                            style: const TextStyle(fontWeight: FontWeight.w500),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Total time: ${_formatDuration(_elapsed)} '
                      '(${processedFps.toStringAsFixed(1)} fps avg)',
                    ),
                    const SizedBox(height: 12),
                    _buildOutputPreview(),
                    const SizedBox(height: 12),
                    if (Platform.isMacOS ||
                        Platform.isLinux ||
                        Platform.isWindows)
                      Align(
                        alignment: Alignment.centerLeft,
                        child: ElevatedButton.icon(
                          onPressed: _openOutputFile,
                          icon: const Icon(Icons.play_circle_outline),
                          label: const Text('Open output video'),
                        ),
                      ),
                  ],
                ),
              ),
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
              Text('Loading preview...'),
            ],
          ),
        ),
      );
    }
    return _OutputVideoPlayer(controller: controller);
  }
}

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
    final pos = value.position;
    final dur = value.duration;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: AspectRatio(
            aspectRatio: value.aspectRatio == 0 ? 16 / 9 : value.aspectRatio,
            child: Stack(
              fit: StackFit.expand,
              children: [
                Container(color: Colors.black),
                VideoPlayer(c),
              ],
            ),
          ),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            IconButton(
              icon: Icon(value.isPlaying ? Icons.pause : Icons.play_arrow),
              onPressed: () {
                if (value.isPlaying) {
                  c.pause();
                } else {
                  c.play();
                }
              },
            ),
            Expanded(
              child: VideoProgressIndicator(
                c,
                allowScrubbing: true,
                padding: const EdgeInsets.symmetric(vertical: 12),
              ),
            ),
            const SizedBox(width: 8),
            Text(
              '${_fmt(pos)} / ${_fmt(dur)}',
              style: const TextStyle(
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ],
        ),
      ],
    );
  }
}

class _OneEuroFilter {
  final double minCutoff;
  final double beta;
  final double dCutoff;

  double? _xPrev;
  double _dxPrev = 0.0;
  double? _tPrev;

  _OneEuroFilter({this.minCutoff = 1.0, this.beta = 0.1, this.dCutoff = 1.0});

  static double _alpha(double cutoff, double dt) {
    final tau = 1.0 / (2 * math.pi * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  double filter(double x, double tSec) {
    if (_xPrev == null || _tPrev == null) {
      _xPrev = x;
      _tPrev = tSec;
      _dxPrev = 0.0;
      return x;
    }
    final dt = (tSec - _tPrev!).clamp(1e-6, 1.0);
    final dx = (x - _xPrev!) / dt;
    final aD = _alpha(dCutoff, dt);
    final dxHat = aD * dx + (1 - aD) * _dxPrev;
    final cutoff = minCutoff + beta * dxHat.abs();
    final a = _alpha(cutoff, dt);
    final xHat = a * x + (1 - a) * _xPrev!;
    _xPrev = xHat;
    _dxPrev = dxHat;
    _tPrev = tSec;
    return xHat;
  }

  void reset() {
    _xPrev = null;
    _tPrev = null;
    _dxPrev = 0.0;
  }
}

class _PoseTrack {
  final Map<PoseLandmarkType, List<_OneEuroFilter>> filters = {};
  double lastLeft = 0, lastTop = 0, lastRight = 0, lastBottom = 0;
  bool hasBox = false;
  int missedFrames = 0;
}

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
    final smoothed = <PoseLandmark>[];
    for (final lm in pose.landmarks) {
      var fs = track.filters[lm.type];
      if (fs == null) {
        fs = [
          _OneEuroFilter(minCutoff: 1.0, beta: 0.1, dCutoff: 1.0),
          _OneEuroFilter(minCutoff: 1.0, beta: 0.1, dCutoff: 1.0),
          _OneEuroFilter(minCutoff: 1.0, beta: 0.1, dCutoff: 1.0),
        ];
        track.filters[lm.type] = fs;
      }
      if (lm.visibility < 0.5) {
        for (final f in fs) {
          f.reset();
        }
        smoothed.add(lm);
        continue;
      }
      final sx = fs[0].filter(lm.x, tSec);
      final sy = fs[1].filter(lm.y, tSec);
      final sz = fs[2].filter(lm.z, tSec);
      smoothed.add(
        PoseLandmark(
          type: lm.type,
          x: sx,
          y: sy,
          z: sz,
          visibility: lm.visibility,
        ),
      );
    }
    return Pose(
      boundingBox: pose.boundingBox,
      score: pose.score,
      landmarks: smoothed,
      imageWidth: pose.imageWidth,
      imageHeight: pose.imageHeight,
    );
  }

  double _iou(Pose a, _PoseTrack b) {
    final box = a.boundingBox;
    final l = math.max(box.left, b.lastLeft);
    final t = math.max(box.top, b.lastTop);
    final r = math.min(box.right, b.lastRight);
    final bo = math.min(box.bottom, b.lastBottom);
    final iw = math.max(0.0, r - l);
    final ih = math.max(0.0, bo - t);
    final inter = iw * ih;
    final aa =
        math.max(0.0, box.right - box.left) *
        math.max(0.0, box.bottom - box.top);
    final bb =
        math.max(0.0, b.lastRight - b.lastLeft) *
        math.max(0.0, b.lastBottom - b.lastTop);
    final union = aa + bb - inter;
    if (union <= 0) return 0;
    return inter / union;
  }
}
