import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:pose_detection/pose_detection.dart';
import 'package:camera/camera.dart';
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
      final List<Pose> poses = await _poseDetector!.detectFromCameraImage(
        image,
        rotation: rotation,
        isBgra: Platform.isMacOS,
        maxDim: maxDim,
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
