import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:pose_detection_tflite/pose_detection_tflite.dart';
import 'package:camera_macos/camera_macos_controller.dart';
import 'package:camera_macos/camera_macos_view.dart';
import 'package:camera_macos/camera_macos_arguments.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(const PoseDetectionApp());
}

class PoseDetectionApp extends StatelessWidget {
  const PoseDetectionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pose Detection Demo',
      theme: ThemeData(
        colorSchemeSeed: Colors.blue,
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Pose Detection Demo'),
      ),
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
                  MaterialPageRoute(
                    builder: (context) => const CameraScreen(),
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
  final PoseDetector _poseDetector = PoseDetector(
    mode: PoseMode.boxesAndLandmarks,
    landmarkModel: PoseLandmarkModel.heavy,
    detectorConf: 0.6,
    detectorIou: 0.4,
    maxDetections: 10,
    minLandmarkScore: 0.5,
    performanceConfig: const PerformanceConfig.xnnpack(), // Enable XNNPACK for 2-5x speedup
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
            Text('Select an image to detect pose',
                style: TextStyle(fontSize: 18, color: Colors.grey[600])),
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
          PoseVisualizerWidget(
            imageFile: _imageFile!,
            results: _results,
          ),
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
                      Text('Detections: ${_results.length} ✓',
                          style: Theme.of(context)
                              .textTheme
                              .titleLarge
                              ?.copyWith(
                                  color: Colors.green,
                                  fontWeight: FontWeight.bold)),
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
            Text('Landmark Details (first pose)',
                style: Theme.of(context).textTheme.headlineSmall),
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
      final Point pixel =
          landmark.toPixel(result.imageWidth, result.imageHeight);
      return Card(
        margin: const EdgeInsets.only(bottom: 8),
        child: ListTile(
          leading: CircleAvatar(
            backgroundColor:
                landmark.visibility > 0.5 ? Colors.green : Colors.orange,
            child: Text(landmark.type.index.toString(),
                style: const TextStyle(fontSize: 12)),
          ),
          title: Text(_landmarkName(landmark.type),
              style: const TextStyle(fontWeight: FontWeight.w500)),
          subtitle: Text(''
              'Position: (${pixel.x}, ${pixel.y})\n'
              'Visibility: ${(landmark.visibility * 100).toStringAsFixed(0)}%'),
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
        .replaceAllMapped(
          RegExp(r'[A-Z]'),
          (match) => ' ${match.group(0)}',
        )
        .trim();
  }
}

class PoseVisualizerWidget extends StatelessWidget {
  final File imageFile;
  final List<Pose> results;

  const PoseVisualizerWidget(
      {super.key, required this.imageFile, required this.results});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, constraints) {
      return Stack(
        children: [
          Image.file(imageFile, fit: BoxFit.contain),
          Positioned.fill(
              child:
                  CustomPaint(painter: MultiOverlayPainter(results: results))),
        ],
      );
    });
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

  void _drawConnections(Canvas canvas, Pose result, double scaleX,
      double scaleY, double offsetX, double offsetY) {
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

  void _drawLandmarks(Canvas canvas, Pose result, double scaleX, double scaleY,
      double offsetX, double offsetY) {
    for (final PoseLandmark l in result.landmarks) {
      if (l.visibility > 0.5) {
        final Offset center =
            Offset(l.x * scaleX + offsetX, l.y * scaleY + offsetY);
        final Paint glow = Paint()..color = Colors.blue.withValues(alpha: 0.3);
        final Paint point = Paint()..color = Colors.red;
        final Paint centerDot = Paint()..color = Colors.white;
        canvas.drawCircle(center, 8, glow);
        canvas.drawCircle(center, 5, point);
        canvas.drawCircle(center, 2, centerDot);
      }
    }
  }

  void _drawBbox(Canvas canvas, Pose r, double scaleX, double scaleY,
      double offsetX, double offsetY) {
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
  CameraMacOSController? _cameraController;
  final PoseDetector _poseDetector = PoseDetector(
    mode: PoseMode.boxes,
    landmarkModel:
        PoseLandmarkModel.lite, // Use lite for better real-time performance
    detectorConf: 0.7,
    detectorIou: 0.4,
    maxDetections: 5,
    minLandmarkScore: 0.5,
    performanceConfig: const PerformanceConfig.xnnpack(), // Enable XNNPACK for real-time performance
  );

  bool _isInitialized = false;
  bool _isProcessing = false;
  List<Pose> _currentPoses = [];
  String? _errorMessage;
  int _frameCount = 0;
  static const int _frameSkip = 10; // Process every 5th frame for performance
  Size? _cameraSize;

  @override
  void initState() {
    super.initState();
    _initializePoseDetector();
  }

  Future<void> _initializePoseDetector() async {
    try {
      // Initialize pose detector
      await _poseDetector.initialize();
      setState(() {
        _isInitialized = true;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to initialize pose detector: $e';
      });
    }
  }

  void _onCameraInitialized(CameraMacOSController controller) {
    setState(() {
      _cameraController = controller;
    });

    // Start image stream
    controller.startImageStream((CameraImageData? imageData) {
      if (imageData != null) {
        _processCameraImage(imageData);
      }
    });
  }

  Future<void> _processCameraImage(CameraImageData imageData) async {
    // Skip frames for performance
    _frameCount++;
    if (_frameCount % _frameSkip != 0) {
      return;
    }

    // Skip if already processing
    if (_isProcessing || !_isInitialized) {
      return;
    }

    _isProcessing = true;

    try {
      // Convert ARGB8888 to PNG bytes
      final Uint8List pngBytes = _convertARGBtoPNG(imageData);

      // Store camera size from image data
      if (_cameraSize == null) {
        setState(() {
          _cameraSize =
              Size(imageData.width.toDouble(), imageData.height.toDouble());
        });
      }

      // Run pose detection
      final List<Pose> poses = await _poseDetector.detect(pngBytes);

      // Update UI with results
      if (mounted) {
        setState(() {
          _currentPoses = poses;
        });
      }
    } catch (e) {
      // Silently ignore errors to avoid spamming the UI
      // print('Detection error: $e');
    } finally {
      _isProcessing = false;
    }
  }

  Uint8List _convertARGBtoPNG(CameraImageData imageData) {
    // Create image from ARGB8888 data
    final img.Image image = img.Image.fromBytes(
      width: imageData.width,
      height: imageData.height,
      bytes: imageData.bytes.buffer,
      order: img.ChannelOrder.argb,
    );

    // Encode to PNG
    return Uint8List.fromList(img.encodePng(image));
  }

  @override
  void dispose() {
    _cameraController?.destroy();
    _poseDetector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Live Pose Detection'),
        actions: [
          if (_isInitialized && _cameraController != null)
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Center(
                child: Text(
                  '${_currentPoses.length} pose(s)',
                  style: const TextStyle(fontSize: 16),
                ),
              ),
            ),
        ],
      ),
      body: _buildBody(),
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
        // Camera preview
        CameraMacOSView(
          onCameraInizialized: _onCameraInitialized,
          cameraMode: CameraMacOSMode.photo,
          enableAudio: false,
        ),

        // Pose overlay
        if (_currentPoses.isNotEmpty && _cameraSize != null)
          CustomPaint(
            painter: CameraPoseOverlayPainter(
              poses: _currentPoses,
              cameraSize: _cameraSize!,
            ),
          ),

        // Status indicator
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
                _isProcessing ? 'Processing...' : 'Ready',
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

  CameraPoseOverlayPainter({
    required this.poses,
    required this.cameraSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (poses.isEmpty) return;

    // Get image dimensions from first pose
    final int imageWidth = poses.first.imageWidth;
    final int imageHeight = poses.first.imageHeight;

    // Calculate scaling to map from image coordinates to preview coordinates
    final double imageAspect = imageWidth / imageHeight;
    final double canvasAspect = size.width / size.height;

    double scaleX, scaleY;
    double offsetX = 0, offsetY = 0;

    if (canvasAspect > imageAspect) {
      scaleY = size.height / imageHeight;
      scaleX = scaleY;
      offsetX = (size.width - imageWidth * scaleX) / 2;
    } else {
      scaleX = size.width / imageWidth;
      scaleY = scaleX;
      offsetY = (size.height - imageHeight * scaleY) / 2;
    }

    for (final pose in poses) {
      _drawBbox(canvas, pose, scaleX, scaleY, offsetX, offsetY);
      if (pose.hasLandmarks) {
        _drawConnections(canvas, pose, scaleX, scaleY, offsetX, offsetY);
        _drawLandmarks(canvas, pose, scaleX, scaleY, offsetX, offsetY);
      }
    }
  }

  void _drawConnections(Canvas canvas, Pose pose, double scaleX, double scaleY,
      double offsetX, double offsetY) {
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
          Offset(start.x * scaleX + offsetX, start.y * scaleY + offsetY),
          Offset(end.x * scaleX + offsetX, end.y * scaleY + offsetY),
          paint,
        );
      }
    }
  }

  void _drawLandmarks(Canvas canvas, Pose pose, double scaleX, double scaleY,
      double offsetX, double offsetY) {
    for (final PoseLandmark l in pose.landmarks) {
      if (l.visibility > 0.5) {
        final Offset center =
            Offset(l.x * scaleX + offsetX, l.y * scaleY + offsetY);
        final Paint glow = Paint()..color = Colors.blue.withValues(alpha: 0.3);
        final Paint point = Paint()..color = Colors.red;
        final Paint centerDot = Paint()..color = Colors.white;
        canvas.drawCircle(center, 8, glow);
        canvas.drawCircle(center, 5, point);
        canvas.drawCircle(center, 2, centerDot);
      }
    }
  }

  void _drawBbox(Canvas canvas, Pose pose, double scaleX, double scaleY,
      double offsetX, double offsetY) {
    final Paint boxPaint = Paint()
      ..color = Colors.orangeAccent.withValues(alpha: 0.9)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final Paint fillPaint = Paint()
      ..color = Colors.orangeAccent.withValues(alpha: 0.08)
      ..style = PaintingStyle.fill;

    final double x1 = pose.boundingBox.left * scaleX + offsetX;
    final double y1 = pose.boundingBox.top * scaleY + offsetY;
    final double x2 = pose.boundingBox.right * scaleX + offsetX;
    final double y2 = pose.boundingBox.bottom * scaleY + offsetY;
    final Rect rect = Rect.fromLTRB(x1, y1, x2, y2);
    canvas.drawRect(rect, fillPaint);
    canvas.drawRect(rect, boxPaint);
  }

  @override
  bool shouldRepaint(CameraPoseOverlayPainter oldDelegate) => true;
}
