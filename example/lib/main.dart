import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:pose_detection/pose_detection.dart';
import 'package:camera/camera.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

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
  final PoseDetector _poseDetector = PoseDetector(
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

  // Performance: Use ValueNotifier for efficient pose overlay updates
  // This avoids full widget rebuilds - only the CustomPaint repaints
  final ValueNotifier<List<Pose>> _poseNotifier = ValueNotifier<List<Pose>>([]);

  // Performance: Dynamic frame throttling based on processing time
  int _lastProcessedTime = 0;
  static const int _minProcessingIntervalMs = 50; // Max ~20 detection FPS

  // Performance metrics (debug)
  int _detectionCount = 0;
  double _avgProcessingTimeMs = 0;

  @override
  void initState() {
    super.initState();
    _initializePoseDetector();
    _initCamera();
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

      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      _cameraController = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _cameraController!.initialize();
      if (!mounted) return;

      _cameraSize = Size(
        _cameraController!.value.previewSize!.height,
        _cameraController!.value.previewSize!.width,
      );

      await _cameraController!.startImageStream(_processCameraImage);
      _isImageStreamStarted = true;

      setState(() {});
    } catch (e) {
      if (mounted) {
        setState(() => _errorMessage = 'Camera init failed: $e');
      }
    }
  }

  /// Converts a CameraImage to BGR cv.Mat for OpenCV processing.
  ///
  /// Handles:
  /// - Desktop BGRA (macOS via camera_desktop): single plane, BGRA byte order
  /// - Desktop RGBA (Linux via camera_desktop): single plane, RGBA byte order
  /// - iOS NV12: 2 planes, YUV420
  /// - Android I420: 3 planes, YUV420
  cv.Mat? _convertCameraImageToMat(CameraImage image) {
    try {
      final int w = image.width;
      final int h = image.height;

      // Desktop: single-plane 4-channel packed format
      if (image.planes.length == 1 &&
          (image.planes[0].bytesPerPixel ?? 1) >= 4) {
        final bytes = image.planes[0].bytes;
        final stride = image.planes[0].bytesPerRow;
        final bgr = Uint8List(w * h * 3);

        int dstIdx = 0;
        for (int y = 0; y < h; y++) {
          final rowStart = y * stride;
          for (int x = 0; x < w; x++) {
            final srcIdx = rowStart + x * 4;
            if (Platform.isMacOS) {
              // BGRA: B=0, G=1, R=2, A=3
              bgr[dstIdx] = bytes[srcIdx];
              bgr[dstIdx + 1] = bytes[srcIdx + 1];
              bgr[dstIdx + 2] = bytes[srcIdx + 2];
            } else {
              // RGBA: R=0, G=1, B=2, A=3
              bgr[dstIdx] = bytes[srcIdx + 2];
              bgr[dstIdx + 1] = bytes[srcIdx + 1];
              bgr[dstIdx + 2] = bytes[srcIdx];
            }
            dstIdx += 3;
          }
        }
        return cv.Mat.fromList(h, w, cv.MatType.CV_8UC3, bgr);
      }

      // Mobile: YUV420 format
      final yRowStride = image.planes[0].bytesPerRow;
      final yPixelStride = image.planes[0].bytesPerPixel ?? 1;
      final bgr = Uint8List(w * h * 3);

      void writePixel(int x, int y, int yp, int up, int vp) {
        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
            .round()
            .clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
        final idx = (y * w + x) * 3;
        bgr[idx] = b;
        bgr[idx + 1] = g;
        bgr[idx + 2] = r;
      }

      if (image.planes.length == 2) {
        // iOS NV12
        final uvRowStride = image.planes[1].bytesPerRow;
        final uvPixelStride = image.planes[1].bytesPerPixel ?? 2;
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            final uvIdx = uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
            final yIdx = y * yRowStride + x * yPixelStride;
            writePixel(
              x,
              y,
              image.planes[0].bytes[yIdx],
              image.planes[1].bytes[uvIdx],
              image.planes[1].bytes[uvIdx + 1],
            );
          }
        }
      } else if (image.planes.length >= 3) {
        // Android I420
        final uvRowStride = image.planes[1].bytesPerRow;
        final uvPixelStride = image.planes[1].bytesPerPixel ?? 1;
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            final uvIdx = uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
            final yIdx = y * yRowStride + x * yPixelStride;
            writePixel(
              x,
              y,
              image.planes[0].bytes[yIdx],
              image.planes[1].bytes[uvIdx],
              image.planes[2].bytes[uvIdx],
            );
          }
        }
      } else {
        return null;
      }

      return cv.Mat.fromList(h, w, cv.MatType.CV_8UC3, bgr);
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
      // Set camera size once (for overlay coordinate mapping)
      if (_cameraSize == null && mounted && !_isDisposed) {
        setState(() {
          _cameraSize = Size(image.width.toDouble(), image.height.toDouble());
        });
      }

      cv.Mat? mat = _convertCameraImageToMat(image);
      if (mat == null) {
        _isProcessing = false;
        return;
      }

      if (_isDisposed) {
        mat.dispose();
        return;
      }

      // Downscale for performance — the detection model internally resizes
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
    } catch (e) {
      // Silently ignore errors to maintain camera feed
    } finally {
      _isProcessing = false;
    }
  }

  @override
  void dispose() {
    _isDisposed = true;
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
    return Scaffold(
      appBar: AppBar(
        title: const Text('Live Pose Detection'),
        actions: [
          if (_isInitialized && _cameraController != null)
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Center(
                // Use ValueListenableBuilder for pose count too
                child: ValueListenableBuilder<List<Pose>>(
                  valueListenable: _poseNotifier,
                  builder: (context, poses, _) => Text(
                    '${poses.length} pose(s)',
                    style: const TextStyle(fontSize: 16),
                  ),
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

  CameraPoseOverlayPainter({required this.poses, required this.cameraSize});

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
      _drawBbox(canvas, pose, scaleX, scaleY, 0, 0);
      if (pose.hasLandmarks) {
        _drawConnections(canvas, pose, scaleX, scaleY, 0, 0);
        _drawLandmarks(canvas, pose, scaleX, scaleY, 0, 0);
      }
    }
  }

  void _drawConnections(
    Canvas canvas,
    Pose pose,
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

  void _drawLandmarks(
    Canvas canvas,
    Pose pose,
    double scaleX,
    double scaleY,
    double offsetX,
    double offsetY,
  ) {
    for (final PoseLandmark l in pose.landmarks) {
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
    Pose pose,
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

    final double x1 = pose.boundingBox.left * scaleX + offsetX;
    final double y1 = pose.boundingBox.top * scaleY + offsetY;
    final double x2 = pose.boundingBox.right * scaleX + offsetX;
    final double y2 = pose.boundingBox.bottom * scaleY + offsetY;
    final Rect rect = Rect.fromLTRB(x1, y1, x2, y2);
    canvas.drawRect(rect, fillPaint);
    canvas.drawRect(rect, boxPaint);
  }

  @override
  bool shouldRepaint(CameraPoseOverlayPainter oldDelegate) {
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
