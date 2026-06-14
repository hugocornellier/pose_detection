import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_litert/flutter_litert.dart' show drawLandmarkMarker;
import '../types.dart';

/// Paints pose detection results over a still image.
///
/// Draws bounding boxes, skeleton connections, and landmark markers for all
/// detected poses.
class MultiOverlayPainter extends CustomPainter {
  /// Poses to render.
  final List<Pose> results;

  late final Paint _glowPaint = Paint()
    ..color = Colors.blue.withValues(alpha: 0.3);
  late final Paint _pointPaint = Paint()..color = Colors.red;
  late final Paint _dotPaint = Paint()..color = Colors.white;

  /// Creates a painter for the given [results].
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
        final double cx = l.x * scaleX + offsetX;
        final double cy = l.y * scaleY + offsetY;
        drawLandmarkMarker(
          canvas,
          cx,
          cy,
          glowPaint: _glowPaint,
          pointPaint: _pointPaint,
          centerPaint: _dotPaint,
        );
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
  bool shouldRepaint(covariant MultiOverlayPainter old) {
    return old.results != results;
  }
}

/// Paints pose detection results over a live camera preview.
///
/// Handles optional horizontal mirroring for front cameras, draws bounding
/// boxes, skeleton connections, and landmark markers.
class CameraPoseOverlayPainter extends CustomPainter {
  /// Poses to render.
  final List<Pose> poses;

  /// Source camera preview size in pixels, retained for repaint comparisons
  /// and API compatibility. Coordinate mapping uses the image dimensions
  /// stored on the first [Pose].
  final Size cameraSize;

  /// When true, flips x-coordinates to match a mirrored front-camera preview.
  final bool mirrorHorizontally;

  late final Paint _glowPaint = Paint()
    ..color = Colors.blue.withValues(alpha: 0.3);
  late final Paint _pointPaint = Paint()..color = Colors.red;
  late final Paint _dotPaint = Paint()..color = Colors.white;

  /// Creates a painter for the given [poses], source [cameraSize], and mirror
  /// flag. The current coordinate mapping uses each pose's source image size.
  CameraPoseOverlayPainter({
    required this.poses,
    required this.cameraSize,
    required this.mirrorHorizontally,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (poses.isEmpty) return;

    final int imageWidth = poses.first.imageWidth;
    final int imageHeight = poses.first.imageHeight;

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
        final double cx = _mirrorX(l.x, scaleX, offsetX, size);
        final double cy = l.y * scaleY + offsetY;
        drawLandmarkMarker(
          canvas,
          cx,
          cy,
          glowPaint: _glowPaint,
          pointPaint: _pointPaint,
          centerPaint: _dotPaint,
        );
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
  bool shouldRepaint(covariant CameraPoseOverlayPainter old) {
    return old.poses != poses ||
        old.cameraSize != cameraSize ||
        old.mirrorHorizontally != mirrorHorizontally;
  }
}

/// Drop-in live-camera overlay: an [AspectRatio] box wrapping the camera
/// preview and a pose overlay painter, sized and aligned the same way across
/// devices. Mirrors the structure of face_detection_tflite's camera overlay so
/// the same device/orientation/mirroring handling carries over unchanged.
///
/// Detection coordinates are mapped from [imageSize] (the post-rotation image
/// the detector ran on) onto the display box with an aspect-fit transform plus
/// optional horizontal mirroring; rotation is handled upstream when the frame
/// is prepared, so this widget only fits and mirrors.
class CameraPoseOverlay extends StatelessWidget {
  /// The camera preview widget (typically a `CameraPreview`).
  final Widget cameraPreview;

  /// Aspect ratio of the raw camera frames (width / height). Retained for
  /// repaint comparisons / API parity; the coordinate mapping uses [imageSize].
  final double cameraAspectRatio;

  /// Aspect ratio used for the display [AspectRatio] box (often the inverse of
  /// [cameraAspectRatio] in portrait).
  final double displayAspectRatio;

  /// Whether the overlay should be mirrored horizontally (front cameras).
  final bool mirrorHorizontally;

  /// Sensor mount orientation in degrees (0/90/180/270). Retained for repaint
  /// comparisons / API parity.
  final int sensorOrientation;

  /// Current device orientation. Retained for repaint comparisons / API parity.
  final Orientation deviceOrientation;

  /// Whether the active camera is front-facing.
  final bool isFrontCamera;

  /// Poses to draw via the overlay painter.
  final List<Pose> poses;

  /// Size of the image used for detection (post-rotation). The overlay painter
  /// is skipped when null.
  final Size? imageSize;

  /// Creates a live-camera pose overlay.
  const CameraPoseOverlay({
    super.key,
    required this.cameraPreview,
    required this.cameraAspectRatio,
    required this.displayAspectRatio,
    required this.mirrorHorizontally,
    required this.sensorOrientation,
    required this.deviceOrientation,
    required this.isFrontCamera,
    required this.poses,
    this.imageSize,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: AspectRatio(
        aspectRatio: displayAspectRatio,
        child: Stack(
          fit: StackFit.expand,
          children: [
            cameraPreview,
            if (imageSize != null)
              CustomPaint(
                painter: _CameraPoseStreamPainter(
                  poses: poses,
                  imageSize: imageSize!,
                  mirrorHorizontally: mirrorHorizontally,
                ),
              ),
          ],
        ),
      ),
    );
  }
}

/// Painter used by [CameraPoseOverlay]. Maps detection coordinates from the
/// source [imageSize] onto the display canvas with an aspect-fit transform and
/// optional horizontal mirroring, then draws bounding boxes, skeleton
/// connections, and landmark markers.
class _CameraPoseStreamPainter extends CustomPainter {
  final List<Pose> poses;
  final Size imageSize;
  final bool mirrorHorizontally;

  late final Paint _glowPaint = Paint()
    ..color = Colors.blue.withValues(alpha: 0.3);
  late final Paint _pointPaint = Paint()..color = Colors.red;
  late final Paint _dotPaint = Paint()..color = Colors.white;

  _CameraPoseStreamPainter({
    required this.poses,
    required this.imageSize,
    required this.mirrorHorizontally,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (poses.isEmpty) return;

    final double sourceWidth = imageSize.width;
    final double sourceHeight = imageSize.height;
    if (sourceWidth <= 0 || sourceHeight <= 0) return;

    final double sourceAspect = sourceWidth / sourceHeight;
    final double viewportAspect = size.width / size.height;

    final double scale;
    double offsetX = 0;
    double offsetY = 0;
    if (sourceAspect > viewportAspect) {
      scale = size.height / sourceHeight;
      offsetX = (size.width - sourceWidth * scale) / 2;
    } else {
      scale = size.width / sourceWidth;
      offsetY = (size.height - sourceHeight * scale) / 2;
    }

    Offset transform(double x, double y) {
      final double mx = mirrorHorizontally ? sourceWidth - x : x;
      return Offset(mx * scale + offsetX, y * scale + offsetY);
    }

    final Paint boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = const Color(0xFF00FFCC);

    final Paint linePaint = Paint()
      ..color = Colors.green.withValues(alpha: 0.8)
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;

    for (final pose in poses) {
      final p1 = transform(pose.boundingBox.left, pose.boundingBox.top);
      final p2 = transform(pose.boundingBox.right, pose.boundingBox.bottom);
      canvas.drawRect(
        Rect.fromLTRB(
          math.min(p1.dx, p2.dx),
          math.min(p1.dy, p2.dy),
          math.max(p1.dx, p2.dx),
          math.max(p1.dy, p2.dy),
        ),
        boxPaint,
      );

      if (!pose.hasLandmarks) continue;

      for (final c in poseLandmarkConnections) {
        final PoseLandmark? a = pose.getLandmark(c[0]);
        final PoseLandmark? b = pose.getLandmark(c[1]);
        if (a != null &&
            b != null &&
            a.visibility > 0.5 &&
            b.visibility > 0.5) {
          canvas.drawLine(transform(a.x, a.y), transform(b.x, b.y), linePaint);
        }
      }

      for (final l in pose.landmarks) {
        if (l.visibility > 0.5) {
          final o = transform(l.x, l.y);
          drawLandmarkMarker(
            canvas,
            o.dx,
            o.dy,
            glowPaint: _glowPaint,
            pointPaint: _pointPaint,
            centerPaint: _dotPaint,
          );
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant _CameraPoseStreamPainter old) {
    return old.poses != poses ||
        old.imageSize != imageSize ||
        old.mirrorHorizontally != mirrorHorizontally;
  }
}
