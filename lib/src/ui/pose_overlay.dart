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

  /// Source camera preview size in pixels (for coord-space mapping).
  final Size cameraSize;

  /// When true, flips x-coordinates to match a mirrored front-camera preview.
  final bool mirrorHorizontally;

  late final Paint _glowPaint = Paint()
    ..color = Colors.blue.withValues(alpha: 0.3);
  late final Paint _pointPaint = Paint()..color = Colors.red;
  late final Paint _dotPaint = Paint()..color = Colors.white;

  /// Creates a painter for the given [poses], source [cameraSize], and mirror
  /// flag.
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
