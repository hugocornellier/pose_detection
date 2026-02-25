export 'pose_detector_unsupported.dart'
    if (dart.library.io) 'pose_detector_native.dart'
    if (dart.library.js_interop) 'pose_detector_web.dart';
