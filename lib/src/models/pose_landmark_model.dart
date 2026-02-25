export 'pose_landmark_model_unsupported.dart'
    if (dart.library.io) 'pose_landmark_model_native.dart'
    if (dart.library.js_interop) 'pose_landmark_model_web.dart';
