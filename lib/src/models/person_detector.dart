export 'person_detector_unsupported.dart'
    if (dart.library.io) 'person_detector_native.dart'
    if (dart.library.js_interop) 'person_detector_web.dart';
