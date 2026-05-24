Pod::Spec.new do |s|
  s.name                  = 'pose_detection'
  s.version               = '3.0.0'
  s.summary               = 'Pose detection via TensorFlow Lite (macOS)'
  s.description           = 'Flutter plugin for on-device pose detection using TensorFlow Lite.'
  s.homepage              = 'https://github.com/your/repo'
  s.license               = { :type => 'MIT' }
  s.authors               = { 'You' => 'you@example.com' }
  s.source                = { :path => '.' }

  s.platform              = :osx, '11.0'

  # TFLite libraries are provided by flutter_litert dependency
end