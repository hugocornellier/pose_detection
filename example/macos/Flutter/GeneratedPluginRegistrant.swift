//
//  Generated file. Do not edit.
//

import FlutterMacOS
import Foundation

import camera_macos
import file_selector_macos
import path_provider_foundation
import pose_detection_tflite

func RegisterGeneratedPlugins(registry: FlutterPluginRegistry) {
  CameraMacosPlugin.register(with: registry.registrar(forPlugin: "CameraMacosPlugin"))
  FileSelectorPlugin.register(with: registry.registrar(forPlugin: "FileSelectorPlugin"))
  PathProviderPlugin.register(with: registry.registrar(forPlugin: "PathProviderPlugin"))
  PoseDetectionTflitePlugin.register(with: registry.registrar(forPlugin: "PoseDetectionTflitePlugin"))
}
