#include "include/pose_detection_tflite/pose_detection_tflite_plugin.h"
#include "pose_detection_tflite_plugin.h"
#include <flutter/plugin_registrar_windows.h>

void PoseDetectionTflitePluginRegisterWithRegistrar(FlutterDesktopPluginRegistrarRef registrar) {
  auto cpp_registrar =
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar);
  pose_detection_tflite::PoseDetectionTflitePlugin::RegisterWithRegistrar(cpp_registrar);
}
