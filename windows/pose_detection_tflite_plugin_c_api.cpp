#include "include/pose_detection_tflite/pose_detection_tflite_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "pose_detection_tflite_plugin.h"

void PoseDetectionTflitePluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  pose_detection_tflite::PoseDetectionTflitePlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
