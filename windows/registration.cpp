#include "include/pose_detection/pose_detection_plugin.h"
#include "pose_detection_plugin.h"
#include <flutter/plugin_registrar_windows.h>

void PoseDetectionPluginRegisterWithRegistrar(FlutterDesktopPluginRegistrarRef registrar) {
  auto cpp_registrar =
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar);
  pose_detection::PoseDetectionPlugin::RegisterWithRegistrar(cpp_registrar);
}
