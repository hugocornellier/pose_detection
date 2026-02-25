#include "include/pose_detection/pose_detection_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "pose_detection_plugin.h"

void PoseDetectionPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  pose_detection::PoseDetectionPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
