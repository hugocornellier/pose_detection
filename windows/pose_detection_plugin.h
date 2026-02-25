#ifndef FLUTTER_PLUGIN_POSE_DETECTION_PLUGIN_H_
#define FLUTTER_PLUGIN_POSE_DETECTION_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace pose_detection {

class PoseDetectionPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows *registrar);

  PoseDetectionPlugin();

  virtual ~PoseDetectionPlugin();

  PoseDetectionPlugin(const PoseDetectionPlugin&) = delete;
  PoseDetectionPlugin& operator=(const PoseDetectionPlugin&) = delete;

  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue> &method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
};

}  // namespace pose_detection

#endif  // FLUTTER_PLUGIN_POSE_DETECTION_PLUGIN_H_
