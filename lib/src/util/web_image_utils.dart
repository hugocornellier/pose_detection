import 'dart:typed_data';

/// Converts RGBA pixel data to a normalized RGB Float32List tensor.
///
/// Normalizes each channel to [0.0, 1.0]. Alpha channel is discarded.
///
/// Parameters:
/// - [rgbaData]: RGBA pixel bytes (length must be a multiple of 4)
/// - [output]: Pre-allocated output buffer (length must be rgbaData.length * 3 / 4)
void rgbaToRgbFloat32(Uint8List rgbaData, Float32List output) {
  const double norm = 1.0 / 255.0;
  int dst = 0;
  for (int src = 0; src < rgbaData.length; src += 4) {
    output[dst++] = rgbaData[src] * norm; // R
    output[dst++] = rgbaData[src + 1] * norm; // G
    output[dst++] = rgbaData[src + 2] * norm; // B
  }
}
