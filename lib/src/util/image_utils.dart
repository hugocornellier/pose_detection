/// Utility functions for image preprocessing and transformations.
///
/// Provides coordinate transformations and tensor conversion utilities
/// used internally by the pose detection pipeline.
class ImageUtils {
  /// Transforms bounding box coordinates from letterbox space back to original image space.
  ///
  /// Reverses the letterbox transformation by removing padding and unscaling coordinates.
  ///
  /// Parameters:
  /// - [xyxy]: Bounding box in letterbox space as [x1, y1, x2, y2]
  /// - [ratio]: Scale ratio from letterbox preprocessing
  /// - [dw]: Horizontal padding from letterbox preprocessing
  /// - [dh]: Vertical padding from letterbox preprocessing
  ///
  /// Returns the bounding box in original image space as [x1, y1, x2, y2].
  static List<double> scaleFromLetterbox(
    List<double> xyxy,
    double ratio,
    int dw,
    int dh,
  ) {
    final double x1 = (xyxy[0] - dw) / ratio;
    final double y1 = (xyxy[1] - dh) / ratio;
    final double x2 = (xyxy[2] - dw) / ratio;
    final double y2 = (xyxy[3] - dh) / ratio;
    return [x1, y1, x2, y2];
  }

  /// Reshapes a flat array into a 4D tensor.
  ///
  /// Converts a 1D flattened array into a 4D nested list structure with the
  /// specified dimensions. Used for converting TensorFlow Lite output buffers
  /// into the expected tensor shape.
  ///
  /// Parameters:
  /// - [flat]: Flat array of values (length must equal dim1 * dim2 * dim3 * dim4)
  /// - [dim1]: First dimension size (batch)
  /// - [dim2]: Second dimension size (height/rows)
  /// - [dim3]: Third dimension size (width/columns)
  /// - [dim4]: Fourth dimension size (channels/features)
  ///
  /// Returns a 4D tensor [dim1, dim2, dim3, dim4] populated from [flat] in row-major order.
  static List<List<List<List<double>>>> reshapeToTensor4D(
    List<double> flat,
    int dim1,
    int dim2,
    int dim3,
    int dim4,
  ) {
    final List<List<List<List<double>>>> result = List.generate(
      dim1,
      (_) => List.generate(
        dim2,
        (_) => List.generate(dim3, (_) => List<double>.filled(dim4, 0.0)),
      ),
    );

    int index = 0;
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
        for (int k = 0; k < dim3; k++) {
          for (int l = 0; l < dim4; l++) {
            result[i][j][k][l] = flat[index++];
          }
        }
      }
    }

    return result;
  }
}
