package com.hugocornellier.pose_detection_example;

import androidx.test.rule.ActivityTestRule;
import dev.flutter.plugins.integration_test.FlutterTestRunner;
import org.junit.Rule;
import org.junit.runner.RunWith;

/**
 * Runs the Dart integration-test target baked into the app APK (selected at
 * build time via {@code flutter build apk --target integration_test/...}) as
 * an Android instrumentation test, locally or on Firebase Test Lab.
 */
@RunWith(FlutterTestRunner.class)
public final class MainActivityTest {
  @Rule
  public ActivityTestRule<MainActivity> rule =
      new ActivityTestRule<>(MainActivity.class, true, false);
}
