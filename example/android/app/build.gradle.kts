plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.hugocornellier.pose_detection_example"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    // Build the androidTest instrumentation APK against the profile variant:
    // it pairs with the profile app APK sent to Firebase Test Lab, and the
    // camerax plugin's debug javac currently fails on a missing
    // androidx.concurrent compile-classpath entry that profile doesn't hit.
    testBuildType = "profile"

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.hugocornellier.pose_detection_example"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
        // Runs the Dart integration_test targets as instrumentation tests
        // (locally and on Firebase Test Lab physical devices).
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

flutter {
    source = "../.."
}

dependencies {
    // Pinned to the versions integration_test's Android library already pulls
    // into the app runtime classpath; AGP consistent resolution rejects newer.
    androidTestImplementation("androidx.test:runner:1.3.0")
    androidTestImplementation("androidx.test:rules:1.2.0")
}
