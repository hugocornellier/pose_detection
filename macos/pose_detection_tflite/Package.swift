// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "pose_detection_tflite",
    platforms: [
        .macOS("11.0")
    ],
    products: [
        .library(name: "pose-detection-tflite", targets: ["pose_detection_tflite"])
    ],
    dependencies: [],
    targets: [
        .target(
            name: "pose_detection_tflite",
            dependencies: [],
            resources: [
                .process("PrivacyInfo.xcprivacy"),
            ]
        )
    ]
)
