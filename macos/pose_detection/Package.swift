// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "pose_detection",
    platforms: [
        .macOS("11.0")
    ],
    products: [
        .library(name: "pose-detection", targets: ["pose_detection"])
    ],
    dependencies: [],
    targets: [
        .target(
            name: "pose_detection",
            dependencies: [],
            resources: [
                .process("PrivacyInfo.xcprivacy"),
            ]
        )
    ]
)
