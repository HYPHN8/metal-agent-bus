// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "metal-agent-bus",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "MetalAgentBus",
            targets: ["MetalAgentBus"]
        ),
        .library(
            name: "MetalAgentBusCore",
            targets: ["MetalAgentBusCore"]
        ),
        .library(
            name: "MetalAgentBusC",
            targets: ["MetalAgentBusC"]
        ),
        .executable(
            name: "metal-agent-bus-bench",
            targets: ["MetalAgentBusBench"]
        )
    ],
    targets: [
        .target(
            name: "MetalAgentBusCore",
            linkerSettings: [
                .linkedFramework("Metal", .when(platforms: [.macOS])),
                .linkedFramework("Accelerate", .when(platforms: [.macOS]))
            ]
        ),
        .target(
            name: "MetalAgentBus",
            dependencies: ["MetalAgentBusCore"]
        ),
        .target(
            name: "MetalAgentBusC",
            dependencies: ["MetalAgentBus", "MetalAgentBusCore"]
        ),
        .executableTarget(
            name: "MetalAgentBusBench",
            dependencies: ["MetalAgentBus", "MetalAgentBusCore"]
        ),
        .testTarget(
            name: "MetalAgentBusTests",
            dependencies: ["MetalAgentBus", "MetalAgentBusCore", "MetalAgentBusC"]
        )
    ]
)
