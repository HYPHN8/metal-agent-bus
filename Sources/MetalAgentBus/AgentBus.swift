import Foundation
import MetalAgentBusCore

/// Message emitted onto the semantic multi-agent bus.
public struct BusMessage: Sendable, Equatable {
    public let id: String
    public let senderID: String
    public let payload: String
    public let embedding: [Float]
    public let timestamp: Date

    public init(
        id: String = UUID().uuidString,
        senderID: String,
        payload: String,
        embedding: [Float],
        timestamp: Date = Date()
    ) {
        self.id = id
        self.senderID = senderID
        self.payload = payload
        self.embedding = embedding
        self.timestamp = timestamp
    }
}

/// Delivered message received by a subscribed agent along with its GPU-computed relevance score.
public struct RoutedMessage: Sendable, Equatable {
    public let message: BusMessage
    public let recipientID: String
    public let similarity: Float

    public init(message: BusMessage, recipientID: String, similarity: Float) {
        self.message = message
        self.recipientID = recipientID
        self.similarity = similarity
    }
}

/// Metal-accelerated semantic pub/sub bus for multi-agent fan-out routing in unified memory.
public actor AgentBus {
    public let dimension: Int
    public let engine: MetalComputeEngine

    private struct Subscription {
        let agentID: String
        let normalizedIntent: [Float]
        let threshold: Float
        let continuation: AsyncStream<RoutedMessage>.Continuation
    }

    private var subscriptions: [Subscription] = []
    private let subscriberSlab: SharedSlab<Float>
    private let thresholdSlab: SharedSlab<Float>
    private var slabsDirty = false

    public init(dimension: Int, engine: MetalComputeEngine, initialSubscribers: Int = 64) {
        precondition(dimension > 0, "dimension must be positive")
        self.dimension = dimension
        self.engine = engine
        self.subscriberSlab = SharedSlab<Float>(device: engine.device, initialCapacity: max(initialSubscribers, 16) * dimension)
        self.thresholdSlab = SharedSlab<Float>(device: engine.device, initialCapacity: max(initialSubscribers, 16))
    }

    public var subscriberCount: Int { subscriptions.count }

    /// Registers (or replaces) an agent's semantic intent subscription and returns its async event stream.
    public func subscribe(
        agentID: String,
        intentEmbedding: [Float],
        threshold: Float = 0.65
    ) throws -> AsyncStream<RoutedMessage> {
        guard intentEmbedding.count == dimension else {
            throw StoreError.dimensionMismatch(expected: dimension, got: intentEmbedding.count)
        }
        // Finish existing subscription if re-subscribing the same agent.
        if let idx = subscriptions.firstIndex(where: { $0.agentID == agentID }) {
            subscriptions[idx].continuation.finish()
            subscriptions.remove(at: idx)
        }

        let norm = VectorMath.normalize(intentEmbedding)
        var capturedContinuation: AsyncStream<RoutedMessage>.Continuation!
        let stream = AsyncStream<RoutedMessage> { continuation in
            capturedContinuation = continuation
        }
        subscriptions.append(
            Subscription(
                agentID: agentID,
                normalizedIntent: norm,
                threshold: threshold,
                continuation: capturedContinuation
            )
        )
        slabsDirty = true
        return stream
    }

    /// Unsubscribes an agent and finishes its `AsyncStream`.
    public func unsubscribe(agentID: String) {
        if let idx = subscriptions.firstIndex(where: { $0.agentID == agentID }) {
            subscriptions[idx].continuation.finish()
            subscriptions.remove(at: idx)
            slabsDirty = true
        }
    }

    /// Publishes a single message and returns the list of routed deliveries.
    @discardableResult
    public func publish(_ message: BusMessage, allowSelfDelivery: Bool = false) throws -> [RoutedMessage] {
        try publishBatch([message], allowSelfDelivery: allowSelfDelivery)
    }

    /// Publishes a batch of messages in a single Metal GPU dispatch across all subscribed agents.
    @discardableResult
    public func publishBatch(
        _ messages: [BusMessage],
        allowSelfDelivery: Bool = false
    ) throws -> [RoutedMessage] {
        guard !messages.isEmpty, !subscriptions.isEmpty else { return [] }
        for msg in messages {
            guard msg.embedding.count == dimension else {
                throw StoreError.dimensionMismatch(expected: dimension, got: msg.embedding.count)
            }
        }

        syncSlabsIfNeeded()

        var flatMessages: [Float] = []
        flatMessages.reserveCapacity(messages.count * dimension)
        for msg in messages {
            flatMessages.append(contentsOf: VectorMath.normalize(msg.embedding))
        }

        let numSubs = subscriptions.count
        let (scores, matches) = try engine.semanticRoute(
            messages: flatMessages,
            numMessages: messages.count,
            subscribers: subscriberSlab,
            thresholds: thresholdSlab,
            numSubscribers: numSubs,
            dim: dimension
        )

        var delivered: [RoutedMessage] = []
        for m in 0..<messages.count {
            let msg = messages[m]
            for s in 0..<numSubs {
                let idx = m * numSubs + s
                guard matches[idx] != 0 else { continue }
                let sub = subscriptions[s]
                if !allowSelfDelivery && sub.agentID == msg.senderID {
                    continue
                }
                let routed = RoutedMessage(message: msg, recipientID: sub.agentID, similarity: scores[idx])
                sub.continuation.yield(routed)
                delivered.append(routed)
            }
        }
        return delivered
    }

    private func syncSlabsIfNeeded() {
        guard slabsDirty else { return }
        subscriberSlab.removeAll(keepingCapacity: true)
        thresholdSlab.removeAll(keepingCapacity: true)
        for sub in subscriptions {
            subscriberSlab.append(contentsOf: sub.normalizedIntent)
            thresholdSlab.append(sub.threshold)
        }
        slabsDirty = false
    }
}
