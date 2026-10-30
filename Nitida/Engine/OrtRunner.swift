import Foundation
import OnnxRuntimeBindings

enum EngineError: LocalizedError {
    case badOutput(String)
    case imageDecode

    var errorDescription: String? {
        switch self {
        case .badOutput(let m): return "El modelo devolvió un resultado inesperado (\(m))."
        case .imageDecode: return "No se pudo leer la foto."
        }
    }
}

/// Thin wrapper around an ONNX Runtime session with a single float input.
final class OrtRunner {
    private static var sharedEnv: ORTEnv?
    private static func env() throws -> ORTEnv {
        if let e = sharedEnv { return e }
        let e = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
        sharedEnv = e
        return e
    }

    let session: ORTSession
    let inputName: String
    let outputNames: [String]
    private(set) var usesNeuralEngine = false

    init(modelURL: URL, useNeuralEngine: Bool) throws {
        let env = try OrtRunner.env()

        func makeOptions(neural: Bool) throws -> ORTSessionOptions {
            let o = try ORTSessionOptions()
            try o.setGraphOptimizationLevel(ORTGraphOptimizationLevel.all)
            let threads = max(1, min(ProcessInfo.processInfo.activeProcessorCount, 6))
            try o.setIntraOpNumThreads(Int32(threads))
            if neural {
                let coreML = ORTCoreMLExecutionProviderOptions()
                coreML.createMLProgram = true
                coreML.enableOnSubgraphs = true
                try o.appendCoreMLExecutionProvider(with: coreML)
            }
            return o
        }

        var created: ORTSession?
        var neural = false
        if useNeuralEngine {
            // If Core ML can't take this model, fall back to the CPU silently.
            if let o = try? makeOptions(neural: true),
               let s = try? ORTSession(env: env, modelPath: modelURL.path, sessionOptions: o) {
                created = s
                neural = true
            }
        }
        let session: ORTSession
        if let created {
            session = created
        } else {
            session = try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: try makeOptions(neural: false))
        }
        let inputs = try session.inputNames()
        let outputs = try session.outputNames()
        self.session = session
        self.inputName = inputs.first ?? "input"
        self.outputNames = outputs
        self.usesNeuralEngine = neural
    }

    /// Runs the model on a float tensor and returns every output keyed by name.
    func run(_ input: [Float], shape: [Int]) throws -> [String: ORTValue] {
        let data = input.withUnsafeBufferPointer { buf in
            NSMutableData(bytes: buf.baseAddress, length: buf.count * MemoryLayout<Float>.stride)
        }
        let tensor = try ORTValue(tensorData: data,
                                  elementType: ORTTensorElementDataType.float,
                                  shape: shape.map { NSNumber(value: $0) })
        return try session.run(withInputs: [inputName: tensor],
                               outputNames: Set(outputNames),
                               runOptions: nil)
    }

    /// Convenience: runs the model and returns the first output's raw floats and shape.
    func runFirst(_ input: [Float], shape: [Int]) throws -> (Data, [Int]) {
        let outputs = try run(input, shape: shape)
        guard let name = outputNames.first, let value = outputs[name] else {
            throw EngineError.badOutput("sin salida")
        }
        return try OrtRunner.floats(value)
    }

    static func floats(_ value: ORTValue) throws -> (Data, [Int]) {
        let data = try value.tensorData() as Data
        let shape = try value.tensorTypeAndShapeInfo().shape.map { $0.intValue }
        return (data, shape)
    }
}
