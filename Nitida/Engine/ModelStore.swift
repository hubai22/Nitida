import Foundation

/// Downloads the AI models once (≈355 MB) and keeps them in Application Support.
@MainActor
final class ModelStore: ObservableObject {
    struct Model {
        let file: String
        let size: Int64
        var url: URL {
            URL(string: "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/\(file)")!
        }
    }

    struct Paths {
        let detector: URL
        let upscaler: URL
        let faceRestorer: URL
    }

    enum State: Equatable {
        case missing
        case downloading(Double)
        case ready
        case failed(String)
    }

    nonisolated static let detector = Model(file: "scrfd_2.5g.onnx", size: 3_295_067)
    // Nomos8k SC: x4 super-resolution trained to synthesize real-world texture (skin, hair, fabric).
    nonisolated static let upscaler = Model(file: "nomos8k_sc_x4.onnx", size: 67_002_465)
    // GPEN-BFR-1024: face restoration that keeps pores and fine detail instead of smoothing.
    nonisolated static let faceRestorer = Model(file: "gpen_bfr_1024.onnx", size: 285_203_703)
    nonisolated static let all = [detector, upscaler, faceRestorer]
    nonisolated static var totalBytes: Int64 { all.reduce(0) { $0 + $1.size } }

    @Published private(set) var state: State = .missing

    private let folder: URL
    private var observation: NSKeyValueObservation?
    private lazy var session: URLSession = {
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 120
        cfg.timeoutIntervalForResource = 3 * 60 * 60
        cfg.waitsForConnectivity = true
        return URLSession(configuration: cfg)
    }()

    init() {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        folder = base.appendingPathComponent("Models", isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var f = folder
        try? f.setResourceValues(values)
        // Remove models from earlier versions that are no longer used, to free space.
        let keep = Set(Self.all.map(\.file))
        if let files = try? FileManager.default.contentsOfDirectory(atPath: folder.path) {
            for file in files where !keep.contains(file) {
                try? FileManager.default.removeItem(at: folder.appendingPathComponent(file))
            }
        }
        state = Self.all.allSatisfy(isInstalled) ? .ready : .missing
    }

    var paths: Paths {
        Paths(detector: path(Self.detector), upscaler: path(Self.upscaler), faceRestorer: path(Self.faceRestorer))
    }

    /// Bytes still to download (only missing models are fetched).
    var remainingBytes: Int64 {
        Self.all.filter { !isInstalled($0) }.reduce(0) { $0 + $1.size }
    }

    private func path(_ m: Model) -> URL { folder.appendingPathComponent(m.file) }

    private func isInstalled(_ m: Model) -> Bool {
        let attrs = try? FileManager.default.attributesOfItem(atPath: path(m).path)
        return (attrs?[.size] as? NSNumber)?.int64Value == m.size
    }

    func download() {
        if case .downloading = state { return }
        Task {
            do {
                var finished: Int64 = Self.all.filter(isInstalled).reduce(0) { $0 + $1.size }
                state = .downloading(Double(finished) / Double(Self.totalBytes))
                for m in Self.all where !isInstalled(m) {
                    try await fetch(m, alreadyDone: finished)
                    finished += m.size
                }
                state = Self.all.allSatisfy(isInstalled) ? .ready : .failed("La descarga quedó incompleta. Inténtalo de nuevo.")
            } catch {
                state = .failed("No se pudo descargar: \(error.localizedDescription)")
            }
        }
    }

    private func fetch(_ m: Model, alreadyDone: Int64) async throws {
        let destination = path(m)
        let total = Double(Self.totalBytes)
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            let task = session.downloadTask(with: m.url) { tmp, response, error in
                if let error { cont.resume(throwing: error); return }
                guard let tmp, let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    cont.resume(throwing: URLError(.badServerResponse)); return
                }
                do {
                    try? FileManager.default.removeItem(at: destination)
                    try FileManager.default.moveItem(at: tmp, to: destination)
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
            let throttle = Throttle()
            observation = task.progress.observe(\.fractionCompleted) { [weak self] p, _ in
                let fraction = (Double(alreadyDone) + p.fractionCompleted * Double(m.size)) / total
                guard throttle.shouldReport(fraction) else { return }
                Task { @MainActor in self?.state = .downloading(fraction) }
            }
            task.resume()
        }
        observation = nil
        guard isInstalled(m) else { throw URLError(.cannotDecodeContentData) }
    }
}

/// Limits progress updates to roughly every 0.2 %.
private final class Throttle: @unchecked Sendable {
    private var last = -1.0
    private let lock = NSLock()

    func shouldReport(_ value: Double) -> Bool {
        lock.lock(); defer { lock.unlock() }
        if value - last < 0.002 && value < 1 { return false }
        last = value
        return true
    }
}
