import SwiftUI
import Photos

@MainActor
final class EditorModel: ObservableObject {
    enum Phase: Equatable {
        case empty
        case picked
        case working(Double, String)
        case done
        case failed(String)
    }

    @Published var phase: Phase = .empty
    @Published private(set) var original: UIImage?
    @Published private(set) var enhanced: UIImage?
    @Published private(set) var facesFound = 0
    @Published var strength: Double = 1.0 {
        didSet { if oldValue != strength { rerender() } }
    }
    @Published var notice: String?

    private var result: EnhanceResult?
    private var renderTask: Task<Void, Never>?

    /// Set while a run is in progress; if the app dies mid-run we turn off acceleration next time.
    nonisolated private static let inFlightKey = "runInFlight"
    nonisolated static let neuralEngineKey = "useNeuralEngine"
    nonisolated static let qualityKey = "quality"

    init() {
        let d = UserDefaults.standard
        if d.bool(forKey: Self.inFlightKey) {
            d.set(false, forKey: Self.inFlightKey)
            d.set(false, forKey: Self.neuralEngineKey)
            notice = "La última mejora se cerró inesperadamente, así que desactivé la aceleración por Neural Engine. Puedes volver a activarla en Ajustes."
        }
    }

    func load(_ image: UIImage) {
        renderTask?.cancel()
        original = image
        enhanced = nil
        result = nil
        facesFound = 0
        phase = .picked
    }

    func reset() {
        renderTask?.cancel()
        original = nil
        enhanced = nil
        result = nil
        phase = .empty
    }

    func enhance(models: ModelStore.Paths) {
        guard let image = original else { return }
        if case .working = phase { return }
        let useNE = UserDefaults.standard.object(forKey: Self.neuralEngineKey) as? Bool ?? true
        let quality = EnhanceQuality(rawValue: UserDefaults.standard.string(forKey: Self.qualityKey) ?? "") ?? .fast
        let strength = Float(self.strength)
        phase = .working(0, "Preparando…")
        UserDefaults.standard.set(true, forKey: Self.inFlightKey)
        UIApplication.shared.isIdleTimerDisabled = true

        Task {
            let outcome: Result<(EnhanceResult, UIImage?), Error> = await Task.detached(priority: .userInitiated) {
                do {
                    let r = try Enhancer.run(image: image, models: models, quality: quality, useNeuralEngine: useNE) { p, label in
                        Task { @MainActor [weak self] in
                            guard let self, case .working = self.phase else { return }
                            self.phase = .working(p, label)
                        }
                    }
                    return .success((r, r.render(strength: strength)))
                } catch {
                    return .failure(error)
                }
            }.value

            UserDefaults.standard.set(false, forKey: Self.inFlightKey)
            UIApplication.shared.isIdleTimerDisabled = false
            switch outcome {
            case .success(let (r, img)):
                result = r
                facesFound = r.facesFound
                enhanced = img
                phase = .done
            case .failure(let error):
                phase = .failed(error.localizedDescription)
            }
        }
    }

    private func rerender() {
        guard let r = result else { return }
        let s = Float(strength)
        renderTask?.cancel()
        renderTask = Task {
            try? await Task.sleep(nanoseconds: 60_000_000)
            if Task.isCancelled { return }
            let img = await Task.detached(priority: .userInitiated) { r.render(strength: s) }.value
            if !Task.isCancelled, let img { enhanced = img }
        }
    }

    /// Saves the enhanced photo to the library. Returns an error message, or nil on success.
    func save() async -> String? {
        guard let img = enhanced, let data = img.jpegData(compressionQuality: 0.95) else {
            return "No hay foto para guardar."
        }
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else {
            return "Nítida no tiene permiso para guardar en Fotos. Actívalo en Ajustes › Nítida › Fotos."
        }
        do {
            try await PHPhotoLibrary.shared().performChanges {
                PHAssetCreationRequest.forAsset().addResource(with: .photo, data: data, options: nil)
            }
            return nil
        } catch {
            return "No se pudo guardar: \(error.localizedDescription)"
        }
    }
}
