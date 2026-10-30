import UIKit

/// Output of a run. Detail layers and faces are kept separately so the sliders re-blend instantly.
struct EnhanceResult {
    let original: UIImage
    let base: RGBAImage
    let detail: [Int8]
    let faces: [RestoredFace]
    let facesFound: Int

    func render(strength: Float, texture: Float) -> UIImage? {
        let textured = Upscaler.applyDetail(base: base, detail: detail, amount: texture)
        return FaceRestorer.composite(base: textured, faces: faces, strength: strength, texture: texture).uiImage()
    }
}

enum EnhanceQuality: String, CaseIterable {
    /// Same models, smaller working size: faces get the same full restoration.
    case fast
    /// Larger working size: more detail in the rest of the photo, can take minutes.
    case max

    var label: String { self == .fast ? "Rápida" : "Máxima calidad" }
    /// Long-side cap for the input; output is at most maxInputSide * 4.
    var maxInputSide: Int { self == .fast ? 384 : 1024 }
}

enum Enhancer {
    static func run(image: UIImage, models: ModelStore.Paths, quality: EnhanceQuality, useNeuralEngine: Bool,
                    progress: @escaping (Double, String) -> Void) throws -> EnhanceResult {
        progress(0, "Preparando la foto…")
        guard let input = RGBAImage.from(image, maxSide: quality.maxInputSide) else { throw EngineError.imageDecode }

        // 1. Find faces on the original.
        progress(0.02, "Buscando caras…")
        let faces: [DetectedFace]
        do {
            let detector = try FaceDetector(modelURL: models.detector, useNeuralEngine: useNeuralEngine)
            faces = try detector.detect(input)
        }

        // 2. Restore and enlarge the whole photo (clean tones + texture layer).
        let up: Upscaler.Output
        do {
            let upscaler = try Upscaler(cleanURL: models.upscaler, textureURL: models.textureUpscaler,
                                        useNeuralEngine: useNeuralEngine)
            up = try upscaler.upscale(input) { p in
                progress(0.04 + (faces.isEmpty ? 0.94 : 0.6) * p, "Mejorando toda la foto…")
            }
        }

        // 3. Rebuild facial detail, working from the textured photo.
        var restored: [RestoredFace] = []
        if !faces.isEmpty {
            let label = faces.count == 1 ? "Reconstruyendo la cara…" : "Reconstruyendo \(faces.count) caras…"
            progress(0.65, label)
            let canvas = Upscaler.applyDetail(base: up.base, detail: up.detail, amount: 1)
            let restorer = try FaceRestorer(cleanURL: models.faceRestorer, detailURL: models.faceTexture)
            restored = try restorer.restore(canvas, faces: faces, landmarkScale: Float(Upscaler.scale)) { p in
                progress(0.65 + 0.33 * p, label)
            }
        }
        progress(1, "Listo")
        return EnhanceResult(original: image, base: up.base, detail: up.detail, faces: restored, facesFound: faces.count)
    }
}
