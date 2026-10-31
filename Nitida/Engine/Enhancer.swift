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
    /// Light, very fast model for the whole photo; faces get the same full restoration.
    case fast
    /// Heavier models with a texture layer on the whole photo; takes longer.
    case max

    var label: String { self == .fast ? "Rápida" : "Máxima calidad" }
    /// Largest side the photo is processed at. The photo is never shrunk below this, so its real
    /// detail is kept; output is 4x (small photos) or 2x (larger ones), at most ~4096 px.
    var maxInputSide: Int { self == .fast ? 2048 : 1280 }
}

enum Enhancer {
    static func run(image: UIImage, models: ModelStore.Paths, quality: EnhanceQuality, useNeuralEngine: Bool,
                    progress: @escaping (Double, String) -> Void) throws -> EnhanceResult {
        progress(0, "Preparando la foto…")
        guard let input = RGBAImage.from(image, maxSide: quality.maxInputSide) else { throw EngineError.imageDecode }
        let longSide = max(input.width, input.height)
        let outScale = longSide <= 1024 ? 4 : 2

        // 1. Find faces on the original.
        progress(0.02, "Buscando caras…")
        let faces: [DetectedFace]
        do {
            let detector = try FaceDetector(modelURL: models.detector, useNeuralEngine: useNeuralEngine)
            faces = try detector.detect(input)
        }

        // 2. Enhance and enlarge the whole photo.
        let upShare = faces.isEmpty ? 0.94 : (quality == .fast ? 0.45 : 0.75)
        let faceStart = 0.04 + upShare
        let up: Upscaler.Output
        do {
            let upscaler = quality == .fast
                ? Upscaler(cleanURL: models.fastUpscaler, textureURL: nil, useNeuralEngine: useNeuralEngine)
                : Upscaler(cleanURL: models.upscaler, textureURL: models.textureUpscaler,
                           useNeuralEngine: useNeuralEngine)
            up = try upscaler.upscale(input, outScale: outScale) { p in
                progress(0.04 + upShare * p, "Mejorando toda la foto…")
            }
        }

        // 3. Restore faces from the original and fuse them with the enhanced photo.
        var restored: [RestoredFace] = []
        if !faces.isEmpty {
            let label = faces.count == 1 ? "Mejorando la cara…" : "Mejorando \(faces.count) caras…"
            progress(faceStart, label)
            let canvas = Upscaler.applyDetail(base: up.base, detail: up.detail, amount: 1)
            let restorer = FaceRestorer(modelURL: models.faceRestorer)
            restored = try restorer.restore(original: input, canvas: canvas, faces: faces, outScale: outScale) { p in
                progress(faceStart + (0.98 - faceStart) * p, label)
            }
        }
        progress(1, "Listo")
        return EnhanceResult(original: image, base: up.base, detail: up.detail, faces: restored, facesFound: faces.count)
    }
}
