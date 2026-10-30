import UIKit

/// Output of a run. Faces are kept separately so the face intensity slider re-blends instantly.
struct EnhanceResult {
    let original: UIImage
    let base: RGBAImage
    let faces: [RestoredFace]
    let facesFound: Int

    func render(strength: Float) -> UIImage? {
        FaceRestorer.composite(base: base, faces: faces, strength: strength).uiImage()
    }
}

enum Enhancer {
    /// Inputs are capped so the 4x output stays within ~4096 px on the long side.
    static let maxInputSide = 1024

    static func run(image: UIImage, models: ModelStore.Paths, useNeuralEngine: Bool,
                    progress: @escaping (Double, String) -> Void) throws -> EnhanceResult {
        progress(0, "Preparando la foto…")
        guard let input = RGBAImage.from(image, maxSide: maxInputSide) else { throw EngineError.imageDecode }

        // 1. Find faces on the original.
        progress(0.02, "Buscando caras…")
        let faces: [DetectedFace]
        do {
            let detector = try FaceDetector(modelURL: models.detector, useNeuralEngine: useNeuralEngine)
            faces = try detector.detect(input)
        }

        // 2. Restore and enlarge the whole photo.
        let base: RGBAImage
        do {
            let upscaler = try Upscaler(modelURL: models.upscaler, useNeuralEngine: useNeuralEngine)
            base = try upscaler.upscale(input) { p in
                progress(0.05 + 0.65 * p, "Mejorando toda la foto…")
            }
        }

        // 3. Rebuild facial detail.
        var restored: [RestoredFace] = []
        if !faces.isEmpty {
            let label = faces.count == 1 ? "Reconstruyendo la cara…" : "Reconstruyendo \(faces.count) caras…"
            progress(0.72, label)
            // GFPGAN stays on the CPU: its modulated convolutions can overflow the Neural Engine's
            // 16-bit math and produce blank faces. It's one 512x512 pass per face, so it's quick anyway.
            let restorer = try FaceRestorer(modelURL: models.faceRestorer, useNeuralEngine: false)
            restored = try restorer.restore(base, faces: faces, landmarkScale: Float(Upscaler.scale)) { p in
                progress(0.72 + 0.26 * p, label)
            }
        }
        progress(1, "Listo")
        return EnhanceResult(original: image, base: base, faces: restored, facesFound: faces.count)
    }
}
