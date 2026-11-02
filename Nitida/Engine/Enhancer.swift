import UIKit

/// Difference that the face restoration makes inside one face's rectangle (after clarity).
struct FaceDelta {
    let x0: Int, y0: Int, width: Int, height: Int
    let delta: [Int16]   // interleaved RGB
}

/// Output of a run: the finished photo plus two adjustable layers, so the sliders are instant.
struct EnhanceResult {
    let original: UIImage
    let base: RGBAImage        // cleaned/enlarged photo with faces at 100 %, before definition
    let definition: [Int8]     // luminance layer of the detail boost (at 100 %)
    let faces: [FaceDelta]
    let skin: [Int8]           // luminance offset of the synthesized skin texture (at 100 %)
    let facesFound: Int

    func render(strength: Float, texture: Float, definition amount: Float) -> UIImage? {
        var out = Clarity.add(definition, to: base, amount: amount)
        let W = out.width
        out.pixels.withUnsafeMutableBufferPointer { o in
            if strength != 1 {
                let k = strength - 1
                for f in faces {
                    for y in 0..<f.height {
                        for x in 0..<f.width {
                            let i = (y * f.width + x) * 3
                            let j = ((f.y0 + y) * W + f.x0 + x) * 4
                            o[j] = toByte(Float(o[j]) + k * Float(f.delta[i]))
                            o[j + 1] = toByte(Float(o[j + 1]) + k * Float(f.delta[i + 1]))
                            o[j + 2] = toByte(Float(o[j + 2]) + k * Float(f.delta[i + 2]))
                        }
                    }
                }
            }
            if texture != 0, skin.count == base.width * base.height {
                for i in 0..<skin.count where skin[i] != 0 {
                    let d = texture * Float(skin[i])
                    let j = i * 4
                    o[j] = toByte(Float(o[j]) + d)
                    o[j + 1] = toByte(Float(o[j + 1]) + d)
                    o[j + 2] = toByte(Float(o[j + 2]) + d)
                }
            }
        }
        return out.uiImage()
    }
}

enum EnhanceQuality: String, CaseIterable {
    case fast
    case max

    var label: String { self == .fast ? "Rápida" : "Máxima calidad" }
    /// Largest side the photo is processed at. The photo is never shrunk below this, so its real
    /// detail is kept; output is 4x (small photos) or 2x (larger ones), at most ~4096 px.
    var maxInputSide: Int { 2048 }
}

enum Enhancer {
    private static let cancelLock = NSLock()
    private static var cancelFlag = false

    /// Asks the running enhancement to stop at the next step (thread-safe).
    static func requestCancel() { cancelLock.lock(); cancelFlag = true; cancelLock.unlock() }
    static func resetCancel() { cancelLock.lock(); cancelFlag = false; cancelLock.unlock() }
    /// Throws if a cancel was requested; called between tiles, strips and faces.
    static func checkCancelled() throws {
        cancelLock.lock(); let c = cancelFlag; cancelLock.unlock()
        if c { throw EngineError.cancelled }
    }

    static func run(image: UIImage, models: ModelStore.Paths, quality: EnhanceQuality, useNeuralEngine: Bool,
                    progress: @escaping (Double, String) -> Void) throws -> EnhanceResult {
        try checkCancelled()
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

        // 2. Clean up and enlarge the whole photo.
        var up: RGBAImage
        do {
            let upscaler = Upscaler(cleanURL: models.fastUpscaler, textureURL: nil, useNeuralEngine: useNeuralEngine)
            up = try upscaler.upscale(input, outScale: outScale) { p in
                progress(0.04 + 0.5 * p, "Mejorando toda la foto…")
            }.base
        }

        // 3. Faces: restored from the original, keeping the person's identity and real skin.
        var restored: [RestoredFace] = []
        if !faces.isEmpty {
            let label = faces.count == 1 ? "Mejorando la cara…" : "Mejorando \(faces.count) caras…"
            progress(0.55, label)
            let restorer = FaceRestorer(modelURL: models.faceRestorer)
            restored = try restorer.restore(original: input, canvas: up, faces: faces, outScale: outScale) { p in
                progress(0.55 + 0.2 * p, label)
            }
        }
        let withFaces = FaceRestorer.composite(base: up, faces: restored, strength: 1, texture: 1)

        // 4. Detail and definition for the whole photo (hair, fabric, background, skin).
        progress(0.76, "Dando definición…")
        let definition = try Clarity.layer(withFaces, scale: outScale, cancelCheck: checkCancelled)
        let base = Clarity.add(definition, to: withFaces, amount: 1)

        // Face layers for the intensity slider: clarity of the photo without the face fix, per face.
        var deltas: [FaceDelta] = []
        let margin = 18 * outScale + 8
        for f in restored {
            try checkCancelled()
            let x0 = max(0, f.x0 - margin), y0 = max(0, f.y0 - margin)
            let x1 = min(up.width, f.x0 + f.width + margin), y1 = min(up.height, f.y0 + f.height + margin)
            let plain = try Clarity.apply(up.cropped(x: x0, y: y0, width: x1 - x0, height: y1 - y0), scale: outScale)
            var d = [Int16](repeating: 0, count: f.width * f.height * 3)
            base.pixels.withUnsafeBufferPointer { b in
                plain.pixels.withUnsafeBufferPointer { p in
                    for y in 0..<f.height {
                        for x in 0..<f.width {
                            let bi = ((f.y0 + y) * base.width + f.x0 + x) * 4
                            let pi = ((f.y0 - y0 + y) * plain.width + (f.x0 - x0 + x)) * 4
                            let di = (y * f.width + x) * 3
                            for c in 0..<3 { d[di + c] = Int16(b[bi + c]) - Int16(p[pi + c]) }
                        }
                    }
                }
            }
            deltas.append(FaceDelta(x0: f.x0, y0: f.y0, width: f.width, height: f.height, delta: d))
        }
        up = RGBAImage(width: 1, height: 1)   // free memory

        // Synthetic skin texture was removed: it produced a visible artificial pattern.
        let skin: [Int8] = []

        progress(1, "Listo")
        return EnhanceResult(original: image, base: withFaces, definition: definition, faces: deltas,
                             skin: skin, facesFound: faces.count)
    }
}
