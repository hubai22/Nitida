import Foundation

/// A rebuilt face: clean tones and features from GFPGAN (512 px) plus fine texture (pores, lashes,
/// lip lines) taken from GPEN-1024 as a high-pass layer. Both live in the same aligned crop.
struct RestoredFace {
    let imageToCrop: Affine   // image -> 1024 crop space
    let clean: [UInt8]        // 512*512 interleaved RGB (GFPGAN)
    let detail: [Int8]        // 1024*1024 planar RGB high-pass (GPEN)
}

/// Face restoration in the style of Remini: GFPGAN gives natural, flattering eyes and skin tone;
/// GPEN-1024 adds real skin texture on top so the face doesn't look plastic.
final class FaceRestorer {
    static let cropSize = 1024
    static let cleanSize = 512
    private static let detailSigma: Float = 2.5

    // FFHQ five-point template (eyes, nose tip, mouth corners), normalized.
    private static let normalizedTemplate: [(Float, Float)] = [
        (0.37691676, 0.46864664), (0.62285697, 0.46912813), (0.50123859, 0.61331904),
        (0.39308822, 0.72541100), (0.61150205, 0.72490465)
    ]

    private let cleanRunner: OrtRunner
    private let detailRunner: OrtRunner

    init(cleanURL: URL, detailURL: URL) throws {
        // Both stay on the CPU: StyleGAN-style modulated convolutions can overflow the Neural
        // Engine's 16-bit math and produce blank faces.
        cleanRunner = try OrtRunner(modelURL: cleanURL, useNeuralEngine: false)
        detailRunner = try OrtRunner(modelURL: detailURL, useNeuralEngine: false)
    }

    /// `landmarkScale` converts detector coordinates into `canvas` coordinates.
    func restore(_ canvas: RGBAImage, faces: [DetectedFace], landmarkScale: Float,
                 progress: (Double) -> Void) throws -> [RestoredFace] {
        var results: [RestoredFace] = []
        for (index, face) in faces.enumerated() {
            try autoreleasepool {
                let pts = face.landmarks.map { ($0.0 * landmarkScale, $0.1 * landmarkScale) }
                let toCrop = Affine.similarity(from: pts, to: Self.template(Self.cropSize))
                let toClean = Affine.similarity(from: pts, to: Self.template(Self.cleanSize))

                // Clean face (GFPGAN, 512).
                let cOut = try run(cleanRunner, canvas: canvas, toCrop: toClean, size: Self.cleanSize)
                var clean = [UInt8](repeating: 0, count: Self.cleanSize * Self.cleanSize * 3)
                let cp = Self.cleanSize * Self.cleanSize
                for i in 0..<cp {
                    clean[i * 3] = toByte(cOut[i])
                    clean[i * 3 + 1] = toByte(cOut[cp + i])
                    clean[i * 3 + 2] = toByte(cOut[2 * cp + i])
                }
                progress((Double(index) + 0.35) / Double(faces.count))

                // Textured face (GPEN, 1024) -> keep only its fine detail.
                let dOut = try run(detailRunner, canvas: canvas, toCrop: toCrop, size: Self.cropSize)
                let detail = highPassToInt8(dOut, channels: 3, width: Self.cropSize, height: Self.cropSize,
                                            sigma: Self.detailSigma)

                results.append(RestoredFace(imageToCrop: toCrop, clean: clean, detail: detail))
            }
            progress(Double(index + 1) / Double(faces.count))
        }
        return results
    }

    private static func template(_ size: Int) -> [(Float, Float)] {
        normalizedTemplate.map { ($0.0 * Float(size), $0.1 * Float(size)) }
    }

    /// Warps the aligned crop out of `canvas`, runs a [-1, 1] face model, returns planar RGB 0...255.
    private func run(_ runner: OrtRunner, canvas: RGBAImage, toCrop: Affine, size C: Int) throws -> [Float] {
        let plane = C * C
        var input = [Float](repeating: 0, count: 3 * plane)
        let toImage = toCrop.inverted()
        canvas.pixels.withUnsafeBufferPointer { buf in
            let p = buf.baseAddress!
            var rgb: (Float, Float, Float) = (0, 0, 0)
            for v in 0..<C {
                for u in 0..<C {
                    let (sx, sy) = toImage.apply(Float(u), Float(v))
                    sampleBilinear(p, width: canvas.width, height: canvas.height, channels: 4,
                                   x: sx, y: sy, out: &rgb)
                    let di = v * C + u
                    input[di] = rgb.0 / 127.5 - 1
                    input[plane + di] = rgb.1 / 127.5 - 1
                    input[2 * plane + di] = rgb.2 / 127.5 - 1
                }
            }
        }
        let (data, shape) = try runner.runFirst(input, shape: [1, 3, C, C])
        guard shape.count == 4, shape[2] == C, shape[3] == C else {
            throw EngineError.badOutput("face \(shape)")
        }
        var out = [Float](repeating: 0, count: 3 * plane)
        data.withUnsafeBytes { raw in
            let f = raw.bindMemory(to: Float.self)
            for i in 0..<(3 * plane) { out[i] = (min(max(f[i], -1), 1) + 1) * 127.5 }
        }
        return out
    }

    /// Feathered square mask over the crop: 0 at the border, 1 inside.
    @inline(__always)
    private static func mask(_ u: Float, _ v: Float) -> Float {
        let edge = Float(cropSize - 1)
        let d = min(min(u, v), min(edge - u, edge - v))
        let t = min(max((d - 8) / 80, 0), 1)
        return t * t * (3 - 2 * t)
    }

    /// Blends restored faces onto `base`.
    /// `strength` 0...1: how much of the AI face is used. `texture` scales the fine-detail layer.
    static func composite(base: RGBAImage, faces: [RestoredFace], strength: Float, texture: Float) -> RGBAImage {
        var out = base
        guard strength > 0, !faces.isEmpty else { return out }
        let C = cropSize, CS = cleanSize
        let ratio = Float(CS) / Float(C)
        let W = out.width, H = out.height
        out.pixels.withUnsafeMutableBufferPointer { dstBuf in
            let dst = dstBuf.baseAddress!
            for face in faces {
                let toImage = face.imageToCrop.inverted()
                let corners = [toImage.apply(0, 0), toImage.apply(Float(C), 0),
                               toImage.apply(0, Float(C)), toImage.apply(Float(C), Float(C))]
                let minX = max(0, Int(corners.map(\.0).min()!.rounded(.down)))
                let maxX = min(W - 1, Int(corners.map(\.0).max()!.rounded(.up)))
                let minY = max(0, Int(corners.map(\.1).min()!.rounded(.down)))
                let maxY = min(H - 1, Int(corners.map(\.1).max()!.rounded(.up)))
                guard minX <= maxX, minY <= maxY else { continue }

                face.clean.withUnsafeBufferPointer { cBuf in
                    face.detail.withUnsafeBufferPointer { dBuf in
                        let cp = cBuf.baseAddress!, dp = dBuf.baseAddress!
                        var rgb: (Float, Float, Float) = (0, 0, 0)
                        var det: (Float, Float, Float) = (0, 0, 0)
                        for y in minY...maxY {
                            for x in minX...maxX {
                                let (u, v) = face.imageToCrop.apply(Float(x), Float(y))
                                if u < 0 || v < 0 || u > Float(C - 1) || v > Float(C - 1) { continue }
                                let m = mask(u, v) * strength
                                if m <= 0 { continue }
                                // Same point in the 512 clean crop (its transform is the 1024 one scaled by 0.5).
                                sampleBilinear(cp, width: CS, height: CS, channels: 3,
                                               x: u * ratio, y: v * ratio, out: &rgb)
                                if texture != 0 {
                                    sampleBilinearPlanarI8(dp, width: C, height: C, x: u, y: v, out: &det)
                                    rgb.0 += texture * det.0
                                    rgb.1 += texture * det.1
                                    rgb.2 += texture * det.2
                                }
                                let di = (y * W + x) * 4
                                dst[di] = toByte(Float(dst[di]) * (1 - m) + rgb.0 * m)
                                dst[di + 1] = toByte(Float(dst[di + 1]) * (1 - m) + rgb.1 * m)
                                dst[di + 2] = toByte(Float(dst[di + 2]) * (1 - m) + rgb.2 * m)
                            }
                        }
                    }
                }
            }
        }
        return out
    }
}
