import Foundation
import os

/// A rebuilt face: clean tones and features from GFPGAN plus fine texture (pores, lashes,
/// lip lines) taken from GPEN as a high-pass layer. Both live in the same 512 aligned crop.
struct RestoredFace {
    let imageToCrop: Affine   // image -> crop space
    let clean: [UInt8]        // 512*512 interleaved RGB (GFPGAN)
    let detail: [Int8]        // 512*512 planar RGB high-pass (GPEN)
}

/// Face restoration in the style of Remini: GFPGAN gives natural, flattering eyes and skin tone;
/// GPEN-1024 adds real skin texture on top so the face doesn't look plastic.
final class FaceRestorer {
    static let cropSize = 512
    static let cleanSize = 512
    private static let detailSigma: Float = 1.25

    // FFHQ five-point template (eyes, nose tip, mouth corners), normalized.
    private static let normalizedTemplate: [(Float, Float)] = [
        (0.37691676, 0.46864664), (0.62285697, 0.46912813), (0.50123859, 0.61331904),
        (0.39308822, 0.72541100), (0.61150205, 0.72490465)
    ]

    // Both run on the CPU (StyleGAN-style modulated convolutions can overflow the Neural Engine's
    // 16-bit math and produce blank faces), and never at the same time, to keep memory low.
    private let cleanURL: URL
    private let detailURL: URL

    init(cleanURL: URL, detailURL: URL) {
        self.cleanURL = cleanURL
        self.detailURL = detailURL
    }

    /// `landmarkScale` converts detector coordinates into `canvas` coordinates.
    func restore(_ canvas: RGBAImage, faces: [DetectedFace], landmarkScale: Float,
                 progress: (Double) -> Void) throws -> [RestoredFace] {
        let C = Self.cropSize, plane = C * C
        let transforms = faces.map { face -> Affine in
            let pts = face.landmarks.map { ($0.0 * landmarkScale, $0.1 * landmarkScale) }
            return Affine.similarity(from: pts, to: Self.template(C))
        }
        let n = Double(max(faces.count, 1))

        // Pass 1: clean faces (GFPGAN), then release the model.
        var cleans: [[UInt8]] = []
        try autoreleasepool {
            let runner = try OrtRunner(modelURL: cleanURL, useNeuralEngine: false)
            for (i, toCrop) in transforms.enumerated() {
                try autoreleasepool {
                    let out = try run(runner, canvas: canvas, toCrop: toCrop, size: C)
                    var clean = [UInt8](repeating: 0, count: plane * 3)
                    for k in 0..<plane {
                        clean[k * 3] = toByte(out[k])
                        clean[k * 3 + 1] = toByte(out[plane + k])
                        clean[k * 3 + 2] = toByte(out[2 * plane + k])
                    }
                    cleans.append(clean)
                }
                progress(0.5 * Double(i + 1) / n)
            }
        }

        // Pass 2: texture (GPEN high-pass), then release the model.
        // Safety: if iOS is low on memory for us, skip the texture layer instead of being killed.
        var results: [RestoredFace] = []
        if os_proc_available_memory() < 900 * 1024 * 1024 {
            return zip(transforms, cleans).map {
                RestoredFace(imageToCrop: $0.0, clean: $0.1, detail: [Int8](repeating: 0, count: plane * 3))
            }
        }
        try autoreleasepool {
            let runner = try OrtRunner(modelURL: detailURL, useNeuralEngine: false)
            for (i, toCrop) in transforms.enumerated() {
                try autoreleasepool {
                    let out = try run(runner, canvas: canvas, toCrop: toCrop, size: C)
                    let detail = highPassToInt8(out, channels: 3, width: C, height: C, sigma: Self.detailSigma)
                    results.append(RestoredFace(imageToCrop: toCrop, clean: cleans[i], detail: detail))
                }
                progress(0.5 + 0.5 * Double(i + 1) / n)
            }
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
