import Foundation

/// A restored face, already mapped into the output photo:
/// inside `rect`, final = low + texture * high, blended with `mask` (0...255) * strength.
struct RestoredFace {
    let x0: Int, y0: Int, width: Int, height: Int
    let low: [UInt8]    // interleaved RGB: restored face (its clean structure, tones, eyes)
    let high: [Int8]    // interleaved RGB: real fine detail of the enhanced photo at this spot
    let mask: [UInt8]   // face oval, feathered
}

/// Identity-preserving face restoration (CodeFormer, fidelity 0.9).
/// The face is taken from the ORIGINAL photo at full resolution, so the model never has to invent
/// a face from a tiny one. When the restored 512 px face is smaller than the face in the output,
/// it only contributes its broad structure, and the fine detail comes from the enhanced photo:
/// no blur, no "plastic" look, and the person stays the same.
final class FaceRestorer {
    static let cropSize = 512

    // FFHQ five-point template (eyes, nose tip, mouth corners), normalized.
    private static let normalizedTemplate: [(Float, Float)] = [
        (0.37691676, 0.46864664), (0.62285697, 0.46912813), (0.50123859, 0.61331904),
        (0.39308822, 0.72541100), (0.61150205, 0.72490465)
    ]

    private let modelURL: URL

    init(modelURL: URL) {
        self.modelURL = modelURL
    }

    /// - original: the input photo (detector coordinates).
    /// - canvas: the enhanced photo, `outScale` times larger.
    func restore(original: RGBAImage, canvas: RGBAImage, faces: [DetectedFace], outScale: Int,
                 progress: (Double) -> Void) throws -> [RestoredFace] {
        let C = Self.cropSize, plane = C * C
        let template = Self.normalizedTemplate.map { ($0.0 * Float(C), $0.1 * Float(C)) }
        var results: [RestoredFace] = []

        // Runs on the CPU: this kind of model can overflow the Neural Engine's 16-bit math.
        let runner = try OrtRunner(modelURL: modelURL, useNeuralEngine: false)
        for (index, face) in faces.enumerated() {
            try Enhancer.checkCancelled()
            try autoreleasepool {
                let toCrop = Affine.similarity(from: face.landmarks, to: template)   // original -> crop
                let input = Self.cropInput(original, toCrop: toCrop, size: C)
                let (data, shape) = try runner.runFirst(input, shape: [1, 3, C, C])
                guard shape.count == 4, shape[2] == C, shape[3] == C else {
                    throw EngineError.badOutput("face \(shape)")
                }
                var restored = [UInt8](repeating: 0, count: plane * 3)
                data.withUnsafeBytes { raw in
                    let f = raw.bindMemory(to: Float.self)
                    for i in 0..<plane {
                        for ch in 0..<3 {
                            let v = min(max(f[ch * plane + i], -1), 1)
                            restored[i * 3 + ch] = toByte((v + 1) * 127.5)
                        }
                    }
                }
                let s = Float(outScale)
                let outToCrop = Affine(a: toCrop.a / s, b: toCrop.b / s, c: toCrop.c / s, d: toCrop.d / s,
                                       tx: toCrop.tx, ty: toCrop.ty)
                if let r = Self.fuse(restored: restored, outToCrop: outToCrop, canvas: canvas) {
                    results.append(r)
                }
            }
            progress(Double(index + 1) / Double(max(faces.count, 1)))
        }
        return results
    }

    /// Samples the aligned crop from the original, normalized to [-1, 1], planar RGB.
    /// Supersamples when the face is larger than the crop to avoid aliasing.
    private static func cropInput(_ img: RGBAImage, toCrop: Affine, size C: Int) -> [Float] {
        let plane = C * C
        var input = [Float](repeating: 0, count: 3 * plane)
        let toImage = toCrop.inverted()
        let scale = sqrt(abs(toImage.a * toImage.d - toImage.b * toImage.c))   // image px per crop px
        let n = min(4, max(1, Int(scale.rounded(.up))))
        let inv = 1 / Float(n * n)
        img.pixels.withUnsafeBufferPointer { buf in
            let p = buf.baseAddress!
            var rgb: (Float, Float, Float) = (0, 0, 0)
            for v in 0..<C {
                for u in 0..<C {
                    var r: Float = 0, g: Float = 0, b: Float = 0
                    for sy in 0..<n {
                        for sx in 0..<n {
                            let fu = Float(u) + (Float(sx) + 0.5) / Float(n) - 0.5
                            let fv = Float(v) + (Float(sy) + 0.5) / Float(n) - 0.5
                            let (ix, iy) = toImage.apply(fu, fv)
                            sampleBilinear(p, width: img.width, height: img.height, channels: 4, x: ix, y: iy, out: &rgb)
                            r += rgb.0; g += rgb.1; b += rgb.2
                        }
                    }
                    let di = v * C + u
                    input[di] = r * inv / 127.5 - 1
                    input[plane + di] = g * inv / 127.5 - 1
                    input[2 * plane + di] = b * inv / 127.5 - 1
                }
            }
        }
        return input
    }

    /// Feathered face oval in crop coordinates (keeps ears, earrings and hair untouched).
    @inline(__always)
    private static func ovalMask(_ u: Float, _ v: Float) -> Float {
        let C = Float(cropSize)
        let du = (u - 0.5 * C) / (0.3516 * C), dv = (v - 0.5469 * C) / (0.4336 * C)
        let e = (du * du + dv * dv).squareRoot()
        let t = min(max((1.05 - e) / 0.25, 0), 1)
        return t * t * (3 - 2 * t)
    }

    /// Maps the restored crop into the output and splits it into low (restored structure) and
    /// high (photo detail) bands at the crop's own resolution limit.
    private static func fuse(restored: [UInt8], outToCrop: Affine, canvas: RGBAImage) -> RestoredFace? {
        let C = cropSize
        let W = canvas.width, H = canvas.height
        let toOut = outToCrop.inverted()
        let corners = [toOut.apply(0, 0), toOut.apply(Float(C), 0), toOut.apply(0, Float(C)), toOut.apply(Float(C), Float(C))]
        let x0 = max(0, Int(corners.map(\.0).min()!.rounded(.down)))
        let x1 = min(W - 1, Int(corners.map(\.0).max()!.rounded(.up)))
        let y0 = max(0, Int(corners.map(\.1).min()!.rounded(.down)))
        let y1 = min(H - 1, Int(corners.map(\.1).max()!.rounded(.up)))
        guard x0 < x1, y0 < y1 else { return nil }
        let bw = x1 - x0 + 1, bh = y1 - y0 + 1, n = bw * bh

        // Output pixels per crop pixel: how much the restored face gets stretched.
        let stretch = 1 / sqrt(abs(outToCrop.a * outToCrop.d - outToCrop.b * outToCrop.c))
        let sigma = 0.6 * stretch

        var rest = [Float](repeating: 0, count: 3 * n)   // planar
        var photo = [Float](repeating: 0, count: 3 * n)  // planar
        var mask = [UInt8](repeating: 0, count: n)
        restored.withUnsafeBufferPointer { rBuf in
            canvas.pixels.withUnsafeBufferPointer { cBuf in
                let rp = rBuf.baseAddress!, cp = cBuf.baseAddress!
                var rgb: (Float, Float, Float) = (0, 0, 0)
                for y in 0..<bh {
                    for x in 0..<bw {
                        let i = y * bw + x
                        let ci = ((y0 + y) * W + (x0 + x)) * 4
                        photo[i] = Float(cp[ci]); photo[n + i] = Float(cp[ci + 1]); photo[2 * n + i] = Float(cp[ci + 2])
                        let (u, v) = outToCrop.apply(Float(x0 + x), Float(y0 + y))
                        if u >= 0, v >= 0, u <= Float(C - 1), v <= Float(C - 1) {
                            mask[i] = toByte(ovalMask(u, v) * 255)
                            sampleBilinear(rp, width: C, height: C, channels: 3, x: u, y: v, out: &rgb)
                            rest[i] = rgb.0; rest[n + i] = rgb.1; rest[2 * n + i] = rgb.2
                        } else {
                            rest[i] = photo[i]; rest[n + i] = photo[n + i]; rest[2 * n + i] = photo[2 * n + i]
                        }
                    }
                }
            }
        }

        var low = [UInt8](repeating: 0, count: 3 * n)
        var high = [Int8](repeating: 0, count: 3 * n)
        if sigma > 0.5 {
            gaussianBlurPlanes(&rest, channels: 3, width: bw, height: bh, sigma: sigma)
            var photoBlur = photo
            gaussianBlurPlanes(&photoBlur, channels: 3, width: bw, height: bh, sigma: sigma)
            for i in 0..<n {
                for ch in 0..<3 {
                    low[i * 3 + ch] = toByte(rest[ch * n + i])
                    let h = (photo[ch * n + i] - photoBlur[ch * n + i]).rounded()
                    high[i * 3 + ch] = Int8(min(max(h, -127), 127))
                }
            }
        } else {
            for i in 0..<n {
                for ch in 0..<3 { low[i * 3 + ch] = toByte(rest[ch * n + i]) }
            }
        }
        return RestoredFace(x0: x0, y0: y0, width: bw, height: bh, low: low, high: high, mask: mask)
    }

    /// Blends restored faces onto `base`. `strength` 0...1; `texture` scales the fine-detail band.
    static func composite(base: RGBAImage, faces: [RestoredFace], strength: Float, texture: Float) -> RGBAImage {
        var out = base
        guard strength > 0, !faces.isEmpty else { return out }
        let W = out.width
        out.pixels.withUnsafeMutableBufferPointer { dstBuf in
            let dst = dstBuf.baseAddress!
            for f in faces {
                for y in 0..<f.height {
                    for x in 0..<f.width {
                        let i = y * f.width + x
                        let m = Float(f.mask[i]) / 255 * strength
                        if m <= 0 { continue }
                        let di = ((f.y0 + y) * W + (f.x0 + x)) * 4
                        for ch in 0..<3 {
                            var v = Float(f.low[i * 3 + ch])
                            if !f.high.isEmpty { v += texture * Float(f.high[i * 3 + ch]) }
                            dst[di + ch] = toByte(Float(dst[di + ch]) * (1 - m) + v * m)
                        }
                    }
                }
            }
        }
        return out
    }
}
