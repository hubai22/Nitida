import Foundation

/// Whole-photo x4 super-resolution with two models, processed in fixed-size tiles so memory stays
/// bounded and the Neural Engine sees one static input shape:
/// - a clean model (Real-ESRGAN) gives natural tones and edges,
/// - a texture model (Nomos8k SC) contributes only its fine detail (high-pass), stored separately
///   so the "Textura" slider can dial it in later without re-running anything.
final class Upscaler {
    static let scale = 4
    private let tile = 192
    private let pad = 32
    private let detailSigma: Float = 2.0
    private let clean: OrtRunner
    private let texture: OrtRunner

    struct Output {
        let base: RGBAImage
        let detail: [Int8]   // interleaved RGB high-pass, base.width * base.height * 3
    }

    init(cleanURL: URL, textureURL: URL, useNeuralEngine: Bool) throws {
        clean = try OrtRunner(modelURL: cleanURL, useNeuralEngine: useNeuralEngine)
        texture = try OrtRunner(modelURL: textureURL, useNeuralEngine: useNeuralEngine)
    }

    func upscale(_ img: RGBAImage, progress: (Double) -> Void) throws -> Output {
        let sc = Upscaler.scale
        let w = img.width, h = img.height
        let S = tile + 2 * pad
        let plane = S * S
        let W = w * sc
        var out = RGBAImage(width: W, height: h * sc)
        var detail = [Int8](repeating: 0, count: W * h * sc * 3)
        var input = [Float](repeating: 0, count: 3 * plane)

        let tilesX = (w + tile - 1) / tile
        let tilesY = (h + tile - 1) / tile
        let total = tilesX * tilesY
        var done = 0
        progress(0)

        try img.pixels.withUnsafeBufferPointer { srcBuf in
            let src = srcBuf.baseAddress!
            for ty in 0..<tilesY {
                for tx in 0..<tilesX {
                    try autoreleasepool {
                        let x0 = tx * tile, y0 = ty * tile

                        // Gather the padded tile (edges replicated), RGB planar, 0...1.
                        for yy in 0..<S {
                            let sy = min(max(y0 - pad + yy, 0), h - 1)
                            for xx in 0..<S {
                                let sx = min(max(x0 - pad + xx, 0), w - 1)
                                let si = (sy * w + sx) * 4
                                let di = yy * S + xx
                                input[di] = Float(src[si]) / 255
                                input[plane + di] = Float(src[si + 1]) / 255
                                input[2 * plane + di] = Float(src[si + 2]) / 255
                            }
                        }

                        let OS = S * sc
                        let oplane = OS * OS
                        let vw = min(tile, w - x0), vh = min(tile, h - y0)

                        // Clean pass -> base pixels.
                        let (cData, cShape) = try clean.runFirst(input, shape: [1, 3, S, S])
                        guard cShape.count == 4, cShape[2] == OS, cShape[3] == OS else {
                            throw EngineError.badOutput("upscaler \(cShape)")
                        }
                        out.pixels.withUnsafeMutableBufferPointer { dstBuf in
                            let dst = dstBuf.baseAddress!
                            cData.withUnsafeBytes { raw in
                                let f = raw.bindMemory(to: Float.self).baseAddress!
                                for yy in 0..<(vh * sc) {
                                    let oy = pad * sc + yy
                                    let dy = y0 * sc + yy
                                    for xx in 0..<(vw * sc) {
                                        let si = oy * OS + pad * sc + xx
                                        let di = (dy * W + x0 * sc + xx) * 4
                                        dst[di] = toByte(f[si] * 255)
                                        dst[di + 1] = toByte(f[oplane + si] * 255)
                                        dst[di + 2] = toByte(f[2 * oplane + si] * 255)
                                    }
                                }
                            }
                        }

                        // Texture pass -> high-pass detail only.
                        let (tData, tShape) = try texture.runFirst(input, shape: [1, 3, S, S])
                        guard tShape.count == 4, tShape[2] == OS, tShape[3] == OS else {
                            throw EngineError.badOutput("texture \(tShape)")
                        }
                        var planes = [Float](repeating: 0, count: 3 * oplane)
                        tData.withUnsafeBytes { raw in
                            let f = raw.bindMemory(to: Float.self)
                            for i in 0..<(3 * oplane) { planes[i] = min(max(f[i], 0), 1) * 255 }
                        }
                        let hp = highPassToInt8(planes, channels: 3, width: OS, height: OS, sigma: detailSigma)
                        detail.withUnsafeMutableBufferPointer { dBuf in
                            let d = dBuf.baseAddress!
                            for yy in 0..<(vh * sc) {
                                let oy = pad * sc + yy
                                let dy = y0 * sc + yy
                                for xx in 0..<(vw * sc) {
                                    let si = oy * OS + pad * sc + xx
                                    let di = (dy * W + x0 * sc + xx) * 3
                                    d[di] = hp[si]
                                    d[di + 1] = hp[oplane + si]
                                    d[di + 2] = hp[2 * oplane + si]
                                }
                            }
                        }
                    }
                    done += 1
                    progress(Double(done) / Double(total))
                }
            }
        }
        return Output(base: out, detail: detail)
    }

    /// base + amount * detail, for the whole photo.
    static func applyDetail(base: RGBAImage, detail: [Int8], amount: Float) -> RGBAImage {
        var out = base
        guard amount != 0 else { return out }
        let n = base.width * base.height
        out.pixels.withUnsafeMutableBufferPointer { oBuf in
            detail.withUnsafeBufferPointer { dBuf in
                let o = oBuf.baseAddress!, d = dBuf.baseAddress!
                for i in 0..<n {
                    let oi = i * 4, di = i * 3
                    o[oi] = toByte(Float(o[oi]) + amount * Float(d[di]))
                    o[oi + 1] = toByte(Float(o[oi + 1]) + amount * Float(d[di + 1]))
                    o[oi + 2] = toByte(Float(o[oi + 2]) + amount * Float(d[di + 2]))
                }
            }
        }
        return out
    }
}
