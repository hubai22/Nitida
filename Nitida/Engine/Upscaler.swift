import Foundation

/// Whole-photo x4 super-resolution with two models, processed in fixed-size tiles so memory stays
/// bounded and the Neural Engine sees one static input shape:
/// - a clean model (Real-ESRGAN) gives natural tones and edges,
/// - a texture model (Nomos8k SC) contributes only its fine detail (high-pass), stored separately
///   so the "Textura" slider can dial it in later without re-running anything.
/// The two models run one after the other (never loaded together) to keep peak memory low.
final class Upscaler {
    static let scale = 4
    private let tile = 192
    private let pad = 32
    private let detailSigma: Float = 2.0
    private let cleanURL: URL
    private let textureURL: URL?
    private let useNeuralEngine: Bool

    struct Output {
        let base: RGBAImage
        let detail: [Int8]   // interleaved RGB high-pass, base.width * base.height * 3
    }

    /// `textureURL` nil skips the texture layer (fast mode).
    init(cleanURL: URL, textureURL: URL?, useNeuralEngine: Bool) {
        self.cleanURL = cleanURL
        self.textureURL = textureURL
        self.useNeuralEngine = useNeuralEngine
    }

    /// Model scale is always 4; `outScale` (1, 2 or 4) box-averages each tile down so large photos
    /// keep all their real detail without producing gigantic outputs.
    func upscale(_ img: RGBAImage, outScale: Int, progress: (Double) -> Void) throws -> Output {
        let sc = Upscaler.scale
        let os = outScale
        let fct = sc / os                     // model pixels per output pixel (per side)
        let norm = 1 / Float(fct * fct)
        let W = img.width * os, H = img.height * os
        var out = RGBAImage(width: W, height: H)
        var detail = [Int8](repeating: 0, count: textureURL == nil ? 0 : W * H * 3)
        let firstShare = textureURL == nil ? 1.0 : 0.5

        // Pass 1: clean base.
        try autoreleasepool {
            let runner = try OrtRunner(modelURL: cleanURL, useNeuralEngine: useNeuralEngine)
            try forEachTile(img, runner: runner, progress: { progress(firstShare * $0) }) { f, OS, oplane, x0, y0, vw, vh in
                out.pixels.withUnsafeMutableBufferPointer { dstBuf in
                    let dst = dstBuf.baseAddress!
                    let base = self.pad * sc
                    for yy in 0..<(vh * os) {
                        let dy = y0 * os + yy
                        for xx in 0..<(vw * os) {
                            var r: Float = 0, g: Float = 0, b: Float = 0
                            for ky in 0..<fct {
                                let row = (base + yy * fct + ky) * OS + base + xx * fct
                                for kx in 0..<fct {
                                    let si = row + kx
                                    r += f[si]; g += f[oplane + si]; b += f[2 * oplane + si]
                                }
                            }
                            let di = (dy * W + x0 * os + xx) * 4
                            dst[di] = toByte(r * norm * 255)
                            dst[di + 1] = toByte(g * norm * 255)
                            dst[di + 2] = toByte(b * norm * 255)
                        }
                    }
                }
            }
        }

        // Pass 2: texture layer (high-pass only).
        guard let textureURL = textureURL else { return Output(base: out, detail: detail) }
        try autoreleasepool {
            let runner = try OrtRunner(modelURL: textureURL, useNeuralEngine: useNeuralEngine)
            try forEachTile(img, runner: runner, progress: { progress(0.5 + 0.5 * $0) }) { f, OS, oplane, x0, y0, vw, vh in
                var planes = [Float](repeating: 0, count: 3 * oplane)
                for i in 0..<(3 * oplane) { planes[i] = min(max(f[i], 0), 1) * 255 }
                // Detail scale follows the output resolution.
                let hp = highPassToInt8(planes, channels: 3, width: OS, height: OS,
                                        sigma: self.detailSigma * Float(fct))
                detail.withUnsafeMutableBufferPointer { dBuf in
                    let d = dBuf.baseAddress!
                    let base = self.pad * sc
                    for yy in 0..<(vh * os) {
                        let dy = y0 * os + yy
                        for xx in 0..<(vw * os) {
                            var r: Float = 0, g: Float = 0, b: Float = 0
                            for ky in 0..<fct {
                                let row = (base + yy * fct + ky) * OS + base + xx * fct
                                for kx in 0..<fct {
                                    let si = row + kx
                                    r += Float(hp[si]); g += Float(hp[oplane + si]); b += Float(hp[2 * oplane + si])
                                }
                            }
                            let di = (dy * W + x0 * os + xx) * 3
                            d[di] = Int8(min(max((r * norm).rounded(), -127), 127))
                            d[di + 1] = Int8(min(max((g * norm).rounded(), -127), 127))
                            d[di + 2] = Int8(min(max((b * norm).rounded(), -127), 127))
                        }
                    }
                }
            }
        }
        return Output(base: out, detail: detail)
    }

    /// Runs `runner` over every padded tile and hands the raw planar output to `handle`:
    /// (floats, output tile side, plane size, tile x0, tile y0, valid width, valid height).
    private func forEachTile(_ img: RGBAImage, runner: OrtRunner, progress: (Double) -> Void,
                             handle: (UnsafePointer<Float>, Int, Int, Int, Int, Int, Int) throws -> Void) throws {
        let sc = Upscaler.scale
        let w = img.width, h = img.height
        let S = tile + 2 * pad
        let plane = S * S
        let OS = S * sc
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
                        let (data, shape) = try runner.runFirst(input, shape: [1, 3, S, S])
                        guard shape.count == 4, shape[2] == OS, shape[3] == OS else {
                            throw EngineError.badOutput("upscaler \(shape)")
                        }
                        let vw = min(tile, w - x0), vh = min(tile, h - y0)
                        try data.withUnsafeBytes { raw in
                            let f = raw.bindMemory(to: Float.self).baseAddress!
                            try handle(f, OS, OS * OS, x0, y0, vw, vh)
                        }
                    }
                    done += 1
                    progress(Double(done) / Double(total))
                }
            }
        }
    }

    /// base + amount * detail, for the whole photo.
    static func applyDetail(base: RGBAImage, detail: [Int8], amount: Float) -> RGBAImage {
        var out = base
        let n = base.width * base.height
        guard amount != 0, detail.count == n * 3 else { return out }
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
