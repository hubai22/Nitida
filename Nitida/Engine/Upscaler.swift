import Foundation

/// Whole-photo restoration + super-resolution (Real-ESRGAN x2 or x4), processed in fixed-size tiles
/// so memory stays bounded and the Neural Engine sees one static input shape.
final class Upscaler {
    let scale: Int
    private let tile = 192
    private let pad = 32
    private let runner: OrtRunner

    init(modelURL: URL, scale: Int, useNeuralEngine: Bool) throws {
        self.scale = scale
        runner = try OrtRunner(modelURL: modelURL, useNeuralEngine: useNeuralEngine)
    }

    func upscale(_ img: RGBAImage, progress: (Double) -> Void) throws -> RGBAImage {
        let sc = scale
        let w = img.width, h = img.height
        let S = tile + 2 * pad
        let plane = S * S
        let W = w * sc
        var out = RGBAImage(width: W, height: h * sc)
        var input = [Float](repeating: 0, count: 3 * plane)

        let tilesX = (w + tile - 1) / tile
        let tilesY = (h + tile - 1) / tile
        let total = tilesX * tilesY
        var done = 0
        progress(0)

        try img.pixels.withUnsafeBufferPointer { srcBuf in
            let src = srcBuf.baseAddress!
            try out.pixels.withUnsafeMutableBufferPointer { dstBuf in
                let dst = dstBuf.baseAddress!
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

                            let (data, shape) = try runner.runFirst(input, shape: [1, 3, S, S])
                            guard shape.count == 4, shape[2] == S * sc, shape[3] == S * sc else {
                                throw EngineError.badOutput("upscaler \(shape)")
                            }
                            let OS = S * sc
                            let oplane = OS * OS
                            let vw = min(tile, w - x0), vh = min(tile, h - y0)

                            data.withUnsafeBytes { raw in
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
                        done += 1
                        progress(Double(done) / Double(total))
                    }
                }
            }
        }
        return out
    }
}
