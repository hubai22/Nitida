import Foundation

/// In-place separable Gaussian blur of `channels` planar float images of size `width` × `height`
/// (edges clamped). Used to split model output into smooth tone and fine detail (high-pass).
func gaussianBlurPlanes(_ data: inout [Float], channels: Int, width w: Int, height h: Int, sigma: Float) {
    let r = max(1, Int((sigma * 3).rounded(.up)))
    var kernel = [Float](repeating: 0, count: 2 * r + 1)
    var sum: Float = 0
    for i in -r...r {
        let v = expf(-Float(i * i) / (2 * sigma * sigma))
        kernel[i + r] = v
        sum += v
    }
    for i in 0..<kernel.count { kernel[i] /= sum }

    var tmp = [Float](repeating: 0, count: w * h)
    data.withUnsafeMutableBufferPointer { dBuf in
        tmp.withUnsafeMutableBufferPointer { tBuf in
            kernel.withUnsafeBufferPointer { kBuf in
                let d = dBuf.baseAddress!, t = tBuf.baseAddress!, k = kBuf.baseAddress!
                for c in 0..<channels {
                    let p = d + c * w * h
                    // Horizontal pass: p -> t
                    for y in 0..<h {
                        let row = y * w
                        for x in 0..<w {
                            var acc: Float = 0
                            for i in -r...r {
                                let xx = min(max(x + i, 0), w - 1)
                                acc += p[row + xx] * k[i + r]
                            }
                            t[row + x] = acc
                        }
                    }
                    // Vertical pass: t -> p
                    for y in 0..<h {
                        for x in 0..<w {
                            var acc: Float = 0
                            for i in -r...r {
                                let yy = min(max(y + i, 0), h - 1)
                                acc += t[yy * w + x] * k[i + r]
                            }
                            p[y * w + x] = acc
                        }
                    }
                }
            }
        }
    }
}

/// High-pass of planar float data (value - blurred value), stored as signed bytes.
/// Input values are in 0...255 units; output is clamped to ±127.
func highPassToInt8(_ planes: [Float], channels: Int, width w: Int, height h: Int, sigma: Float) -> [Int8] {
    var blurred = planes
    gaussianBlurPlanes(&blurred, channels: channels, width: w, height: h, sigma: sigma)
    var out = [Int8](repeating: 0, count: planes.count)
    for i in 0..<planes.count {
        let v = (planes[i] - blurred[i]).rounded()
        out[i] = Int8(min(max(v, -127), 127))
    }
    return out
}

/// Bilinear sampling of a planar Int8 image (3 planes of w×h) with edge clamping.
@inline(__always)
func sampleBilinearPlanarI8(_ p: UnsafePointer<Int8>, width: Int, height: Int,
                            x: Float, y: Float, out: inout (Float, Float, Float)) {
    let xc = min(max(x, 0), Float(width - 1))
    let yc = min(max(y, 0), Float(height - 1))
    let x0 = Int(xc), y0 = Int(yc)
    let x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1)
    let fx = xc - Float(x0), fy = yc - Float(y0)
    let w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy
    let plane = width * height
    let i00 = y0 * width + x0, i10 = y0 * width + x1, i01 = y1 * width + x0, i11 = y1 * width + x1
    out.0 = Float(p[i00]) * w00 + Float(p[i10]) * w10 + Float(p[i01]) * w01 + Float(p[i11]) * w11
    let q = p + plane
    out.1 = Float(q[i00]) * w00 + Float(q[i10]) * w10 + Float(q[i01]) * w01 + Float(q[i11]) * w11
    let s = p + 2 * plane
    out.2 = Float(s[i00]) * w00 + Float(s[i10]) * w10 + Float(s[i01]) * w01 + Float(s[i11]) * w11
}
