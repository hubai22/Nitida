import Foundation
import Accelerate

/// Fast planar-float filters (vImage) used by the detail and skin-texture stages.
enum Filters {
    static func gaussianKernel(_ sigma: Float) -> [Float] {
        let r = max(1, Int((sigma * 3).rounded(.up)))
        var k = (-r...r).map { expf(-Float($0 * $0) / (2 * sigma * sigma)) }
        let s = k.reduce(0, +)
        for i in k.indices { k[i] /= s }
        return k
    }

    static func boxKernel(_ r: Int) -> [Float] {
        [Float](repeating: 1 / Float(2 * r + 1), count: 2 * r + 1)
    }

    /// Separable convolution with edge extension.
    static func convolve(_ src: [Float], width w: Int, height h: Int, kernel: [Float]) -> [Float] {
        var input = src
        var output = [Float](repeating: 0, count: w * h)
        input.withUnsafeMutableBytes { ib in
            output.withUnsafeMutableBytes { ob in
                var s = vImage_Buffer(data: ib.baseAddress, height: vImagePixelCount(h),
                                      width: vImagePixelCount(w), rowBytes: w * MemoryLayout<Float>.stride)
                var d = vImage_Buffer(data: ob.baseAddress, height: vImagePixelCount(h),
                                      width: vImagePixelCount(w), rowBytes: w * MemoryLayout<Float>.stride)
                kernel.withUnsafeBufferPointer { kb in
                    _ = vImageSepConvolve_PlanarF(&s, &d, nil, 0, 0,
                                                  kb.baseAddress!, UInt32(kernel.count),
                                                  kb.baseAddress!, UInt32(kernel.count),
                                                  0, 0, vImage_Flags(kvImageEdgeExtend))
                }
            }
        }
        return output
    }

    static func gauss(_ src: [Float], _ w: Int, _ h: Int, _ sigma: Float) -> [Float] {
        convolve(src, width: w, height: h, kernel: gaussianKernel(sigma))
    }

    static func box(_ src: [Float], _ w: Int, _ h: Int, _ r: Int) -> [Float] {
        convolve(src, width: w, height: h, kernel: boxKernel(r))
    }

    /// Runs `body` over horizontal strips with `margin` extra rows above/below so filters see
    /// enough context. `body(y0, rows, top)` gets the strip's first row in the image, the total
    /// rows (with margins) and how many margin rows lie above the strip's own rows.
    static func strips(height: Int, strip: Int, margin: Int,
                       _ body: (_ a: Int, _ rows: Int, _ own0: Int, _ ownRows: Int) throws -> Void) rethrows {
        var y = 0
        while y < height {
            let own = min(strip, height - y)
            let a = max(0, y - margin)
            let b = min(height, y + own + margin)
            try body(a, b - a, y - a, own)
            y += own
        }
    }
}

/// Multi-band "clarity": boosts fine, medium and coarse detail on luminance, like the difference
/// between a phone photo and a Remini/Topaz result. Edge-aware (guided filter base, no halos) and
/// noise-aware (fine detail is only boosted where there is real structure).
enum Clarity {
    static let gains: (fine: Float, mid: Float, coarse: Float) = (3.0, 1.5, 1.2)

    static func layer(_ img: RGBAImage, scale: Int, cancelCheck: () throws -> Void = {}) rethrows -> [Int8] {
        var out = [Int8](repeating: 0, count: img.width * img.height)
        let W = img.width, H = img.height, s = Float(scale)
        let r = 8 * scale
        let margin = 18 * scale + 8
        let eps: Float = 1e-3
        try Filters.strips(height: H, strip: 384, margin: margin) { a, rows, own0, ownRows in
            try cancelCheck()
            let n = W * rows
            var L = [Float](repeating: 0, count: n)
            img.pixels.withUnsafeBufferPointer { p in
                for i in 0..<n {
                    let j = (a * W + i) * 4
                    L[i] = (0.299 * Float(p[j]) + 0.587 * Float(p[j + 1]) + 0.114 * Float(p[j + 2])) / 255
                }
            }
            let b1 = Filters.gauss(L, W, rows, 1.0 * s)
            let b3 = Filters.gauss(L, W, rows, 3.0 * s)
            // Guided filter (I = p = L) for an edge-preserving coarse base.
            var LL = [Float](repeating: 0, count: n)
            for i in 0..<n { LL[i] = L[i] * L[i] }
            let mI = Filters.box(L, W, rows, r)
            let mII = Filters.box(LL, W, rows, r)
            var A = [Float](repeating: 0, count: n), B = [Float](repeating: 0, count: n)
            for i in 0..<n {
                let v = max(mII[i] - mI[i] * mI[i], 0)
                A[i] = v / (v + eps)
                B[i] = mI[i] - A[i] * mI[i]
            }
            let mA = Filters.box(A, W, rows, r)
            let mB = Filters.box(B, W, rows, r)
            var absMid = [Float](repeating: 0, count: n)
            for i in 0..<n { absMid[i] = abs(b1[i] - b3[i]) }
            let act = Filters.gauss(absMid, W, rows, 2 * s)

            let g = gains
            out.withUnsafeMutableBufferPointer { o in
                for y in own0..<(own0 + ownRows) {
                    for x in 0..<W {
                        let i = y * W + x
                        let b8 = mA[i] * L[i] + mB[i]
                        let fine = L[i] - b1[i], mid = b1[i] - b3[i], coarse = b3[i] - b8
                        let gate = min(max((act[i] - 0.0015) / 0.006, 0), 1)
                        let gf = 1 + (g.fine - 1) * gate
                        let v = b8 + 0.15 * tanhf(coarse * g.coarse / 0.15)
                                   + 0.10 * tanhf(mid * g.mid / 0.10)
                                   + 0.08 * tanhf(fine * gf / 0.08)
                        let d = (v - L[i]) * 255
                        o[a * W + i] = Int8(min(max(d.rounded(), -127), 127))
                    }
                }
            }
        }
        return out
    }
    }

    /// Adds a luminance layer (times `amount`) to an image.
    static func add(_ layer: [Int8], to img: RGBAImage, amount: Float) -> RGBAImage {
        var out = img
        guard amount != 0, layer.count == img.width * img.height else { return out }
        out.pixels.withUnsafeMutableBufferPointer { o in
            for i in 0..<layer.count where layer[i] != 0 {
                let d = amount * Float(layer[i]), j = i * 4
                o[j] = toByte(Float(o[j]) + d)
                o[j + 1] = toByte(Float(o[j + 1]) + d)
                o[j + 2] = toByte(Float(o[j + 2]) + d)
            }
        }
        return out
    }

    static func apply(_ img: RGBAImage, scale: Int, cancelCheck: () throws -> Void = {}) rethrows -> RGBAImage {
        add(try layer(img, scale: scale, cancelCheck: cancelCheck), to: img, amount: 1)
    }
}

/// Synthesizes fine pore texture on smooth skin only (where the photo has no real skin detail),
/// returned as a luminance offset layer so it can be dialed in instantly.
enum SkinTexture {
    @inline(__always)
    private static func noise(_ x: Int, _ y: Int) -> Float {
        var h = UInt32(truncatingIfNeeded: (x &* 73_856_093) ^ (y &* 19_349_663) ^ 0x9E37_79B9)
        h ^= h >> 16; h = h &* 0x7FEB_352D; h ^= h >> 15; h = h &* 0x846C_A68B; h ^= h >> 16
        return (Float(h) / Float(UInt32.max) - 0.5) * 3.4641   // uniform, variance 1
    }

    static func layer(_ img: RGBAImage, scale: Int, amount: Float = 1.6,
                      cancelCheck: () throws -> Void = {}) rethrows -> [Int8] {
        let W = img.width, H = img.height, s = Float(scale)
        var result = [Int8](repeating: 0, count: W * H)
        let margin = 18 * scale + 8
        let s1 = 0.55 * s, s2 = 1.4 * s
        // Std of a difference-of-Gaussians of unit white noise.
        let dogVar = 1 / (4 * Float.pi * s1 * s1) + 1 / (4 * Float.pi * s2 * s2) - 1 / (Float.pi * (s1 * s1 + s2 * s2))
        let dogSD = max(dogVar, 1e-6).squareRoot()
        let target: Float = 0.032

        try Filters.strips(height: H, strip: 384, margin: margin) { a, rows, own0, ownRows in
            try cancelCheck()
            let n = W * rows
            var L = [Float](repeating: 0, count: n)
            var skin = [Float](repeating: 0, count: n)
            var nz = [Float](repeating: 0, count: n)
            img.pixels.withUnsafeBufferPointer { p in
                for i in 0..<n {
                    let j = (a * W + i) * 4
                    let R = Float(p[j]), G = Float(p[j + 1]), Bl = Float(p[j + 2])
                    let Y = 0.299 * R + 0.587 * G + 0.114 * Bl
                    let Cr = (R - Y) * 0.713 + 128, Cb = (Bl - Y) * 0.564 + 128
                    L[i] = Y / 255
                    skin[i] = (Cr > 133 && Cr < 180 && Cb > 77 && Cb < 132 && Y > 45 && Y < 245) ? 1 : 0
                    nz[i] = noise(i % W, a + i / W)
                }
            }
            skin = Filters.gauss(skin, W, rows, 3 * s)
            let b1 = Filters.gauss(L, W, rows, s)
            var f2 = [Float](repeating: 0, count: n)
            for i in 0..<n { let f = L[i] - b1[i]; f2[i] = f * f }
            let local = Filters.gauss(f2, W, rows, 4 * s)
            let e1 = Filters.gauss(L, W, rows, 1.5 * s)
            let e4 = Filters.gauss(L, W, rows, 4 * s)
            let p1 = Filters.gauss(nz, W, rows, s1)
            let p2 = Filters.gauss(nz, W, rows, s2)

            for y in own0..<(own0 + ownRows) {
                for x in 0..<W {
                    let i = y * W + x
                    let need = min(max((target - local[i].squareRoot()) / target, 0), 1)
                    let flat = min(max(1 - abs(e1[i] - e4[i]) / 0.03, 0), 1)
                    var pore = (p1[i] - p2[i]) / dogSD
                    pore = pore < 0 ? pore * 1.25 : pore * 0.7      // pores read as small dark dots
                    let amp = 0.016 * amount * skin[i] * need * flat * (0.6 + 0.8 * L[i])
                    let d = 255 * pore * amp * (L[i] / max(L[i], 0.15))
                    result[(a + y) * W + x] = Int8(min(max(d.rounded(), -127), 127))
                }
            }
        }
        return result
    }
}

extension RGBAImage {
    /// Sub-image copy (rect clamped to the image).
    func cropped(x: Int, y: Int, width w: Int, height h: Int) -> RGBAImage {
        var out = RGBAImage(width: w, height: h)
        pixels.withUnsafeBufferPointer { src in
            out.pixels.withUnsafeMutableBufferPointer { dst in
                for row in 0..<h {
                    let s = ((y + row) * width + x) * 4
                    let d = row * w * 4
                    for k in 0..<(w * 4) { dst[d + k] = src[s + k] }
                }
            }
        }
        return out
    }
}
