import UIKit

/// 8-bit RGBA bitmap (alpha always 255), rows top-to-bottom.
struct RGBAImage {
    let width: Int
    let height: Int
    var pixels: [UInt8]

    init(width: Int, height: Int, pixels: [UInt8]) {
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    init(width: Int, height: Int) {
        self.init(width: width, height: height, pixels: [UInt8](repeating: 255, count: width * height * 4))
    }

    private static let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

    /// Draws the image upright (EXIF orientation applied), scaled so its long side is at most `maxSide`.
    static func from(_ image: UIImage, maxSide: Int) -> RGBAImage? {
        let w0 = image.size.width * image.scale
        let h0 = image.size.height * image.scale
        guard w0 > 0, h0 > 0 else { return nil }
        let f = min(1.0, CGFloat(maxSide) / max(w0, h0))
        let w = max(1, Int((w0 * f).rounded()))
        let h = max(1, Int((h0 * f).rounded()))

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        format.preferredRange = .standard
        let upright = UIGraphicsImageRenderer(size: CGSize(width: w, height: h), format: format).image { ctx in
            ctx.cgContext.interpolationQuality = .high
            image.draw(in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        guard let cg = upright.cgImage else { return nil }

        var px = [UInt8](repeating: 255, count: w * h * 4)
        let ok = px.withUnsafeMutableBytes { buf -> Bool in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h,
                                      bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        return ok ? RGBAImage(width: w, height: h, pixels: px) : nil
    }

    func cgImage() -> CGImage? {
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: Self.colorSpace,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }

    func uiImage() -> UIImage? {
        guard let cg = cgImage() else { return nil }
        return UIImage(cgImage: cg)
    }
}

/// Bilinear sampling of an interleaved 8-bit buffer with edge clamping.
/// `channels` is the pixel stride (4 for RGBA, 3 for RGB); writes 3 values (R,G,B) as Float 0...255.
@inline(__always)
func sampleBilinear(_ p: UnsafePointer<UInt8>, width: Int, height: Int, channels: Int,
                    x: Float, y: Float, out: inout (Float, Float, Float)) {
    let xc = min(max(x, 0), Float(width - 1))
    let yc = min(max(y, 0), Float(height - 1))
    let x0 = Int(xc), y0 = Int(yc)
    let x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1)
    let fx = xc - Float(x0), fy = yc - Float(y0)
    let w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy
    let i00 = (y0 * width + x0) * channels
    let i10 = (y0 * width + x1) * channels
    let i01 = (y1 * width + x0) * channels
    let i11 = (y1 * width + x1) * channels
    out.0 = Float(p[i00]) * w00 + Float(p[i10]) * w10 + Float(p[i01]) * w01 + Float(p[i11]) * w11
    out.1 = Float(p[i00 + 1]) * w00 + Float(p[i10 + 1]) * w10 + Float(p[i01 + 1]) * w01 + Float(p[i11 + 1]) * w11
    out.2 = Float(p[i00 + 2]) * w00 + Float(p[i10 + 2]) * w10 + Float(p[i01 + 2]) * w01 + Float(p[i11 + 2]) * w11
}

@inline(__always)
func toByte(_ v: Float) -> UInt8 {
    if !(v > 0) { return 0 }   // also catches NaN
    if v >= 255 { return 255 }
    return UInt8(v + 0.5)
}

/// 2D similarity/affine transform: x' = a*x + b*y + tx, y' = c*x + d*y + ty.
struct Affine {
    var a: Float, b: Float, c: Float, d: Float, tx: Float, ty: Float

    @inline(__always)
    func apply(_ x: Float, _ y: Float) -> (Float, Float) {
        (a * x + b * y + tx, c * x + d * y + ty)
    }

    func inverted() -> Affine {
        let det = a * d - b * c
        let ia = d / det, ib = -b / det, ic = -c / det, id = a / det
        return Affine(a: ia, b: ib, c: ic, d: id,
                      tx: -(ia * tx + ib * ty), ty: -(ic * tx + id * ty))
    }

    /// Least-squares similarity (rotation + uniform scale + translation, no reflection) mapping src -> dst.
    static func similarity(from src: [(Float, Float)], to dst: [(Float, Float)]) -> Affine {
        let n = Float(src.count)
        let msx = src.map(\.0).reduce(0, +) / n, msy = src.map(\.1).reduce(0, +) / n
        let mdx = dst.map(\.0).reduce(0, +) / n, mdy = dst.map(\.1).reduce(0, +) / n
        var num1: Float = 0, num2: Float = 0, den: Float = 0
        for i in 0..<src.count {
            let sx = src[i].0 - msx, sy = src[i].1 - msy
            let dx = dst[i].0 - mdx, dy = dst[i].1 - mdy
            num1 += sx * dx + sy * dy
            num2 += sx * dy - sy * dx
            den += sx * sx + sy * sy
        }
        let cs = num1 / den, sn = num2 / den
        return Affine(a: cs, b: -sn, c: sn, d: cs,
                      tx: mdx - (cs * msx - sn * msy),
                      ty: mdy - (sn * msx + cs * msy))
    }
}
