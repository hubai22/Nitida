import Foundation

/// A face rebuilt by the restoration model, plus the transform from image space into its aligned crop.
struct RestoredFace {
    let imageToCrop: Affine
    let pixels: [UInt8]   // cropSize*cropSize RGB
}

/// GPEN-BFR-1024 face restoration: aligns each face to the FFHQ template, regenerates realistic
/// skin texture, eyes, lips and hair at 1024 px, and blends it back with a feathered mask.
final class FaceRestorer {
    static let cropSize = 1024
    // FFHQ five-point template (eyes, nose tip, mouth corners).
    private static let normalizedTemplate: [(Float, Float)] = [
        (0.37691676, 0.46864664), (0.62285697, 0.46912813), (0.50123859, 0.61331904),
        (0.39308822, 0.72541100), (0.61150205, 0.72490465)
    ]
    private static let template: [(Float, Float)] = normalizedTemplate.map {
        ($0.0 * Float(cropSize), $0.1 * Float(cropSize))
    }

    private let runner: OrtRunner

    init(modelURL: URL, useNeuralEngine: Bool) throws {
        runner = try OrtRunner(modelURL: modelURL, useNeuralEngine: useNeuralEngine)
    }

    /// `landmarkScale` converts detector coordinates into `canvas` coordinates.
    func restore(_ canvas: RGBAImage, faces: [DetectedFace], landmarkScale: Float,
                 progress: (Double) -> Void) throws -> [RestoredFace] {
        let C = FaceRestorer.cropSize, plane = C * C
        var results: [RestoredFace] = []
        var input = [Float](repeating: 0, count: 3 * plane)

        for (index, face) in faces.enumerated() {
            try autoreleasepool {
                let pts = face.landmarks.map { ($0.0 * landmarkScale, $0.1 * landmarkScale) }
                let toCrop = Affine.similarity(from: pts, to: FaceRestorer.template)
                let toImage = toCrop.inverted()

                // Warp the face into the aligned crop, normalized to [-1, 1].
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
                var px = [UInt8](repeating: 0, count: plane * 3)
                data.withUnsafeBytes { raw in
                    let f = raw.bindMemory(to: Float.self).baseAddress!
                    for i in 0..<plane {
                        for ch in 0..<3 {
                            let v = min(max(f[ch * plane + i], -1), 1)
                            px[i * 3 + ch] = toByte((v + 1) * 127.5)
                        }
                    }
                }
                results.append(RestoredFace(imageToCrop: toCrop, pixels: px))
            }
            progress(Double(index + 1) / Double(faces.count))
        }
        return results
    }

    /// Feathered square mask over the crop: 0 at the border, 1 inside.
    @inline(__always)
    private static func mask(_ u: Float, _ v: Float) -> Float {
        let edge = Float(cropSize - 1)
        let d = min(min(u, v), min(edge - u, edge - v))
        let t = min(max((d - 8) / 80, 0), 1)
        return t * t * (3 - 2 * t)
    }

    /// Blends restored faces onto `base`. `strength` 0...1 controls how much of the AI face is used.
    static func composite(base: RGBAImage, faces: [RestoredFace], strength: Float) -> RGBAImage {
        var out = base
        guard strength > 0, !faces.isEmpty else { return out }
        let C = cropSize
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

                face.pixels.withUnsafeBufferPointer { fBuf in
                    let fp = fBuf.baseAddress!
                    var rgb: (Float, Float, Float) = (0, 0, 0)
                    for y in minY...maxY {
                        for x in minX...maxX {
                            let (u, v) = face.imageToCrop.apply(Float(x), Float(y))
                            if u < 0 || v < 0 || u > Float(C - 1) || v > Float(C - 1) { continue }
                            let m = mask(u, v) * strength
                            if m <= 0 { continue }
                            sampleBilinear(fp, width: C, height: C, channels: 3, x: u, y: v, out: &rgb)
                            let di = (y * W + x) * 4
                            dst[di] = toByte(Float(dst[di]) * (1 - m) + rgb.0 * m)
                            dst[di + 1] = toByte(Float(dst[di + 1]) * (1 - m) + rgb.1 * m)
                            dst[di + 2] = toByte(Float(dst[di + 2]) * (1 - m) + rgb.2 * m)
                        }
                    }
                }
            }
        }
        return out
    }
}
