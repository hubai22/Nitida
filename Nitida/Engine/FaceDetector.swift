import Foundation

struct DetectedFace {
    var x1: Float, y1: Float, x2: Float, y2: Float
    var landmarks: [(Float, Float)]   // left eye, right eye, nose, mouth left, mouth right
    var score: Float
}

/// SCRFD face detector (640x640 letterboxed input, 5 facial landmarks per face).
final class FaceDetector {
    private let size = 640
    private let strides = [8, 16, 32]
    private let anchorsPerCell = 2
    private let runner: OrtRunner

    init(modelURL: URL, useNeuralEngine: Bool) throws {
        runner = try OrtRunner(modelURL: modelURL, useNeuralEngine: useNeuralEngine)
    }

    func detect(_ img: RGBAImage, threshold: Float = 0.5) throws -> [DetectedFace] {
        let S = size, plane = S * S
        let w = img.width, h = img.height
        let sc = Float(S) / Float(max(w, h))
        let nw = min(S, Int((Float(w) * sc).rounded()))
        let nh = min(S, Int((Float(h) * sc).rounded()))

        // Letterbox (top-left), normalized (v - 127.5) / 128, RGB planar.
        let padValue: Float = -127.5 / 128
        var input = [Float](repeating: padValue, count: 3 * plane)
        img.pixels.withUnsafeBufferPointer { buf in
            let p = buf.baseAddress!
            var rgb: (Float, Float, Float) = (0, 0, 0)
            for y in 0..<nh {
                let sy = (Float(y) + 0.5) / sc - 0.5
                for x in 0..<nw {
                    let sx = (Float(x) + 0.5) / sc - 0.5
                    sampleBilinear(p, width: w, height: h, channels: 4, x: sx, y: sy, out: &rgb)
                    let di = y * S + x
                    input[di] = (rgb.0 - 127.5) / 128
                    input[plane + di] = (rgb.1 - 127.5) / 128
                    input[2 * plane + di] = (rgb.2 - 127.5) / 128
                }
            }
        }

        let outputs = try runner.run(input, shape: [1, 3, S, S])
        // Outputs are scores (x3 strides), boxes (x3), landmarks (x3), in model order.
        var names = runner.outputNames
        if names.allSatisfy({ Int($0) != nil }) { names.sort { Int($0)! < Int($1)! } }
        guard names.count >= 9 else { throw EngineError.badOutput("detector outputs \(names.count)") }

        var faces: [DetectedFace] = []
        for (i, stride) in strides.enumerated() {
            guard let sv = outputs[names[i]], let bv = outputs[names[i + 3]], let kv = outputs[names[i + 6]] else {
                throw EngineError.badOutput("detector missing output")
            }
            let (sData, _) = try OrtRunner.floats(sv)
            let (bData, _) = try OrtRunner.floats(bv)
            let (kData, _) = try OrtRunner.floats(kv)
            let fs = S / stride
            let count = fs * fs * anchorsPerCell
            guard sData.count >= count * 4, bData.count >= count * 16, kData.count >= count * 40 else {
                throw EngineError.badOutput("detector size")
            }
            sData.withUnsafeBytes { sRaw in
                bData.withUnsafeBytes { bRaw in
                    kData.withUnsafeBytes { kRaw in
                        let s = sRaw.bindMemory(to: Float.self)
                        let b = bRaw.bindMemory(to: Float.self)
                        let k = kRaw.bindMemory(to: Float.self)
                        let st = Float(stride)
                        for a in 0..<count where s[a] >= threshold {
                            let cell = a / anchorsPerCell
                            let cx = Float(cell % fs) * st
                            let cy = Float(cell / fs) * st
                            var lm: [(Float, Float)] = []
                            for j in 0..<5 {
                                lm.append(((cx + k[a * 10 + 2 * j] * st) / sc,
                                           (cy + k[a * 10 + 2 * j + 1] * st) / sc))
                            }
                            faces.append(DetectedFace(
                                x1: (cx - b[a * 4] * st) / sc,
                                y1: (cy - b[a * 4 + 1] * st) / sc,
                                x2: (cx + b[a * 4 + 2] * st) / sc,
                                y2: (cy + b[a * 4 + 3] * st) / sc,
                                landmarks: lm, score: s[a]))
                        }
                    }
                }
            }
        }

        // Non-maximum suppression; drop tiny detections.
        faces.sort { $0.score > $1.score }
        var kept: [DetectedFace] = []
        for f in faces where (f.x2 - f.x1) >= 10 && (f.y2 - f.y1) >= 10 {
            if kept.allSatisfy({ iou($0, f) < 0.4 }) { kept.append(f) }
        }
        return kept
    }

    private func iou(_ a: DetectedFace, _ b: DetectedFace) -> Float {
        let ix = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
        let iy = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
        let inter = ix * iy
        let ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return ua > 0 ? inter / ua : 0
    }
}
