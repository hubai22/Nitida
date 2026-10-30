import SwiftUI

/// Before/after viewer: drag the handle to reveal the original on the left side.
struct CompareView: View {
    let before: UIImage
    let after: UIImage
    @State private var split: CGFloat = 0.5

    var body: some View {
        GeometryReader { geo in
            let fit = fittedSize(for: after.size, in: geo.size)
            ZStack(alignment: .leading) {
                Image(uiImage: after)
                    .resizable()
                    .frame(width: fit.width, height: fit.height)
                Image(uiImage: before)
                    .resizable()
                    .frame(width: fit.width, height: fit.height)
                    .mask(alignment: .leading) {
                        Rectangle().frame(width: fit.width * split)
                    }
                handle(height: fit.height)
                    .offset(x: fit.width * split - 22)
            }
            .frame(width: fit.width, height: fit.height)
            .overlay(alignment: .topLeading) { tag("Antes").opacity(split > 0.12 ? 1 : 0) }
            .overlay(alignment: .topTrailing) { tag("Después").opacity(split < 0.88 ? 1 : 0) }
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0).onChanged { value in
                    split = min(max(value.location.x / fit.width, 0), 1)
                }
            )
            .position(x: geo.size.width / 2, y: geo.size.height / 2)
        }
    }

    private func handle(height: CGFloat) -> some View {
        ZStack {
            Rectangle()
                .fill(.white)
                .frame(width: 2, height: height)
                .shadow(color: .black.opacity(0.4), radius: 2)
            Circle()
                .fill(.white)
                .frame(width: 34, height: 34)
                .shadow(color: .black.opacity(0.35), radius: 4)
                .overlay(
                    Image(systemName: "arrow.left.and.right")
                        .font(.system(size: 14, weight: .bold))
                        .foregroundStyle(.black)
                )
        }
        .frame(width: 44)
    }

    private func tag(_ text: String) -> some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(.ultraThinMaterial, in: Capsule())
            .padding(10)
    }

    private func fittedSize(for image: CGSize, in box: CGSize) -> CGSize {
        guard image.width > 0, image.height > 0, box.width > 0, box.height > 0 else { return box }
        let s = min(box.width / image.width, box.height / image.height)
        return CGSize(width: image.width * s, height: image.height * s)
    }
}
