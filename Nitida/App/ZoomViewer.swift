import SwiftUI
import UIKit

/// Full-screen viewer: pinch or double-tap to zoom; hold the button to see the original
/// at the same zoom and position.
struct ZoomViewer: View {
    let before: UIImage
    let after: UIImage
    @Environment(\.dismiss) private var dismiss
    @State private var showBefore = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            ZoomableImage(image: after, overlay: before, showOverlay: showBefore)
                .ignoresSafeArea()

            VStack {
                HStack {
                    Text(showBefore ? "Antes" : "Después")
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                    Spacer()
                    Button { dismiss() } label: {
                        Image(systemName: "xmark")
                            .font(.headline)
                            .padding(10)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                }
                .padding()
                Spacer()
                Text("Mantén pulsado para ver el original")
                    .font(.subheadline.weight(.semibold))
                    .padding(.horizontal, 18).padding(.vertical, 12)
                    .background(.ultraThinMaterial, in: Capsule())
                    .scaleEffect(showBefore ? 0.96 : 1)
                    .onLongPressGesture(minimumDuration: 0, maximumDistance: 60, perform: {}, onPressingChanged: { pressing in
                        showBefore = pressing
                    })
                    .padding(.bottom, 24)
            }
            .foregroundStyle(.white)
        }
        .preferredColorScheme(.dark)
    }
}

/// UIScrollView-based zoom (smooth pinch, pan and double-tap like the Photos app).
private struct ZoomableImage: UIViewRepresentable {
    let image: UIImage
    let overlay: UIImage
    let showOverlay: Bool

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> UIScrollView {
        let scroll = UIScrollView()
        scroll.delegate = context.coordinator
        scroll.minimumZoomScale = 1
        scroll.maximumZoomScale = 10
        scroll.showsHorizontalScrollIndicator = false
        scroll.showsVerticalScrollIndicator = false
        scroll.contentInsetAdjustmentBehavior = .never
        scroll.backgroundColor = .black

        let container = UIView()
        let main = UIImageView(image: image)
        let top = UIImageView(image: overlay)
        for v in [main, top] {
            v.contentMode = .scaleAspectFit
            v.translatesAutoresizingMaskIntoConstraints = false
            container.addSubview(v)
            NSLayoutConstraint.activate([
                v.leadingAnchor.constraint(equalTo: container.leadingAnchor),
                v.trailingAnchor.constraint(equalTo: container.trailingAnchor),
                v.topAnchor.constraint(equalTo: container.topAnchor),
                v.bottomAnchor.constraint(equalTo: container.bottomAnchor)
            ])
        }
        top.isHidden = true
        scroll.addSubview(container)
        context.coordinator.container = container
        context.coordinator.overlay = top

        let doubleTap = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.doubleTapped(_:)))
        doubleTap.numberOfTapsRequired = 2
        scroll.addGestureRecognizer(doubleTap)
        return scroll
    }

    func updateUIView(_ scroll: UIScrollView, context: Context) {
        context.coordinator.overlay?.isHidden = !showOverlay
        if let container = context.coordinator.container, container.frame.size != scroll.bounds.size,
           scroll.zoomScale == 1 {
            container.frame = CGRect(origin: .zero, size: scroll.bounds.size)
            scroll.contentSize = scroll.bounds.size
        }
        DispatchQueue.main.async {
            if let container = context.coordinator.container, scroll.zoomScale == 1,
               container.frame.size != scroll.bounds.size {
                container.frame = CGRect(origin: .zero, size: scroll.bounds.size)
                scroll.contentSize = scroll.bounds.size
            }
        }
    }

    final class Coordinator: NSObject, UIScrollViewDelegate {
        weak var container: UIView?
        weak var overlay: UIImageView?

        func viewForZooming(in scrollView: UIScrollView) -> UIView? { container }

        @objc func doubleTapped(_ g: UITapGestureRecognizer) {
            guard let scroll = g.view as? UIScrollView else { return }
            if scroll.zoomScale > 1 {
                scroll.setZoomScale(1, animated: true)
            } else {
                let p = g.location(in: container)
                let size = CGSize(width: scroll.bounds.width / 3, height: scroll.bounds.height / 3)
                scroll.zoom(to: CGRect(x: p.x - size.width / 2, y: p.y - size.height / 2,
                                       width: size.width, height: size.height), animated: true)
            }
        }
    }
}
