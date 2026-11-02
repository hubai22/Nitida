import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var models = ModelStore()
    @StateObject private var editor = EditorModel()
    @State private var pickerItem: PhotosPickerItem?
    @State private var showSettings = false
    @State private var toast: String?
    @State private var showZoom = false

    private let accent = Color("AccentColor")

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                if let notice = editor.notice {
                    noticeBanner(notice)
                }
                if models.state != .ready {
                    ModelSetupCard(store: models)
                }
                canvas
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                controls
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 12)
            .background(LinearGradient(colors: [Color(red: 0.03, green: 0.07, blue: 0.17), Color(red: 0.01, green: 0.03, blue: 0.09)],
                                       startPoint: .top, endPoint: .bottom).ignoresSafeArea())
            .navigationTitle("Nítida")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button { showSettings = true } label: { Image(systemName: "gearshape") }
                }
                if editor.original != nil {
                    ToolbarItem(placement: .navigationBarLeading) {
                        Button("Nueva") { editor.reset() }
                    }
                }
            }
            .sheet(isPresented: $showSettings) { SettingsView() }
            .overlay(alignment: .bottom) { toastView }
        }
        .preferredColorScheme(.dark)
        .tint(accent)
        .onChange(of: pickerItem) { item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    editor.load(image)
                } else {
                    show("No se pudo abrir esa foto.")
                }
                pickerItem = nil
            }
        }
    }

    private var isWorking: Bool {
        if case .working = editor.phase { return true }
        return false
    }

    // MARK: - Main area

    @ViewBuilder
    private var canvas: some View {
        switch editor.phase {
        case .empty:
            emptyState
        case .picked, .failed:
            if let img = editor.original {
                Image(uiImage: img)
                    .resizable()
                    .scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        case .working(let p, let label):
            ZStack {
                if let img = editor.original {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        .opacity(0.35)
                        .blur(radius: 6)
                }
                VStack(spacing: 14) {
                    ProgressView(value: p)
                        .progressViewStyle(.linear)
                        .frame(maxWidth: 240)
                    Text(label).font(.headline)
                    Text("\(Int(p * 100)) %  ·  Puedes salir de la app, seguirá trabajando")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Button(role: .cancel) { editor.cancel() } label: {
                        Label("Cancelar", systemImage: "xmark")
                    }
                    .buttonStyle(.bordered)
                }
                .padding(24)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            }
        case .done:
            if let before = editor.original, let after = editor.enhanced {
                CompareView(before: before, after: after)
                    .overlay(alignment: .bottomTrailing) {
                        Button { showZoom = true } label: {
                            Image(systemName: "arrow.up.left.and.arrow.down.right")
                                .font(.headline)
                                .padding(10)
                                .background(.ultraThinMaterial, in: Circle())
                        }
                        .padding(10)
                    }
                    .fullScreenCover(isPresented: $showZoom) {
                        ZoomViewer(before: before, after: after)
                    }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 14) {
            Image(systemName: "sparkles")
                .font(.system(size: 56, weight: .light))
                .foregroundStyle(accent)
            Text("Devuelve el detalle a tus fotos")
                .font(.title3.weight(.semibold))
            Text("Elige una foto borrosa o de baja calidad. La IA mejora toda la imagen y reconstruye las caras con más nitidez. Todo se procesa en tu iPhone.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Controls

    @ViewBuilder
    private var controls: some View {
        switch editor.phase {
        case .empty:
            picker(label: "Elegir foto", prominent: true)
        case .picked:
            VStack(spacing: 10) {
                Button {
                    editor.enhance(models: models.paths)
                } label: {
                    Label("Mejorar foto", systemImage: "wand.and.stars").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(models.state != .ready)
                if models.state != .ready {
                    Text("Primero descarga el motor de IA (arriba).")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                picker(label: "Elegir otra foto", prominent: false)
            }
        case .working:
            EmptyView()
        case .failed(let message):
            VStack(spacing: 10) {
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                Button("Reintentar") { editor.enhance(models: models.paths) }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
            }
        case .done:
            VStack(spacing: 12) {
                sliderRow("Definición", value: $editor.definition, range: 0...2)
                if editor.facesFound > 0 {
                    sliderRow("Intensidad en caras", value: $editor.strength, range: 0...1)
                } else {
                    Text("No se detectaron caras: se mejoró la foto completa.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                HStack(spacing: 10) {
                    Button {
                        Task {
                            if let error = await editor.save() { show(error) } else { show("Guardada en Fotos ✓") }
                        }
                    } label: {
                        Label("Guardar", systemImage: "square.and.arrow.down").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)

                    if let img = editor.enhanced {
                        ShareLink(item: Image(uiImage: img),
                                  preview: SharePreview("Foto mejorada", image: Image(uiImage: img))) {
                            Label("Compartir", systemImage: "square.and.arrow.up").frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                    }
                }
                HStack(spacing: 10) {
                    Button { editor.backToOptions() } label: {
                        Label("Volver", systemImage: "arrow.uturn.backward").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    Button { showZoom = true } label: {
                        Label("Zoom", systemImage: "magnifyingglass").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                }
                .controlSize(.regular)
            }
        }
    }

    private func sliderRow(_ title: String, value: Binding<Double>, range: ClosedRange<Double>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title).font(.subheadline.weight(.medium))
                Spacer()
                Text("\(Int((value.wrappedValue * 100).rounded())) %")
                    .font(.subheadline.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(value: value, in: range)
        }
    }

    @ViewBuilder
    private func picker(label: String, prominent: Bool) -> some View {
        let picker = PhotosPicker(selection: $pickerItem, matching: .images, photoLibrary: .shared()) {
            Label(label, systemImage: "photo.on.rectangle").frame(maxWidth: .infinity)
        }
        .controlSize(.large)
        if prominent {
            picker.buttonStyle(.borderedProminent)
        } else {
            picker.buttonStyle(.bordered)
        }
    }

    private func noticeBanner(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "info.circle.fill").foregroundStyle(accent)
            Text(text).font(.footnote)
            Spacer(minLength: 0)
            Button { editor.notice = nil } label: { Image(systemName: "xmark").font(.footnote) }
        }
        .padding(12)
        .background(accent.opacity(0.14), in: RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Toast

    @ViewBuilder
    private var toastView: some View {
        if let toast {
            Text(toast)
                .font(.subheadline.weight(.medium))
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial, in: Capsule())
                .padding(.bottom, 90)
                .transition(.opacity)
        }
    }

    private func show(_ message: String) {
        withAnimation { toast = message }
        Task {
            try? await Task.sleep(nanoseconds: 2_500_000_000)
            withAnimation { if toast == message { toast = nil } }
        }
    }
}

// MARK: - Model download card

struct ModelSetupCard: View {
    @ObservedObject var store: ModelStore

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Motor de IA", systemImage: "cpu").font(.headline)
            switch store.state {
            case .missing:
                Text("Se descarga una sola vez (\(sizeText)). Te recomiendo usar Wi-Fi.")
                    .font(.footnote).foregroundStyle(.secondary)
                Button("Descargar") { store.download() }
                    .buttonStyle(.borderedProminent)
            case .downloading(let p):
                ProgressView(value: p)
                Text("Descargando… \(Int(p * 100)) %  ·  Mantén la app abierta")
                    .font(.footnote).foregroundStyle(.secondary)
            case .failed(let message):
                Text(message).font(.footnote).foregroundStyle(.red)
                Button("Reintentar") { store.download() }
                    .buttonStyle(.borderedProminent)
            case .ready:
                EmptyView()
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private var sizeText: String {
        ByteCountFormatter.string(fromByteCount: store.remainingBytes, countStyle: .file)
    }
}

// MARK: - Settings

struct SettingsView: View {
    @AppStorage(EditorModel.neuralEngineKey) private var useNeuralEngine = true
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("Acelerar con Neural Engine", isOn: $useNeuralEngine)
                } footer: {
                    Text("Usa el chip de IA del iPhone para procesar más rápido. Si alguna foto sale rara o la app se cierra al mejorar, desactívalo.")
                }
                Section("Cómo funciona") {
                    Text("1. La foto completa se limpia y amplía (×2 o ×4) sin reducirla antes.")
                    Text("2. Las caras se mejoran con CodeFormer conservando la identidad; en caras grandes y nítidas actúa poco para no alisar la piel.")
                    Text("3. Se refuerza el detalle de toda la foto (pelo, ropa, fondo, piel) sin halos ni ruido.")
                    Text("Todo ocurre en el iPhone; tus fotos no salen del teléfono.")
                        .foregroundStyle(.secondary)
                }
                .font(.footnote)
                Section("Créditos") {
                    Text("CodeFormer (S-Lab, uso no comercial), SPAN Clear Reality, SCRFD (InsightFace), ONNX Runtime (Microsoft). Modelos de facefusion-assets.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Ajustes")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Listo") { dismiss() } }
            }
        }
        .preferredColorScheme(.dark)
    }
}
