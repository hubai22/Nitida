import UIKit
import BackgroundTasks

/// Keeps an enhancement running when the user leaves the app (switches apps, takes a call).
/// - iOS 26+: a continued-processing task, shown by the system with a progress indicator.
/// - Older iOS: a standard background task (gives the app some extra time before suspension).
/// If iOS still pauses the app, the work simply resumes when the user comes back.
final class BackgroundRun {
    static let identifierPrefix = "nitida.enhance"

    private let lock = NSLock()
    private var legacyID: UIBackgroundTaskIdentifier = .invalid
    private var systemTask: AnyObject?
    private var fraction: Double = 0
    private var finished = false
    private var succeeded = false

    @MainActor
    func start() {
        legacyID = UIApplication.shared.beginBackgroundTask(withName: "Mejorar foto") { [weak self] in
            self?.endLegacy()
        }
        #if compiler(>=6.2)
        if #available(iOS 26.0, *) { startContinued() }
        #endif
    }

    /// Thread-safe; call from the processing thread.
    func update(_ value: Double) {
        lock.lock()
        fraction = value
        let task = systemTask
        lock.unlock()
        #if compiler(>=6.2)
        if #available(iOS 26.0, *), let task = task as? BGContinuedProcessingTask {
            task.progress.completedUnitCount = Int64(value * 1000)
        }
        #endif
    }

    @MainActor
    func finish(success: Bool) {
        lock.lock()
        finished = true
        succeeded = success
        let task = systemTask
        systemTask = nil
        lock.unlock()
        #if compiler(>=6.2)
        if #available(iOS 26.0, *), let task = task as? BGContinuedProcessingTask {
            task.progress.completedUnitCount = 1000
            task.setTaskCompleted(success: success)
        }
        #endif
        endLegacy()
    }

    private func endLegacy() {
        DispatchQueue.main.async {
            if self.legacyID != .invalid {
                UIApplication.shared.endBackgroundTask(self.legacyID)
                self.legacyID = .invalid
            }
        }
    }

    #if compiler(>=6.2)
    @available(iOS 26.0, *)
    @MainActor
    private func startContinued() {
        // Each run needs its own identifier: a handler can only be registered once per identifier.
        let identifier = "\(Self.identifierPrefix).\(UUID().uuidString)"
        let registered = BGTaskScheduler.shared.register(forTaskWithIdentifier: identifier, using: nil) { [weak self] task in
            guard let task = task as? BGContinuedProcessingTask else {
                task.setTaskCompleted(success: false)
                return
            }
            task.progress.totalUnitCount = 1000
            task.expirationHandler = { [weak self] in
                // The system needs the time back; the work will resume when the app is reopened.
                self?.lock.lock()
                self?.systemTask = nil
                self?.lock.unlock()
                task.setTaskCompleted(success: false)
            }
            guard let self else { task.setTaskCompleted(success: false); return }
            self.lock.lock()
            if self.finished {
                // The run ended before the system started the task.
                let ok = self.succeeded
                self.lock.unlock()
                task.progress.completedUnitCount = 1000
                task.setTaskCompleted(success: ok)
                return
            }
            self.systemTask = task
            task.progress.completedUnitCount = Int64(self.fraction * 1000)
            self.lock.unlock()
        }
        guard registered else { return }
        let request = BGContinuedProcessingTaskRequest(identifier: identifier,
                                                       title: "Nítida",
                                                       subtitle: "Mejorando tu foto…")
        request.strategy = .fail
        try? BGTaskScheduler.shared.submit(request)
    }
    #endif
}
