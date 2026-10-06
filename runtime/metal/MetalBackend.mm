// Modified by meowkernels.
#import "MetalBackend.hpp"
#include "AwakeClock.hpp"
#include "CommandWatchdog.hpp"
#include "EnvSwitch.hpp"
#include "Residency.hpp"
#ifdef SPLASH_BACKEND_INSTRUMENTATION
#include "BackendInstrumentation.hpp"
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <IOKit/IOKitLib.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <unistd.h>

namespace splash::metal {
namespace {

// The accelerator entry that backs a Metal device publishes gpu-core-count.
// The device's registry ID names that entry or a child of it; the first
// IOAccelerator service is the fallback, since Apple silicon Macs have one
// GPU. Zero means the property was not found anywhere.
uint32_t gpuCoreCountForDevice(uint64_t registryId) noexcept {
    uint32_t count = 0;
    const auto read = [&](io_registry_entry_t entry) {
        if (!entry) return false;
        CFTypeRef value = IORegistryEntryCreateCFProperty(
            entry, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (value) {
            int64_t number = 0;
            if (CFGetTypeID(value) == CFNumberGetTypeID() &&
                CFNumberGetValue(static_cast<CFNumberRef>(value),
                                 kCFNumberSInt64Type, &number) &&
                number > 0 && number <= 4096) {
                count = static_cast<uint32_t>(number);
            }
            CFRelease(value);
        }
        return count != 0;
    };
    io_registry_entry_t entry = IOServiceGetMatchingService(
        kIOMainPortDefault, IORegistryEntryIDMatching(registryId));
    for (int depth = 0; entry && depth < 4 && !read(entry); ++depth) {
        io_registry_entry_t parent = MACH_PORT_NULL;
        if (IORegistryEntryGetParentEntry(entry, kIOServicePlane, &parent) !=
            KERN_SUCCESS) {
            parent = MACH_PORT_NULL;
        }
        IOObjectRelease(entry);
        entry = parent;
    }
    if (entry) IOObjectRelease(entry);
    if (!count) {
        io_registry_entry_t accelerator = IOServiceGetMatchingService(
            kIOMainPortDefault, IOServiceMatching("IOAccelerator"));
        if (accelerator) {
            read(accelerator);
            IOObjectRelease(accelerator);
        }
    }
    return count;
}

std::string stringFromNSString(NSString *value) {
    if (!value) return {};
    const char *utf8 = value.UTF8String;
    return utf8 ? utf8 : "";
}

std::string errorDescription(NSError *error) {
    if (!error) return "unknown Metal error";
    std::string result = stringFromNSString(error.localizedDescription);
    return result.empty() ? "unknown Metal error" : result;
}

void readMacosVersion(DeviceCapabilities &capabilities) {
    const NSOperatingSystemVersion os =
        NSProcessInfo.processInfo.operatingSystemVersion;
    const auto component = [](NSInteger value) {
        return value > 0 ? static_cast<uint32_t>(value) : 0U;
    };
    capabilities.macosMajor = component(os.majorVersion);
    capabilities.macosMinor = component(os.minorVersion);
    capabilities.macosPatch = component(os.patchVersion);
}

// The backend and probeDeviceCapabilities() share one reading of the device,
// so the probe judges a Mac by the values the engine validates.
void readDeviceCapabilities(id<MTLDevice> device,
                            DeviceCapabilities &capabilities) {
    capabilities.deviceName = stringFromNSString(device.name);
    capabilities.gpuCoreCount = gpuCoreCountForDevice(device.registryID);
    // Apple GPU families nest, so the device's is the last one supported
    // counting up from Apple7.
    uint32_t family = 0;
    for (uint32_t next = 7;
         [device supportsFamily:static_cast<MTLGPUFamily>(1000 + next)]; ++next)
        family = next;
    capabilities.appleGpuFamily = family;
    capabilities.physicalMemoryBytes = NSProcessInfo.processInfo.physicalMemory;
    capabilities.recommendedMaxWorkingSetBytes =
        device.recommendedMaxWorkingSetSize;
    capabilities.maxBufferLengthBytes = device.maxBufferLength;
    capabilities.maxThreadgroupMemoryBytes = device.maxThreadgroupMemoryLength;
    MTLSize maximumThreads = device.maxThreadsPerThreadgroup;
    capabilities.maxThreadgroupWidth = maximumThreads.width;
    capabilities.hasUnifiedMemory = device.hasUnifiedMemory;
}

// Every uint64_t size, offset and length passes to Metal as it is.
static_assert(sizeof(NSUInteger) == sizeof(uint64_t),
              "Splash builds for arm64 only");

MTLSize metalSize(const DispatchSize &size, std::string_view field) {
    if (!size.x || !size.y || !size.z) {
        throw MetalBackendError(std::string(field) + " must be non-zero");
    }
    return MTLSizeMake(size.x, size.y, size.z);
}

bool multiplyOverflows(uint64_t left, uint64_t right) {
    return right && left > std::numeric_limits<uint64_t>::max() / right;
}

// Entries of a kernel's buffer argument table on every Apple GPU family.
constexpr uint32_t kBufferArgumentEntries = 31;
// How long a ticket waits for its command before it asks the watchdog.
constexpr auto kTicketWaitSlice = std::chrono::seconds(1);
// The queue's outstanding command buffers; a command takes one per event
// signal and one more.
constexpr NSUInteger kMaximumCommandBuffers = 512;

// Refuses the event steps of a command of `dispatches` dispatches that its
// submission cannot encode.
void checkEvents(std::span<const EventStep> events, size_t dispatches) {
    size_t before = 0;
    NSUInteger signals = 0;
    for (const EventStep &step : events) {
        if (!step.event || !step.value)
            throw MetalBackendError(
                "event step needs an event and a nonzero value");
        if (step.before < before)
            throw MetalBackendError("event steps are out of dispatch order");
        if (step.before > dispatches)
            throw MetalBackendError(
                "event step follows more dispatches than its command has");
        before = step.before;
        if (step.kind == EventStep::Kind::Signal) ++signals;
    }
    if (signals >= kMaximumCommandBuffers)
        throw MetalBackendError("Metal command signals more events than its "
                                "queue holds command buffers");
}

double awakeSeconds() noexcept {
    return std::chrono::duration<double>(AwakeClock::now().time_since_epoch()).count();
}

const char *commandStatusName(MTLCommandBufferStatus status) noexcept {
    switch (status) {
    case MTLCommandBufferStatusNotEnqueued: return "not_enqueued";
    case MTLCommandBufferStatusEnqueued: return "enqueued";
    case MTLCommandBufferStatusCommitted: return "committed";
    case MTLCommandBufferStatusScheduled: return "scheduled";
    case MTLCommandBufferStatusCompleted: return "completed";
    case MTLCommandBufferStatusError: return "error";
    }
    return "unknown";
}

template <typename T>
void raisePeak(std::atomic<T> &peak, T value) noexcept {
    T current = peak.load(std::memory_order_relaxed);
    while (value > current &&
           !peak.compare_exchange_weak(current, value,
                                       std::memory_order_relaxed)) {}
}

// Hashes pipeline names as views, so a cache lookup builds no string.
struct PipelineNameHash {
    using is_transparent = void;
    size_t operator()(std::string_view name) const noexcept {
        return std::hash<std::string_view>{}(name);
    }
};

NSString *checkedNSString(std::string_view value, std::string_view field) {
    NSString *result = [[NSString alloc]
        initWithBytes:value.data()
        length:value.size()
        encoding:NSUTF8StringEncoding];
    if (!result) {
        throw MetalBackendError(std::string(field) + " is not UTF-8");
    }
    return result;
}

}  // namespace

struct AllocationAccounting {
    std::atomic<uint64_t> allocatedBytes{0};
    std::atomic<uint64_t> peakAllocatedBytes{0};
};

struct MetalAllocation {
    // Nil while the memory is released (MetalBackend::releaseMemory).
    __strong id<MTLBuffer> buffer = nil;
    std::shared_ptr<AllocationAccounting> accounting;
    // What the buffer adds to the accounting: its allocated size, or zero
    // while released.
    uint64_t bytes = 0;
    // The length, storage and label a restored buffer is allocated with.
    uint64_t length = 0;
    BufferStorage storage = BufferStorage::Shared;
    __strong NSString *label = nil;
    // The owner of wrapped memory (MetalBackend::wrapSharedMemory), which is
    // never released: our views keep it as well as Metal's deallocator.
    std::shared_ptr<void> owner;
    // The residency set the buffer belongs to, held weakly as allocations
    // may outlive the backend. The set retains the buffer, and with it its
    // memory, so the last view takes it out.
    std::weak_ptr<Residency> residency;

    // Takes the buffer, which joins the residency set and the accounting.
    void attach(id<MTLBuffer> allocated) {
        buffer = allocated;
        bytes = allocated.allocatedSize;
        if (auto kept = residency.lock()) kept->add(buffer);
        raisePeak(accounting->peakAllocatedBytes,
                  accounting->allocatedBytes.fetch_add(
                      bytes, std::memory_order_relaxed) + bytes);
    }
    // Lets the buffer go: it leaves the residency set and the accounting.
    void detach() noexcept {
        if (!buffer) return;
        if (auto kept = residency.lock()) kept->remove(buffer);
        accounting->allocatedBytes.fetch_sub(bytes, std::memory_order_relaxed);
        buffer = nil;
        bytes = 0;
    }

    ~MetalAllocation() { detach(); }
};

// Every Impl has an allocation, whose buffer is nil only while its memory is
// released: only allocateBuffer, wrapSharedMemory and view create one.
struct MetalBuffer::Impl {
    std::shared_ptr<MetalAllocation> allocation;
    uint64_t offsetBytes = 0;
    uint64_t lengthBytes = 0;
};

struct SharedEvent::Impl {
    __strong id<MTLSharedEvent> event = nil;
    // The listener of the backend that created the event, which runs its
    // notify() callbacks; the backend and each of its events hold it.
    __strong MTLSharedEventListener *listener = nil;
};

struct BackendAsyncState {
    explicit BackendAsyncState(double commandTimeoutSeconds)
        : commandWatchdog(commandTimeoutSeconds) {}

    __strong id<MTLDevice> device = nil;
    mutable std::atomic<uint64_t> deviceCurrentAllocatedBytes{0};
    mutable std::atomic<uint64_t> devicePeakAllocatedBytes{0};
    std::atomic<bool> healthy{true};
    mutable std::mutex healthMutex;
    std::string healthReason;
    mutable std::mutex gateMutex;
    uint64_t nextSequence = 0;
    uint64_t activeSequence = 0;
    size_t activeDispatchCount = 0;
    __weak id<MTLCommandBuffer> activeCommand = nil;
    std::function<void(id<MTLCommandBuffer>)> activeCompletion;
    CommandWatchdog commandWatchdog;
    bool stopping = false;
    // MetalBackend::setWaitInterrupt's predicate.
    std::function<bool()> waitInterrupt;

    void sampleDeviceMemory() const noexcept {
        if (!device) return;
        uint64_t current = static_cast<uint64_t>(device.currentAllocatedSize);
        deviceCurrentAllocatedBytes.store(current, std::memory_order_relaxed);
        raisePeak(devicePeakAllocatedBytes, current);
    }

    [[noreturn]] void throwUnhealthy() const {
        std::lock_guard lock(healthMutex);
        throw MetalBackendError("Metal backend is unhealthy: " + healthReason);
    }

    void ensureHealthy() const {
        if (!healthy.load(std::memory_order_acquire)) throwUnhealthy();
    }

    void markUnhealthy(std::string reason) {
        {
            std::lock_guard lock(healthMutex);
            if (healthReason.empty()) healthReason = std::move(reason);
        }
        healthy.store(false, std::memory_order_release);
    }

    uint64_t beginSubmission(size_t dispatchCount) {
        ensureHealthy();
        std::lock_guard lock(gateMutex);
        if (stopping)
            throw MetalBackendError("Metal backend is stopping");
        if (activeSequence) {
            throw MetalBackendError(
                "Metal backend already has an in-flight command");
        }
        activeSequence = ++nextSequence;
        activeDispatchCount = dispatchCount;
        return activeSequence;
    }

    // `leading` excludes buffers already committed (fastkernel streamed heads).
    void commitSubmission(uint64_t sequence,
                          std::span<const id<MTLCommandBuffer>> leading,
                          id<MTLCommandBuffer> command,
                          std::function<void(id<MTLCommandBuffer>)> completion) {
        std::lock_guard lock(gateMutex);
        activeCommand = command;
        activeCompletion = std::move(completion);
        commandWatchdog.start(sequence, awakeSeconds());
        for (id<MTLCommandBuffer> earlier : leading) [earlier commit];
        [command commit];
    }

    void releaseSubmission(uint64_t sequence) noexcept {
        std::lock_guard lock(gateMutex);
        commandWatchdog.complete(sequence);
        if (activeSequence == sequence) {
            activeSequence = 0;
            activeCommand = nil;
            activeCompletion = {};
        }
    }

    void completeSubmission(uint64_t sequence) noexcept {
        std::lock_guard lock(gateMutex);
        commandWatchdog.complete(sequence);
    }

    // Runs the command watchdog. A terminal command whose callback is late
    // is completed here; one still running past its timeout marks the
    // backend unhealthy, and the answer is then true: the backend gave up on
    // that command.
    [[nodiscard]] bool commandAbandoned() noexcept {
        id<MTLCommandBuffer> command = nil;
        std::function<void(id<MTLCommandBuffer>)> complete;
        {
            std::lock_guard lock(gateMutex);
            if (!commandWatchdog.expired(awakeSeconds())) return false;
            command = activeCommand;
            const auto status = command ? command.status
                                        : MTLCommandBufferStatusNotEnqueued;
            // Recover terminal results even if the driver has not delivered
            // its callback. Finish outside the gate: it takes the ticket lock.
            if (command && (status == MTLCommandBufferStatusCompleted ||
                            status == MTLCommandBufferStatusError)) {
                complete = activeCompletion;
            } else {
                // Waits that must not throw run this too: the reason drops
                // its details when they cannot be formatted.
                std::string reason = "Metal command completion timed out";
                try {
                    std::ostringstream message;
                    message << reason << " after "
                            << commandWatchdog.timeoutSeconds()
                            << " seconds (sequence=" << activeSequence
                            << ", status=" << (command ? commandStatusName(status)
                                                       : "unavailable")
                            << ", dispatches=" << activeDispatchCount << ')';
                    reason = message.str();
                } catch (const std::bad_alloc &) {
                }
                markUnhealthy(std::move(reason));
                return true;
            }
        }
        if (complete) complete(command);
        return false;
    }

    void checkCommandHealth() {
        static_cast<void>(commandAbandoned());
        ensureHealthy();
    }

    // True when the process is shutting down; the waiter decides whether
    // that gives its command up.
    [[nodiscard]] bool waitInterrupted() const noexcept {
        return waitInterrupt && waitInterrupt();
    }

    [[nodiscard]] bool hasActiveSubmission() const noexcept {
        std::lock_guard lock(gateMutex);
        return activeSequence != 0;
    }
};

struct CommandTicket::State {
    std::shared_ptr<BackendAsyncState> backend;
    std::vector<std::shared_ptr<MetalAllocation>> retainedAllocations;
    CommandCompletion completion;
    mutable std::mutex mutex;
    std::condition_variable condition;
    uint64_t sequence = 0;
    CommandTiming timing;
    AwakeClock::time_point wallStart;
    // Command buffers committed before the last one, split at event signals.
    std::vector<id<MTLCommandBuffer>> leadingCommands;
    std::string error;
    bool completed = false;
    bool released = false;

    void finishCommand(id<MTLCommandBuffer> command) {
        auto wallEnd = AwakeClock::now();
        CommandTiming timing;
        const double gpuStart = leadingCommands.empty()
            ? command.GPUStartTime : leadingCommands.front().GPUStartTime;
        timing.gpuSeconds = command.GPUEndTime - gpuStart;
        if (!std::isfinite(timing.gpuSeconds) || timing.gpuSeconds < 0.0) {
            timing.gpuSeconds = 0.0;
        }
        timing.wallSeconds =
            std::chrono::duration<double>(wallEnd - wallStart).count();
        // fastkernel diagnostic, default off: SPLASH_GPU_GAP_LOG=1 prints each
        // command's GPU span, from its first buffer's start to its last
        // buffer's end; the GPU idles between one line's end and the next's
        // start.
        static const bool gapLog = std::getenv("SPLASH_GPU_GAP_LOG") != nullptr;
        if (gapLog)
            std::fprintf(stderr, "gpu_cmd %llu %.9f %.9f %.9f %zu\n",
                         static_cast<unsigned long long>(sequence), gpuStart,
                         command.GPUEndTime, timing.wallSeconds,
                         leadingCommands.size() + 1);

        std::string error;
        // The first buffer that failed names the error: the buffers after a
        // failed one still run, as its signal is delivered (EventStep).
        id<MTLCommandBuffer> failed =
            command.status != MTLCommandBufferStatusCompleted ? command : nil;
        for (id<MTLCommandBuffer> earlier : leadingCommands) {
            if (earlier.status == MTLCommandBufferStatusError) {
                failed = earlier;
                break;
            }
        }
        if (failed) {
            std::ostringstream message;
            message << "Metal command " << sequence << " failed";
            if (failed.error) {
                message << ": " << errorDescription(failed.error);
            }
            error = message.str();
        }

        finish(timing, std::move(error));
    }

    void finish(CommandTiming result, std::string failure) {
        CommandCompletion notify;
        {
            std::lock_guard lock(mutex);
            // Host recovery, late callbacks, and discarded commands all share
            // this completion path; only the first result may publish or notify.
            if (completed) return;
            backend->completeSubmission(sequence);
            if (!failure.empty()) backend->markUnhealthy(failure);
            timing = result;
            error = std::move(failure);
            completed = true;
            notify = completion;
        }
        if (notify) {
            try {
                notify();
            } catch (...) {
                backend->markUnhealthy(
                    "Metal completion callback threw an exception");
            }
        }
        condition.notify_all();
    }

    void release() noexcept {
        bool shouldRelease = false;
        {
            std::lock_guard lock(mutex);
            if (!released) {
                released = true;
                retainedAllocations.clear();
                shouldRelease = true;
            }
        }
        if (shouldRelease && backend) {
            // Refresh the cached device telemetry, which status reports, on
            // the consuming thread after GPU completion; admission samples
            // its own (refreshMemoryStats).
            if (backend->healthy.load(std::memory_order_acquire))
                backend->sampleDeviceMemory();
            backend->releaseSubmission(sequence);
        }
    }

    // Waits for the command in kTicketWaitSlice slices. Between them, outside
    // `mutex` (the watchdog may finish this ticket through finishCommand), it
    // asks the backend whether to stop, and with honorShutdown also whether
    // the process is shutting down. False when the backend gave up on a
    // command that never completed: the GPU may still use the retained
    // allocations, which the command's completion handler keeps alive with
    // this state.
    [[nodiscard]] bool awaitCompletion(bool honorShutdown) noexcept {
        std::unique_lock lock(mutex);
        while (!condition.wait_until(lock, AwakeClock::now() + kTicketWaitSlice,
                                     [this] { return completed; })) {
            lock.unlock();
            const bool abandoned = backend->commandAbandoned();
            const bool interrupted =
                !abandoned && honorShutdown && backend->waitInterrupted();
            lock.lock();
            if (completed) break;
            // A shutdown gives the command up only here, where the lock shows
            // it unfinished: one that completed meanwhile leaves the backend
            // healthy.
            if (interrupted) backend->markUnhealthy(shutdownReason());
            if (abandoned || interrupted) return false;
        }
        return true;
    }

    // Why the backend is unhealthy once a shutdown gave this command up.
    [[nodiscard]] std::string shutdownReason() const noexcept {
        std::string reason = "shutdown requested while waiting for a Metal command";
        try {
            reason = "shutdown requested while waiting for Metal command " +
                     std::to_string(sequence);
        } catch (const std::bad_alloc &) {
        }
        return reason;
    }

    // An abandoned command keeps its allocations until its completion
    // handler lets go of this state; the unhealthy backend admits no more.
    // A shutdown does not give up this wait: teardown waits for the command,
    // as long as the watchdog lets it.
    void abandon() noexcept {
        if (awaitCompletion(false)) release();
    }
};

struct MetalBackend::Impl {
    explicit Impl(double commandTimeoutSeconds)
        : asyncState(std::make_shared<BackendAsyncState>(commandTimeoutSeconds)) {}

    // One dispatch of a command, validated and resolved to its pipeline.
    struct PreparedDispatch {
        const ComputeDispatch *source = nullptr;
        MTLSize groups{};
        MTLSize threads{};
        uint64_t threadCount = 0;
        // The argument table entries its buffers take, one bit each.
        uint32_t bufferIndices = 0;
        __strong id<MTLComputePipelineState> pipeline = nil;
    };

    // A command as submission encodes it: its dispatches and every
    // allocation they bind, which its ticket retains.
    struct PreparedCommand {
        std::vector<PreparedDispatch> dispatches;
        std::vector<std::shared_ptr<MetalAllocation>> retainedAllocations;
    };

    std::function<void()> operationGuard;

#ifdef SPLASH_BACKEND_INSTRUMENTATION
    bool dispatchProfiling = false;
    std::vector<DispatchTiming> dispatchProfile;
#endif
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    // Runs the notify() callbacks of the backend's shared events, on a
    // serial dispatch queue.
    __strong MTLSharedEventListener *eventListener = nil;
    // Allocations hold it weakly: they may outlive the backend.
    std::shared_ptr<Residency> residency;
    // SPLASH_DRAFT_AHEAD: the last trailing command buffer and its completion
    // signal (submitting thread only).
    __strong id<MTLCommandBuffer> trailing = nil;
    __strong dispatch_semaphore_t trailingDone = nil;
    __strong id<MTLLibrary> library = nil;
    // Looked up for every dispatch when its command is prepared, which the
    // GPU may be waiting for; a hit allocates nothing. Used only by the
    // submitting thread.
    std::unordered_map<std::string, id<MTLComputePipelineState>,
                       PipelineNameHash, std::equal_to<>>
        pipelines;

    DeviceCapabilities capabilities;
    std::shared_ptr<AllocationAccounting> accounting =
        std::make_shared<AllocationAccounting>();
    std::shared_ptr<BackendAsyncState> asyncState;

    void sampleDeviceMemory() const noexcept {
        asyncState->sampleDeviceMemory();
    }

    void ensureHealthy() const {
        asyncState->ensureHealthy();
    }

    void markUnhealthy(std::string reason) {
        asyncState->markUnhealthy(std::move(reason));
    }

    id<MTLBuffer> newBuffer(uint64_t bytes, BufferStorage storage,
                            NSString *label) {
        MTLResourceOptions options = storage == BufferStorage::Shared
            ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
        id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                                   options:options];
        if (!buffer)
            throw MetalAllocationError("Metal buffer allocation failed");
        if (label) buffer.label = label;
        return buffer;
    }

    // The base allocation of a buffer, which must be a whole buffer of this
    // backend.
    MetalAllocation &baseAllocation(const MetalBuffer &buffer) const {
        if (!buffer.impl_ ||
            buffer.impl_->allocation->accounting.get() != accounting.get())
            throw MetalBackendError("Metal buffer is not a buffer of this backend");
        MetalAllocation &allocation = *buffer.impl_->allocation;
        if (buffer.impl_->offsetBytes ||
            buffer.impl_->lengthBytes != allocation.length)
            throw MetalBackendError("a view's memory is its base buffer's");
        return allocation;
    }

    id<MTLComputePipelineState> pipeline(std::string_view name) {
        if (name.empty()) {
            throw MetalBackendError("Metal pipeline name must not be empty");
        }
        if (const auto cached = pipelines.find(name); cached != pipelines.end())
            return cached->second;
        id<MTLComputePipelineState> result = newPipeline(name);
        pipelines.emplace(name, result);
        sampleDeviceMemory();
        return result;
    }

    id<MTLComputePipelineState> newPipeline(std::string_view name) {
        NSString *key = checkedNSString(name, "pipeline name");
        id<MTLFunction> function = [library newFunctionWithName:key];
        if (!function) {
            throw MetalBackendError(
                "missing Metal function: " + std::string(name));
        }
        NSError *error = nil;
        id<MTLComputePipelineState> result =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!result) {
            throw MetalBackendError(
                "unable to create Metal pipeline " + std::string(name) +
                ": " + errorDescription(error));
        }
        return result;
    }

    PreparedCommand prepare(std::span<const ComputeDispatch> dispatches) {
        if (dispatches.empty()) {
            throw MetalBackendError("Metal command must contain a dispatch");
        }
        PreparedCommand command;
        command.dispatches.reserve(dispatches.size());
        size_t bindings = 0;
        for (const ComputeDispatch &dispatch : dispatches) {
            PreparedDispatch item;
            item.source = &dispatch;
            item.groups = metalSize(dispatch.threadgroups, "threadgroups");
            item.threads = metalSize(
                dispatch.threadsPerThreadgroup, "threadsPerThreadgroup");
            if (multiplyOverflows(dispatch.threadsPerThreadgroup.x,
                                  dispatch.threadsPerThreadgroup.y) ||
                multiplyOverflows(dispatch.threadsPerThreadgroup.x *
                                      dispatch.threadsPerThreadgroup.y,
                                  dispatch.threadsPerThreadgroup.z)) {
                throw MetalBackendError("threadsPerThreadgroup size overflows");
            }
            item.threadCount = dispatch.threadsPerThreadgroup.x *
                dispatch.threadsPerThreadgroup.y *
                dispatch.threadsPerThreadgroup.z;

            // Each binding takes its own entry of the argument table.
            uint32_t indices = 0;
            const auto claim = [&](uint32_t index) {
                if (index >= kBufferArgumentEntries) {
                    throw MetalBackendError(
                        "compute binding index exceeds the argument table");
                }
                if (indices & (uint32_t{1} << index)) {
                    throw MetalBackendError("duplicate compute binding index");
                }
                indices |= uint32_t{1} << index;
            };
            for (const BufferBinding &binding : dispatch.buffers) {
                if (!binding.buffer.impl_) {
                    std::ostringstream message;
                    message << "compute dispatch '" << dispatch.pipelineName
                            << "' contains an empty buffer at index "
                            << binding.index;
                    throw MetalBackendError(message.str());
                }
                if (binding.buffer.impl_->allocation->accounting.get() !=
                    accounting.get()) {
                    throw MetalBackendError(
                        "compute dispatch buffer belongs to another backend");
                }
                if (!binding.buffer.impl_->allocation->buffer) {
                    std::ostringstream message;
                    message << "compute dispatch '" << dispatch.pipelineName
                            << "' binds released memory at index "
                            << binding.index;
                    throw MetalBackendError(message.str());
                }
                claim(binding.index);
                item.bufferIndices |= uint32_t{1} << binding.index;
            }
            for (const BytesBinding &binding : dispatch.bytes) {
                if (!binding.data || !binding.sizeBytes) {
                    throw MetalBackendError("compute byte binding is empty");
                }
                claim(binding.index);
            }
            bindings += dispatch.buffers.size();
            command.dispatches.push_back(item);
        }

        // fastkernel, exact scheduling, read at every submission so one
        // binary serves both arms of an in-process A/B: the two M8 N32
        // split-K entries run as their twins. SPLASH_SPLIT4_FOOTER (default
        // on, _ftr) skips each threadgroup's last-tile scratch-reuse barrier
        // (lockstep -0.154 ms/step); SPLASH_SPLIT4_HOIST (default on, _hoist)
        // issues each matmul pair's metadata loads first, with the stock FMA
        // order (lockstep -0.312 ms/step). Byte identity was shown with this
        // Mac's GPU compiler (fastkernel 1.0.0); re-check per chip.
        const bool footer = envSwitch("SPLASH_SPLIT4_FOOTER");
        const bool hoist = envSwitch("SPLASH_SPLIT4_HOIST");
        const std::string_view split4Suffix = hoist ? (footer ? "_hoist_ftr" : "_hoist") : "_ftr";
        for (PreparedDispatch &item : command.dispatches) {
            const std::string &name = item.source->pipelineName;
            item.pipeline = (footer || hoist) &&
                    (name == "decode_linear_q4_n32_split4_precomputed_sums" ||
                     name == "decode_linear_q4_n32_split4_precomputed_sums_residual")
                ? pipeline(name + std::string(split4Suffix))
                : pipeline(name);
            if (item.threadCount >
                item.pipeline.maxTotalThreadsPerThreadgroup) {
                throw MetalBackendError(
                    "threadsPerThreadgroup exceeds pipeline capability");
            }
        }

        // Each allocation the command binds, once.
        std::vector<const std::shared_ptr<MetalAllocation> *> bound;
        bound.reserve(bindings);
        for (const ComputeDispatch &dispatch : dispatches) {
            for (const BufferBinding &binding : dispatch.buffers)
                bound.push_back(&binding.buffer.impl_->allocation);
        }
        const auto allocation =
            [](const std::shared_ptr<MetalAllocation> *owner) {
                return owner->get();
            };
        std::ranges::sort(bound, {}, allocation);
        const auto repeated = std::ranges::unique(bound, {}, allocation);
        bound.erase(repeated.begin(), repeated.end());
        command.retainedAllocations.reserve(bound.size());
        for (const std::shared_ptr<MetalAllocation> *owner : bound)
            command.retainedAllocations.push_back(*owner);
        return command;
    }

    // fastkernel SPLASH_STREAMED_SUBMIT / SPLASH_CHUNKED_SUBMIT: the first
    // Metal command buffer of the next command, committed before the rest
    // was encoded (streamHead). Its ticket is the command's; the rest waits
    // for it through headFence.
    struct StreamedHead {
        std::shared_ptr<CommandTicket::State> ticket;
        __strong id<MTLCommandBuffer> command = nil;
        size_t dispatches = 0;
        size_t events = 0;
    };
    // Used only by the submitting thread, like `pipelines`.
    std::optional<StreamedHead> streamed;
    __strong id<MTLFence> headFence = nil;

    // Binds and dispatches one prepared dispatch.
    static void encodeDispatch(id<MTLComputeCommandEncoder> encoder,
                               const PreparedDispatch &item) {
        const ComputeDispatch &dispatch = *item.source;
        // Indexed by argument table entry. The ticket and the dispatches
        // keep the buffers alive.
        __unsafe_unretained id<MTLBuffer> buffers[kBufferArgumentEntries];
        NSUInteger offsets[kBufferArgumentEntries];
        [encoder setComputePipelineState:item.pipeline];
        for (const BufferBinding &binding : dispatch.buffers) {
            const MetalBuffer::Impl &buffer = *binding.buffer.impl_;
            buffers[binding.index] = buffer.allocation->buffer;
            offsets[binding.index] = buffer.offsetBytes;
        }
        // One call per run of consecutive entries: a command graph's
        // dispatch binds a single run.
        for (uint32_t unbound = item.bufferIndices; unbound;) {
            const uint32_t first = std::countr_zero(unbound);
            const uint32_t count = std::countr_one(unbound >> first);
            [encoder setBuffers:buffers + first
                        offsets:offsets + first
                      withRange:NSMakeRange(first, count)];
            unbound &= ~(((uint32_t{1} << count) - 1) << first);
        }
        for (const BytesBinding &binding : dispatch.bytes) {
            [encoder setBytes:binding.data
                       length:binding.sizeBytes
                      atIndex:binding.index];
        }
        [encoder dispatchThreadgroups:item.groups
                 threadsPerThreadgroup:item.threads];
    }

    // Encodes a prepared head and the event steps after it into one Metal
    // command buffer and commits it at once; the ticket retains `retained`.
    StreamedHead commitHead(std::span<const PreparedDispatch> dispatches,
                            std::span<const EventStep> events,
                            std::vector<std::shared_ptr<MetalAllocation>> retained) {
        if (!headFence) {
            headFence = [device newFence];
            if (!headFence) throw MetalBackendError("unable to create Metal fence");
        }
        StreamedHead head;
        head.ticket = std::make_shared<CommandTicket::State>();
        head.ticket->backend = asyncState;
        head.ticket->retainedAllocations = std::move(retained);
        head.ticket->sequence = asyncState->beginSubmission(dispatches.size());
        head.ticket->wallStart = AwakeClock::now();
        head.dispatches = dispatches.size();
        head.events = events.size();
        const auto fail = [&](std::string message) {
            markUnhealthy(message);
            asyncState->releaseSubmission(head.ticket->sequence);
            throw MetalBackendError(std::move(message));
        };
        @autoreleasepool {
            head.command = [queue commandBuffer];
            if (!head.command) fail("unable to create Metal command buffer");
            id<MTLComputeCommandEncoder> encoder = [head.command computeCommandEncoder];
            if (!encoder) fail("unable to create Metal compute encoder");
            for (const PreparedDispatch &item : dispatches) encodeDispatch(encoder, item);
            [encoder updateFence:headFence];
            [encoder endEncoding];
            for (const EventStep &step : events) {
                id<MTLSharedEvent> event =
                    (__bridge id<MTLSharedEvent>)step.event.nativeHandle();
                if (step.kind == EventStep::Kind::Wait) {
                    [head.command encodeWaitForEvent:event value:step.value];
                    continue;
                }
                [head.command encodeSignalEvent:event value:step.value];
                // As commit() does: a failed head still delivers its signal.
                const SharedEvent signaled = step.event;
                const uint64_t value = step.value;
                [head.command addCompletedHandler:^(id<MTLCommandBuffer> ended) {
                    if (ended.status == MTLCommandBufferStatusError)
                        signaled.signal(value);
                }];
            }
            residency->use();
            [head.command commit];
        }
        return head;
    }

    // Waits (bounded by the command watchdog) for a head whose submission
    // will not come, then ends its submission.
    void abandonHead(StreamedHead &head) noexcept {
        const double deadline =
            awakeSeconds() + asyncState->commandWatchdog.timeoutSeconds();
        while (head.command.status < MTLCommandBufferStatusCompleted &&
               awakeSeconds() < deadline)
            usleep(100);
        if (head.command.status < MTLCommandBufferStatusCompleted)
            markUnhealthy("a streamed head did not finish after its command was abandoned");
        asyncState->releaseSubmission(head.ticket->sequence);
    }

    // SPLASH_DRAFT_AHEAD: one more command buffer behind the submission just
    // committed, outside the one-in-flight gate. Every buffer is
    // hazard-tracked, so Metal orders it after that submission's writes and
    // the next submission after its own. Its completion handler holds its
    // allocations and signals `trailingDone`.
    // Invariant: it binds device-allocated (tracked) buffers only; memory
    // reached by address (KV page extents) is untracked and needs fences.
    void commitTrailing(const PreparedCommand &prepared) {
        @autoreleasepool {
            id<MTLCommandBuffer> trail = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = trail ? [trail computeCommandEncoder] : nil;
            if (!encoder) {
                markUnhealthy("unable to create the Metal trailing command");
                return;
            }
            for (const PreparedDispatch &item : prepared.dispatches)
                encodeDispatch(encoder, item);
            [encoder endEncoding];
            auto retained = std::make_shared<std::vector<std::shared_ptr<MetalAllocation>>>(
                prepared.retainedAllocations);
            dispatch_semaphore_t done = dispatch_semaphore_create(0);
            std::weak_ptr<BackendAsyncState> state = asyncState;
            [trail addCompletedHandler:^(id<MTLCommandBuffer> ended) {
                retained->clear();
                if (ended.status != MTLCommandBufferStatusCompleted)
                    if (auto alive = state.lock())
                        alive->markUnhealthy("Metal trailing command failed");
                dispatch_semaphore_signal(done);
            }];
            residency->use();
            [trail commit];
            trailing = trail;
            trailingDone = done;
        }
    }

    // Encodes and commits prepared dispatches and the event steps between
    // them; the ticket retains `retained` until it is consumed. With a
    // streamed `head`, `dispatches` and `events` are the command's rest (event
    // steps still count dispatches from the command's first).
    CommandTicket commit(std::span<const PreparedDispatch> dispatches,
                         std::span<const EventStep> events,
                         std::vector<std::shared_ptr<MetalAllocation>> retained,
                         CommandCompletion completion,
                         std::optional<StreamedHead> head = std::nullopt) {
        auto ticketState = head ? head->ticket : std::make_shared<CommandTicket::State>();
        ticketState->backend = asyncState;
        ticketState->completion = std::move(completion);
        if (head) {
            ticketState->retainedAllocations.insert(
                ticketState->retainedAllocations.end(),
                std::make_move_iterator(retained.begin()),
                std::make_move_iterator(retained.end()));
        } else {
            ticketState->retainedAllocations = std::move(retained);
            ticketState->sequence =
                asyncState->beginSubmission(dispatches.size());
        }
        const size_t first = head ? head->dispatches : 0;

        auto failBeforeCommit = [&](std::string message) {
            markUnhealthy(message);
            asyncState->releaseSubmission(ticketState->sequence);
            throw MetalBackendError(std::move(message));
        };

        auto wallStart = head ? ticketState->wallStart : AwakeClock::now();
        // Metal may autorelease the command and its encoder, and the serving
        // loop's pool never drains, so their temporary ownership ends with
        // this submission (under the validation layer an autoreleased
        // command holds every member of the residency set). The command
        // retains everything the GPU still needs.
        @autoreleasepool {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            if (!command) {
                failBeforeCommit("unable to create Metal command buffer");
            }
            ticketState->wallStart = wallStart;
            // The buffers before `command`, each ended by an event signal (a
            // streamed head first, already committed).
            std::vector<id<MTLCommandBuffer>> leading;
            if (head) leading.push_back(head->command);
            const size_t committed = leading.size();
            bool waitHead = head.has_value();
            id<MTLComputeCommandEncoder> encoder = nil;
            // Encodes the event steps that follow the first `encoded`
            // dispatches.
            auto step = events.begin();
            const auto encodeSteps = [&](size_t encoded) {
                for (; step != events.end() && step->before == first + encoded; ++step) {
                    if (encoder) {
                        [encoder endEncoding];
                        encoder = nil;
                    }
                    id<MTLSharedEvent> event =
                        (__bridge id<MTLSharedEvent>)step->event.nativeHandle();
                    if (step->kind == EventStep::Kind::Wait) {
                        [command encodeWaitForEvent:event value:step->value];
                        continue;
                    }
                    [command encodeSignalEvent:event value:step->value];
                    // A buffer that fails may end without its signal; the
                    // work after it, waiting on an agent that waits on the
                    // signal, would then stall until a timeout ends it. The
                    // CPU delivers the signal instead, so the command ends
                    // with the failure at once.
                    const SharedEvent signaled = step->event;
                    const uint64_t value = step->value;
                    [command addCompletedHandler:^(id<MTLCommandBuffer> ended) {
                        if (ended.status == MTLCommandBufferStatusError)
                            signaled.signal(value);
                    }];
                    leading.push_back(command);
                    command = [queue commandBuffer];
                    if (!command) {
                        failBeforeCommit("unable to create Metal command buffer");
                    }
                }
            };
            for (size_t index = 0; index < dispatches.size(); ++index) {
                encodeSteps(index);
                if (!encoder) {
                    encoder = [command computeCommandEncoder];
                    if (!encoder) {
                        failBeforeCommit("unable to create Metal compute encoder");
                    }
                    if (std::exchange(waitHead, false))
                        [encoder waitForFence:headFence];
                }
                encodeDispatch(encoder, dispatches[index]);
            }
            encodeSteps(dispatches.size());
            if (encoder) [encoder endEncoding];
            ticketState->leadingCommands = leading;

            // Driver callbacks only complete the ticket. Device-wide memory
            // telemetry is sampled on the host when consuming the result. The
            // handler holds the ticket's state strongly: once a waiter gives
            // up on the command, it keeps the retained allocations until the
            // GPU ends.
            [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                ticketState->finishCommand(completed);
            }];
            residency->use();
            asyncState->commitSubmission(ticketState->sequence,
                std::span(leading).subspan(committed), command,
                [weakTicket = std::weak_ptr(ticketState)](
                    id<MTLCommandBuffer> completed) {
                    if (auto ticket = weakTicket.lock())
                        ticket->finishCommand(completed);
                });
        }
        return CommandTicket(std::move(ticketState));
    }

#ifdef SPLASH_BACKEND_INSTRUMENTATION
    // Commits every dispatch of the command as its own command and waits
    // for it, then hands back an already-completed ticket with the summed
    // timing, so callers observe the usual asynchronous contract.
    CommandTicket submitProfiled(std::span<const ComputeDispatch> dispatches,
                                 CommandCompletion completion) {
        const PreparedCommand command = prepare(dispatches);
        CommandTiming total;
        for (const PreparedDispatch &item : command.dispatches) {
            const CommandTiming timing =
                commit({&item, 1}, {}, command.retainedAllocations, {}).wait();
            dispatchProfile.push_back(
                {item.source->pipelineName, timing.gpuSeconds});
            total.gpuSeconds += timing.gpuSeconds;
            total.wallSeconds += timing.wallSeconds;
        }
        auto ticketState = std::make_shared<CommandTicket::State>();
        ticketState->backend = asyncState;
        ticketState->sequence =
            asyncState->beginSubmission(command.dispatches.size());
        ticketState->timing = total;
        ticketState->completed = true;
        if (completion) completion();
        return CommandTicket(std::move(ticketState));
    }
#endif
};

SharedEvent::SharedEvent() = default;
SharedEvent::~SharedEvent() = default;
SharedEvent::SharedEvent(const SharedEvent &) = default;
SharedEvent &SharedEvent::operator=(const SharedEvent &) = default;
SharedEvent::SharedEvent(SharedEvent &&) noexcept = default;
SharedEvent &SharedEvent::operator=(SharedEvent &&) noexcept = default;
SharedEvent::SharedEvent(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

SharedEvent::operator bool() const noexcept { return impl_ && impl_->event; }

void *SharedEvent::nativeHandle() const noexcept {
    return impl_ ? (__bridge void *)impl_->event : nullptr;
}

// Metal ignores a value below the event's, so the write alone raises it.
void SharedEvent::signal(uint64_t value) const noexcept {
    if (*this) impl_->event.signaledValue = value;
}

void SharedEvent::notify(uint64_t value, std::function<void()> callback) const {
    if (!*this) throw MetalBackendError("an empty shared event cannot notify");
    [impl_->event notifyListener:impl_->listener
                         atValue:value
                           block:^(id<MTLSharedEvent>, uint64_t) { callback(); }];
}

MetalBuffer::MetalBuffer() = default;
MetalBuffer::~MetalBuffer() = default;
MetalBuffer::MetalBuffer(const MetalBuffer &) = default;
MetalBuffer &MetalBuffer::operator=(const MetalBuffer &) = default;
MetalBuffer::MetalBuffer(MetalBuffer &&) noexcept = default;
MetalBuffer &MetalBuffer::operator=(MetalBuffer &&) noexcept = default;

MetalBuffer::MetalBuffer(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MetalBuffer::operator bool() const noexcept {
    return impl_ != nullptr;
}

uint64_t MetalBuffer::sizeBytes() const noexcept {
    return impl_ ? impl_->lengthBytes : 0;
}

uint64_t MetalBuffer::allocatedBytes() const noexcept {
    return impl_ ? impl_->allocation->bytes : 0;
}

bool MetalBuffer::sameView(const MetalBuffer &other) const noexcept {
    if (impl_ == other.impl_) return true;
    return impl_ && other.impl_ &&
           impl_->allocation == other.impl_->allocation &&
           impl_->offsetBytes == other.impl_->offsetBytes &&
           impl_->lengthBytes == other.impl_->lengthBytes;
}

BufferStorage MetalBuffer::storage() const noexcept {
    return impl_ ? impl_->allocation->storage : BufferStorage::Shared;
}

void *MetalBuffer::contents() const noexcept {
    if (!impl_ || impl_->allocation->storage != BufferStorage::Shared ||
        !impl_->allocation->buffer) {
        return nullptr;
    }
    return static_cast<uint8_t *>(impl_->allocation->buffer.contents) +
           impl_->offsetBytes;
}

uint64_t MetalBuffer::gpuAddress() const noexcept {
    if (!impl_ || !impl_->allocation->buffer) return 0;
    return impl_->allocation->buffer.gpuAddress + impl_->offsetBytes;
}

CommandTicket::CommandTicket() = default;

CommandTicket::CommandTicket(std::shared_ptr<State> state)
    : state_(std::move(state)) {}

CommandTicket::~CommandTicket() {
    if (state_) state_->abandon();
}

CommandTicket::CommandTicket(CommandTicket &&) noexcept = default;

CommandTicket &CommandTicket::operator=(CommandTicket &&other) noexcept {
    if (this == &other) return *this;
    if (state_) state_->abandon();
    state_ = std::move(other.state_);
    return *this;
}

bool CommandTicket::ready() const noexcept {
    if (!state_) return false;
    std::lock_guard lock(state_->mutex);
    return state_->completed;
}

CommandTiming CommandTicket::wait() {
    if (!state_) throw MetalBackendError("Metal command ticket is empty");
    if (!state_->awaitCompletion(true)) {
        // Let go first, so that unwinding does not wait again.
        auto backend = state_->backend;
        state_.reset();
        backend->throwUnhealthy();
    }
    CommandTiming timing;
    std::string error;
    {
        std::lock_guard lock(state_->mutex);
        timing = state_->timing;
        error = state_->error;
    }
    state_->release();
    if (!error.empty()) throw MetalBackendError(error);
    return timing;
}

MetalBackend::MetalBackend(std::string metallibPath, double residencyKeepAliveSeconds,
                           double commandTimeoutSeconds)
    : impl_(std::make_unique<Impl>(commandTimeoutSeconds)) {
    @autoreleasepool {
        if (metallibPath.empty()) {
            throw MetalBackendError("metallib path must not be empty");
        }
        if (!(residencyKeepAliveSeconds > 0.0)) {
            throw MetalBackendError("residency keep-alive must be positive");
        }
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) {
            throw MetalBackendError("Metal device unavailable");
        }
        // Check the OS floor before loading Metal resources so an unsupported
        // system reports the version requirement first, in a message that
        // names the device as the device check does.
        readMacosVersion(impl_->capabilities);
        readDeviceCapabilities(impl_->device, impl_->capabilities);
        if (!impl_->capabilities.meetsMinimumMacos()) {
            throw MetalBackendError(*impl_->capabilities.validationMessage());
        }
        impl_->asyncState->device = impl_->device;
        // A command takes one Metal command buffer per event signal
        // (EventStep) and one more, all created before the first is
        // committed, and creating one waits while the queue's limit of them
        // is outstanding: at the default of 64, a 64-layer prefill with a
        // Neural Engine step each would wait forever.
        impl_->queue = [impl_->device
            newCommandQueueWithMaxCommandBufferCount:kMaximumCommandBuffers];
        if (!impl_->queue) {
            throw MetalBackendError("unable to create Metal command queue");
        }
        impl_->eventListener = [[MTLSharedEventListener alloc]
            initWithDispatchQueue:dispatch_queue_create(
                "splash.metal.events",
                dispatch_queue_attr_make_with_autorelease_frequency(
                    DISPATCH_QUEUE_SERIAL,
                    DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM))];

        NSString *path = checkedNSString(metallibPath, "metallib path");
        NSError *error = nil;
        NSData *fileData = [NSData dataWithContentsOfFile:path
                                                 options:0
                                                   error:&error];
        if (!fileData) {
            throw MetalBackendError(
                "unable to read metallib " + metallibPath + ": " +
                errorDescription(error));
        }
        // The library keeps the bytes read here, whatever later replaces the
        // path; the dispatch data retains them rather than copying them.
        dispatch_data_t data = dispatch_data_create(
            fileData.bytes, fileData.length, nullptr, ^{ (void)fileData; });
        error = nil;
        impl_->library =
            [impl_->device newLibraryWithData:data error:&error];
        if (!impl_->library) {
            throw MetalBackendError(
                "unable to load metallib " + metallibPath + ": " +
                errorDescription(error));
        }
        // Ending residency dispatches a kernel built here, so no pipeline or
        // driver program is compiled when a keep-alive lapses.
        impl_->residency = std::make_shared<Residency>(
            impl_->device, impl_->queue,
            impl_->newPipeline(Residency::kKickPipeline), residencyKeepAliveSeconds);
    }
    impl_->sampleDeviceMemory();
}

MetalBackend::~MetalBackend() = default;

void MetalBackend::stop() noexcept {
    std::lock_guard lock(impl_->asyncState->gateMutex);
    impl_->asyncState->stopping = true;
}

const DeviceCapabilities &MetalBackend::capabilities() const noexcept {
    return impl_->capabilities;
}

DeviceCapabilities probeDeviceCapabilities() {
    @autoreleasepool {
        DeviceCapabilities capabilities;
        readMacosVersion(capabilities);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw MetalBackendError("Metal device unavailable");
        readDeviceCapabilities(device, capabilities);
        return capabilities;
    }
}

void MetalBackend::checkOperation() const {
    impl_->ensureHealthy();
    if (impl_->operationGuard) impl_->operationGuard();
}

void MetalBackend::setOperationGuard(std::function<void()> guard) {
    impl_->operationGuard = std::move(guard);
}

void MetalBackend::setWaitInterrupt(std::function<bool()> shuttingDown) {
    impl_->asyncState->waitInterrupt = std::move(shuttingDown);
}

MetalBuffer MetalBackend::allocateBuffer(uint64_t bytes,
                                         BufferStorage storage,
                                         std::string_view label) {
    checkOperation();
    if (!bytes) throw MetalBackendError("Metal buffer size must be positive");
    if (bytes > impl_->capabilities.maxBufferLengthBytes) {
        throw MetalBackendError("Metal buffer exceeds maxBufferLength");
    }
    auto allocation = std::make_shared<MetalAllocation>();
    allocation->accounting = impl_->accounting;
    allocation->length = bytes;
    allocation->storage = storage;
    if (!label.empty()) allocation->label = checkedNSString(label, "buffer label");
    allocation->residency = impl_->residency;
    allocation->attach(impl_->newBuffer(bytes, storage, allocation->label));
    impl_->sampleDeviceMemory();
    auto result = std::make_shared<MetalBuffer::Impl>();
    result->lengthBytes = bytes;
    result->allocation = std::move(allocation);
    return MetalBuffer(std::move(result));
}

MetalBuffer MetalBackend::wrapSharedMemory(void *address, uint64_t bytes,
                                           std::shared_ptr<void> owner,
                                           std::string_view label) {
    checkOperation();
    const uint64_t page = static_cast<uint64_t>(getpagesize());
    if (!address || !bytes || !owner ||
        reinterpret_cast<uintptr_t>(address) % page || bytes % page) {
        throw MetalBackendError(
            "wrapped memory needs an owner and whole pages");
    }
    if (bytes > impl_->capabilities.maxBufferLengthBytes) {
        throw MetalBackendError("Metal buffer exceeds maxBufferLength");
    }
    auto allocation = std::make_shared<MetalAllocation>();
    allocation->accounting = impl_->accounting;
    allocation->length = bytes;
    allocation->owner = owner;
    if (!label.empty()) allocation->label = checkedNSString(label, "buffer label");
    allocation->residency = impl_->residency;
    // Metal may keep the buffer past our last view, so its deallocator holds
    // the owner too.
    id<MTLBuffer> buffer = [impl_->device
        newBufferWithBytesNoCopy:address
                          length:bytes
                         options:MTLResourceStorageModeShared
                     deallocator:^(void *, NSUInteger) { (void)owner; }];
    if (!buffer)
        throw MetalAllocationError("zero-copy Metal buffer creation failed");
    if (allocation->label) buffer.label = allocation->label;
    allocation->attach(buffer);
    impl_->sampleDeviceMemory();
    auto result = std::make_shared<MetalBuffer::Impl>();
    result->lengthBytes = bytes;
    result->allocation = std::move(allocation);
    return MetalBuffer(std::move(result));
}

void MetalBackend::releaseMemory(const MetalBuffer &buffer) {
    MetalAllocation &allocation = impl_->baseAllocation(buffer);
    if (allocation.owner)
        throw MetalBackendError("wrapped memory is its owner's to release");
    if (!allocation.buffer)
        throw MetalBackendError("Metal buffer memory is already released");
    if (commandInFlight())
        throw MetalBackendError(
            "Metal buffer memory is released while a command is in flight");
    allocation.detach();
    impl_->sampleDeviceMemory();
}

void MetalBackend::restoreMemory(const MetalBuffer &buffer) {
    checkOperation();
    MetalAllocation &allocation = impl_->baseAllocation(buffer);
    if (allocation.buffer)
        throw MetalBackendError("Metal buffer memory is not released");
    allocation.attach(impl_->newBuffer(allocation.length, allocation.storage,
                                       allocation.label));
    impl_->sampleDeviceMemory();
}

MetalBuffer MetalBackend::view(const MetalBuffer &base,
                               uint64_t offsetBytes,
                               uint64_t lengthBytes) const {
    impl_->ensureHealthy();
    if (!base.impl_) {
        throw MetalBackendError("cannot view an empty Metal buffer");
    }
    if (base.impl_->allocation->accounting.get() != impl_->accounting.get()) {
        throw MetalBackendError("Metal buffer belongs to another backend");
    }
    if (!lengthBytes || offsetBytes > base.impl_->lengthBytes ||
        lengthBytes > base.impl_->lengthBytes - offsetBytes) {
        std::ostringstream message;
        message << "Metal buffer view is out of range: offset=" << offsetBytes
                << " length=" << lengthBytes
                << " base_length=" << base.impl_->lengthBytes;
        throw MetalBackendError(message.str());
    }
    auto result = std::make_shared<MetalBuffer::Impl>();
    result->allocation = base.impl_->allocation;
    result->offsetBytes = base.impl_->offsetBytes + offsetBytes;
    result->lengthBytes = lengthBytes;
    return MetalBuffer(std::move(result));
}

SharedEvent MetalBackend::newSharedEvent() {
    checkOperation();
    auto result = std::make_shared<SharedEvent::Impl>();
    result->event = [impl_->device newSharedEvent];
    if (!result->event) {
        throw MetalBackendError("unable to create Metal shared event");
    }
    result->listener = impl_->eventListener;
    return SharedEvent(std::move(result));
}

CommandTiming MetalBackend::submit(const ComputeDispatch &dispatch) {
    return submitCommandAsync({&dispatch, 1}).wait();
}

CommandTicket MetalBackend::submitCommandAsync(
    std::span<const ComputeDispatch> dispatches,
    CommandCompletion completion) {
    return submitCommandAsync(Command{dispatches, {}}, std::move(completion));
}

CommandTicket MetalBackend::submitCommandAsync(const Command &command,
                                               CommandCompletion completion) {
    checkOperation();
#ifdef SPLASH_BACKEND_INSTRUMENTATION
    if (impl_->dispatchProfiling) {
        if (!command.events.empty())
            throw MetalBackendError(
                "dispatch profiling does not replay event steps");
        return impl_->submitProfiled(command.dispatches, std::move(completion));
    }
#endif
    checkEvents(command.events, command.dispatches.size());
    std::optional<Impl::StreamedHead> head = std::exchange(impl_->streamed, std::nullopt);
    // fastkernel SPLASH_CHUNKED_SUBMIT=N (default 48, 0 = off, read per
    // submission): a command without event steps and of more than 2N
    // dispatches commits its first N as a head before the rest is encoded,
    // so the GPU starts sooner. Scheduling only.
    if (!head && command.events.empty()) {
        const char *value = std::getenv("SPLASH_CHUNKED_SUBMIT");
        const size_t chunk = value ? std::strtoul(value, nullptr, 10) : 48;
        if (chunk && command.dispatches.size() > 2 * chunk) {
            Impl::PreparedCommand prepared =
                impl_->prepare(command.dispatches.first(chunk));
            head = impl_->commitHead(prepared.dispatches, {},
                                     std::move(prepared.retainedAllocations));
        }
    }
    if (!head) {
        Impl::PreparedCommand prepared = impl_->prepare(command.dispatches);
        return impl_->commit(prepared.dispatches, command.events,
                             std::move(prepared.retainedAllocations),
                             std::move(completion));
    }
    // The rest of a command whose head is committed: a failure from here on
    // leaves the head running, so it waits for it.
    try {
        if (head->dispatches >= command.dispatches.size() ||
            head->events > command.events.size() ||
            (head->events && command.events[head->events - 1].before != head->dispatches) ||
            (head->events < command.events.size() &&
             command.events[head->events].before < head->dispatches))
            throw MetalBackendError("a streamed head does not fit its command");
        Impl::PreparedCommand prepared =
            impl_->prepare(command.dispatches.subspan(head->dispatches));
        return impl_->commit(prepared.dispatches, command.events.subspan(head->events),
                             std::move(prepared.retainedAllocations),
                             std::move(completion), std::move(head));
    } catch (...) {
        // commit() took the head over (its own failures end the submission).
        if (head && head->ticket) {
            impl_->markUnhealthy("a command failed after its head was committed");
            impl_->abandonHead(*head);
        }
        throw;
    }
}

bool MetalBackend::streamHead(const Command &head) {
    checkOperation();
#ifdef SPLASH_BACKEND_INSTRUMENTATION
    if (impl_->dispatchProfiling) return false;
#endif
    if (impl_->streamed)
        throw MetalBackendError("a streamed head is already pending");
    checkEvents(head.events, head.dispatches.size());
    for (const EventStep &step : head.events) {
        if (step.before != head.dispatches.size())
            throw MetalBackendError("a streamed head's event steps must follow it");
    }
    Impl::PreparedCommand prepared = impl_->prepare(head.dispatches);
    impl_->streamed = impl_->commitHead(prepared.dispatches, head.events,
                                        std::move(prepared.retainedAllocations));
    return true;
}

void MetalBackend::abandonStreamedHead() noexcept {
    if (auto head = std::exchange(impl_->streamed, std::nullopt))
        impl_->abandonHead(*head);
}

CommandTicket MetalBackend::submitCommandAsync(const Command &command,
                                               CommandCompletion completion,
                                               const TrailingBuilder &trailing,
                                               bool &trailingCommitted) {
    trailingCommitted = false;
#ifdef SPLASH_BACKEND_INSTRUMENTATION
    // Profiled replay commits dispatch by dispatch: nothing trails it.
    if (impl_->dispatchProfiling)
        return submitCommandAsync(command, std::move(completion));
#endif
    CommandTicket ticket = submitCommandAsync(command, std::move(completion));
    // Built only once the command is committed, so building it never delays
    // the command's GPU start.
    if (!trailing)
        return ticket;
    const std::span<const ComputeDispatch> dispatches = trailing();
    if (dispatches.empty())
        return ticket;
    impl_->commitTrailing(impl_->prepare(dispatches));
    trailingCommitted = impl_->trailing != nil;
    return ticket;
}

void MetalBackend::awaitTrailing() {
    id<MTLCommandBuffer> trail = impl_->trailing;
    dispatch_semaphore_t done = impl_->trailingDone;
    impl_->trailing = nil;
    impl_->trailingDone = nil;
    if (!trail)
        return;
    // Bounded like the command watchdog: a stuck block must not hang the
    // engine thread until the OS GPU timeout.
    const double timeout = impl_->asyncState->commandWatchdog.timeoutSeconds();
    if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW,
            static_cast<int64_t>(timeout * NSEC_PER_SEC))) != 0) {
        impl_->markUnhealthy("Metal trailing command timed out");
        throw MetalBackendError("Metal trailing command timed out");
    }
    if (trail.status != MTLCommandBufferStatusCompleted) {
        std::string message = "Metal trailing command failed";
        if (trail.error) message += ": " + errorDescription(trail.error);
        impl_->markUnhealthy(message);
        throw MetalBackendError(message);
    }
}

void MetalBackend::preparePipelines(
    std::span<const ComputeDispatch> dispatches) {
    checkOperation();
    static_cast<void>(impl_->prepare(dispatches));
}

MetalMemoryStats MetalBackend::memoryStats() const noexcept {
    // Reading MTLDevice.currentAllocatedSize can synchronize with an active
    // command on some Apple GPUs. Every allocation and command lifecycle
    // boundary already samples it, so status must use the cached atomic value
    // rather than turning a control-plane query into a GPU barrier.
    return {
        impl_->accounting->allocatedBytes.load(std::memory_order_relaxed),
        impl_->accounting->peakAllocatedBytes.load(std::memory_order_relaxed),
        impl_->asyncState->deviceCurrentAllocatedBytes.load(
            std::memory_order_relaxed),
        impl_->asyncState->devicePeakAllocatedBytes.load(
            std::memory_order_relaxed),
    };
}

MetalMemoryStats MetalBackend::refreshMemoryStats() const noexcept {
    impl_->sampleDeviceMemory();
    return memoryStats();
}

bool MetalBackend::commandInFlight() const noexcept {
    return impl_->asyncState->hasActiveSubmission();
}

void MetalBackend::checkHealth() {
    impl_->asyncState->checkCommandHealth();
}

bool MetalBackend::healthy() const noexcept {
    return impl_->asyncState->healthy.load(std::memory_order_acquire);
}

std::string MetalBackend::unhealthyReason() const {
    std::lock_guard lock(impl_->asyncState->healthMutex);
    return impl_->asyncState->healthReason;
}

#ifdef SPLASH_BACKEND_INSTRUMENTATION
uint64_t BackendInstrumentation::submittedCommands(
    const MetalBackend &backend) {
    std::lock_guard lock(backend.impl_->asyncState->gateMutex);
    return backend.impl_->asyncState->nextSequence;
}

size_t BackendInstrumentation::cachedPipelines(const MetalBackend &backend) {
    return backend.impl_->pipelines.size();
}

std::vector<std::string> BackendInstrumentation::cachedPipelineNames(const MetalBackend &backend) {
    std::vector<std::string> names;
    for (const auto &entry : backend.impl_->pipelines)
        names.emplace_back(entry.first);
    return names;
}

void BackendInstrumentation::setDispatchProfiling(MetalBackend &backend,
                                                  bool enabled) {
    backend.impl_->dispatchProfiling = enabled;
}

std::vector<DispatchTiming>
BackendInstrumentation::takeDispatchProfile(MetalBackend &backend) {
    return std::exchange(backend.impl_->dispatchProfile, {});
}
#endif

}  // namespace splash::metal
