#include "ggml.h"
#include "whisper.h"

#include <emscripten.h>
#include <emscripten/bind.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int N_THREAD = 8;
constexpr size_t kMaxContexts = 4;

struct audio_job {
    std::vector<float> samples;
    std::string language;
};

std::array<whisper_context *, kMaxContexts> g_contexts{nullptr, nullptr, nullptr, nullptr};
std::array<std::thread, kMaxContexts> g_workers;
std::array<std::atomic<bool>, kMaxContexts> g_running;

std::array<std::mutex, kMaxContexts> g_queue_mutex;
std::array<std::condition_variable, kMaxContexts> g_queue_cv;
std::array<std::deque<audio_job>, kMaxContexts> g_audio_queues;

std::array<std::mutex, kMaxContexts> g_status_mutex;
std::array<std::string, kMaxContexts> g_status;
std::array<std::string, kMaxContexts> g_status_forced;
std::array<std::string, kMaxContexts> g_transcribed;
std::array<std::string, kMaxContexts> g_default_language;

struct global_init {
    global_init() {
        for (auto &flag : g_running) {
            flag.store(false);
        }
        for (auto &queue : g_audio_queues) {
            queue.clear();
        }
        for (auto &status : g_status) {
            status.clear();
        }
        for (auto &status : g_status_forced) {
            status.clear();
        }
        for (auto &transcribed : g_transcribed) {
            transcribed.clear();
        }
        for (auto &lang : g_default_language) {
            lang.clear();
        }
    }
} global_initializer;

inline bool is_valid_index(size_t index) {
    return index < kMaxContexts && g_contexts[index] != nullptr;
}

void stream_set_status(size_t index, const std::string & status) {
    std::lock_guard<std::mutex> lock(g_status_mutex[index]);
    g_status[index] = status;
}

void stream_main(size_t index) {
    auto & ctx = g_contexts[index];
    if (!ctx) {
        return;
    }

    const bool is_multilingual = whisper_is_multilingual(ctx);
    const int64_t window_samples = 15 * WHISPER_SAMPLE_RATE; // 15 second rolling window for richer context

    stream_set_status(index, "waiting for audio ...");

    while (g_running[index]) {
        audio_job job;
        {
            std::unique_lock<std::mutex> lock(g_queue_mutex[index]);
            g_queue_cv[index].wait(lock, [index]() {
                return !g_running[index] || !g_audio_queues[index].empty();
            });

            if (!g_running[index] && g_audio_queues[index].empty()) {
                break;
            }

            job = std::move(g_audio_queues[index].front());
            g_audio_queues[index].pop_front();
        }

        if (job.samples.empty()) {
            continue;
        }

        // If the audio job is longer than the processing window, keep only the last window samples.
        if (job.samples.size() > static_cast<size_t>(window_samples)) {
            job.samples.erase(job.samples.begin(), job.samples.end() - window_samples);
        }

        std::string lang_selected = job.language.empty() ? g_default_language[index] : job.language;
        if (!is_multilingual) {
            lang_selected = "en";
        }

        whisper_full_params wparams = whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);
        wparams.n_threads        = std::min(N_THREAD, (int) std::thread::hardware_concurrency());
        wparams.offset_ms        = 0;
        wparams.translate        = false;
        wparams.no_context       = false;
        wparams.single_segment   = false;
        wparams.print_realtime   = false;
        wparams.print_progress   = false;
        wparams.print_timestamps = true;
        wparams.print_special    = false;
        wparams.max_tokens       = 128;
        wparams.audio_ctx        = 0;   // use full encoder context
        wparams.temperature_inc  = 0.2f; // enable fallback decoding when confidence is low
        wparams.language         = lang_selected.empty() ? nullptr : lang_selected.c_str();

        stream_set_status(index, "running whisper ...");

        const auto t_start = std::chrono::high_resolution_clock::now();
        const int ret = whisper_full(ctx, wparams, job.samples.data(), job.samples.size());
        const auto t_end = std::chrono::high_resolution_clock::now();

        if (ret != 0) {
            stream_set_status(index, "whisper_full failed");
            printf("whisper_full() failed: %d\n", ret);
            continue;
        }

        printf("stream[%zu]: whisper_full() returned %d in %f seconds\n", index, ret,
               std::chrono::duration<double>(t_end - t_start).count());

        std::string text_heard;
        const int n_segments = whisper_full_n_segments(ctx);
        if (n_segments > 0) {
            const char * text = whisper_full_get_segment_text(ctx, n_segments - 1);
            if (text) {
                text_heard += text;
                printf("transcribed[%zu]: %s\n", index, text);
            }
        }

        {
            std::lock_guard<std::mutex> lock(g_status_mutex[index]);
            g_transcribed[index] = std::move(text_heard);
            g_status[index] = "waiting for audio ...";
        }
    }

    whisper_free(ctx);
    g_contexts[index] = nullptr;
}

size_t acquire_context(const std::string & path_model, const std::string & lang) {
    for (size_t i = 0; i < kMaxContexts; ++i) {
        if (g_contexts[i] == nullptr) {
            whisper_context_params params = whisper_context_default_params();
            g_contexts[i] = whisper_init_from_file_with_params(path_model.c_str(), params);
            if (!g_contexts[i]) {
                return 0;
            }

            g_default_language[i] = lang;
            g_audio_queues[i].clear();
            {
                std::lock_guard<std::mutex> lock(g_status_mutex[i]);
                g_status[i].clear();
                g_status_forced[i].clear();
                g_transcribed[i].clear();
            }

            g_running[i] = true;
            g_workers[i] = std::thread(stream_main, i);
            return i + 1; // 1-based handle for JS side
        }
    }

    return 0;
}

void release_context(size_t index) {
    if (!is_valid_index(index)) {
        return;
    }

    if (g_running[index]) {
        g_running[index] = false;
        g_queue_cv[index].notify_all();
    }

    if (g_workers[index].joinable()) {
        g_workers[index].join();
    }
}

int enqueue_audio(size_t index, const std::string & language, const float * audio_data, size_t audio_len) {
    if (!is_valid_index(index) || !audio_data || audio_len == 0) {
        return -1;
    }

    audio_job job;
    job.samples.assign(audio_data, audio_data + audio_len);
    job.language = language;

    {
        std::lock_guard<std::mutex> lock(g_queue_mutex[index]);
        g_audio_queues[index].push_back(std::move(job));
    }

    g_queue_cv[index].notify_one();
    return static_cast<int>(g_audio_queues[index].size());
}

} // namespace

EMSCRIPTEN_BINDINGS(stream) {
    emscripten::function("init", emscripten::optional_override([](const std::string & path_model, const std::string & lang) {
        return acquire_context(path_model, lang);
    }));

    emscripten::function("free", emscripten::optional_override([](size_t handle) {
        if (handle == 0 || handle > kMaxContexts) {
            return;
        }
        release_context(handle - 1);
    }));

    emscripten::function("queue_audio", emscripten::optional_override([](size_t handle, const std::string & language, uintptr_t audio_ptr, size_t audio_len) {
        if (handle == 0 || handle > kMaxContexts) {
            return -1;
        }
        const float * audio_data = reinterpret_cast<const float *>(audio_ptr);
        return enqueue_audio(handle - 1, language, audio_data, audio_len);
    }));

    emscripten::function("pcm_audio_feed", emscripten::optional_override([](size_t handle, uintptr_t audio_ptr, size_t audio_len) {
        if (handle == 0 || handle > kMaxContexts) {
            return -1;
        }
        const float * audio_data = reinterpret_cast<const float *>(audio_ptr);
        return enqueue_audio(handle - 1, std::string{}, audio_data, audio_len);
    }));

    emscripten::function("get_transcribed", emscripten::optional_override([](size_t handle) {
        if (handle == 0 || handle > kMaxContexts) {
            return std::string();
        }
        const size_t index = handle - 1;
        std::lock_guard<std::mutex> lock(g_status_mutex[index]);
        return g_transcribed[index];
    }));

    emscripten::function("get_status", emscripten::optional_override([](size_t handle) {
        if (handle == 0 || handle > kMaxContexts) {
            return std::string();
        }
        const size_t index = handle - 1;
        std::lock_guard<std::mutex> lock(g_status_mutex[index]);
        const std::string & forced = g_status_forced[index];
        return forced.empty() ? g_status[index] : forced;
    }));

    emscripten::function("set_status", emscripten::optional_override([](size_t handle, const std::string & status) {
        if (handle == 0 || handle > kMaxContexts) {
            return;
        }
        const size_t index = handle - 1;
        std::lock_guard<std::mutex> lock(g_status_mutex[index]);
        g_status_forced[index] = status;
    }));
}
