#pragma once
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <cstdarg>

namespace gomoku {

// Non-blocking async logger: log() enqueues a formatted line and returns
// immediately; a background thread flushes the queue to stdout. Formatting
// matches the Python trainer output so logs are familiar ([PROGRESS], kl:...).
class AsyncLogger {
public:
    AsyncLogger() { worker_ = std::thread(&AsyncLogger::run, this); }
    ~AsyncLogger() {
        stop_ = true;
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        flush();
    }

    // printf-style, non-blocking
    void log(const char* fmt, ...);

private:
    void run();
    void flush();

    std::queue<std::string> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    std::atomic<bool> stop_{false};
};

extern AsyncLogger logger;

}  // namespace gomoku
