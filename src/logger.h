#pragma once
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <cstdarg>

namespace gomoku {

// Training logger. log() flushes completed lines immediately.
class AsyncLogger {
public:
    AsyncLogger() { worker_ = std::thread(&AsyncLogger::run, this); }
    ~AsyncLogger() {
        stop_ = true;
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        flush();
    }

    // printf-style
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
