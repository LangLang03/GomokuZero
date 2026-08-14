#include "logger.h"
#include <cstring>
#include <cstdio>
#include <vector>

namespace gomoku {

AsyncLogger logger;

void AsyncLogger::log(const char* fmt, ...) {
    // synchronous write: training logs are low-frequency (a few per batch)
    // and must survive crashes; async buffering loses them on SIGSEGV
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    std::fwrite(buf, 1, strlen(buf), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void AsyncLogger::run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_) {
        if (!queue_.empty()) {
            std::string line = std::move(queue_.front());
            queue_.pop();
            lock.unlock();
            std::fwrite(line.c_str(), 1, line.size(), stdout);
            std::fputc('\n', stdout);
            std::fflush(stdout);
            lock.lock();
        } else {
            cv_.wait_for(lock, std::chrono::milliseconds(50));
        }
    }
}

void AsyncLogger::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
        std::string line = std::move(queue_.front());
        queue_.pop();
        std::fwrite(line.c_str(), 1, line.size(), stdout);
        std::fputc('\n', stdout);
    }
    std::fflush(stdout);
}

}  // namespace gomoku
