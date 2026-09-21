#pragma once
#include <cstdarg>
#include <cstdio>
#include <string>

namespace gomoku {

// Training logger. Writes to stdout, and additionally to the file named by
// the GOMOK_LOG_FILE environment variable when set.
//
// The earlier implementation started a worker thread and polled a queue that
// log() never filled: the thread did nothing and every call wrote to stdout
// directly. This keeps the direct write and adds the optional file sink.
class Logger {
public:
    Logger();
    ~Logger();

    // printf-style.
    void log(const char* format, ...);

private:
    std::FILE* file_ = nullptr;
    bool owns_file_ = false;
};

extern Logger logger;

}  // namespace gomoku
