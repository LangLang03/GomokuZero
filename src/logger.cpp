#include "logger.h"
#include <cstdlib>
#include <cstring>

namespace gomoku {

Logger logger;

Logger::Logger() {
    const char* path = std::getenv("GOMOK_LOG_FILE");
    if (path == nullptr || path[0] == '\0') return;
    file_ = std::fopen(path, "w");
    if (file_ != nullptr) {
        owns_file_ = true;
        std::fprintf(stderr, "[log] also writing to %s\n", path);
    } else {
        std::fprintf(stderr, "[log] cannot open %s for writing\n", path);
    }
}

Logger::~Logger() {
    if (owns_file_ && file_ != nullptr) std::fclose(file_);
}

void Logger::log(const char* format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    // Flush every line so progress survives a crash.
    std::fputs(buffer, stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
    if (file_ != nullptr) {
        std::fputs(buffer, file_);
        std::fputc('\n', file_);
        std::fflush(file_);
    }
}

}  // namespace gomoku
