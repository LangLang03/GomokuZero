#include "train.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace gomoku;
namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f) { bool caught = false; try { f(); } catch (const std::invalid_argument&) { caught = true; } check(caught, "invalid config accepted"); }
}
int main() {
    try {
        TrainConfig c;
        SearchConfig s;
        validate_train_config(c, s);
        for (double bad : {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            auto altered = c; altered.lr_base = bad;
            rejects([&] { validate_train_config(altered, s); });
        }
        { auto x = c; x.batch_size = 0; rejects([&] { validate_train_config(x,s); }); }
        { auto x = c; x.game_batch_num = 0; rejects([&] { validate_train_config(x,s); }); }
        { auto x = c; x.pretrain_epochs = 1; rejects([&] { validate_train_config(x,s); }); }
        { auto x = c; x.mix_ratio = 1.1; rejects([&] { validate_train_config(x,s); }); }
        { auto x = c; x.eval_games = 3; x.baseline_dir="somewhere"; rejects([&] { validate_train_config(x,s); }); }
        { auto x = s; x.eval_opening_moves = 9; rejects([&] { validate_train_config(c,x); }); }
        { auto x = s; x.virtual_loss = 2000; rejects([&] { validate_train_config(c,x); }); }
        { auto x = c; x.game_batch_num = 0; x.pretrain_epochs=1; x.mix_data="dataset"; validate_train_config(x,s); }
        const double base = 0.002, floor = 0.0002;
        auto lr = [&](int step) { return scheduled_learning_rate(base, .1, 2, 10, step); };
        check(std::abs(lr(0) - .0011) < 1e-10 && std::abs(lr(1) - base) < 1e-10, "warmup");
        check(std::abs(lr(9) - floor) < 1e-10 && std::abs(lr(100) - floor) < 1e-10, "final floor");
        for (int i=3;i<10;++i) check(lr(i) <= lr(i-1), "decay must be monotonic");
        check(std::isfinite(scheduled_learning_rate(base,.1,20,1,0)), "single step schedule");
        check(std::abs(scheduled_learning_rate(base,.1,20,2,1)-base) < 1e-10, "short phase warmup is bounded");
        std::cout << "trainer config/schedule tests passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
