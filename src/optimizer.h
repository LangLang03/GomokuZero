#pragma once
#include <vector>
#include <cmath>
#include <cstddef>

namespace gomoku {

// Hand-written Adam optimizer (matches torch::optim::Adam semantics:
//   m = beta1*m + (1-beta1)*g
//   v = beta2*v + (1-beta2)*g^2
//   w -= lr * m_hat / (sqrt(v_hat) + eps)
// with l2 weight decay added to the gradient (weight_decay).
// Also implements gradient clipping (clip_grad_norm_ equivalent).
class Adam {
public:
    Adam(double lr, double beta1 = 0.9, double beta2 = 0.999,
         double eps = 1e-8, double weight_decay = 0.0)
        : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps),
          weight_decay_(weight_decay) {}

    // register a parameter tensor (returns its index for step())
    size_t add_param(std::vector<float>& param) {
        size_t idx = params_.size();
        params_.push_back(&param);
        m_.emplace_back(param.size(), 0.0f);
        v_.emplace_back(param.size(), 0.0f);
        return idx;
    }

    void zero_grad() {
        for (size_t i = 0; i < grads_.size(); ++i) {
            std::fill(grads_[i]->begin(), grads_[i]->end(), 0.0f);
        }
    }

    // register a gradient tensor that mirrors a param
    size_t add_grad(std::vector<float>& grad) {
        size_t idx = grads_.size();
        grads_.push_back(&grad);
        return idx;
    }

    // clip total gradient norm to max_norm (in-place scaling)
    void clip_grad_norm(float max_norm) {
        float sum_sq = 0.0f;
        for (auto* g : grads_) {
            for (float v : *g) sum_sq += v * v;
        }
        float norm = std::sqrt(sum_sq);
        if (norm > max_norm && norm > 0) {
            float scale = max_norm / norm;
            for (auto* g : grads_) {
                for (auto& v : *g) v *= scale;
            }
        }
    }

    // apply one Adam step for all params (assumes grads_ filled)
    void step() {
        ++step_;
        const float bias1 = 1.0f - std::pow(static_cast<float>(beta1_), step_);
        const float bias2 = 1.0f - std::pow(static_cast<float>(beta2_), step_);
        for (size_t i = 0; i < params_.size(); ++i) {
            auto* p = params_[i];
            auto* g = grads_[i];
            auto& m = m_[i];
            auto& v = v_[i];
            for (size_t j = 0; j < p->size(); ++j) {
                float gj = (*g)[j];
                if (weight_decay_ > 0) gj += weight_decay_ * (*p)[j];
                m[j] = static_cast<float>(beta1_) * m[j] +
                       (1.0f - static_cast<float>(beta1_)) * gj;
                v[j] = static_cast<float>(beta2_) * v[j] +
                       (1.0f - static_cast<float>(beta2_)) * gj * gj;
                const float m_hat = m[j] / bias1;
                const float v_hat = v[j] / bias2;
                (*p)[j] -= static_cast<float>(lr_) * m_hat /
                           (std::sqrt(v_hat) + static_cast<float>(eps_));
            }
        }
    }

    void set_lr(double lr) { lr_ = lr; }
    double lr() const { return lr_; }

private:
    double lr_, beta1_, beta2_, eps_, weight_decay_;
    int step_ = 0;
    std::vector<std::vector<float>*> params_;
    std::vector<std::vector<float>*> grads_;
    std::vector<std::vector<float>> m_, v_;
};

}  // namespace gomoku
