#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

static std::vector<double> weights(double d, int width) {
    std::vector<double> w(width);
    w[0] = 1.0;
    for (int k = 1; k < width; ++k) {
        w[k] = -w[k - 1] * (d - k + 1.0) / k;
    }
    std::reverse(w.begin(), w.end());
    return w;
}

int main() {
    constexpr int n = 200000;
    constexpr int width = 256;
    constexpr double d = 0.45;

    std::vector<double> x(n), y(n, NAN);
    for (int i = 0; i < n; ++i) {
        x[i] = std::log(100.0 + 0.01 * i + std::sin(i * 0.01));
    }

    auto w = weights(d, width);
    auto start = std::chrono::steady_clock::now();

    for (int i = width - 1; i < n; ++i) {
        double sum = 0.0;
        for (int k = 0; k < width; ++k) {
            sum += w[k] * x[i - width + 1 + k];
        }
        y[i] = sum;
    }

    double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start
    ).count();

    std::cout << std::fixed << std::setprecision(6)
              << "events=" << n
              << " width=" << width
              << " d=" << d
              << " elapsed_s=" << seconds
              << " events_per_s=" << (n - width + 1) / seconds
              << " checksum=" << y.back()
              << "\n";
}