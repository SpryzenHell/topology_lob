#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void fractional_diff_kernel(
    const double* x,
    const double* w,
    double* out,
    int n,
    int m
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || i < m - 1) return;

    double acc = 0.0;
    for (int k = 0; k < m; ++k) {
        acc += w[k] * x[i - m + 1 + k];
    }
    out[i] = acc;
}

static void check(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(error));
        std::exit(1);
    }
}

int main() {
    constexpr int n = 200000;
    constexpr int width = 256;
    constexpr double d = 0.45;

    std::vector<double> host_x(n), host_w(width), host_out(n, NAN);
    for (int i = 0; i < n; ++i) {
        host_x[i] = std::log(
            100.0 + 0.001 * i + 0.01 * std::sin(i * 0.01)
        );
    }

    host_w[0] = 1.0;
    for (int k = 1; k < width; ++k) {
        host_w[k] = host_w[k - 1] * ((k - 1.0 - d) / k);
    }

    double* dev_x = nullptr;
    double* dev_w = nullptr;
    double* dev_out = nullptr;
    check(cudaMalloc(&dev_x, n * sizeof(double)), "cudaMalloc x");
    check(cudaMalloc(&dev_w, width * sizeof(double)), "cudaMalloc weights");
    check(cudaMalloc(&dev_out, n * sizeof(double)), "cudaMalloc output");

    check(
        cudaMemcpy(dev_x, host_x.data(), n * sizeof(double), cudaMemcpyHostToDevice),
        "copy x"
    );
    check(
        cudaMemcpy(dev_w, host_w.data(), width * sizeof(double), cudaMemcpyHostToDevice),
        "copy weights"
    );

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event start");
    check(cudaEventCreate(&stop), "event stop");

    check(cudaEventRecord(start), "event record start");
    fractional_diff_kernel<<<(n + 255) / 256, 256>>>(dev_x, dev_w, dev_out, n, width);
    check(cudaGetLastError(), "kernel launch");
    check(cudaEventRecord(stop), "event record stop");
    check(cudaEventSynchronize(stop), "event sync");

    float milliseconds = 0.0f;
    check(cudaEventElapsedTime(&milliseconds, start, stop), "event elapsed");
    check(
        cudaMemcpy(host_out.data(), dev_out, n * sizeof(double), cudaMemcpyDeviceToHost),
        "copy output"
    );

    std::printf(
        "events=%d width=%d d=%.2f kernel_ms=%.4f events_per_s=%.2f checksum=%.12g\n",
        n, width, d, milliseconds, 1000.0 * n / milliseconds, host_out.back()
    );

    cudaFree(dev_x);
    cudaFree(dev_w);
    cudaFree(dev_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}