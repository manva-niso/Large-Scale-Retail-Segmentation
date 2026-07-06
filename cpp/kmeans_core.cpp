#include "kmeans_core.h"
#include <thread>
#include <limits>

static void assign_chunk(const double* points, const double* centroids,
                          size_t n_clusters, size_t n_dims,
                          std::vector<int>& labels, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
        const double* p = points + i * n_dims;
        double best_dist = std::numeric_limits<double>::max();
        int best_k = 0;
        for (size_t k = 0; k < n_clusters; ++k) {
            const double* c = centroids + k * n_dims;
            double dist = 0.0;
            for (size_t d = 0; d < n_dims; ++d) {
                double diff = p[d] - c[d];
                dist += diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; best_k = (int)k; }
        }
        labels[i] = best_k;
    }
}

std::vector<int> assign_parallel(const double* points, size_t n_points,
                                  const double* centroids, size_t n_clusters,
                                  size_t n_dims, int num_threads) {
    if (num_threads < 1) num_threads = 1;
    if ((size_t)num_threads > n_points) num_threads = n_points > 0 ? (int)n_points : 1;

    std::vector<int> labels(n_points);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    size_t chunk = n_points / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = (t == num_threads - 1) ? n_points : start + chunk;
        threads.emplace_back(assign_chunk, points, centroids, n_clusters, n_dims,
                              std::ref(labels), start, end);
    }
    for (auto& th : threads) th.join();
    return labels;
}
