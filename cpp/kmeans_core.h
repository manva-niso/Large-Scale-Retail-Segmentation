#pragma once
#include <vector>
#include <cstddef>

// Assigns each point to its nearest centroid (squared Euclidean distance).
// Parallelized across points (rows), split into contiguous chunks per thread.
// points: flat row-major array of shape (n_points, n_dims)
// centroids: flat row-major array of shape (n_clusters, n_dims)
// Returns a vector<int> of length n_points with the assigned cluster index.
std::vector<int> assign_parallel(
    const double* points, size_t n_points,
    const double* centroids, size_t n_clusters,
    size_t n_dims, int num_threads
);
