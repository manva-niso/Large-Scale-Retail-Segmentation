#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cstddef>

struct RFM {
    int32_t last_seen_day = -1;
    int32_t frequency = 0;
    double monetary = 0.0;
};

// Computes per-customer RFM (Recency-input, Frequency, Monetary) in parallel.
// customer_ids/days/prices are flat arrays of length n (n transactions).
// num_threads worker threads split the array into contiguous chunks.
//
// Dense-array version: customer_ids are assumed to be small dense integers
// (0..n_customers-1, e.g. after a factorize step in Python). Each thread
// accumulates into its OWN full-size local array (no hashing, no locking),
// then partial arrays are summed element-wise. This trades memory
// (n_customers * n_threads * sizeof(RFM)) for speed -- a standard technique
// when the key space is dense integers rather than sparse/arbitrary.
std::vector<RFM> compute_rfm_dense_parallel(
    const int32_t* customer_idx,   // dense index in [0, n_customers)
    const int32_t* days,
    const double* prices,
    size_t n,
    size_t n_customers,
    int num_threads
);
