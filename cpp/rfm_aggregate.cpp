#include "rfm_aggregate.h"
#include <thread>
#include <algorithm>

static void worker(const int32_t* customer_idx, const int32_t* days, const double* prices,
                    size_t start, size_t end, size_t n_customers,
                    std::vector<RFM>& local) {
    local.assign(n_customers, RFM{});
    for (size_t i = start; i < end; ++i) {
        int32_t idx = customer_idx[i];
        auto& r = local[idx];
        r.frequency += 1;
        r.monetary += prices[i];
        if (days[i] > r.last_seen_day) r.last_seen_day = days[i];
    }
}

std::vector<RFM> compute_rfm_dense_parallel(
        const int32_t* customer_idx, const int32_t* days, const double* prices,
        size_t n, size_t n_customers, int num_threads) {
    if (num_threads < 1) num_threads = 1;
    if ((size_t)num_threads > n) num_threads = n > 0 ? (int)n : 1;

    std::vector<std::vector<RFM>> partials(num_threads);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    size_t chunk = n / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = (t == num_threads - 1) ? n : start + chunk;
        threads.emplace_back(worker, customer_idx, days, prices, start, end, n_customers,
                              std::ref(partials[t]));
    }
    for (auto& th : threads) th.join();

    std::vector<RFM> result(n_customers);
    for (auto& part : partials) {
        for (size_t c = 0; c < n_customers; ++c) {
            result[c].frequency += part[c].frequency;
            result[c].monetary  += part[c].monetary;
            result[c].last_seen_day = std::max(result[c].last_seen_day, part[c].last_seen_day);
        }
    }
    return result;
}
