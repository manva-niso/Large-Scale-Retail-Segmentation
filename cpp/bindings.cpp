#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "rfm_aggregate.h"
#include "kmeans_core.h"

namespace py = pybind11;

// Wraps compute_rfm_dense_parallel: customer_idx must be dense integers in
// [0, n_customers). Python side is responsible for factorizing raw customer_id
// -> dense idx (pandas .factorize() or similar) before calling this.
// Returns parallel numpy arrays: recency, frequency, monetary, indexed by
// the same dense idx (i.e. result[i] corresponds to customer_idx == i).
py::dict compute_rfm(py::array_t<int32_t> customer_idx,
                      py::array_t<int32_t> days,
                      py::array_t<double> prices,
                      int n_customers,
                      int reference_day,
                      int num_threads) {
    auto idx_buf = customer_idx.request();
    auto day_buf = days.request();
    auto price_buf = prices.request();

    size_t n = idx_buf.shape[0];
    auto result = compute_rfm_dense_parallel(
        static_cast<const int32_t*>(idx_buf.ptr),
        static_cast<const int32_t*>(day_buf.ptr),
        static_cast<const double*>(price_buf.ptr),
        n, (size_t)n_customers, num_threads
    );

    auto out_recency = py::array_t<int32_t>(n_customers);
    auto out_freq = py::array_t<int32_t>(n_customers);
    auto out_monetary = py::array_t<double>(n_customers);

    auto rec_ptr = static_cast<int32_t*>(out_recency.request().ptr);
    auto freq_ptr = static_cast<int32_t*>(out_freq.request().ptr);
    auto mon_ptr = static_cast<double*>(out_monetary.request().ptr);

    for (int c = 0; c < n_customers; ++c) {
        rec_ptr[c] = result[c].frequency > 0 ? (reference_day - result[c].last_seen_day) : -1;
        freq_ptr[c] = result[c].frequency;
        mon_ptr[c] = result[c].monetary;
    }

    py::dict out;
    out["recency"] = out_recency;
    out["frequency"] = out_freq;
    out["monetary"] = out_monetary;
    return out;
}

// Wraps assign_parallel for K-Means: points and centroids are 2D numpy arrays.
py::array_t<int> kmeans_assign(py::array_t<double, py::array::c_style | py::array::forcecast> points,
                                py::array_t<double, py::array::c_style | py::array::forcecast> centroids,
                                int num_threads) {
    auto p_buf = points.request();
    auto c_buf = centroids.request();
    size_t n_points = p_buf.shape[0];
    size_t n_dims = p_buf.shape[1];
    size_t n_clusters = c_buf.shape[0];

    auto labels = assign_parallel(
        static_cast<const double*>(p_buf.ptr), n_points,
        static_cast<const double*>(c_buf.ptr), n_clusters,
        n_dims, num_threads
    );

    auto out = py::array_t<int>(n_points);
    auto out_ptr = static_cast<int*>(out.request().ptr);
    for (size_t i = 0; i < n_points; ++i) out_ptr[i] = labels[i];
    return out;
}

PYBIND11_MODULE(retail_cpp, m) {
    m.doc() = "C++ acceleration layer for retail segmentation (RFM aggregation + K-Means assignment)";
    m.def("compute_rfm", &compute_rfm,
          py::arg("customer_idx"), py::arg("days"), py::arg("prices"),
          py::arg("n_customers"), py::arg("reference_day"), py::arg("num_threads") = 4,
          "Multithreaded dense-array RFM aggregation over raw transactions");
    m.def("kmeans_assign", &kmeans_assign,
          py::arg("points"), py::arg("centroids"), py::arg("num_threads") = 4,
          "Multithreaded nearest-centroid assignment step for K-Means");
}
