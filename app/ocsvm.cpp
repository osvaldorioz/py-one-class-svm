#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

// Implementación simple de One-Class SVM para detección de anomalías
class OneClassSVM {
public:
    OneClassSVM(double nu, double gamma) : nu_(nu), gamma_(gamma) {}

    void fit(torch::Tensor data) {
        data_ = data;
        int n = data.size(0);

        // Calcular la matriz de kernel utilizando un kernel RBF
        kernel_matrix_ = torch::empty({n, n});
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dist = torch::norm(data[i] - data[j]).item<double>();
                kernel_matrix_[i][j] = std::exp(-gamma_ * dist * dist);
            }
        }

        // Aproximación simple para encontrar el umbral de decisión usando nu
        double threshold_index = static_cast<int>(std::floor(n * (1 - nu_)));
        auto scores_tuple = kernel_matrix_.sum(1).sort();
        auto sorted_scores = std::get<0>(scores_tuple);
        threshold_ = sorted_scores[threshold_index].item<double>();
    }

    std::vector<int> predict(torch::Tensor data) {
        int n = data.size(0);
        std::vector<int> labels(n, 1);  // 1 = normal, -1 = anomalía

        for (int i = 0; i < n; i++) {
            double score = 0.0;
            for (int j = 0; j < data_.size(0); j++) {
                double dist = torch::norm(data[i] - data_[j]).item<double>();
                score += std::exp(-gamma_ * dist * dist);
            }
            if (score < threshold_) {
                labels[i] = -1;  // Anomalía
            }
        }

        return labels;
    }

private:
    double nu_;
    double gamma_;
    double threshold_;
    torch::Tensor data_;
    torch::Tensor kernel_matrix_;
};

PYBIND11_MODULE(one_class_svm, m) {
    py::class_<OneClassSVM>(m, "OneClassSVM")
        .def(py::init<double, double>())
        .def("fit", &OneClassSVM::fit, "Train the One-Class SVM model")
        .def("predict", &OneClassSVM::predict, "Predict anomalies using One-Class SVM");
}
