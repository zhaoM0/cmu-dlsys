#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace py = pybind11;

void matmul(const float *A, const float *B, float *dst, int m, int k, int n) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {

            float sum = 0;
            for (int i = 0; i < k; i++) 
                sum += A[row * k + i] * B[i * n + col];
            dst[row * n + col] = sum;

        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    char* y_one_hot = (char *)malloc(m * k * sizeof(char));
    memset(y_one_hot, 0, m * k);

    float* dst = (float *)malloc(batch * k * sizeof(float));

    for (int offset_r = 0, i = 0; offset_r < (m * k); offset_r += k, i++)    
        y_one_hot[offset_r + y[i]] = 1;

    for (int start = 0; start < m; start += batch) {
        int offset_x = start * n;
        int offset_y = start * k;

        float* batch_x = X + offset_x;              // len is batch x n
        float* batch_y = y_one_hot + offset_y;      // len is batch x k

        // exp
        matmul(batch_x, batch_y, dst, batch, n, k);
        for (int i = 0; i < batch * k; i++)
            dst[i] = exp(dst[i]);

        // norm 
        for (int i = 0; i < batch; i++) {
            float sum = 0;
            for (int j = 0; j < k; j++)  sum += dst[i * k + j]; 
            for (int j = 0; j < k; j++)  dst[i * k + j] /= sum;
        }   

        // grad 
        matmul();     
    }

}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
