#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "george/parser.h"
#include "george/kernels.h"
#include "george/exceptions.h"

#include <vector>
#include <tuple>
#include <Eigen/Sparse>



namespace py = pybind11;

class KernelInterface {
  public:
    KernelInterface (py::object kernel_spec) : kernel_spec_(kernel_spec) {
      kernel_ = george::parse_kernel_spec(kernel_spec_);
    };
    ~KernelInterface () {
      delete kernel_;
    };
    size_t ndim () const { return kernel_->get_ndim(); };
    size_t size () const { return kernel_->size(); };
    double value (const double* x1, const double* x2) const { return kernel_->value(x1, x2); };
    void gradient (const double* x1, const double* x2, const unsigned* which, double* grad) const {
      return kernel_->gradient(x1, x2, which, grad);
    };
    void x1_gradient (const double* x1, const double* x2, double* grad) const {
      return kernel_->x1_gradient(x1, x2, grad);
    };
    void x2_gradient (const double* x1, const double* x2, double* grad) const {
      return kernel_->x2_gradient(x1, x2, grad);
    };
    py::object kernel_spec () const { return kernel_spec_; };

  private:
    py::object kernel_spec_;
    george::kernels::Kernel* kernel_;
};


PYBIND11_MODULE(kernel_interface, m) {

  m.doc() = R"delim(
Docs...
)delim";

  py::class_<KernelInterface> interface(m, "KernelInterface");
  interface.def(py::init<py::object>());

  interface.def("value_general", [](KernelInterface& self, py::array_t<double> x1, py::array_t<double> x2) {
    auto x1p = x1.unchecked<2>();
    auto x2p = x2.unchecked<2>();
    size_t n1 = x1p.shape(0), n2 = x2p.shape(0);
    if (x1p.shape(1) != py::ssize_t(self.ndim()) || x2p.shape(1) != py::ssize_t(self.ndim())) throw george::dimension_mismatch();
    py::array_t<double> result({n1, n2});
    auto resultp = result.mutable_unchecked<2>();
    for (size_t i = 0; i < n1; ++i) {
      for (size_t j = 0; j < n2; ++j) {
        resultp(i, j) = self.value(&(x1p(i, 0)), &(x2p(j, 0)));
      }
    }
    return result;
  });

  interface.def("value_symmetric", [](KernelInterface& self, py::array_t<double> x) {
    auto xp = x.unchecked<2>();
    size_t n = xp.shape(0);
    if (xp.shape(1) != py::ssize_t(self.ndim())) throw george::dimension_mismatch();
    py::array_t<double> result({n, n});
    auto resultp = result.mutable_unchecked<2>();
    for (size_t i = 0; i < n; ++i) {
      resultp(i, i) = self.value(&(xp(i, 0)), &(xp(i, 0)));
      for (size_t j = i+1; j < n; ++j) {
        double value = self.value(&(xp(i, 0)), &(xp(j, 0)));
        resultp(i, j) = value;
        resultp(j, i) = value;
      }
    }
    return result;
  });
 

interface.def("value_sparse", [](KernelInterface& self, py::array_t<double> x, const std::vector<std::vector<size_t>>& neighbors) {
    auto xp = x.unchecked<2>();
    size_t n = xp.shape(0);
    if (xp.shape(1) != py::ssize_t(self.ndim())) throw george::dimension_mismatch();
    
    // Initialize a sparse matrix
    Eigen::SparseMatrix<double> result(n, n);
    
    
    // Use lists to store row, column and value for sparse representation
    std::vector<Eigen::Triplet<double>> tripletList; // For holding non-zero elements
    
    for (size_t i = 0; i < n; ++i) {
        // Calculate the covariance for the points in the neighborhood
        double self_value = self.value(&(xp(i, 0)), &(xp(i, 0)));
        tripletList.emplace_back(i, i, self_value); // Diagonal element

        for (size_t j : neighbors[i]) { // Use the list of neighbors
            if (i < j) { // Only compute for upper triangle
                double value = self.value(&(xp(i, 0)), &(xp(j, 0)));
                tripletList.emplace_back(i, j, value);
                tripletList.emplace_back(j, i, value); // Symmetric entry
            }
        }
    }
    
    result.setFromTriplets(tripletList.begin(), tripletList.end()); // Construct sparse matrix from triplet list

    return result; // Make sure this returns a format compatible with Python (e.g., via PyEigen)
});










  interface.def("value_diagonal", [](KernelInterface& self, py::array_t<double> x1, py::array_t<double> x2) {
    auto x1p = x1.unchecked<2>();
    auto x2p = x2.unchecked<2>();
    size_t n = x1p.shape(0);
    if (py::ssize_t(n) != x2p.shape(0) || x1p.shape(1) != py::ssize_t(self.ndim()) || x2p.shape(1) != py::ssize_t(self.ndim())) throw george::dimension_mismatch();
    py::array_t<double> result(n);
    auto resultp = result.mutable_unchecked<1>();
    for (size_t i = 0; i < n; ++i) {
      resultp(i) = self.value(&(x1p(i, 0)), &(x2p(i, 0)));
    }
    return result;
  });

  interface.def("gradient_general", [](KernelInterface& self, py::array_t<unsigned> which, py::array_t<double> x1, py::array_t<double> x2) {
    auto x1p = x1.unchecked<2>();
    auto x2p = x2.unchecked<2>();
    size_t n1 = x1p.shape(0), n2 = x2p.shape(0), size = self.size();
    if (x1p.shape(1) != py::ssize_t(self.ndim()) || x2p.shape(1) != py::ssize_t(self.ndim())) throw george::dimension_mismatch();
    py::array_t<double> result({n1, n2, size});
    auto resultp = result.mutable_unchecked<3>();
    auto w = which.unchecked<1>();
    unsigned* wp = (unsigned*)&(w(0));
    for (size_t i = 0; i < n1; ++i) {
      for (size_t j = 0; j < n2; ++j) {
        self.gradient(&(x1p(i, 0)), &(x2p(j, 0)), wp, &(resultp(i, j, 0)));
      }
    }
    return result;
  });

  interface.def("gradient_symmetric", [](KernelInterface& self, py::array_t<unsigned> which, py::array_t<double> x) {
    auto xp = x.unchecked<2>();
    size_t n = xp.shape(0), size = self.size();
    if (xp.shape(1) != py::ssize_t(self.ndim())) throw george::dimension_mismatch();
    py::array_t<double> result({n, n, size});
    auto resultp = result.mutable_unchecked<3>();
    auto w = which.unchecked<1>();
    unsigned* wp = (unsigned*)&(w(0));
    for (size_t i = 0; i < n; ++i) {
      self.gradient(&(xp(i, 0)), &(xp(i, 0)), wp, &(resultp(i, i, 0)));
      for (size_t j = i+1; j < n; ++j) {
        self.gradient(&(xp(i, 0)), &(xp(j, 0)), wp, &(resultp(i, j, 0)));
        for (size_t k = 0; k < size; ++k) resultp(j, i, k) = resultp(i, j, k);
      }
    }
    return result;
  });


  interface.def("gradient_sparse", [](KernelInterface& self, py::array_t<unsigned> which, py::array_t<double> x, const std::vector<std::vector<size_t>>& neighbors) {
      auto xp = x.unchecked<2>();
      size_t n = xp.shape(0);
      size_t size = self.size();
      if (xp.shape(1) != py::ssize_t(self.ndim())) throw george::dimension_mismatch();
      
      // Initialize a sparse matrix for each gradient output
      std::vector<Eigen::SparseMatrix<double>> result(size, Eigen::SparseMatrix<double>(n, n));

      // Initialize triplet list for storing non-zero elements
      std::vector<std::vector<Eigen::Triplet<double>>> tripletList(size);  // Size initialized

      auto w = which.unchecked<1>();
      unsigned* wp = (unsigned*)&(w(0));
      std::vector<double> grad_vector(size);

      for (size_t i = 0; i < n; ++i) {
          // Calculate the covariance for the points in the neighborhood
          self.gradient(&(xp(i, 0)), &(xp(i, 0)), wp, grad_vector.data());
          for (size_t k = 0; k < size; ++k) {
              tripletList[k].emplace_back(i, i, grad_vector[k]); // Diagonal element
          }

          for (size_t j : neighbors[i]) { // Use the list of neighbors
              if (i < j) { // Only compute for upper triangle
                  self.gradient(&(xp(i, 0)), &(xp(j, 0)), wp, grad_vector.data());
                  for (size_t k = 0; k < size; ++k) {
                      tripletList[k].emplace_back(i, j, grad_vector[k]);
                      tripletList[k].emplace_back(j, i, grad_vector[k]); // Symmetric entry
                  } 
              }
          }
      }

      // Construct sparse matrix from triplet list
      for (size_t k = 0; k < size; ++k) {
          result[k].setFromTriplets(tripletList[k].begin(), tripletList[k].end());
      }

      return result; // Return vector of sparse matrices
  });


  interface.def("x1_gradient_general", [](KernelInterface& self, py::array_t<double> x1, py::array_t<double> x2) {
    auto x1p = x1.unchecked<2>();
    auto x2p = x2.unchecked<2>();
    size_t n1 = x1p.shape(0), n2 = x2p.shape(0), ndim = self.ndim();
    if (x1p.shape(1) != py::ssize_t(ndim) || x2p.shape(1) != py::ssize_t(ndim)) throw george::dimension_mismatch();
    py::array_t<double> result({n1, n2, ndim});
    auto resultp = result.mutable_unchecked<3>();
    for (size_t i = 0; i < n1; ++i) {
      for (size_t j = 0; j < n2; ++j) {
        for (size_t k = 0; k < ndim; ++k) resultp(i, j, k) = 0.0;
        self.x1_gradient(&(x1p(i, 0)), &(x2p(j, 0)), &(resultp(i, j, 0)));
      }
    }
    return result;
  });

  interface.def("x2_gradient_general", [](KernelInterface& self, py::array_t<double> x1, py::array_t<double> x2) {
    auto x1p = x1.unchecked<2>();
    auto x2p = x2.unchecked<2>();
    size_t n1 = x1p.shape(0), n2 = x2p.shape(0), ndim = self.ndim();
    if (x1p.shape(1) != py::ssize_t(ndim) || x2p.shape(1) != py::ssize_t(ndim)) throw george::dimension_mismatch();
    py::array_t<double> result({n1, n2, ndim});
    auto resultp = result.mutable_unchecked<3>();
    for (size_t i = 0; i < n1; ++i) {
      for (size_t j = 0; j < n2; ++j) {
        for (size_t k = 0; k < ndim; ++k) resultp(i, j, k) = 0.0;
        self.x2_gradient(&(x1p(i, 0)), &(x2p(j, 0)), &(resultp(i, j, 0)));
      }
    }
    return result;
  });

  interface.def(py::pickle(
      [](const KernelInterface& self) {
        return py::make_tuple(self.kernel_spec());
      },
      [](py::tuple t) {
        if (t.size() != 1) throw std::runtime_error("Invalid state!");
        return new KernelInterface(t[0]);
      }
  ));


  //interface.def("__getstate__", [](const KernelInterface& self) {
  //  return std::make_tuple(self.kernel_spec());
  //});

  //interface.def("__setstate__", [](KernelInterface& self, py::tuple t) {
  //  if (t.size() != 1) throw std::runtime_error("Invalid state!");
  //  new (&self) KernelInterface(t[0]);
  //});
}
