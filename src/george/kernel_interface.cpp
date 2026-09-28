#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "george/parser.h"
#include "george/kernels.h"
#include "george/exceptions.h"

#include <vector>
#include <algorithm>
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
    double get_parameter (size_t i) const { return kernel_->get_parameter(i); };
    double get_cutoff () const { return kernel_->get_cutoff(); };
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
    {
      py::gil_scoped_release release;
      #pragma omp parallel for schedule(static)
      for (py::ssize_t i = 0; i < py::ssize_t(n1); ++i) {
        for (size_t j = 0; j < n2; ++j) {
          resultp(i, j) = self.value(&(x1p(i, 0)), &(x2p(j, 0)));
        }
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


  // Sparsity pattern of the (symmetric) covariance matrix of a compactly supported kernel: the diagonal and
  // the candidate neighbours (nbr_idx, row_ptr: CSR, e.g. from BallTree.query_radius with a radius >= the
  // cutoff) whose Euclidean distance is <= radius, sorted in each row. Returns (indices, indptr), which
  // are both the CSR and the CSC arrays of the symmetric matrix.
  interface.def("sparse_pattern", [](KernelInterface& self, py::array_t<double> x, py::array_t<int64_t> nbr_idx, py::array_t<int64_t> row_ptr, double radius) {
    auto xp = x.unchecked<2>();
    auto idx = nbr_idx.unchecked<1>();
    auto ptr = row_ptr.unchecked<1>();
    const py::ssize_t n = xp.shape(0), ndim = xp.shape(1);
    const double r2max = radius * radius;
    auto within = [&](py::ssize_t i, int64_t j) {
      double r2 = 0;
      for (py::ssize_t d = 0; d < ndim; ++d) r2 += (xp(i, d) - xp(j, d)) * (xp(i, d) - xp(j, d));
      return j != i && r2 <= r2max;
    };
    py::array_t<int64_t> indptr(n + 1);
    auto ip = indptr.mutable_unchecked<1>();
    std::vector<int64_t> counts(n);
    {
      py::gil_scoped_release release;
      #pragma omp parallel for schedule(dynamic, 4096)
      for (py::ssize_t i = 0; i < n; ++i) {
        int64_t c = 1; // the diagonal
        for (int64_t k = ptr(i); k < ptr(i + 1); ++k) if (within(i, idx(k))) ++c;
        counts[i] = c;
      }
    }
    ip(0) = 0;
    for (py::ssize_t i = 0; i < n; ++i) ip(i + 1) = ip(i) + counts[i];
    py::array_t<int32_t> indices(ip(n));
    auto ind = indices.mutable_unchecked<1>();
    {
      py::gil_scoped_release release;
      #pragma omp parallel for schedule(dynamic, 4096)
      for (py::ssize_t i = 0; i < n; ++i) {
        int64_t pos = ip(i);
        ind(pos++) = int32_t(i);
        for (int64_t k = ptr(i); k < ptr(i + 1); ++k) if (within(i, idx(k))) ind(pos++) = int32_t(idx(k));
        std::sort(&ind(ip(i)), &ind(ip(i)) + (ip(i + 1) - ip(i)));
      }
    }
    return py::make_tuple(indices, indptr);
  });

  // Values of the kernel matrix at the entries of a sparsity pattern (see sparse_pattern), in its order.
  interface.def("value_sparse", [](KernelInterface& self, py::array_t<double> x, py::array_t<int32_t> indices, py::array_t<int64_t> indptr) {
    auto xp = x.unchecked<2>();
    auto ind = indices.unchecked<1>();
    auto ip = indptr.unchecked<1>();
    if (xp.shape(1) != py::ssize_t(self.ndim())) throw george::dimension_mismatch();
    const py::ssize_t n = ip.shape(0) - 1;
    py::array_t<double> values(ind.shape(0));
    auto v = values.mutable_unchecked<1>();
    {
      py::gil_scoped_release release;
      #pragma omp parallel for schedule(dynamic, 4096)
      for (py::ssize_t i = 0; i < n; ++i) {
        for (int64_t k = ip(i); k < ip(i + 1); ++k) v(k) = self.value(&(xp(i, 0)), &(xp(ind(k), 0)));
      }
    }
    return values;
  });

  // Gradients of the kernel matrix with respect to the parameters at the entries of a sparsity pattern
  // (see sparse_pattern): an array (number of parameters, number of entries).
  interface.def("gradient_sparse", [](KernelInterface& self, py::array_t<unsigned> which, py::array_t<double> x, py::array_t<int32_t> indices, py::array_t<int64_t> indptr) {
    auto xp = x.unchecked<2>();
    auto ind = indices.unchecked<1>();
    auto ip = indptr.unchecked<1>();
    if (xp.shape(1) != py::ssize_t(self.ndim())) throw george::dimension_mismatch();
    const py::ssize_t n = ip.shape(0) - 1, nnz = ind.shape(0);
    const size_t size = self.size();
    auto w = which.unchecked<1>();
    const unsigned* wp = (const unsigned*)&(w(0));
    py::array_t<double> result({py::ssize_t(size), nnz});
    auto g = result.mutable_unchecked<2>();
    {
      py::gil_scoped_release release;
      #pragma omp parallel
      {
        std::vector<double> grad(size);
        #pragma omp for schedule(dynamic, 4096)
        for (py::ssize_t i = 0; i < n; ++i) {
          for (int64_t k = ip(i); k < ip(i + 1); ++k) {
            std::fill(grad.begin(), grad.end(), 0.0);
            self.gradient(&(xp(i, 0)), &(xp(ind(k), 0)), wp, grad.data());
            for (size_t p = 0; p < size; ++p) g(p, k) = grad[p];
          }
        }
      }
    }
    return result;
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


  interface.def("get_cutoff", [](const KernelInterface& self) {
   return self.get_cutoff();
  });

  //interface.def("__getstate__", [](const KernelInterface& self) {
  //  return std::make_tuple(self.kernel_spec());
  //});

  //interface.def("__setstate__", [](KernelInterface& self, py::tuple t) {
  //  if (t.size() != 1) throw std::runtime_error("Invalid state!");
  //  new (&self) KernelInterface(t[0]);
  //});
}
