# -*- coding: utf-8 -*-
from __future__ import division, print_function

__all__ = ["BasicSolver"]

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.sparse import csc_matrix, coo_matrix, issparse
from scipy.sparse.linalg import splu
# from pdbridge import *
from dPy_BPACK_wrapper import *
from ..metrics import Metric
from ..kernels import Product, ConstantKernel, ExpSquaredKernel, ExpKernel, Matern32Kernel, Matern52Kernel
import copy
import scipy
import time
import os


class BasicSolver(object):
    """
    This is the most basic solver built using :func:`scipy.linalg.cholesky`.

    kernel (george.kernels.Kernel): A subclass of :class:`Kernel` specifying
        the kernel function.

    """
    def __init__(self, kernel, verbose=0, INT64=0, algo3d=0, compute_grad=0, model_sparse=0, model_bpack=0, debug=0, sym=0, nprobe=10, bpack_scaled_geometry=0, bpack_s_op=1):
        self.kernel = kernel
        self.nprobe = nprobe # number of random probe vectors for the trace terms of the sparse gradient
        self._computed = False
        self._log_det = None
        self.verbose = verbose
        self.INT64 = INT64
        self.algo3d = algo3d
        self.debug = debug
        self.sym = sym
        self.compute_grad = compute_grad
        self.model_sparse = model_sparse
        self.model_bpack = model_bpack
        self.bpack_scaled_geometry = bpack_scaled_geometry # whether to divide each dimension of the points passed to butterflypack by the kernel length scale
        self.bpack_s_op = bpack_s_op # whether to give each butterflypack H2 matrix the absolute floor of its ID rank rule (H2_s_op, see _bpack_s_op)
        self.Kg = None
        self.K = None

    def _bpack_meta(self, x, yerr, gid):
        """
        The metadata passed to butterflypack. ``coordinates`` are the points used to build the
        butterflypack cluster tree, while the kernel entries are always evaluated with the original
        points (``kernel_coordinates`` if present, otherwise ``coordinates``). With
        bpack_scaled_geometry=1, each dimension of ``coordinates`` is divided by the (axis-aligned)
        kernel length scale, so that the admissibility condition sees the same distances as the
        anisotropic kernel.
        """
        meta = {
            "coordinates": x,
            "kernel": self.kernel,
            "yerr": yerr.astype(np.float64),
            "id": gid
        }
        if self.bpack_scaled_geometry == 1:
            meta["coordinates"] = np.ascontiguousarray(x * self._bpack_geometry_scale(x.shape[1]), dtype=np.float64)
            meta["kernel_coordinates"] = x
        return meta

    # The entry evaluator of the GPU backends (payload["gpu_entry"], registered by
    # Py_BPACK_worker.py; doc/gpu_kernels.md): K or one of its hyperparameter derivatives,
    # selected by params[1] (mode): with s_d the inverse squared length scales of the
    # coordinates passed to butterflypack, t_d = s_d (x_d - y_d)^2, q = sum_d t_d (r^2 of the
    # george kernels) and A the amplitude, the kernel k(q) and g(q) = -dk/dq are
    #   squared exponential: k = A exp(-q/2),                      g = k/2
    #   exponential:         k = A exp(-a), a = sqrt(q),           g = k/(2a) (0 at a = 0)
    #   Matern-3/2:          k = A (1 + a) exp(-a), a = sqrt(3q),  g = 3/2 A exp(-a)
    #   Matern-5/2:          k = A (1 + a + a^2/3) exp(-a), a = sqrt(5q),
    #                        g = 5/6 A (1 + a) exp(-a)
    # and, since d q / d log_M_d = -t_d,
    #   mode 0: k, plus yerr_i^2 when i == j      (the kernel K)
    #   mode 1: k                                 (d K / d log_constant)
    #   mode 2+d: g t_d                           (d K / d log_M_d, axis-aligned metrics)
    #   mode 5: g q                               (d K / d log_M, isotropic metrics)
    _BPACK_GPU_ENTRY_SOURCE = r"""
__device__ double bpack_entry(const double* x, long long i, const double* y, long long j,
                              const double* params, int dim) {
    const int mode = (int)params[1];
    double q = 0.0;
    for (int d = 0; d < dim; ++d) {
        const double t = x[d] - y[d];
        q += params[2 + d] * t * t;
    }
    double v = params[0] * exp(-0.5 * q);
    if (mode == 0) {
        if (i == j) v += params[2 + dim + i];
        return v;
    }
    if (mode == 1) return v;
    if (mode == 5) return 0.5 * v * q;
    const double t = x[mode - 2] - y[mode - 2];
    return 0.5 * v * params[mode] * t * t;
}
"""
    _BPACK_GPU_ENTRY_SOURCE_EXP = r"""
__device__ double bpack_entry(const double* x, long long i, const double* y, long long j,
                              const double* params, int dim) {
    const int mode = (int)params[1];
    double q = 0.0;
    for (int d = 0; d < dim; ++d) {
        const double t = x[d] - y[d];
        q += params[2 + d] * t * t;
    }
    const double a = sqrt(q);
    const double e = params[0] * exp(-a);
    if (mode <= 1) {
        double v = e;
        if (mode == 0 && i == j) v += params[2 + dim + i];
        return v;
    }
    if (a == 0.0) return 0.0;
    const double g = 0.5 * e / a;
    if (mode == 5) return g * q;
    const double t = x[mode - 2] - y[mode - 2];
    return g * params[mode] * t * t;
}
"""
    _BPACK_GPU_ENTRY_SOURCE_MATERN32 = r"""
__device__ double bpack_entry(const double* x, long long i, const double* y, long long j,
                              const double* params, int dim) {
    const int mode = (int)params[1];
    double q = 0.0;
    for (int d = 0; d < dim; ++d) {
        const double t = x[d] - y[d];
        q += params[2 + d] * t * t;
    }
    const double a = sqrt(3.0 * q);
    const double e = params[0] * exp(-a);
    if (mode <= 1) {
        double v = (1.0 + a) * e;
        if (mode == 0 && i == j) v += params[2 + dim + i];
        return v;
    }
    const double g = 1.5 * e;
    if (mode == 5) return g * q;
    const double t = x[mode - 2] - y[mode - 2];
    return g * params[mode] * t * t;
}
"""
    _BPACK_GPU_ENTRY_SOURCE_MATERN52 = r"""
__device__ double bpack_entry(const double* x, long long i, const double* y, long long j,
                              const double* params, int dim) {
    const int mode = (int)params[1];
    double q = 0.0;
    for (int d = 0; d < dim; ++d) {
        const double t = x[d] - y[d];
        q += params[2 + d] * t * t;
    }
    const double a = sqrt(5.0 * q);
    const double e = params[0] * exp(-a);
    if (mode <= 1) {
        double v = (1.0 + a + a * a / 3.0) * e;
        if (mode == 0 && i == j) v += params[2 + dim + i];
        return v;
    }
    const double g = (5.0 / 6.0) * (1.0 + a) * e;
    if (mode == 5) return g * q;
    const double t = x[mode - 2] - y[mode - 2];
    return g * params[mode] * t * t;
}
"""

    def _bpack_gpu_entry(self, x, yerr, gid, geometry_scale):
        """
        The GPU entry evaluator of the kernel for butterflypack's GPU backends, as CUDA source
        text the library compiles with NVRTC (payload["gpu_entry"], doc/gpu_kernels.md): a
        constant times an axis-aligned or isotropic squared exponential, exponential,
        Matern-3/2 or Matern-5/2 kernel of the coordinates passed to butterflypack (the points times
        geometry_scale), for gid = 0 with yerr^2 on the diagonal, else its derivative in
        parameter gid-1 of the kernel. None if the kernel has no such form.
        """
        ndim = x.shape[1]
        if ndim > 3:
            return None
        parts = [self.kernel.k1, self.kernel.k2] if isinstance(self.kernel, Product) else [self.kernel]
        constants = [k for k in parts if isinstance(k, ConstantKernel)]
        sources = {ExpSquaredKernel: self._BPACK_GPU_ENTRY_SOURCE,
                   ExpKernel: self._BPACK_GPU_ENTRY_SOURCE_EXP,
                   Matern32Kernel: self._BPACK_GPU_ENTRY_SOURCE_MATERN32,
                   Matern52Kernel: self._BPACK_GPU_ENTRY_SOURCE_MATERN52}
        squared = [k for k in parts if type(k) in sources]
        if len(squared) != 1 or len(constants) + 1 != len(parts) or squared[0].block is not None:
            return None
        metric = squared[0].metric
        if metric.metric_type == 2:
            return None
        inverse_metric = np.zeros(ndim)
        for i, axis in enumerate(metric.axes):
            inverse_metric[axis] = 1.0 / (np.diag(metric.to_matrix())[i] * geometry_scale[axis]**2)
        if gid == 0:
            mode = 0
        else:
            name = self.kernel.get_parameter_names(include_frozen=True)[gid-1]
            if name.endswith("log_constant"):
                mode = 1
            elif ":log_M_" in name:
                mode = 5 if metric.metric_type == 0 else 2 + metric.axes[int(name.split("_")[-1])]
            else:
                return None
        # (a ConstantKernel is ndim exp(log_constant): take its value)
        amplitude = float(np.prod([k.get_value(np.zeros((1, k.ndim)))[0, 0] for k in constants]))
        params = [amplitude, float(mode)] + inverse_metric.tolist()
        if mode == 0:
            params = np.concatenate((params, np.atleast_1d(yerr).astype(np.float64)**2 * np.ones(x.shape[0])))
        return {"source": sources[type(squared[0])], "params": params,
                "flags": 3}  # BPACK_GPU_SYMMETRIC | BPACK_GPU_COORDINATES

    # The largest entry of the length-scale derivative operators g t_d and g q (the GPU sources above),
    # max_q q g(q), in units of the amplitude: squared exponential 1/e (at r = sqrt(2)), exponential
    # 1/(2e) (r = 1), Matern-3/2 2/e^2 (r = 2/sqrt(3)), Matern-5/2 (1+sqrt(3))^2 (2+sqrt(3))
    # exp(-1-sqrt(3)) / 6 (r = (1+sqrt(3))/sqrt(5)). Separating two points along axis d only gives
    # t_d = q, so every axis has the same largest entry.
    _BPACK_DERIVATIVE_MAX_ENTRY = {
        ExpSquaredKernel: np.exp(-1.0),
        ExpKernel: 0.5 * np.exp(-1.0),
        Matern32Kernel: 2.0 * np.exp(-2.0),
        Matern52Kernel: (1.0 + np.sqrt(3.0))**2 * (2.0 + np.sqrt(3.0)) * np.exp(-1.0 - np.sqrt(3.0)) / 6.0,
    }

    def _bpack_s_op(self, yerr, gid):
        """
        butterflypack's H2_s_op of one matrix (bpack_s_op=1): every H2 ID then keeps a column while
        |R_kk| > max(tol_comp |R_11|, tol_comp s_op), so blocks negligible next to the matrix get rank 0
        instead of being compressed to tol_comp of their own size. s_op is in the units of the entries:
          gid = 0, the factorization of K: min(yerr^2), the smallest diagonal term (the white noise and
            the jitter), a lower bound of K's smallest eigenvalue 1/||K^-1||, which the solves and the
            log-determinant are accurate against;
          gid > 0, the compression-only derivative in kernel parameter gid-1: its largest entry (at most
            its norm), the amplitude for log_constant and the amplitude times
            _BPACK_DERIVATIVE_MAX_ENTRY for log_M.
        0 (the relative rule only) when bpack_s_op is 0 or the kernel is not a constant times one of
        the kernels of _bpack_gpu_entry.
        """
        if not self.bpack_s_op:
            return 0.0
        if gid == 0:
            return float(np.min(np.atleast_1d(yerr).astype(np.float64)**2))
        parts = [self.kernel.k1, self.kernel.k2] if isinstance(self.kernel, Product) else [self.kernel]
        constants = [k for k in parts if isinstance(k, ConstantKernel)]
        stationary = [k for k in parts if type(k) in self._BPACK_DERIVATIVE_MAX_ENTRY]
        if len(stationary) != 1 or len(constants) + 1 != len(parts):
            return 0.0
        # (a ConstantKernel is ndim exp(log_constant): take its value)
        amplitude = float(np.prod([k.get_value(np.zeros((1, k.ndim)))[0, 0] for k in constants]))
        name = self.kernel.get_parameter_names(include_frozen=True)[gid-1]
        if name.endswith("log_constant"):
            return amplitude
        if ":log_M_" in name:
            return amplitude * float(self._BPACK_DERIVATIVE_MAX_ENTRY[type(stationary[0])])
        return 0.0

    def _bpack_geometry_scale(self, ndim):
        """The per-dimension scaling of the coordinates passed to butterflypack (1 unless bpack_scaled_geometry)."""
        if self.bpack_scaled_geometry == 1:
            return self._inverse_lengthscales(ndim)
        return np.ones(ndim)

    def _inverse_lengthscales(self, ndim):
        metrics = []
        stack = [self.kernel]
        while stack:
            k = stack.pop()
            if isinstance(k, Metric):
                metrics.append(k)
            elif hasattr(k, "models"):
                stack.extend(k.models.values())
        if len(metrics) != 1 or metrics[0].metric_type == 2:
            raise NotImplementedError("bpack_scaled_geometry requires a kernel with exactly one axis-aligned or isotropic metric")
        metric = metrics[0]
        scale = np.ones(ndim)
        scale[list(metric.axes)] = 1.0 / np.sqrt(np.diag(metric.to_matrix()))
        return scale

    @property
    def computed(self):
        """
        A flag indicating whether or not the covariance matrix was computed
        and factorized (using the :func:`compute` method).

        """
        return self._computed

    @computed.setter
    def computed(self, v):
        self._computed = v

    @property
    def log_determinant(self):
        """
        The log-determinant of the covariance matrix. This will only be
        non-``None`` after calling the :func:`compute` method.

        """
        return self._log_det

    @log_determinant.setter
    def log_determinant(self, v):
        self._log_det = v

    def compute(self, x, nns, yerr):
        """
        Compute and factorize the covariance matrix.

        Args:
            x (ndarray[nsamples, ndim]): The independent coordinates of the
                data points.
            yerr (ndarray[nsamples] or float): The Gaussian uncertainties on
                the data points at coordinates ``x``. These values will be
                added in quadrature to the diagonal of the covariance matrix.
        """
        # Compute the kernel matrix.

        if self.model_bpack == 1:
            K=None
            self._n = x.shape[0] 
            start = time.time()
            meta = self._bpack_meta(x, yerr, 0)
            if(self.verbose==1 and self.bpack_scaled_geometry==1):
                print("bpack scaled geometry, inverse length scales: ", self._inverse_lengthscales(x.shape[1]))
            payload = {
                "block_func_filepath": os.path.abspath(__file__), ## this assumes user_block_funcs_kernel.py is located in the same directory as basic.py
                "block_func_module": "user_block_funcs_kernel",
                "block_func_name": "compute_block",
                "meta": meta
            }
            gpu_entry = self._bpack_gpu_entry(x, yerr, 0, self._bpack_geometry_scale(x.shape[1]))
            if gpu_entry is not None:
                payload["gpu_entry"] = gpu_entry
            # this matrix's H2_s_op (the same as bpack_factor(..., h2_s_op=); ignored by HODLR)
            payload["_butterflypack_H2_s_op"] = self._bpack_s_op(yerr, 0)
            if(self.verbose==1 and self.bpack_s_op):
                print("bpack H2_s_op: K %.3e" % payload["_butterflypack_H2_s_op"])
            bpack_factor(payload, fid=0)
            end = time.time()
            if(self.verbose==1):
                print(f"Time spent in compress and invert K: {end - start} seconds")
            if(self.compute_grad==1):
                start = time.time()
                for g in range(self.kernel.full_size):
                    meta = self._bpack_meta(x, yerr, g+1)
                    payload = {
                        "block_func_filepath": os.path.abspath(__file__),
                        "block_func_module": "user_block_funcs_kernel",
                        "block_func_name": "compute_block",
                        "meta": meta
                    }
                    gpu_entry = self._bpack_gpu_entry(x, yerr, g+1, self._bpack_geometry_scale(x.shape[1]))
                    if gpu_entry is not None:
                        payload["gpu_entry"] = gpu_entry
                    payload["_butterflypack_H2_s_op"] = self._bpack_s_op(yerr, g+1)
                    if(self.verbose==1 and self.bpack_s_op):
                        print("bpack H2_s_op: d K / d %s %.3e" % (self.kernel.get_parameter_names(include_frozen=True)[g],
                                                                  payload["_butterflypack_H2_s_op"]))
                    bpack_factor(payload, nofactor=True, fid=g+1)
                end = time.time()
                if(self.verbose==1):
                    print(f"Time spent in compress Kgs: {end - start} seconds")
        else:
            start = time.time()
            if self.model_sparse == 1:      
                K = self.kernel.get_value(x,nns=nns) 
                print('initial K.nnz',K.nnz)
                # K_coo=K.tocoo()
                # row_indices = K_coo.row
                # col_indices = K_coo.col
                # nonzero_mask = K_coo.data != 0
                # K = csc_matrix((K_coo.data[nonzero_mask], (row_indices[nonzero_mask], col_indices[nonzero_mask])), shape=K.shape)
                # print('final K.nnz',K.nnz)
            else:
                K = self.kernel.get_value(x)
            end = time.time()
            if(self.verbose==1):
                print(f"Time spent in assembling K: {end - start} seconds")

            self._n = x.shape[0]     

            if self.model_sparse == 1 :
                # diag_yerr = csc_matrix(np.diag(yerr ** 2))
                # K = K + diag_yerr
                K.setdiag(K.diagonal() + yerr**2)
                # the gradient matrices are only needed by grad_log_likelihood, not by the
                # likelihood-only evaluations of the line search, so assemble them on first use
                self.Kg = None
                self._Kg_args = (x, nns)


                # K_copy = copy.deepcopy(K)
                # K_dense = K_copy.toarray()
                # self._factor = (cholesky(K_dense, overwrite_a=True, lower=False), False)
                # print("logdet_dense: ",self._calculate_log_determinant(K_dense))
            else:
                eye = np.eye(K.shape[0])
                K += eye * (yerr ** 2)  # Adjust K with yerr
            
            self.K=K
            
            # Factor the matrix using sparse Cholesky factorization if K is sparse
            if self.model_sparse == 1:
                # self._factor = splu(K)
                superlu_factor(K, self.INT64, self.algo3d, self.verbose)
            else:
                self._factor = (cholesky(K, overwrite_a=True, lower=False), False)


        self.log_determinant = self._calculate_log_determinant(K)
        # print("self.log_determinant: ",self.log_determinant)
        self.computed = True



    def _calculate_log_determinant(self, K):
        """
        Calculate the log-determinant of the covariance matrix.
        Uses the determinant of the Cholesky factor.

        Args:
            K (ndarray or csr_matrix): The covariance matrix.

        Returns:
            float: The log-determinant value.
        """
        # A hierarchical factorisation of a covariance matrix can come back indefinite at
        # hyperparameters where the compression breaks down, which the sign of the determinant
        # reports.  The log-determinant itself is legitimately negative (every eigenvalue of a
        # covariance matrix with a small nugget is below one), so the sign is the only usable
        # signal; it is recorded here so that a sampler can reject such a state. The value used is
        # log|det K| whatever the sign: multiplying it by the sign (as before) negated the
        # log-determinant whenever a solver reported a negative sign, which butterflypack's logdet
        # did for SPD matrices at some sizes (its factorization's own logdet had the sign 1 and the
        # same magnitude), and the likelihood then disagreed with its gradient.
        self.log_determinant_sign = 1.0
        if self.model_bpack == 1:
            sign,logdet = bpack_logdet(fid=0)
            self.log_determinant_sign = sign
            if sign < 0 and self.verbose == 1:
                print("bpack logdet: negative determinant sign reported (log|det| %.10e used)" % logdet)
            log_det = logdet
        else:    
            if self.model_sparse == 1:
                # For sparse K, splu doesn't provide logdet. Use slogdet instead. 
                # sign, logdet = np.linalg.slogdet(K.toarray())
                sign,logdet = superlu_logdet(self.verbose)
                self.log_determinant_sign = sign
                log_det = logdet  # log|det K| (see above)
            else:
                # For dense K
                log_det = 2 * np.sum(np.log(np.diag(self._factor[0])))
        
        return log_det

    def get_Kg(self):
        """
        The gradient matrices dK/dtheta of the sparse covariance matrix, assembled on first use
        after compute().
        """
        if self.Kg is None:
            start = time.time()
            x, nns = self._Kg_args
            self.Kg = self.kernel.get_gradient(x,nns=nns)
            end = time.time()
            if(self.verbose==1):
                print(f"Time spent in assembling Kgs: {end - start} seconds")
        return self.Kg

    def apply_forward(self,x,i):
        if self.model_bpack == 1:
            y=bpack_mult(x, "N", fid=i)
            # print("in apply apply_forward:",np.linalg.norm(x),np.linalg.norm(y))
            return y
        else:
            if self.model_sparse == 1:
                if(i==0):
                    return self.K@x
                else:
                    return self.get_Kg()[i-1]@x
            else:
                if(i==0):
                    return self.K@x    
                else:
                    raise Exception('self.Kg has not been computed when self.model_sparse is 0 in apply_forward')        

    def apply_inverse(self, y, in_place=False):
        """
        Apply the inverse of the covariance matrix to the input by solving

        .. math::

            K\,x = y

        Args:
            y (ndarray[nsamples] or ndarray[nsamples, nrhs]): The vector or
                matrix :math:`y`.
            in_place (Optional[bool]): Should the data in ``y`` be overwritten
                with the result :math:`x`? (default: ``False``)

        Returns:
            ndarray or sparse matrix: The result of the operation.
        """
        if self.model_bpack == 1:
            x = bpack_solve(y,fid=0)
            if in_place:
                np.copyto(y, x)
                return y
            else:
                return x
        else:
            if self.model_sparse == 1:
                if in_place:
                    superlu_solve(y, self.verbose)
                    return y
                else:
                    x=copy.deepcopy(y)
                    superlu_solve(x, self.verbose)
                    return x
            else:
                return cho_solve(self._factor, y, overwrite_b=in_place)

    def dot_solve(self, y):
        """
        Compute the inner product of a vector with the inverse of the
        covariance matrix applied to itself:

        .. math::

            y\,K^{-1}\,y

        Args:
            y (ndarray[nsamples]): The vector :math:`y`.

        Returns:
            float: Result of the inner product.
        """
        return np.dot(y.T, self.apply_inverse(y))

    def apply_sqrt(self, r):
        """
        Apply the Cholesky square root of the covariance matrix to the input
        vector or matrix.

        Args:
            r (ndarray[nsamples] or ndarray[nsamples, nrhs]): The input vector
                or matrix.

        Returns:
            ndarray or sparse matrix: Result of the multiplication with the Cholesky factor.
        """
        if self.model_bpack == 1:
            raise NotImplementedError("apply_sqrt is not implemented for model_bpack yet")
        else:    
            if self.model_sparse == 1:
                raise NotImplementedError("apply_sqrt is not implemented for sparse matrix yet")
            else:
                return np.dot(r, self._factor[0])  # Dense multiplication

    def get_inverse(self):
        """
        Get the dense inverse covariance matrix. This is used for computing
        gradients, but it is not recommended in general.
        """
        return self.apply_inverse(np.eye(self._n), in_place=True)


    def get_full(self,i):
        """
        Get the dense covariance matrix or its derivative. This is used for computing
        gradients, but it is not recommended in general.
        """
        return self.apply_forward(np.eye(self._n), i=i)