# -*- coding: utf-8 -*-
from __future__ import division, print_function

__all__ = ["BasicSolver"]

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.sparse import csc_matrix, coo_matrix, issparse
from scipy.sparse.linalg import splu
from pdbridge import *
from dPy_BPACK_wrapper import *
import copy
import scipy
import time


class BasicSolver(object):
    """
    This is the most basic solver built using :func:`scipy.linalg.cholesky`.

    kernel (george.kernels.Kernel): A subclass of :class:`Kernel` specifying
        the kernel function.

    """
    def __init__(self, kernel, verbose=0, INT64=0, algo3d=0, compute_grad=0, model_sparse=0, model_bpack=0, debug=0, sym=0):
        self.kernel = kernel
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
        self.Kg = None
        self.K = None

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
            start = time.time()
            meta = {
                "coordinates": x,
                "kernel": self.kernel,
                "yerr": yerr.astype(np.float64),
                "id": 0
            }
            payload = {
                "block_func_module": "user_block_funcs_kernel",
                "block_func_name": "compute_block",
                "meta": meta
            }
            bpack_factor(payload, self.verbose, fid=0)
            end = time.time()
            if(self.verbose==1):
                print(f"Time spent in compress and invert K: {end - start} seconds")
            if(self.compute_grad==1):
                start = time.time()
                for g in range(self.kernel.full_size):
                    meta = {
                        "coordinates": x,
                        "kernel": self.kernel,
                        "yerr": yerr.astype(np.float64),
                        "id": g+1
                    }
                    payload = {
                        "block_func_module": "user_block_funcs_kernel",
                        "block_func_name": "compute_block",
                        "meta": meta
                    }
                    bpack_factor(payload, self.verbose, nofactor=True, fid=g+1)                    
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
                if(self.compute_grad==1):
                    start = time.time()
                    if self.model_sparse == 1:      
                        Kgs = self.kernel.get_gradient(x,nns=nns) 
                        # for i in range(len(Kgs)):
                        #     # print('initial Kgs[i].nnz',Kgs[i].nnz)
                        #     K_coo=Kgs[i].tocoo()
                        #     row_indices = K_coo.row
                        #     col_indices = K_coo.col
                        #     nonzero_mask = K_coo.data != 0
                        #     Kgs[i] = csc_matrix((K_coo.data[nonzero_mask], (row_indices[nonzero_mask], col_indices[nonzero_mask])), shape=K_coo.shape)
                        #     # print('final Kgs[i].nnz',Kgs[i].nnz)
                        self.Kg = Kgs
                    else: 
                        Kg = self.kernel.get_gradient(x)          
                        self.Kg = [csc_matrix(Kg[:, :, i]) for i in range(Kg.shape[-1])]                
                
                    end = time.time()
                    if(self.verbose==1):
                        print(f"Time spent in assembling Kgs: {end - start} seconds")


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
        if self.model_bpack == 1:
            sign,logdet = bpack_logdet(self.verbose,fid=0)
            log_det = sign*logdet            
        else:    
            if self.model_sparse == 1:
                # For sparse K, splu doesn't provide logdet. Use slogdet instead. 
                # sign, logdet = np.linalg.slogdet(K.toarray())
                sign,logdet = superlu_logdet(self.verbose)
                log_det = sign*logdet
            else:
                # For dense K
                log_det = 2 * np.sum(np.log(np.diag(self._factor[0])))
        
        return log_det

    def apply_forward(self,x,i):
        if self.model_bpack == 1:
            y=x.copy()
            bpack_mult(y, "N", verbosity=self.verbose,fid=i)
            return y
        else:
            if self.model_sparse == 1:
                if(i==0):
                    return self.K@x
                else:
                    return self.Kg[i-1]@x    
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
            if in_place:
                bpack_solve(y, self.verbose,fid=0)
                return y
            else:
                x=copy.deepcopy(y)
                bpack_solve(x, self.verbose,fid=0)
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
