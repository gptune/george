# -*- coding: utf-8 -*-
"""
Gaussian process regression with the INLA/SPDE approach on a tensor-product lattice (2D and 3D).

The latent function x is a Gaussian Markov random field on the nodes of a tensor-product lattice
that covers the inputs, extended on each side by a buffer whose cells grow geometrically (the SPDE
is solved with Neumann boundary conditions, which inflate the variance within about one range of
the boundary). Its precision matrix

    Q = tau^2 C p(S),    S = C^-1 sum_d Lambda_d G_d,

discretizes (1 - div Lambda grad)^(alpha/2) (tau x) = W, the stationary solution of which is a Matern
field with smoothness nu = alpha - d/2, marginal variance sigma^2 and length scale
l_d = sqrt(2 nu Lambda_d) along dimension d (correlation M_nu(sqrt(2 nu) r), r = |(s - s') / l|).
C (diagonal) and G_d are the lumped mass matrix and the stiffness matrix of dimension d of the
multilinear finite elements on the lattice. For an integer alpha, p(s) = (1 + s)^alpha (alpha = 2:
Q = tau^2 L C^-1 L with L = C + sum_d Lambda_d G_d); otherwise p is the polynomial of degree
ceil(alpha) of the "parsimonious" approximation of R-INLA (Lindgren et al. 2011), which minimizes a
weighted L2 error of the spectral density. The default is alpha = 2 (nu = 1 in 2D, nu = 1/2 in 3D).
tau^2 is set so that the marginal variance of the field of spectral density 1 / p is sigma^2.

The observations are y = mean + A x + e, e ~ N(0, s^2 I), A interpolating multilinearly from the
lattice nodes (a selection matrix for inputs on the nodes), and mean is the sample mean of y. With
this Gaussian likelihood, the log marginal likelihood computed by INLA is exact:

    log p(y) = 1/2 log|Q| - 1/2 log|Q_c| - n/2 log s^2 - |r|^2 / (2 s^2) + 1/2 b' Q_c^-1 b - n/2 log 2 pi

with Q_c = Q + A'A / s^2, r = y - mean and b = A'r / s^2. On the lattice, log|Q| is analytic (the
eigenvalues of C^-1 L are sums of the generalized eigenvalues of the 1D matrices), so Q_c is the only
matrix factored, with SuperLU_DIST through pdbridge. Its sparsity pattern does not depend on the
hyperparameters, so after the first factorization pdbridge refactors it reusing its ordering and
symbolic factorization.

The predictive mean is A_* Q_c^-1 b + mean. The predictive variances (of the latent function, without
the noise) come from nsamples samples x_s ~ N(0, Q_c^-1) of the posterior of x, drawn by solving
Q_c x_s = w_s with w_s = tau C V p^1/2 z1 + A' z2 / s ~ N(0, Q_c) (V below). For every lattice cell, the
covariance block of its 2^d nodes is estimated with the Rao-Blackwellized estimator

    Sigma_BB = Q_BB^-1 + mean_s z_s z_s',   z_s = Q_BB^-1 (Q_c x_s)_B - (x_s)_B = -E[x_B | x_-B],

restricted to the cell nodes, B being the cell and a layer of nodes around it (4^d nodes). Its first
term is exact: the samples only estimate the variance of E[x_B | x_-B].

The gradient of the log-likelihood is exact except for two traces of matrices of the size of the
observations, tr(A Q_c^-1 A') and tr((Q^-1 - Q_c^-1) dQ) = w tr(A Q_c^-1 dQ Q^-1 A') (the part of
tr(Q_c^-1 dQ) not cancelled by the derivative of log|Q|), Hutchinson estimates with fixed probes from
one multi-RHS solve, exact if there are fewer observations than probes. The Fisher information (for
the MALA sampler) is estimated in the same way. Q^-1 is applied with the M_d-orthonormal generalized
eigenvectors V_d of the 1D matrices (G_d V_d = M_d V_d mu_d): Q^-1 = tau^-2 V diag(1 / p(s)) V',
V = kron(V_d), s = sum_d Lambda_d mu_d, and log|Q| = n log tau^2 + log|C| + sum log p(s).
"""

from __future__ import division, print_function

__all__ = ["INLAGP"]

import itertools
import time
from math import comb, factorial

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import coo_matrix, csc_matrix, diags, identity, kron
from scipy.special import gamma as gamma_function

# pdbridge holds a single factorization: every INLAGP factorization increments this count, and an
# INLAGP whose last factorization is not the latest one refactors before solving
_factorization_count = [0]


def _superlu():
    """
    The pdbridge functions: the ones attached to george.solvers.basic (as done by the drivers using
    the sparse george kernels), otherwise pdbridge's.
    """
    from .solvers import basic
    try:
        return basic.superlu_factor, basic.superlu_logdet, basic.superlu_solve
    except AttributeError:
        import pdbridge
        return pdbridge.superlu_factor, pdbridge.superlu_logdet, pdbridge.superlu_solve


class _INLAKernel(object):
    """The attributes Model_George reads from the kernel of its model."""
    kernel_type = "INLA"

    def __init__(self, ndim, full_size):
        self.ndim = ndim
        self.full_size = full_size


def _infer_axis(values):
    """
    The evenly spaced grid, between the smallest and largest values, of which all the values are
    nodes (up to round-off). Missing nodes are allowed.
    """
    u = np.sort(values)
    width = u[-1] - u[0]
    if width <= 0:
        raise ValueError("INLA: all the inputs have the same coordinate along a dimension")
    u = u[np.concatenate([[True], np.diff(u) > 1e-9 * width])]
    h = np.min(np.diff(u))
    count = int(round(width / h)) + 1
    if np.max(np.abs(u - (u[0] + np.rint((u - u[0]) / h) * h))) > 1e-6 * h:
        raise ValueError("INLA: the inputs are not on a lattice, set model_inla_shape (the number "
                         "of lattice nodes per dimension)")
    return np.linspace(u[0], u[-1], count)


def _buffer_offsets(h, width, ratio):
    """Distances to the boundary of the nodes of a buffer whose cells grow by ratio from h."""
    offsets = []
    total = 0.0
    step = h
    while total < width:
        step *= ratio
        total += step
        offsets.append(total)
    return np.array(offsets)


def _fem_1d(z):
    """
    The lumped mass (diagonal), and the diagonal and off-diagonal of the stiffness matrix of the
    linear finite elements on the nodes z (natural boundary conditions).
    """
    h = np.diff(z)
    mass = np.zeros(len(z))
    mass[:-1] += h / 2
    mass[1:] += h / 2
    stiff_diag = np.zeros(len(z))
    stiff_diag[:-1] += 1 / h
    stiff_diag[1:] += 1 / h
    return mass, stiff_diag, -1 / h


def _kron(factors):
    out = factors[0]
    for f in factors[1:]:
        out = kron(out, f, format="csr")
    return out


def _smoothness_polynomial(alpha):
    """
    The coefficients c_0, ..., c_m (m = ceil(alpha)) of p(s) = sum_j c_j s^j approximating the spectral
    function (1 + s)^alpha of the SPDE: exact for an integer alpha, otherwise the "parsimonious"
    approximation of R-INLA (inla.spde2.matern, for 0 < alpha < 2) generalized to any alpha:
    p(s) = sum_i a_i (1 + s)^i where, with t = 1 / (1 + s), q(t) = sum_i a_i t^(m-i) is the least-squares
    approximation of t^(m-alpha) on [0, 1] with the weight t^(lambda-1), lambda = alpha - floor(alpha).
    """
    m = int(np.ceil(alpha - 1e-12))
    if abs(alpha - round(alpha)) < 1e-12:
        return np.array([comb(m, j) for j in range(m + 1)], dtype=float)
    lam = alpha - np.floor(alpha)
    i = np.arange(m + 1)
    a = np.linalg.solve(1.0 / (2 * m - i[:, None] - i[None, :] + lam), 1.0 / (2 * m - i + lam - alpha))
    return np.array([sum(a[k] * comb(k, j) for k in range(j, m + 1)) for j in range(m + 1)])


def _variance_integral(coefficients, alpha, ndim):
    """
    int_0^inf s^(d/2-1) / p(s) ds, with which the marginal variance of the field of spectral density
    1 / p(|w|^2) is 1 / ((4 pi)^(d/2) Gamma(d/2) tau^2 sqrt(det Lambda)) times it.
    """
    if abs(alpha - round(alpha)) < 1e-12:
        return gamma_function(ndim / 2.0) * gamma_function(alpha - ndim / 2.0) / gamma_function(alpha)
    value, _ = quad(lambda s: s ** (ndim / 2.0 - 1) / np.polynomial.polynomial.polyval(s, coefficients), 0, np.inf, limit=200)
    return value


class INLAGP(object):
    """
    A Gaussian process with the INLA/SPDE Matern model on a tensor-product lattice, with the parts of
    the interface of george.GP used by GPTune's Model_George. The parameter vector is
    [log s^2, log sigma^2, log l_1^2, ..., log l_d^2] (a single length scale if isotropic).

    Args:
        ndim (int): 2 or 3.
        nu (None or float): the smoothness nu > 0 of the Matern field (exact if nu + ndim/2 is an
            integer, approximated otherwise); None for alpha = 2 (nu = 2 - ndim/2).
        noise_variance, amplitude (float): the initial s^2 and sigma^2.
        lengthscale (float or array): the initial length scale(s).
        isotropic (bool): one length scale for all the dimensions.
        shape (None, int or list of ints): the number of lattice nodes per dimension between the
            bounds. If None, the inputs must be on an evenly spaced lattice, which is used.
        bounds (None or list of (low, high)): the region covered by the lattice when shape is given
            (extended to contain the inputs); the bounding box of the inputs if None.
        buffer (float): the width of the buffer added on each side of the lattice, which should be
            at least the largest range 2 l of the model.
        buffer_ratio (float): the ratio of the sizes of consecutive buffer cells.
        nsamples (int): the number of posterior samples of the variance estimator.
        batch (int): the number of samples drawn by one multi-RHS solve.
        rb_ring (int): the number of layers of nodes around a cell added to its conditioning block
            in the variance estimator ((2 + 2 rb_ring)^d nodes).
        nprobe (int): the number of Hutchinson probe vectors of the traces in the gradient and the
            Fisher information (the traces are exact if nprobe is at least the dimension).
        seed (int): the seed of the random numbers of the samples.
        INT64, algo3d: passed to superlu_factor.
    """

    def __init__(self, ndim, noise_variance=1e-6, amplitude=1.0, lengthscale=0.1, isotropic=False,
                 shape=None, bounds=None, buffer=0.5, buffer_ratio=1.2, nsamples=128, batch=32,
                 rb_ring=1, nprobe=64, seed=0, verbose=0, INT64=0, algo3d=0, nu=None):
        if ndim not in (2, 3):
            raise ValueError("INLA supports only 2D and 3D inputs, got %d dimensions" % ndim)
        self.ndim = ndim
        self.nu = 2.0 - ndim / 2.0 if nu is None else float(nu)
        if self.nu <= 0:
            raise ValueError("INLA: the smoothness nu must be positive, got %g" % self.nu)
        self.alpha = self.nu + ndim / 2.0
        # p(s) and the constant of tau^2 = constant / (sigma^2 sqrt(det Lambda))
        self._polynomial = _smoothness_polynomial(self.alpha)
        self._tau2_constant = (_variance_integral(self._polynomial, self.alpha, ndim)
                               / ((4 * np.pi) ** (ndim / 2.0) * gamma_function(ndim / 2.0)))
        self.isotropic = bool(isotropic)
        nlength = 1 if self.isotropic else ndim
        lengthscale = np.broadcast_to(np.asarray(lengthscale, dtype=float), (nlength,))
        self._params = np.concatenate([[np.log(noise_variance), np.log(amplitude)], np.log(lengthscale ** 2)])
        self.kernel = _INLAKernel(ndim, len(self._params))
        if shape is not None:
            shape = [int(s) for s in np.broadcast_to(np.asarray(shape), (ndim,))]
            if min(shape) < 2:
                raise ValueError("INLA: the lattice needs at least 2 nodes per dimension")
        self.shape = shape
        self.bounds = bounds
        self.buffer = float(buffer)
        self.buffer_ratio = float(buffer_ratio)
        self.nsamples = int(nsamples)
        self.batch = int(batch)
        self.rb_ring = int(rb_ring)
        self.nprobe = int(nprobe)
        self.seed = seed
        self.verbose = verbose
        self.INT64 = INT64
        self.algo3d = algo3d
        self._x = None
        self._yerr2 = 0.0
        self._factor_id = None  # value of _factorization_count at the factorization of Q_c
        self._factor_params = None  # parameters of that factorization
        self._xhat = None  # (factor id, y, Q_c^-1 b)
        self._blocks = None  # (factor id, y, covariance blocks of the cells)

    # ------------------------------------------------------------------ parameters

    def get_parameter_vector(self, include_frozen=False):
        return self._params.copy()

    def set_parameter_vector(self, vector, include_frozen=False):
        self._params = np.array(vector, dtype=float)

    def _hyperparameters(self, params=None):
        """s^2 (with the jitter), sigma^2, tau^2 and Lambda."""
        p = self._params if params is None else params
        s2 = np.exp(p[0]) + self._yerr2
        sigma2 = np.exp(p[1])
        lengthscale2 = np.broadcast_to(np.exp(p[2:]), (self.ndim,))
        lam = lengthscale2 / (2 * self.nu)
        tau2 = self._tau2_constant / (sigma2 * np.sqrt(np.prod(lam)))
        return s2, sigma2, tau2, lam

    @property
    def spacing(self):
        """The spacing of the lattice (without the buffer) along each dimension."""
        return np.array(self._spacing)

    # ------------------------------------------------------------------ lattice

    def compute(self, x, nns=None, yerr=0.0, **kwargs):
        """
        Build the lattice covering the inputs x (nsamples, ndim), the interpolation matrix and the
        matrices whose linear combinations are Q_c. nns is not used; yerr (a scalar) is added in
        quadrature to the noise.
        """
        start = time.time()
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != self.ndim:
            raise ValueError("INLA: the inputs have %d dimensions, expected %d" % (x.shape[1], self.ndim))
        self._x = x
        self._yerr2 = float(yerr) ** 2
        d = self.ndim

        # the nodes of the lattice and its buffer along each dimension
        self._nodes = []
        self._spacing = []
        self._buffer_cells = []
        for k in range(d):
            if self.shape is None:
                grid = _infer_axis(x[:, k])
            else:
                low, high = (x[:, k].min(), x[:, k].max()) if self.bounds is None else self.bounds[k]
                grid = np.linspace(min(low, x[:, k].min()), max(high, x[:, k].max()), self.shape[k])
            h = grid[1] - grid[0]
            offsets = _buffer_offsets(h, self.buffer, self.buffer_ratio)
            self._nodes.append(np.concatenate([grid[0] - offsets[::-1], grid, grid[-1] + offsets]))
            self._spacing.append(h)
            self._buffer_cells.append(len(offsets))
        self._shape = tuple(len(z) for z in self._nodes)
        self._cell_shape = tuple(m - 1 for m in self._shape)
        n = int(np.prod(self._shape))
        self._n = n

        # multilinear interpolation: the 2^d corners of a cell are its first node plus these offsets
        strides = np.array([int(np.prod(self._shape[k + 1:])) for k in range(d)])
        self._corners = np.array(list(itertools.product((0, 1), repeat=d)))
        self._corner_offsets = self._corners @ strides
        nodes, weights, _ = self._interpolation(x)
        rows = np.repeat(np.arange(x.shape[0]), weights.shape[1])
        keep = weights.ravel() != 0
        self._A = coo_matrix((weights.ravel()[keep], (rows[keep], nodes.ravel()[keep])),
                             shape=(x.shape[0], n)).tocsr()

        # 1D matrices, their generalized eigenvalues mu and M-orthonormal eigenvectors V (G V = M V mu,
        # V' M V = I), for log|Q| and Q^-1 = tau^-2 V diag(1 / p(sum_d Lambda_d mu_d)) V' with V = kron(V_d),
        # and B_j = M (M^-1 G)^j, j = 0, ..., m (B_0 = M, B_1 = G, B_2 = G M^-1 G, ...)
        order = len(self._polynomial) - 1
        masses, powers = [], []
        self._eigenvalues = []
        self._eigenvectors = []
        for z in self._nodes:
            mass, stiff_diag, stiff_off = _fem_1d(z)
            stiff = diags([stiff_off, stiff_diag, stiff_off], [-1, 0, 1], format="csr")
            masses.append(mass)
            B = [diags(mass, format="csr")]
            for _ in range(order):
                B.append((stiff @ diags(1 / mass) @ B[-1]).tocsr())
            powers.append(B)
            scaled_off = stiff_off / np.sqrt(mass[:-1] * mass[1:])
            eig, vectors = eigh_tridiagonal(stiff_diag / mass, scaled_off)
            self._eigenvalues.append(np.maximum(eig, 0.0))
            self._eigenvectors.append(vectors / np.sqrt(mass)[:, None])
        self._cdiag = _kron([diags(m) for m in masses]).diagonal()
        self._logdet_c = sum(np.sum(np.log(masses[k])) * n / self._shape[k] for k in range(d))

        # Q / tau^2 = C p(S) = sum_j c_j C (sum_d Lambda_d S_d)^j, S_d = C^-1 G_d acting on dimension d: the
        # term of a multi-index k (|k| <= m) is kron_d(B_{k_d}) with the coefficient
        # c_|k| multinomial(|k|; k) prod_d Lambda_d^k_d
        self._multi_indices = [k for k in itertools.product(range(order + 1), repeat=d) if sum(k) <= order]
        terms = [(("k", k), _kron([powers[dim][k[dim]] for dim in range(d)])) for k in self._multi_indices]
        self._AtA = (self._A.T @ self._A).tocsr()
        terms.append(("data", self._AtA))

        # the common sparsity pattern (CSC with sorted indices, as factored), and the positions of
        # the entries of every matrix in it
        pattern = identity(n, format="csc")
        for _, mat in terms:
            pattern = pattern + abs(mat)
        pattern = pattern.tocsc()
        pattern.sum_duplicates()
        pattern.sort_indices()
        self._indptr = pattern.indptr
        self._indices = pattern.indices
        cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(pattern.indptr))
        self._keys = cols * n + pattern.indices
        self._terms = {}
        for name, mat in terms:
            mat = mat.tocoo()
            mat.sum_duplicates()
            self._terms[name] = (self._positions(mat.row, mat.col), mat.data.copy())
        self._qc = None
        self._factor_id = None
        self._xhat = None
        self._blocks = None
        if self.verbose:
            print("INLA lattice: %s nodes (%s without the %s buffer cells per side), %d nonzeros in Q_c, "
                  "%.2f s" % ("x".join(map(str, self._shape)),
                              "x".join(str(m - 2 * b) for m, b in zip(self._shape, self._buffer_cells)),
                              "/".join(map(str, self._buffer_cells)), len(self._keys), time.time() - start))

    def _positions(self, rows, cols):
        """The positions of the entries (rows, cols) in the pattern; -1 for the ones not in it."""
        keys = np.asarray(cols, dtype=np.int64) * self._n + np.asarray(rows, dtype=np.int64)
        pos = np.searchsorted(self._keys, keys)
        pos = np.minimum(pos, len(self._keys) - 1)
        return np.where(self._keys[pos] == keys, pos, -1)

    def _interpolation(self, t):
        """The nodes (npoints, 2^d) and multilinear weights of the points t, and their cells."""
        first = []
        local = []
        outside = False
        for k in range(self.ndim):
            z = self._nodes[k]
            cell = np.clip(np.searchsorted(z, t[:, k], side="right") - 1, 0, len(z) - 2)
            u = (t[:, k] - z[cell]) / (z[cell + 1] - z[cell])
            outside = outside or np.any((u < -1e-9) | (u > 1 + 1e-9))
            u = np.clip(u, 0.0, 1.0)
            # points on a node (up to round-off) get the weight 1 exactly
            u[u < 1e-9] = 0.0
            u[u > 1 - 1e-9] = 1.0
            first.append(cell)
            local.append(u)
        if outside:
            print("INLA warning: points outside the lattice are moved to its boundary")
        cells = np.ravel_multi_index(first, self._cell_shape)
        nodes = np.ravel_multi_index(first, self._shape)[:, None] + self._corner_offsets[None, :]
        weights = np.ones(nodes.shape)
        for j, corner in enumerate(self._corners):
            for k in range(self.ndim):
                weights[:, j] *= local[k] if corner[k] else 1 - local[k]
        return nodes, weights, cells

    # ------------------------------------------------------------------ likelihood

    def _combine(self, coefficients):
        """The matrix (CSC, with the common pattern) sum of the terms with these coefficients."""
        data = np.zeros(len(self._keys))
        for name, coef in coefficients.items():
            if coef != 0:
                pos, val = self._terms[name]
                data[pos] += coef * val
        return csc_matrix((data, self._indices, self._indptr), shape=(self._n, self._n))

    def _prior_coefficients(self, tau2, lam):
        """The coefficients of the terms of Q = tau^2 C p(S)."""
        coefficients = {}
        for k in self._multi_indices:
            multinomial = factorial(sum(k)) / np.prod([factorial(kd) for kd in k])
            coefficients[("k", k)] = tau2 * self._polynomial[sum(k)] * multinomial * np.prod(lam ** np.array(k))
        return coefficients

    def _assemble(self, params=None, data_term=True):
        """Q_c (CSC, with the common pattern) at the parameters; Q without the data term."""
        s2, _, tau2, lam = self._hyperparameters(params)
        coefficients = self._prior_coefficients(tau2, lam)
        if data_term:
            coefficients["data"] = 1 / s2
        return self._combine(coefficients)

    def _spectrum(self, lam):
        """p(s) at the eigenvalues s = sum_d Lambda_d mu_d of S (array of the shape of the lattice)."""
        s = 0.0
        for k in range(self.ndim):
            axis_shape = [1] * self.ndim
            axis_shape[k] = -1
            s = s + lam[k] * self._eigenvalues[k].reshape(axis_shape)
        return np.polynomial.polynomial.polyval(s, self._polynomial)

    def logdet_prior(self, params=None):
        """log|Q| = n log tau^2 + log|C| + sum log p(s), s the eigenvalues of S."""
        _, _, tau2, lam = self._hyperparameters(params)
        return self._n * np.log(tau2) + self._logdet_c + np.sum(np.log(self._spectrum(lam)))

    def _eigenvector_apply(self, x, transpose):
        """kron(V_d) x (or its transpose), x of the shape of the lattice with trailing columns."""
        for k, vectors in enumerate(self._eigenvectors):
            x = np.moveaxis(np.tensordot(vectors.T if transpose else vectors, x, axes=(1, k)), 0, k)
        return x

    def _spectral_apply(self, rhs, power, transposed_first):
        """V diag(p^power) V' rhs, or V diag(p^power) rhs if not transposed_first, rhs of shape (n,) or (n, ncol)."""
        _, _, tau2, lam = self._hyperparameters()
        spectrum = self._spectrum(lam) ** power
        rhs = np.asarray(rhs, dtype=float)
        x = rhs.reshape(self._shape + rhs.shape[1:])
        if transposed_first:
            x = self._eigenvector_apply(x, True)
        x = x * spectrum.reshape(spectrum.shape + (1,) * (x.ndim - self.ndim))
        return self._eigenvector_apply(x, False).reshape(rhs.shape), tau2

    def _prior_solve(self, rhs):
        """Q^-1 rhs = tau^-2 V diag(1 / p) V' rhs (rhs of shape (n,) or (n, ncol))."""
        x, tau2 = self._spectral_apply(rhs, -1.0, True)
        return x / tau2

    def _prior_sqrt(self, z):
        """F z with F F' = Q: F = tau C V diag(p^1/2), as Q = tau^2 C V diag(p) V' C."""
        x, tau2 = self._spectral_apply(z, 0.5, False)
        return np.sqrt(tau2) * self._cdiag.reshape((-1,) + (1,) * (x.ndim - 1)) * x

    def _precision_derivatives(self):
        """
        The derivatives dQ of Q with respect to the parameters after the noise (log sigma^2, then the log
        length scales squared) at the current parameters.
        """
        _, _, tau2, lam = self._hyperparameters()
        d = self.ndim
        prior = self._prior_coefficients(tau2, lam)
        # tau^2 is proportional to 1 / sigma^2
        derivatives = [self._combine({name: -coef for name, coef in prior.items()})]
        # log l_d^2 = log Lambda_d + const: the coefficient of a term of multi-index k is proportional to
        # tau^2 prod_d Lambda_d^k_d, and tau^2 to prod_d Lambda_d^-1/2
        for group in ([list(range(d))] if self.isotropic else [[k] for k in range(d)]):
            derivatives.append(self._combine({name: coef * sum(name[1][dim] - 0.5 for dim in group)
                                              for name, coef in prior.items()}))
        return derivatives

    @property
    def computed(self):
        return (self._factor_id is not None and self._factor_id == _factorization_count[0]
                and np.array_equal(self._factor_params, self._params))

    def recompute(self, quiet=False, **kwargs):
        """Factor Q_c at the current parameters, unless it is the current factorization."""
        if self._x is None:
            raise RuntimeError("You need to compute the model first")
        if self.computed:
            return True
        superlu_factor, superlu_logdet, _ = _superlu()
        start = time.time()
        self._qc = self._assemble()
        superlu_factor(self._qc, self.INT64, self.algo3d, self.verbose == 1)
        sign, logdet = superlu_logdet(self.verbose == 1)
        _factorization_count[0] += 1
        self._factor_id = _factorization_count[0]
        self._factor_params = self._params.copy()
        self._logdet_qc = logdet if sign == 1 else np.nan
        if self.verbose:
            print("INLA factorization of Q_c: %.2f s" % (time.time() - start))
        if sign != 1:
            if quiet:
                return False
            raise ValueError("INLA: Q_c is not positive definite (sign of the determinant %s)" % sign)
        return True

    def _residual(self, y):
        y = np.ravel(np.asarray(y, dtype=float))
        if len(y) != self._A.shape[0]:
            raise ValueError("Dimension mismatch")
        mean = np.mean(y)
        return y - mean, mean

    def _latent_mean(self, y):
        """Q_c^-1 b at the current factorization, for the observations y (cached)."""
        y = np.ravel(np.asarray(y, dtype=float))
        if self._xhat is not None and self._xhat[0] == self._factor_id and np.array_equal(self._xhat[1], y):
            return self._xhat[2]
        r, _ = self._residual(y)
        s2 = self._hyperparameters()[0]
        xhat = self._solve(self._A.T @ r / s2)
        self._xhat = (self._factor_id, y.copy(), xhat)
        return xhat

    def log_likelihood(self, y, quiet=False):
        """The log marginal likelihood of the observations y at the current parameters."""
        try:
            if not self.recompute(quiet=quiet):
                return -np.inf
            r, _ = self._residual(y)
            s2 = self._hyperparameters()[0]
            b = self._A.T @ r / s2
            xhat = self._latent_mean(y)
            nobs = len(r)
            ll = 0.5 * (self.logdet_prior() - self._logdet_qc - nobs * np.log(s2) - r @ r / s2 + b @ xhat
                        - nobs * np.log(2 * np.pi))
        except (ValueError, np.linalg.LinAlgError):
            if quiet:
                return -np.inf
            raise
        if self.verbose:
            print("INLA log-likelihood: %.10e at %s" % (ll, self._params))
        return ll if np.isfinite(ll) else -np.inf

    def _solve(self, rhs):
        """Q_c^-1 rhs at the current factorization, by multi-RHS solves of at most batch columns."""
        superlu_solve = _superlu()[2]
        rhs = np.asarray(rhs, dtype=float)
        if rhs.ndim == 1:
            x = np.array(rhs)
            superlu_solve(x, self.verbose == 1)
            return x
        x = np.empty_like(rhs)
        for c0 in range(0, rhs.shape[1], self.batch):
            block = np.asfortranarray(rhs[:, c0:c0 + self.batch])
            superlu_solve(block, self.verbose == 1)
            x[:, c0:c0 + self.batch] = block
        return x

    def _probes(self, n):
        """
        The Hutchinson probes of the traces of n x n matrices and the weight of their sum: nprobe
        Rademacher vectors, always the same ones so that the estimates are deterministic, smooth
        functions of the parameters (as the L-BFGS line searches need), or the identity (exact
        traces) if nprobe >= n.
        """
        if self.nprobe >= n:
            return np.eye(n), 1.0
        return np.random.default_rng(0).choice([-1.0, 1.0], size=(n, self.nprobe)), 1.0 / self.nprobe

    def grad_log_likelihood(self, y, quiet=False):
        """
        The gradient of log_likelihood with respect to the parameter vector. With x = Q_c^-1 b,
        w = 1 / s^2, r the residual and P = Q^-1 - Q_c^-1 (the posterior reduction of the covariance
        of the latent field):

            d/d log s^2      = exp(p_0) (-n w + w^2 (tr(A Q_c^-1 A') + |r - A x|^2)) / 2
            d/d theta (in Q) = (dlog|Q| - tr(Q_c^-1 dQ) - x' dQ x) / 2 = (tr(P dQ) - x' dQ x) / 2

        as dlog|Q| = tr(Q^-1 dQ), with tr(P dQ) = w tr(A Q_c^-1 dQ Q^-1 A'). The traces, of matrices of
        the size of the observations, are Hutchinson estimates (exact if nprobe is at least the number
        of observations) from one multi-RHS solve with Q_c and one application of Q^-1.
        """
        try:
            if not self.recompute(quiet=quiet):
                return np.zeros(len(self._params))
            r, _ = self._residual(y)
            xhat = self._latent_mean(y)
            s2 = self._hyperparameters()[0]
            w = 1 / s2
            probes, weight = self._probes(len(r))
            lifted = self._A.T @ probes
            solved = self._solve(lifted)
            prior_solved = self._prior_solve(lifted)
            residual = r - self._A @ xhat
            grad = np.empty(len(self._params))
            trace = weight * np.sum(probes * (self._A @ solved))
            grad[0] = 0.5 * np.exp(self._params[0]) * (-len(r) * w + w * w * (trace + residual @ residual))
            for k, dQ in enumerate(self._precision_derivatives()):
                trace = w * weight * np.sum(solved * (dQ @ prior_solved))
                grad[1 + k] = 0.5 * (trace - xhat @ (dQ @ xhat))
        except (ValueError, np.linalg.LinAlgError):
            if quiet:
                return np.zeros(len(self._params))
            raise
        if self.verbose:
            print("INLA gradient of the log-likelihood: %s" % grad)
        return grad

    def fisher_information(self, quiet=False):
        """
        The Fisher information matrix of the log-likelihood with respect to the parameter vector,
        F_ij = tr(S^-1 dS_i S^-1 dS_j) / 2 with S = A Q^-1 A' + s^2 I the covariance of the
        observations, estimated as george.GP.fisher_information with probes u of the observation
        space: F_ij ~ mean_u (dS_i S^-1 u)' (S^-1 dS_j u) / 2. S^-1 = w - w^2 A Q_c^-1 A' needs one
        solve with Q_c, and dS = -A Q^-1 dQ Q^-1 A' for the parameters of Q, whose Q^-1 is applied
        with the eigenvectors of the 1D matrices: 1 + len(parameters) multi-RHS solves.
        """
        try:
            if not self.recompute(quiet=quiet):
                return np.full((len(self._params), len(self._params)), np.nan)
            s2 = self._hyperparameters()[0]
            w = 1 / s2
            A = self._A

            def covariance_solve(V):
                return w * V - w * w * (A @ self._solve(A.T @ V))

            noise = np.exp(self._params[0])
            derivatives = [lambda V: noise * V]
            for dQ in self._precision_derivatives():
                derivatives.append(lambda V, dQ=dQ: -(A @ self._prior_solve(dQ @ self._prior_solve(A.T @ V))))
            probes, weight = self._probes(A.shape[0])
            solved = covariance_solve(probes)
            W = [dS(solved) for dS in derivatives]
            S = [covariance_solve(dS(probes)) for dS in derivatives]
            F = np.array([[0.5 * weight * np.sum(Wi * Sj) for Sj in S] for Wi in W])
        except (ValueError, np.linalg.LinAlgError):
            if quiet:
                return np.full((len(self._params), len(self._params)), np.nan)
            raise
        return 0.5 * (F + F.T)

    # ------------------------------------------------------------------ prediction

    def _covariance_blocks(self, y):
        """
        The Rao-Blackwellized estimates of the posterior covariance blocks of the nodes of every
        cell (ncells, 2^d, 2^d), at the current factorization (cached). The conditioning block B of
        a cell is the cell and the rb_ring layers of nodes around it (only the cell at the boundary
        of the lattice): larger blocks leave less of the variance to the samples.
        """
        y = np.ravel(np.asarray(y, dtype=float))
        if self._blocks is not None and self._blocks[0] == self._factor_id and np.array_equal(self._blocks[1], y):
            return self._blocks[2]
        start = time.time()
        superlu_solve = _superlu()[2]
        s2, _, tau2, lam = self._hyperparameters()
        d = self.ndim
        ncorner = len(self._corners)
        shape = np.array(self._shape)
        strides = np.array([int(np.prod(self._shape[k + 1:])) for k in range(d)])
        cells = np.indices(self._cell_shape).reshape(d, -1).T
        first = cells @ strides
        ncell = len(first)

        # Q_c[i, j] = values[j, index of the offset multi(i) - multi(j)], the last column being 0
        qc = self._qc
        cols = np.repeat(np.arange(self._n), np.diff(qc.indptr))
        delta = (np.stack(np.unravel_index(qc.indices, self._shape), axis=1)
                 - np.stack(np.unravel_index(cols, self._shape), axis=1))
        offsets, which = np.unique(delta, axis=0, return_inverse=True)
        values = np.zeros((self._n, len(offsets) + 1))
        values[cols, np.ravel(which)] = qc.data
        index = {tuple(o): i for i, o in enumerate(offsets)}

        # the blocks: their nodes relative to the first node of the cell, the rows of the cell in
        # them, and the index of the offset between every two of their nodes
        groups = []
        remaining = np.ones(ncell, dtype=bool)
        for ring in sorted({self.rb_ring, 0}, reverse=True):
            inside = remaining & np.all((cells >= ring) & (cells + 1 + ring <= shape - 1), axis=1)
            remaining &= ~inside
            local = np.array(list(itertools.product(range(-ring, 2 + ring), repeat=d)))
            pairs = np.array([[index.get(tuple(a - b), len(offsets)) for b in local] for a in local])
            rows = np.ravel_multi_index((self._corners + ring).T, (2 + 2 * ring,) * d)
            groups.append((np.nonzero(inside)[0], local @ strides, pairs, rows))

        # the samples, nsamples of N(0, Q_c^-1) by batches of batch solves with right-hand sides of N(0, Q_c)
        rng = np.random.default_rng(self.seed)
        exact = np.empty((ncell, ncorner, ncorner))
        second = np.zeros((ncell, ncorner, ncorner))
        done = 0
        while done < self.nsamples:
            nrhs = min(self.batch, self.nsamples - done)
            rhs = self._prior_sqrt(rng.standard_normal((self._n, nrhs)))
            rhs += self._A.T @ rng.standard_normal((self._A.shape[0], nrhs)) / np.sqrt(s2)
            samples = np.asfortranarray(rhs)
            superlu_solve(samples, self.verbose == 1)
            product = qc @ samples
            for members, block_offsets, pairs, rows in groups:
                size = len(block_offsets)
                chunk = max(1, 2 ** 22 // (size * max(size, nrhs)))
                for c0 in range(0, len(members), chunk):
                    idx = members[c0:c0 + chunk]
                    nodes = first[idx, None] + block_offsets[None, :]
                    # the rows of Q_BB^-1 of the cell nodes
                    inverse = np.linalg.inv(values[nodes[:, None, :], pairs[None, :, :]])[:, rows, :]
                    if done == 0:
                        exact[idx] = inverse[:, :, rows]
                    z = inverse @ product[nodes] - samples[nodes[:, rows]]
                    second[idx] += z @ z.transpose(0, 2, 1)
            done += nrhs
        blocks = exact + second / self.nsamples
        self._blocks = (self._factor_id, y.copy(), blocks)
        if self.verbose:
            print("INLA covariance blocks of %d cells from %d samples: %.2f s" % (ncell, self.nsamples, time.time() - start))
        return blocks

    def predict(self, y, t, return_cov=True, return_var=False, cache=True, kernel=None):
        """
        The predictive distribution of the latent function at the points t (ntest, ndim),
        conditioned on the observations y: mu, (mu, cov) or (mu, var) as george.GP.predict. The
        variances are the Rao-Blackwellized estimates; the covariance matrix is exact (one solve
        with ntest right-hand sides).
        """
        self.recompute()
        t = np.atleast_2d(np.asarray(t, dtype=float))
        _, mean = self._residual(y)
        xhat = self._latent_mean(y)
        nodes, weights, cells = self._interpolation(t)
        mu = np.sum(weights * xhat[nodes], axis=1) + mean
        if return_var:
            blocks = self._covariance_blocks(y)
            var = np.einsum("ni,nij,nj->n", weights, blocks[cells], weights)
            return mu, var
        if not return_cov:
            return mu
        rows = np.repeat(np.arange(t.shape[0]), weights.shape[1])
        At = coo_matrix((weights.ravel(), (rows, nodes.ravel())), shape=(t.shape[0], self._n)).tocsr()
        return mu, At @ self._solve(At.T.toarray())
