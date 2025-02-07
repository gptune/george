#ifndef _GEORGE_KERNELS_H_
#define _GEORGE_KERNELS_H_

#include <cmath>
#include <cfloat>
#include <vector>



#include "george/metrics.h"
#include "george/subspace.h"

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884e+00
#endif

namespace george {

namespace kernels {

class Kernel {
public:
    Kernel () {};
    virtual ~Kernel () {};
    virtual double value (const double* x1, const double* x2) { return 0.0; };
    virtual void gradient (const double* x1, const double* x2,
                           const unsigned* which, double* grad) {};
    virtual void x1_gradient (const double* x1, const double* x2,
                              double* grad) {};
    virtual void x2_gradient (const double* x1, const double* x2,
                              double* grad) {};

    // Parameter vector spec.
    virtual size_t size () const { return 0; }
    virtual size_t get_ndim () const { return 0; }
    virtual void set_parameter (size_t i, double v) {};
    virtual double get_parameter (size_t i) const { return 0.0; };
    virtual void set_metric_parameter (size_t i, double v) {};
    virtual void set_axis (size_t i, size_t v) {};
};


//
// OPERATORS
//

class Operator : public Kernel {
public:
    Operator (Kernel* k1, Kernel* k2) : kernel1_(k1), kernel2_(k2) {};
    ~Operator () {
        delete kernel1_;
        delete kernel2_;
    };
    Kernel* get_kernel1 () const { return kernel1_; };
    Kernel* get_kernel2 () const { return kernel2_; };

    // Parameter vector spec.
    size_t size () const { return kernel1_->size() + kernel2_->size(); };
    size_t get_ndim () const { return kernel1_->get_ndim(); }
    void set_parameter (size_t i, double value) {
        size_t n = kernel1_->size();
        if (i < n) kernel1_->set_parameter(i, value);
        else kernel2_->set_parameter(i-n, value);
    };
    double get_parameter (size_t i) const {
        size_t n = kernel1_->size();
        if (i < n) return kernel1_->get_parameter(i);
        return kernel2_->get_parameter(i-n);
    };

protected:
    Kernel* kernel1_, * kernel2_;
};

class Sum : public Operator {
public:
    Sum (Kernel* k1, Kernel* k2) : Operator(k1, k2) {};
    double value (const double* x1, const double* x2) {
        return this->kernel1_->value(x1, x2) + this->kernel2_->value(x1, x2);
    };
    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        size_t i, n1 = this->kernel1_->size(), n2 = this->size();

        bool any = false;
        for (i = 0; i < n1; ++i) if (which[i]) { any = true; break; }
        if (any) this->kernel1_->gradient(x1, x2, which, grad);

        any = false;
        for (i = n1; i < n2; ++i) if (which[i]) { any = true; break; }
        if (any) this->kernel2_->gradient(x1, x2, &(which[n1]), &(grad[n1]));
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        size_t ndim = this->get_ndim();
        std::vector<double> g1(ndim), g2(ndim);
        this->kernel1_->x1_gradient(x1, x2, &(g1[0]));
        this->kernel2_->x1_gradient(x1, x2, &(g2[0]));
        for (size_t i = 0; i < ndim; ++i) grad[i] = g1[i] + g2[i];
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        size_t ndim = this->get_ndim();
        std::vector<double> g1(ndim), g2(ndim);
        this->kernel1_->x2_gradient(x1, x2, &(g1[0]));
        this->kernel2_->x2_gradient(x1, x2, &(g2[0]));
        for (size_t i = 0; i < ndim; ++i) grad[i] = g1[i] + g2[i];
    };
};

class Product : public Operator {
public:
    Product (Kernel* k1, Kernel* k2) : Operator(k1, k2) {};
    double value (const double* x1, const double* x2) {
        return this->kernel1_->value(x1, x2) * this->kernel2_->value(x1, x2);
    };
    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        bool any;
        size_t i, n1 = this->kernel1_->size(), n2 = this->size();
        double k;

        any = false;
        for (i = 0; i < n1; ++i) if (which[i]) { any = true; break; }
        if (any) {
            k = this->kernel2_->value(x1, x2);
            this->kernel1_->gradient(x1, x2, which, grad);
            for (i = 0; i < n1; ++i) grad[i] *= k;
        }

        any = false;
        for (i = n1; i < n2; ++i) if (which[i]) { any = true; break; }
        if (any) {
            k = this->kernel1_->value(x1, x2);
            this->kernel2_->gradient(x1, x2, &(which[n1]), &(grad[n1]));
            for (i = n1; i < n2; ++i) grad[i] *= k;
        }
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        size_t ndim = this->get_ndim();
        std::vector<double> g1(ndim), g2(ndim);
        double k1 = this->kernel1_->value(x1, x2);
        double k2 = this->kernel2_->value(x1, x2);
        this->kernel1_->x1_gradient(x1, x2, &(g1[0]));
        this->kernel2_->x1_gradient(x1, x2, &(g2[0]));
        for (size_t i = 0; i < ndim; ++i) {
            grad[i] = k2 * g1[i] + k1 * g2[i];
        }
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        size_t ndim = this->get_ndim();
        std::vector<double> g1(ndim), g2(ndim);
        double k1 = this->kernel1_->value(x1, x2);
        double k2 = this->kernel2_->value(x1, x2);
        this->kernel1_->x2_gradient(x1, x2, &(g1[0]));
        this->kernel2_->x2_gradient(x1, x2, &(g2[0]));
        for (size_t i = 0; i < ndim; ++i) {
            grad[i] = k2 * g1[i] + k1 * g2[i];
        }
    };
};


/*
The linear regression kernel

.. math::

    k(\mathbf{x}_i,\,\mathbf{x}_j) =
        \frac{(\mathbf{x}_i \cdot \mathbf{x}_j)^P}{\gamma^2}

:param order:
    The power :math:`P`. This parameter is a *constant*; it is not
    included in the parameter vector.

:param log_gamma2:
    The scale factor :math:`\gamma^2`.
*/

class LinearKernel : public Kernel {
public:
    LinearKernel (
        double log_gamma2,
        double order,
        size_t ndim,
        size_t naxes
    ) :
        size_(1),
        subspace_(ndim, naxes)
        , param_log_gamma2_(log_gamma2)
        , constant_order_(order)
    {
        update_reparams();
    };

    size_t get_ndim () const { return subspace_.get_ndim(); };
    size_t get_axis (size_t i) const { return subspace_.get_axis(i); };
    void set_axis (size_t i, size_t value) { subspace_.set_axis(i, value); };

    double get_parameter (size_t i) const {
        if (i == 0) return param_log_gamma2_;
        return 0.0;
    };
    void set_parameter (size_t i, double value) {
        if (i == 0) {
            param_log_gamma2_ = value;
            update_reparams();
        } else
        ;
    };

    double get_value (
            double log_gamma2,
            double inv_gamma2,
            
            double order,
            double x1, double x2) {
        if (order == 0.0) return inv_gamma2;
        return pow(x1 * x2, order) * inv_gamma2;

    };

    double value (const double* x1, const double* x2) {
        size_t i, j, n = subspace_.get_naxes();
        double value = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            value += get_value(
                param_log_gamma2_,
                reparam_inv_gamma2_,
                
                constant_order_,
                x1[j], x2[j]);
        }
        return value;
    };

    double log_gamma2_gradient (
            double log_gamma2,
            double inv_gamma2,
            
            double order,
            double x1, double x2) {
        if (order == 0.0) return -inv_gamma2;
        return -pow(x1 * x2, order) * inv_gamma2;

    };
    double _x1_gradient (
            double log_gamma2,
            double inv_gamma2,
            
            double order,
            double x1, double x2) {
        if (order == 0.0) return 0.0;
        return x2 * order * pow(x1 * x2, order - 1.0) * inv_gamma2;

    };

    double _x2_gradient (
            double log_gamma2,
            double inv_gamma2,
            
            double order,
            double x1, double x2) {
        if (order == 0.0) return 0.0;
        return x1 * order * pow(x1 * x2, order - 1.0) * inv_gamma2;

    };

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        grad[0] = 0.0;
        

        size_t i, j, n = subspace_.get_naxes();
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            if (which[0])
                grad[0] += log_gamma2_gradient(
                    param_log_gamma2_,
                    reparam_inv_gamma2_,
                    
                    constant_order_,
                    x1[j], x2[j]);
            
        }
        
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x1_gradient(
                param_log_gamma2_,
                reparam_inv_gamma2_,
                
                constant_order_,
                x1[j], x2[j]);
        }
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x2_gradient(
                param_log_gamma2_,
                reparam_inv_gamma2_,
                
                constant_order_,
                x1[j], x2[j]);
        }
    };

    void update_reparams () {
        reparam_inv_gamma2_ = get_reparam_inv_gamma2 (
            param_log_gamma2_,
            constant_order_
        );
        
    };

    double get_reparam_inv_gamma2 (
        double log_gamma2,
        double order
    ) {
        return exp(-log_gamma2);
    }
    

    size_t size () const { return size_; };

private:
    size_t size_;
    george::subspace::Subspace subspace_;
    double param_log_gamma2_;
    
    double reparam_inv_gamma2_;
    
    double constant_order_;
};


/*
This is equivalent to a "scale mixture" of :class:`ExpSquaredKernel`
kernels with different scale lengths drawn from a gamma distribution.
See R&W for more info but here's the equation:

.. math::
    k(r^2) = \left[1 - \frac{r^2}{2\,\alpha} \right]^\alpha

:param log_alpha:
    The Gamma distribution parameter.
*/

template <typename M>
class RationalQuadraticKernel : public Kernel {
public:
    RationalQuadraticKernel (
        double log_alpha,
        int blocked,
        const double* min_block,
        const double* max_block,
        size_t ndim,
        size_t naxes
    ) :
        size_(1),
        metric_(ndim, naxes),
        blocked_(blocked),
        min_block_(naxes),
        max_block_(naxes)
        , param_log_alpha_(log_alpha)
        
    {
        size_t i;
        if (blocked_) {
            for (i = 0; i < naxes; ++i) {
                min_block_[i] = min_block[i];
                max_block_[i] = max_block[i];
            }
        }
        update_reparams();
    };

    size_t get_ndim () const { return metric_.get_ndim(); };

    double get_parameter (size_t i) const {
        if (i == 0) return param_log_alpha_;
        return metric_.get_parameter(i - size_);
    };
    void set_parameter (size_t i, double value) {
        if (i == 0) {
            param_log_alpha_ = value;
            update_reparams();
        } else
        metric_.set_parameter(i - size_, value);
    };

    double get_metric_parameter (size_t i) const {
        return metric_.get_parameter(i);
    };
    void set_metric_parameter (size_t i, double value) {
        metric_.set_parameter(i, value);
    };

    size_t get_axis (size_t i) const {
        return metric_.get_axis(i);
    };
    void set_axis (size_t i, size_t value) {
        metric_.set_axis(i, value);
    };

    double get_value (
            double log_alpha,
            double alpha,
            
            double r2) {
        return pow(1 + 0.5 * r2 / alpha, -alpha);

    };

    double value (const double* x1, const double* x2) {
        if (blocked_) {
            size_t i, j;
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i])
                    return 0.0;
            }
        }
        double r2 = metric_.value(x1, x2);
        return get_value(
            param_log_alpha_,
            reparam_alpha_,
            
            r2);
    };

    double log_alpha_gradient (
            double log_alpha,
            double alpha,
            
            double r2) {
        double t1 = 1.0 + 0.5 * r2 / alpha,
               t2 = 2.0 * alpha * t1;
        return alpha * pow(t1, -alpha) * (r2 / t2 - log(t1));

    };
    double radial_gradient (
            double log_alpha,
            double alpha,
            
            double r2) {
        return -0.5 * pow(1 + 0.5 * r2 / alpha, -alpha-1);

    };

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        bool out = false;
        size_t i, j, n = size();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        if (which[0])
            grad[0] = log_alpha_gradient(
                    param_log_alpha_,
                    reparam_alpha_,
                    
                    r2);
        

        bool any = false;
        for (i = size_; i < size(); ++i) if (which[i]) { any = true; break; }
        if (any) {
            double r2grad = radial_gradient(
                    param_log_alpha_,
                    
                    reparam_alpha_,
                    
                    r2);
            metric_.gradient(x1, x2, &(grad[size_]));
            for (i = size_; i < n; ++i) grad[i] *= r2grad;
        }
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        bool out = false;
        size_t i, j, n = this->get_ndim();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        double r2grad = 2.0 * radial_gradient(
                param_log_alpha_,
                
                reparam_alpha_,
                
                r2);
        metric_.x1_gradient(x1, x2, grad);
        for (i = 0; i < n; ++i) grad[i] *= r2grad;
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        bool out = false;
        size_t i, j, n = this->get_ndim();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        double r2grad = 2.0 * radial_gradient(
                param_log_alpha_,
                
                reparam_alpha_,
                
                r2);
        metric_.x1_gradient(x1, x2, grad);
        for (i = 0; i < n; ++i) grad[i] *= -r2grad;
    };

    size_t size () const { return metric_.size() + size_; };

    void update_reparams () {
        reparam_alpha_ = get_reparam_alpha (
            param_log_alpha_
        );
        
    };

    double get_reparam_alpha (
        double log_alpha
    ) {
        return exp(log_alpha);
    }
    

private:
    size_t size_;
    M metric_;
    int blocked_;
    std::vector<double> min_block_, max_block_;
    double param_log_alpha_;
    
    double reparam_alpha_;
    
};


/*
The exponential kernel is a stationary kernel where the value
at a given radius :math:`r^2` is given by:

.. math::

    k(r^2) = \exp \left ( -\sqrt{r^2} \right )
*/

template <typename M>
class ExpKernel : public Kernel {
public:
    ExpKernel (
        int blocked,
        const double* min_block,
        const double* max_block,
        size_t ndim,
        size_t naxes
    ) :
        size_(0),
        metric_(ndim, naxes),
        blocked_(blocked),
        min_block_(naxes),
        max_block_(naxes)
        
    {
        size_t i;
        if (blocked_) {
            for (i = 0; i < naxes; ++i) {
                min_block_[i] = min_block[i];
                max_block_[i] = max_block[i];
            }
        }
        update_reparams();
    };

    size_t get_ndim () const { return metric_.get_ndim(); };

    double get_parameter (size_t i) const {
        return metric_.get_parameter(i - size_);
    };
    void set_parameter (size_t i, double value) {
        metric_.set_parameter(i - size_, value);
    };

    double get_metric_parameter (size_t i) const {
        return metric_.get_parameter(i);
    };
    void set_metric_parameter (size_t i, double value) {
        metric_.set_parameter(i, value);
    };

    size_t get_axis (size_t i) const {
        return metric_.get_axis(i);
    };
    void set_axis (size_t i, size_t value) {
        metric_.set_axis(i, value);
    };

    double get_value (
            
            double r2) {
        return exp(-sqrt(r2));
    };

    double value (const double* x1, const double* x2) {
        if (blocked_) {
            size_t i, j;
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i])
                    return 0.0;
            }
        }
        double r2 = metric_.value(x1, x2);
        return get_value(
            
            r2);
    };

    double radial_gradient (
            
            double r2) {
        if (r2 < DBL_EPSILON) return 0.0;
        double r = sqrt(r2);
        return -0.5 * exp(-r) / r;

    };

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        bool out = false;
        size_t i, j, n = size();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        

        bool any = false;
        for (i = size_; i < size(); ++i) if (which[i]) { any = true; break; }
        if (any) {
            double r2grad = radial_gradient(
                    
                    
                    r2);
            metric_.gradient(x1, x2, &(grad[size_]));
            for (i = size_; i < n; ++i) grad[i] *= r2grad;
        }
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        bool out = false;
        size_t i, j, n = this->get_ndim();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        double r2grad = 2.0 * radial_gradient(
                
                
                r2);
        metric_.x1_gradient(x1, x2, grad);
        for (i = 0; i < n; ++i) grad[i] *= r2grad;
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        bool out = false;
        size_t i, j, n = this->get_ndim();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        double r2grad = 2.0 * radial_gradient(
                
                
                r2);
        metric_.x1_gradient(x1, x2, grad);
        for (i = 0; i < n; ++i) grad[i] *= -r2grad;
    };

    size_t size () const { return metric_.size() + size_; };

    void update_reparams () {
        
    };

    

private:
    size_t size_;
    M metric_;
    int blocked_;
    std::vector<double> min_block_, max_block_;
    
    
};


/*
A local Gaussian kernel.

.. math::
    k(\mathbf{x}_i,\,\mathbf{x}_j) = \exp\left(
        -\frac{(x_i - x_0)^2 + (x_j - x_0)^2}{2\,w} \right))

:param location:
    The location :math:`x_0` of the Gaussian.

:param log_width:
    The (squared) width :math:`w` of the Gaussian.
*/

class LocalGaussianKernel : public Kernel {
public:
    LocalGaussianKernel (
        double location,
        double log_width,
        size_t ndim,
        size_t naxes
    ) :
        size_(2),
        subspace_(ndim, naxes)
        , param_location_(location)
        , param_log_width_(log_width)
    {
        update_reparams();
    };

    size_t get_ndim () const { return subspace_.get_ndim(); };
    size_t get_axis (size_t i) const { return subspace_.get_axis(i); };
    void set_axis (size_t i, size_t value) { subspace_.set_axis(i, value); };

    double get_parameter (size_t i) const {
        if (i == 0) return param_location_;
        if (i == 1) return param_log_width_;
        return 0.0;
    };
    void set_parameter (size_t i, double value) {
        if (i == 0) {
            param_location_ = value;
            update_reparams();
        } else
        if (i == 1) {
            param_log_width_ = value;
            update_reparams();
        } else
        ;
    };

    double get_value (
            double location,
            double log_width,
            double inv_2w,
            
            double x1, double x2) {
        double d1 = x1 - location, d2 = x2 - location;
        return exp(-(d1*d1 + d2*d2) * inv_2w);

    };

    double value (const double* x1, const double* x2) {
        size_t i, j, n = subspace_.get_naxes();
        double value = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            value += get_value(
                param_location_,
                param_log_width_,
                reparam_inv_2w_,
                
                x1[j], x2[j]);
        }
        return value;
    };

    double location_gradient (
            double location,
            double log_width,
            double inv_2w,
            
            double x1, double x2) {
        double d1 = x1 - location, d2 = x2 - location;
        return 2 * exp(-(d1*d1 + d2*d2) * inv_2w) * inv_2w * (d1 + d2);

    };
    double log_width_gradient (
            double location,
            double log_width,
            double inv_2w,
            
            double x1, double x2) {
        double d1 = x1 - location, d2 = x2 - location,
               arg = (d1*d1 + d2*d2) * inv_2w;
        return exp(-arg) * arg;

    };
    double _x1_gradient (
            double location,
            double log_width,
            double inv_2w,
            
            double x1, double x2) {
        double d1 = x1 - location, d2 = x2 - location;
        return -2.0 * exp(-(d1*d1 + d2*d2) * inv_2w) * d1 * inv_2w;

    };

    double _x2_gradient (
            double location,
            double log_width,
            double inv_2w,
            
            double x1, double x2) {
        double d1 = x1 - location, d2 = x2 - location;
        return -2.0 * exp(-(d1*d1 + d2*d2) * inv_2w) * d2 * inv_2w;

    };

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        grad[0] = 0.0;
        grad[1] = 0.0;
        

        size_t i, j, n = subspace_.get_naxes();
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            if (which[0])
                grad[0] += location_gradient(
                    param_location_,
                    param_log_width_,
                    reparam_inv_2w_,
                    
                    x1[j], x2[j]);
            if (which[1])
                grad[1] += log_width_gradient(
                    param_location_,
                    param_log_width_,
                    reparam_inv_2w_,
                    
                    x1[j], x2[j]);
            
        }
        
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x1_gradient(
                param_location_,
                param_log_width_,
                reparam_inv_2w_,
                
                x1[j], x2[j]);
        }
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x2_gradient(
                param_location_,
                param_log_width_,
                reparam_inv_2w_,
                
                x1[j], x2[j]);
        }
    };

    void update_reparams () {
        reparam_inv_2w_ = get_reparam_inv_2w (
            param_location_,param_log_width_
        );
        
    };

    double get_reparam_inv_2w (
        double location,double log_width
    ) {
        return 0.5 * exp(-log_width);
    }
    

    size_t size () const { return size_; };

private:
    size_t size_;
    george::subspace::Subspace subspace_;
    double param_location_;
    double param_log_width_;
    
    double reparam_inv_2w_;
    
};


/*
This kernel is a no-op*/

class EmptyKernel : public Kernel {
public:
    EmptyKernel (
        size_t ndim,
        size_t naxes
    ) :
        size_(0),
        subspace_(ndim, naxes)
    {
        update_reparams();
    };

    size_t get_ndim () const { return subspace_.get_ndim(); };
    size_t get_axis (size_t i) const { return subspace_.get_axis(i); };
    void set_axis (size_t i, size_t value) { subspace_.set_axis(i, value); };

    double get_parameter (size_t i) const {
        return 0.0;
    };
    void set_parameter (size_t i, double value) {
        ;
    };

    double get_value (
            
            double x1, double x2) {
        return 0.0;

    };

    double value (const double* x1, const double* x2) {
        size_t i, j, n = subspace_.get_naxes();
        double value = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            value += get_value(
                
                x1[j], x2[j]);
        }
        return value;
    };

    double _x1_gradient (
            
            double x1, double x2) {
        return 0.0;
    };

    double _x2_gradient (
            
            double x1, double x2) {
        return 0.0;
    };

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        

        
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x1_gradient(
                
                x1[j], x2[j]);
        }
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x2_gradient(
                
                x1[j], x2[j]);
        }
    };

    void update_reparams () {
        
    };

    

    size_t size () const { return size_; };

private:
    size_t size_;
    george::subspace::Subspace subspace_;
    
    
};


/*
The simplest periodic kernel. This

.. math::
    k(\mathbf{x}_i,\,\mathbf{x}_j) = \cos\left(
        \frac{2\,\pi\,|x_i - x_j|}{P} \right)

where the parameter :math:`P` is the period of the oscillation. This
kernel should probably always be multiplied be a stationary kernel
(e.g. :class:`ExpSquaredKernel`) to allow quasi-periodic variations.

:param log_period:
    The period of the oscillation.
*/

class CosineKernel : public Kernel {
public:
    CosineKernel (
        double log_period,
        size_t ndim,
        size_t naxes
    ) :
        size_(1),
        subspace_(ndim, naxes)
        , param_log_period_(log_period)
    {
        update_reparams();
    };

    size_t get_ndim () const { return subspace_.get_ndim(); };
    size_t get_axis (size_t i) const { return subspace_.get_axis(i); };
    void set_axis (size_t i, size_t value) { subspace_.set_axis(i, value); };

    double get_parameter (size_t i) const {
        if (i == 0) return param_log_period_;
        return 0.0;
    };
    void set_parameter (size_t i, double value) {
        if (i == 0) {
            param_log_period_ = value;
            update_reparams();
        } else
        ;
    };

    double get_value (
            double log_period,
            double factor,
            
            double x1, double x2) {
        return cos((x1 - x2) * factor);

    };

    double value (const double* x1, const double* x2) {
        size_t i, j, n = subspace_.get_naxes();
        double value = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            value += get_value(
                param_log_period_,
                reparam_factor_,
                
                x1[j], x2[j]);
        }
        return value;
    };

    double log_period_gradient (
            double log_period,
            double factor,
            
            double x1, double x2) {
        double r = factor * (x1 - x2);
        return r * sin(r);

    };
    double _x1_gradient (
            double log_period,
            double factor,
            
            double x1, double x2) {
        return -factor*sin(factor * (x1-x2));

    };

    double _x2_gradient (
            double log_period,
            double factor,
            
            double x1, double x2) {
        return factor*sin(factor * (x1-x2));

    };

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        grad[0] = 0.0;
        

        size_t i, j, n = subspace_.get_naxes();
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            if (which[0])
                grad[0] += log_period_gradient(
                    param_log_period_,
                    reparam_factor_,
                    
                    x1[j], x2[j]);
            
        }
        
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x1_gradient(
                param_log_period_,
                reparam_factor_,
                
                x1[j], x2[j]);
        }
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x2_gradient(
                param_log_period_,
                reparam_factor_,
                
                x1[j], x2[j]);
        }
    };

    void update_reparams () {
        reparam_factor_ = get_reparam_factor (
            param_log_period_
        );
        
    };

    double get_reparam_factor (
        double log_period
    ) {
        return 2 * M_PI * exp(-log_period);
    }
    

    size_t size () const { return size_; };

private:
    size_t size_;
    george::subspace::Subspace subspace_;
    double param_log_period_;
    
    double reparam_factor_;
    
};


/*
The Matern-5/2 kernel is stationary kernel where the value at a
given radius :math:`r^2` is given by:

.. math::

    k(r^2) = \left( 1+\sqrt{5\,r^2}+ \frac{5\,r^2}{3} \right)\,
             \exp \left (-\sqrt{5\,r^2} \right )
*/

template <typename M>
class Matern52Kernel : public Kernel {
public:
    Matern52Kernel (
        int blocked,
        const double* min_block,
        const double* max_block,
        size_t ndim,
        size_t naxes
    ) :
        size_(0),
        metric_(ndim, naxes),
        blocked_(blocked),
        min_block_(naxes),
        max_block_(naxes)
        
    {
        size_t i;
        if (blocked_) {
            for (i = 0; i < naxes; ++i) {
                min_block_[i] = min_block[i];
                max_block_[i] = max_block[i];
            }
        }
        update_reparams();
    };

    size_t get_ndim () const { return metric_.get_ndim(); };

    double get_parameter (size_t i) const {
        return metric_.get_parameter(i - size_);
    };
    void set_parameter (size_t i, double value) {
        metric_.set_parameter(i - size_, value);
    };

    double get_metric_parameter (size_t i) const {
        return metric_.get_parameter(i);
    };
    void set_metric_parameter (size_t i, double value) {
        metric_.set_parameter(i, value);
    };

    size_t get_axis (size_t i) const {
        return metric_.get_axis(i);
    };
    void set_axis (size_t i, size_t value) {
        metric_.set_axis(i, value);
    };

    double get_value (
            
            double r2) {
        double r = sqrt(5.0 * r2);
        return (1 + r + 5.0 * r2 / 3.0) * exp(-r);

    };

    double value (const double* x1, const double* x2) {
        if (blocked_) {
            size_t i, j;
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i])
                    return 0.0;
            }
        }
        double r2 = metric_.value(x1, x2);
        return get_value(
            
            r2);
    };

    double radial_gradient (
            
            double r2) {
        double r = sqrt(5.0 * r2);
        return -5 * (1 + r) * exp(-r) / 6.0;

    };

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        bool out = false;
        size_t i, j, n = size();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        

        bool any = false;
        for (i = size_; i < size(); ++i) if (which[i]) { any = true; break; }
        if (any) {
            double r2grad = radial_gradient(
                    
                    
                    r2);
            metric_.gradient(x1, x2, &(grad[size_]));
            for (i = size_; i < n; ++i) grad[i] *= r2grad;
        }
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        bool out = false;
        size_t i, j, n = this->get_ndim();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        double r2grad = 2.0 * radial_gradient(
                
                
                r2);
        metric_.x1_gradient(x1, x2, grad);
        for (i = 0; i < n; ++i) grad[i] *= r2grad;
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        bool out = false;
        size_t i, j, n = this->get_ndim();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        double r2grad = 2.0 * radial_gradient(
                
                
                r2);
        metric_.x1_gradient(x1, x2, grad);
        for (i = 0; i < n; ++i) grad[i] *= -r2grad;
    };

    size_t size () const { return metric_.size() + size_; };

    void update_reparams () {
        
    };

    

private:
    size_t size_;
    M metric_;
    int blocked_;
    std::vector<double> min_block_, max_block_;
    
    
};


/*
The exp-sine-squared kernel is a commonly used periodic kernel. Unlike
the :class:`CosineKernel`, this kernel never has negative covariance
which might be useful for your problem. Here's the equation:

.. math::
    k(\mathbf{x}_i,\,\mathbf{x}_j) =
        \exp \left( -\Gamma\,\sin^2\left[
            \frac{\pi}{P}\,\left|x_i-x_j\right|
        \right] \right)

:param gamma:
    The scale :math:`\Gamma` of the correlations.

:param log_period:
    The log of the period :math:`P` of the oscillation (in the same units
    as :math:`\mathbf{x}`).
*/

class ExpSine2Kernel : public Kernel {
public:
    ExpSine2Kernel (
        double gamma,
        double log_period,
        size_t ndim,
        size_t naxes
    ) :
        size_(2),
        subspace_(ndim, naxes)
        , param_gamma_(gamma)
        , param_log_period_(log_period)
    {
        update_reparams();
    };

    size_t get_ndim () const { return subspace_.get_ndim(); };
    size_t get_axis (size_t i) const { return subspace_.get_axis(i); };
    void set_axis (size_t i, size_t value) { subspace_.set_axis(i, value); };

    double get_parameter (size_t i) const {
        if (i == 0) return param_gamma_;
        if (i == 1) return param_log_period_;
        return 0.0;
    };
    void set_parameter (size_t i, double value) {
        if (i == 0) {
            param_gamma_ = value;
            update_reparams();
        } else
        if (i == 1) {
            param_log_period_ = value;
            update_reparams();
        } else
        ;
    };

    double get_value (
            double gamma,
            double log_period,
            double factor,
            
            double x1, double x2) {
        double s = sin((x1 - x2) * factor);
        return exp(-gamma * s * s);

    };

    double value (const double* x1, const double* x2) {
        size_t i, j, n = subspace_.get_naxes();
        double value = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            value += get_value(
                param_gamma_,
                param_log_period_,
                reparam_factor_,
                
                x1[j], x2[j]);
        }
        return value;
    };

    double gamma_gradient (
            double gamma,
            double log_period,
            double factor,
            
            double x1, double x2) {
        double s = sin((x1 - x2) * factor), s2 = s * s;
        return -s2 * exp(-gamma * s2);

    };
    double log_period_gradient (
            double gamma,
            double log_period,
            double factor,
            
            double x1, double x2) {
        double arg = (x1 - x2) * factor,
               s = sin(arg), c = cos(arg),
               A = exp(-gamma * s * s);
        return 2 * gamma * arg * c * s * A;

    };
    double _x1_gradient (
            double gamma,
            double log_period,
            double factor,
            
            double x1, double x2) {
        double d = x1 - x2;
        double s = sin(d * factor);
        return -exp(-gamma * s * s) * factor * gamma * sin(2.0 * factor * d);

    };

    double _x2_gradient (
            double gamma,
            double log_period,
            double factor,
            
            double x1, double x2) {
        double d = x1 - x2;
        double s = sin(d * factor);
        return exp(-gamma * s * s) * factor * gamma * sin(2.0 * factor * d);

    };

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        grad[0] = 0.0;
        grad[1] = 0.0;
        

        size_t i, j, n = subspace_.get_naxes();
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            if (which[0])
                grad[0] += gamma_gradient(
                    param_gamma_,
                    param_log_period_,
                    reparam_factor_,
                    
                    x1[j], x2[j]);
            if (which[1])
                grad[1] += log_period_gradient(
                    param_gamma_,
                    param_log_period_,
                    reparam_factor_,
                    
                    x1[j], x2[j]);
            
        }
        
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x1_gradient(
                param_gamma_,
                param_log_period_,
                reparam_factor_,
                
                x1[j], x2[j]);
        }
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x2_gradient(
                param_gamma_,
                param_log_period_,
                reparam_factor_,
                
                x1[j], x2[j]);
        }
    };

    void update_reparams () {
        reparam_factor_ = get_reparam_factor (
            param_gamma_,param_log_period_
        );
        
    };

    double get_reparam_factor (
        double gamma,double log_period
    ) {
        return M_PI * exp(-log_period);
    }
    

    size_t size () const { return size_; };

private:
    size_t size_;
    george::subspace::Subspace subspace_;
    double param_gamma_;
    double param_log_period_;
    
    double reparam_factor_;
    
};


/*
This kernel returns the constant

.. math::

    k(\mathbf{x}_i,\,\mathbf{x}_j) = c

where :math:`c` is a parameter.

:param log_constant:
    The log of :math:`c` in the above equation.
*/

class ConstantKernel : public Kernel {
public:
    ConstantKernel (
        double log_constant,
        size_t ndim,
        size_t naxes
    ) :
        size_(1),
        subspace_(ndim, naxes)
        , param_log_constant_(log_constant)
    {
        update_reparams();
    };

    size_t get_ndim () const { return subspace_.get_ndim(); };
    size_t get_axis (size_t i) const { return subspace_.get_axis(i); };
    void set_axis (size_t i, size_t value) { subspace_.set_axis(i, value); };

    double get_parameter (size_t i) const {
        if (i == 0) return param_log_constant_;
        return 0.0;
    };
    void set_parameter (size_t i, double value) {
        if (i == 0) {
            param_log_constant_ = value;
            update_reparams();
        } else
        ;
    };

    double get_value (
            double log_constant,
            double constant,
            
            double x1, double x2) {
        return constant;

    };

    double value (const double* x1, const double* x2) {
        size_t i, j, n = subspace_.get_naxes();
        double value = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            value += get_value(
                param_log_constant_,
                reparam_constant_,
                
                x1[j], x2[j]);
        }
        return value;
    };

    double log_constant_gradient (
            double log_constant,
            double constant,
            
            double x1, double x2) {
        return constant;

    };
    double _x1_gradient (
            double log_constant,
            double constant,
            
            double x1, double x2) {
        return 0.0;

    };

    double _x2_gradient (
            double log_constant,
            double constant,
            
            double x1, double x2) {
        return 0.0;

    };

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        grad[0] = 0.0;
        

        size_t i, j, n = subspace_.get_naxes();
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            if (which[0])
                grad[0] += log_constant_gradient(
                    param_log_constant_,
                    reparam_constant_,
                    
                    x1[j], x2[j]);
            
        }
        
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x1_gradient(
                param_log_constant_,
                reparam_constant_,
                
                x1[j], x2[j]);
        }
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x2_gradient(
                param_log_constant_,
                reparam_constant_,
                
                x1[j], x2[j]);
        }
    };

    void update_reparams () {
        reparam_constant_ = get_reparam_constant (
            param_log_constant_
        );
        
    };

    double get_reparam_constant (
        double log_constant
    ) {
        return exp(log_constant);
    }
    

    size_t size () const { return size_; };

private:
    size_t size_;
    george::subspace::Subspace subspace_;
    double param_log_constant_;
    
    double reparam_constant_;
    
};


/*
The exponential-squared kernel is a stationary kernel where the value
at a given radius :math:`r^2` is given by:

.. math::

    k(r^2) = \exp \left ( -\frac{r^2}{2} \right )
*/

template <typename M>
class ExpSquaredKernel : public Kernel {
public:
    ExpSquaredKernel (
        int blocked,
        const double* min_block,
        const double* max_block,
        size_t ndim,
        size_t naxes
    ) :
        size_(0),
        metric_(ndim, naxes),
        blocked_(blocked),
        min_block_(naxes),
        max_block_(naxes)
        
    {
        size_t i;
        if (blocked_) {
            for (i = 0; i < naxes; ++i) {
                min_block_[i] = min_block[i];
                max_block_[i] = max_block[i];
            }
        }
        update_reparams();
    };

    size_t get_ndim () const { return metric_.get_ndim(); };

    double get_parameter (size_t i) const {
        return metric_.get_parameter(i - size_);
    };
    void set_parameter (size_t i, double value) {
        metric_.set_parameter(i - size_, value);
    };

    double get_metric_parameter (size_t i) const {
        return metric_.get_parameter(i);
    };
    void set_metric_parameter (size_t i, double value) {
        metric_.set_parameter(i, value);
    };

    size_t get_axis (size_t i) const {
        return metric_.get_axis(i);
    };
    void set_axis (size_t i, size_t value) {
        metric_.set_axis(i, value);
    };

    double get_value (
            
            double r2) {
        return exp(-0.5 * r2);
    };

    double value (const double* x1, const double* x2) {
        if (blocked_) {
            size_t i, j;
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i])
                    return 0.0;
            }
        }
        double r2 = metric_.value(x1, x2);
        return get_value(
            
            r2);
    };

    double radial_gradient (
            
            double r2) {
        return -0.5 * exp(-0.5 * r2);
    };

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        bool out = false;
        size_t i, j, n = size();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        

        bool any = false;
        for (i = size_; i < size(); ++i) if (which[i]) { any = true; break; }
        if (any) {
            double r2grad = radial_gradient(
                    
                    
                    r2);
            metric_.gradient(x1, x2, &(grad[size_]));
            for (i = size_; i < n; ++i) grad[i] *= r2grad;
        }
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        bool out = false;
        size_t i, j, n = this->get_ndim();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        double r2grad = 2.0 * radial_gradient(
                
                
                r2);
        metric_.x1_gradient(x1, x2, grad);
        for (i = 0; i < n; ++i) grad[i] *= r2grad;
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        bool out = false;
        size_t i, j, n = this->get_ndim();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        double r2grad = 2.0 * radial_gradient(
                
                
                r2);
        metric_.x1_gradient(x1, x2, grad);
        for (i = 0; i < n; ++i) grad[i] *= -r2grad;
    };

    size_t size () const { return metric_.size() + size_; };

    void update_reparams () {
        
    };

    

private:
    size_t size_;
    M metric_;
    int blocked_;
    std::vector<double> min_block_, max_block_;
    
    
};


/*
The Matern-3/2 kernel is stationary kernel where the value at a
given radius :math:`r^2` is given by:

.. math::

    k(r^2) = \left( 1+\sqrt{3\,r^2} \right)\,
             \exp \left (-\sqrt{3\,r^2} \right )
*/

template <typename M>
class Matern32Kernel : public Kernel {
public:
    Matern32Kernel (
        int blocked,
        const double* min_block,
        const double* max_block,
        size_t ndim,
        size_t naxes
    ) :
        size_(0),
        metric_(ndim, naxes),
        blocked_(blocked),
        min_block_(naxes),
        max_block_(naxes)
        
    {
        size_t i;
        if (blocked_) {
            for (i = 0; i < naxes; ++i) {
                min_block_[i] = min_block[i];
                max_block_[i] = max_block[i];
            }
        }
        update_reparams();
    };

    size_t get_ndim () const { return metric_.get_ndim(); };

    double get_parameter (size_t i) const {
        return metric_.get_parameter(i - size_);
    };
    void set_parameter (size_t i, double value) {
        metric_.set_parameter(i - size_, value);
    };

    double get_metric_parameter (size_t i) const {
        return metric_.get_parameter(i);
    };
    void set_metric_parameter (size_t i, double value) {
        metric_.set_parameter(i, value);
    };

    size_t get_axis (size_t i) const {
        return metric_.get_axis(i);
    };
    void set_axis (size_t i, size_t value) {
        metric_.set_axis(i, value);
    };

    double get_value (
            
            double r2) {
        double r = sqrt(3.0 * r2);
        return (1.0 + r) * exp(-r);

    };

    double value (const double* x1, const double* x2) {
        if (blocked_) {
            size_t i, j;
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i])
                    return 0.0;
            }
        }
        double r2 = metric_.value(x1, x2);
        return get_value(
            
            r2);
    };

    double radial_gradient (
            
            double r2) {
        double r = sqrt(3.0 * r2);
        return -3.0 * 0.5 * exp(-r);

    };

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        bool out = false;
        size_t i, j, n = size();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        

        bool any = false;
        for (i = size_; i < size(); ++i) if (which[i]) { any = true; break; }
        if (any) {
            double r2grad = radial_gradient(
                    
                    
                    r2);
            metric_.gradient(x1, x2, &(grad[size_]));
            for (i = size_; i < n; ++i) grad[i] *= r2grad;
        }
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        bool out = false;
        size_t i, j, n = this->get_ndim();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        double r2grad = 2.0 * radial_gradient(
                
                
                r2);
        metric_.x1_gradient(x1, x2, grad);
        for (i = 0; i < n; ++i) grad[i] *= r2grad;
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        bool out = false;
        size_t i, j, n = this->get_ndim();
        if (blocked_) {
            for (i = 0; i < min_block_.size(); ++i) {
                j = metric_.get_axis(i);
                if (x1[j] < min_block_[i] || x1[j] > max_block_[i] ||
                        x2[j] < min_block_[i] || x2[j] > max_block_[i]) {
                    out = true;
                    break;
                }
            }
            if (out) {
                for (i = 0; i < n; ++i) grad[i] = 0.0;
                return;
            }
        }

        double r2 = metric_.value(x1, x2);
        double r2grad = 2.0 * radial_gradient(
                
                
                r2);
        metric_.x1_gradient(x1, x2, grad);
        for (i = 0; i < n; ++i) grad[i] *= -r2grad;
    };

    size_t size () const { return metric_.size() + size_; };

    void update_reparams () {
        
    };

    

private:
    size_t size_;
    M metric_;
    int blocked_;
    std::vector<double> min_block_, max_block_;
    
    
};



/**
 * A local coregionalization kernel (LCMKernel) storing B_{t,q} and K_{t,q} in *linear space*,
 * but from the perspective of external code, they are log-parameters.
 *
 * The child kernels (like ExpSquaredKernel) see only x1[0..(naxes-2)],
 * x2[0..(naxes-2)] as the "spatial" dimensions. The last coordinate is the task ID.
 *
 * Formula:
 *   K(x1, x2) = sum_{q=0..Q-1}
 *     [ B_{t1,q} * B_{t2,q} + K_{t1,q}*delta_{(t1==t2)} ]
 *     * child_q.value(x1_spatial, x2_spatial)
 */
class LCMKernel : public Kernel
{
public:
    /**
     * @param init_logB   (size T*Q) => B_{t,q} = exp(init_logB[i]) stored in linear space
     * @param init_logK   (size T*Q) => K_{t,q} = exp(init_logK[i]) stored in linear space
     * @param num_tasks   T
     * @param children    Q child kernels, each operating on (naxes-1) dims
     * @param naxes       total dims, including last for the task ID
     */
    LCMKernel(
        const std::vector<double>& init_logB,
        const std::vector<double>& init_logK,
        size_t num_tasks,
        std::vector<Kernel*> children,
        size_t naxes
    )
    : num_tasks_(num_tasks),
      children_(std::move(children)),
      naxes_(naxes)
    {
        num_latent_ = children_.size(); // Q
        const size_t TQ = num_tasks_ * num_latent_;
        if (init_logB.size() != TQ) {
            throw std::invalid_argument("init_logB must have length T*Q.");
        }
        if (init_logK.size() != TQ) {
            throw std::invalid_argument("init_logK must have length T*Q.");
        }

        // Allocate & exponentiate B_, K_ in linear space
        B_.resize(TQ);
        K_.resize(TQ);
        for (size_t i = 0; i < TQ; ++i) {
            B_[i] = std::exp(init_logB[i]);  // store in linear
            K_[i] = std::exp(init_logK[i]);
        }

        // total param size = (TQ + TQ) + sum of children sizes
        size_ = 2*TQ;
        for (auto* c : children_) {
            size_ += c->size();
        }
    }

    ~LCMKernel() {
        // If we truly own them, delete. Otherwise, manage accordingly.
        for (auto* c : children_) {
            delete c;
        }
    }



    size_t get_ndim () const { return naxes_; };


    double value(const double* x1, const double* x2) override {
        // The last coordinate is the task
        size_t t1 = static_cast<size_t>(x1[naxes_ - 1]);
        size_t t2 = static_cast<size_t>(x2[naxes_ - 1]);
        if (t1 >= num_tasks_ || t2 >= num_tasks_) {
            return 0.0;
        }

        // Copy the first (naxes_-1) coords for the child kernels
        std::vector<double> x1_spatial(naxes_ - 1), x2_spatial(naxes_ - 1);
        for (size_t i = 0; i < naxes_ - 1; ++i) {
            x1_spatial[i] = x1[i];
            x2_spatial[i] = x2[i];
        }



        double sumval = 0.0;
        for (size_t q = 0; q < num_latent_; ++q) {
            double bprod = B_[t1 * num_latent_ + q] * B_[t2 * num_latent_ + q];
            double kterm = 0.0;
            if (t1 == t2) {
                kterm = K_[t1 * num_latent_ + q];
            }

            // Evaluate child kernel on the spatial coords
            double cval = children_[q]->value(x1_spatial.data(), x2_spatial.data());
            // if(fabs(x1[0]-0.887218)<1e-5 &&  fabs(x2[0]-0.739814)<1e-5){  
            // std::cout<<" child " <<cval<<std::endl;
            // std::cout<<" B  " <<B_[t1 * num_latent_ + q]<< " "<<B_[t2 * num_latent_ + q] <<std::endl;
            // std::cout<<" K  " <<K_[t1 * num_latent_ + q]<< " "<<K_[t2 * num_latent_ + q] <<std::endl;
            // }

            sumval += (bprod + kterm) * cval;
        }
        

// if(fabs(x1[0]-0.887218)<1e-5 &&  fabs(x2[0]-0.739814)<1e-5){          
//         // for (size_t i = 0; i < naxes_ ; ++i) {
//         //     std::cout<<" x1 " <<x1[i]<<std::endl;
//         // }

//         // for (size_t i = 0; i < naxes_ ; ++i) {
//         //     std::cout<<" x2 " <<x2[i]<<std::endl;
//         // }

//         // for (size_t t = 0; t < num_tasks_; ++t) {
//         //     for (size_t q = 0; q < num_latent_; ++q) {
//         //         std::cout<<" B " <<B_[t * num_latent_ + q]<<std::endl;
//         //     }
//         // }
//         // for (size_t t = 0; t < num_tasks_; ++t) {
//         //     for (size_t q = 0; q < num_latent_; ++q) {
//         //         std::cout<<" K " <<K_[t * num_latent_ + q]<<std::endl;
//         //     }
//         // }
//         std::cout<<x1[0]<<" "<<x1[1]<<" "<<x2[0]<<" "<<x2[1]<<" V " <<sumval<<std::endl;
// }
//         // exit(1);        
        
        return sumval;
        
    }

    void gradient(const double* x1, const double* x2,
                  const unsigned* which, double* grad) override
    {
        // zero out
        std::fill(grad, grad + size_, 0.0);

        size_t t1 = static_cast<size_t>(x1[naxes_ - 1]);
        size_t t2 = static_cast<size_t>(x2[naxes_ - 1]);
        if (t1 >= num_tasks_ || t2 >= num_tasks_) {
            // out-of-range => gradient = 0
            return;
        }

        // Copy spatial part
        std::vector<double> x1_spatial(naxes_ - 1), x2_spatial(naxes_ - 1);
        for (size_t i = 0; i < naxes_ - 1; ++i) {
            x1_spatial[i] = x1[i];
            x2_spatial[i] = x2[i];
        }

        // Precompute child kernel values & child kernel gradients
        std::vector<double> child_vals(num_latent_);
        std::vector< std::vector<double> > child_grads(num_latent_);

        for (size_t q = 0; q < num_latent_; ++q) {
            child_vals[q] = children_[q]->value(x1_spatial.data(), x2_spatial.data());

            child_grads[q].resize(children_[q]->size(), 0.0);
            children_[q]->gradient(x1_spatial.data(), x2_spatial.data(),
                                   which, child_grads[q].data());
        }

        const size_t TQ = num_tasks_ * num_latent_;
        size_t offset = 0;

        // 1) partial wrt logB_{t,q} => B_{t,q} * partial wrt B_{t,q}
        for (size_t i = 0; i < TQ; ++i) {
            if (!which || which[offset + i]) {
                size_t t = i / num_latent_;
                size_t q = i % num_latent_;

                double dFdBtq = 0.0;
                double Bt1q = B_[t1 * num_latent_ + q];
                double Bt2q = B_[t2 * num_latent_ + q];

                if (t1 == t2) {
                    if (t == t1) {
                        // derivative wrt B_{t1,q} => 2*B_{t1,q} * child_vals[q]
                        dFdBtq = 2.0 * Bt1q * child_vals[q];
                    }
                } else {
                    if (t == t1) {
                        dFdBtq = Bt2q * child_vals[q];
                    } else if (t == t2) {
                        dFdBtq = Bt1q * child_vals[q];
                    }
                }
                // chain rule wrt log(B_{t,q}) => multiply by B_{t,q}
                double B_tq = B_[t * num_latent_ + q];
                grad[offset + i] = B_tq * dFdBtq;
            }
        }
        offset += TQ;

        // 2) partial wrt logK_{t,q}
        for (size_t i = 0; i < TQ; ++i) {
            if (!which || which[offset + i]) {
                size_t t = i / num_latent_;
                size_t q = i % num_latent_;

                // derivative is nonzero only if t1 == t2 && t==t1 => child_vals[q]
                if (t1 == t2 && t == t1) {
                    double K_tq = K_[t * num_latent_ + q];
                    grad[offset + i] = K_tq * child_vals[q];
                }
            }
        }
        offset += TQ;

        // 3) partial wrt child kernel parameters
        //    factor = B_{t1,q}*B_{t2,q} + K_{t1,q}*(delta_{(t1==t2)})
        for (size_t q = 0; q < num_latent_; ++q) {
            double Bt1q = B_[t1 * num_latent_ + q];
            double Bt2q = B_[t2 * num_latent_ + q];
            double factor = Bt1q * Bt2q;
            if (t1 == t2) {
                factor += K_[t1 * num_latent_ + q];
            }

            size_t csize = children_[q]->size();
            for (size_t p = 0; p < csize; ++p) {
                if (!which || which[offset + p]) {
                    grad[offset + p] = factor * child_grads[q][p];
                }
            }
            offset += csize;
        }
    }

    // total # of parameters
    size_t size() const override {
        return size_;
    }

    // Externally, B_ and K_ are log-parameters
    double get_parameter(size_t i) const override {
        const size_t TQ = num_tasks_ * num_latent_;
        if (i < TQ) {
            // logB
            return std::log(B_[i]);
        }
        i -= TQ;
        if (i < TQ) {
            // logK
            return std::log(K_[i]);
        }
        i -= TQ;
        // child
        for (auto* c : children_) {
            size_t cs = c->size();
            if (i < cs) {
                return c->get_parameter(i);
            }
            i -= cs;
        }
        throw std::out_of_range("LCMKernel::get_parameter: index out of range");
    }

    void set_parameter(size_t i, double value) override {
        const size_t TQ = num_tasks_ * num_latent_;
        if (i < TQ) {
            // logB
            B_[i] = std::exp(value);
            return;
        }
        i -= TQ;
        if (i < TQ) {
            // logK
            K_[i] = std::exp(value);
            return;
        }
        i -= TQ;
        // child
        for (auto* c : children_) {
            size_t cs = c->size();
            if (i < cs) {
                c->set_parameter(i, value);
                return;
            }
            i -= cs;
        }
        throw std::out_of_range("LCMKernel::set_parameter: index out of range");
    }

private:
    // Internally we store B_, K_ in linear space (size T*Q).
    std::vector<double> B_;
    std::vector<double> K_;

    size_t num_tasks_;       // T
    size_t num_latent_;      // Q
    std::vector<Kernel*> children_;
    size_t naxes_;           // total dims (including last for task)

    size_t size_;            // total param count
};




/*
A polynomial kernel

.. math::

    k(\mathbf{x}_i,\,\mathbf{x}_j) =
        (\mathbf{x}_i \cdot \mathbf{x}_j + \sigma^2)^P

:param order:
    The power :math:`P`. This parameter is a *constant*; it is not
    included in the parameter vector.

:param log_sigma2:
    The variance :math:`\sigma^2 > 0`.
*/

class PolynomialKernel : public Kernel {
public:
    PolynomialKernel (
        double log_sigma2,
        double order,
        size_t ndim,
        size_t naxes
    ) :
        size_(1),
        subspace_(ndim, naxes)
        , param_log_sigma2_(log_sigma2)
        , constant_order_(order)
    {
        update_reparams();
    };

    size_t get_ndim () const { return subspace_.get_ndim(); };
    size_t get_axis (size_t i) const { return subspace_.get_axis(i); };
    void set_axis (size_t i, size_t value) { subspace_.set_axis(i, value); };

    double get_parameter (size_t i) const {
        if (i == 0) return param_log_sigma2_;
        return 0.0;
    };
    void set_parameter (size_t i, double value) {
        if (i == 0) {
            param_log_sigma2_ = value;
            update_reparams();
        } else
        ;
    };

    double get_value (
            double log_sigma2,
            double sigma2,
            
            double order,
            double x1, double x2) {
        if (order == 0.0) return 1.0;
        return pow(x1 * x2 + sigma2, order);

    };

    double value (const double* x1, const double* x2) {
        size_t i, j, n = subspace_.get_naxes();
        double value = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            value += get_value(
                param_log_sigma2_,
                reparam_sigma2_,
                
                constant_order_,
                x1[j], x2[j]);
        }
        return value;
    };

    double log_sigma2_gradient (
            double log_sigma2,
            double sigma2,
            
            double order,
            double x1, double x2) {
        if (order == 0.0) return 0.0;
        return sigma2 * pow(x1 * x2 + sigma2, order-1.0) * order;

    };
    double _x1_gradient (
            double log_sigma2,
            double sigma2,
            
            double order,
            double x1, double x2) {
        if (order == 0.0) return 0.0;
        return x2 * order * pow(x1 * x2 + sigma2, order-1.0);

    };

    double _x2_gradient (
            double log_sigma2,
            double sigma2,
            
            double order,
            double x1, double x2) {
        if (order == 0.0) return 0.0;
        return x1 * order * pow(x1 * x2 + sigma2, order-1.0);

    };

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        grad[0] = 0.0;
        

        size_t i, j, n = subspace_.get_naxes();
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            if (which[0])
                grad[0] += log_sigma2_gradient(
                    param_log_sigma2_,
                    reparam_sigma2_,
                    
                    constant_order_,
                    x1[j], x2[j]);
            
        }
        
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x1_gradient(
                param_log_sigma2_,
                reparam_sigma2_,
                
                constant_order_,
                x1[j], x2[j]);
        }
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x2_gradient(
                param_log_sigma2_,
                reparam_sigma2_,
                
                constant_order_,
                x1[j], x2[j]);
        }
    };

    void update_reparams () {
        reparam_sigma2_ = get_reparam_sigma2 (
            param_log_sigma2_,
            constant_order_
        );
        
    };

    double get_reparam_sigma2 (
        double log_sigma2,
        double order
    ) {
        return exp(log_sigma2);
    }
    

    size_t size () const { return size_; };

private:
    size_t size_;
    george::subspace::Subspace subspace_;
    double param_log_sigma2_;
    
    double reparam_sigma2_;
    
    double constant_order_;
};


/*
The dot product kernel

.. math::

    k(\mathbf{x}_i,\,\mathbf{x}_j) = \mathbf{x}_i \cdot \mathbf{x}_j

with no parameters.
*/

class DotProductKernel : public Kernel {
public:
    DotProductKernel (
        size_t ndim,
        size_t naxes
    ) :
        size_(0),
        subspace_(ndim, naxes)
    {
        update_reparams();
    };

    size_t get_ndim () const { return subspace_.get_ndim(); };
    size_t get_axis (size_t i) const { return subspace_.get_axis(i); };
    void set_axis (size_t i, size_t value) { subspace_.set_axis(i, value); };

    double get_parameter (size_t i) const {
        return 0.0;
    };
    void set_parameter (size_t i, double value) {
        ;
    };

    double get_value (
            
            double x1, double x2) {
        return x1 * x2;
    };

    double value (const double* x1, const double* x2) {
        size_t i, j, n = subspace_.get_naxes();
        double value = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            value += get_value(
                
                x1[j], x2[j]);
        }
        return value;
    };

    double _x1_gradient (
            
            double x1, double x2) {
        return x2;
    };

    double _x2_gradient (
            
            double x1, double x2) {
        return x1;
    };

    void gradient (const double* x1, const double* x2, const unsigned* which,
                   double* grad) {
        

        
    };

    void x1_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x1_gradient(
                
                x1[j], x2[j]);
        }
    };

    void x2_gradient (const double* x1, const double* x2, double* grad) {
        size_t i, j, ndim = this->get_ndim(), n = subspace_.get_naxes();
        for (i = 0; i < ndim; ++i) grad[i] = 0.0;
        for (i = 0; i < n; ++i) {
            j = subspace_.get_axis(i);
            grad[j] = _x2_gradient(
                
                x1[j], x2[j]);
        }
    };

    void update_reparams () {
        
    };

    

    size_t size () const { return size_; };

private:
    size_t size_;
    george::subspace::Subspace subspace_;
    
    
};

}; // namespace kernels
}; // namespace george

#endif