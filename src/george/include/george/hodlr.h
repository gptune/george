#ifndef _GEORGE_HODLR_H_
#define _GEORGE_HODLR_H_

#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <Eigen/SVD>

namespace george {
  namespace hodlr {

template <typename KernelType>
class Node {
private:
  const Eigen::VectorXd& diag_;
  KernelType* kernel_;
  Node<KernelType>* parent_;
  std::vector<Node<KernelType>*> children_;
  int start_, size_, direction_, rank_;
  bool is_leaf_;
  std::vector<Eigen::MatrixXd> U_, V_;
  Eigen::FullPivLU<Eigen::MatrixXd> lu_;
  Eigen::LDLT<Eigen::MatrixXd> ldlt_;
  double log_det_;

public:

  Node (const Eigen::VectorXd& diag,
        KernelType* kernel,
        int start,
        int size,
        int min_size,
        double tol,
        double tol_abs,
        int verbose,
        std::mt19937& random,
        int direction = 0,
        Node<KernelType>* parent = NULL)
    : diag_(diag)
    , kernel_(kernel)
    , parent_(parent)
    , children_(2)
    , start_(start)
    , size_(size)
    , direction_(direction)
    , U_(2)
    , V_(2)
  {
    int half = size_ / 2;
    if (half >= min_size) {
      is_leaf_ = false;

      // Low-rank approximation
      rank_ = low_rank_approx(start+half, size-half, start, half, tol, tol_abs, random, U_[1], V_[0]);
      U_[0] = V_[0];
      V_[1] = U_[1];
      if(parent_==NULL && verbose==1)
          std::cout<<"top-level rank: "<<rank_<<std::endl;

      if(rank_==half){
        Eigen::MatrixXd S;
        S = get_exact_matrix();

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd singular_values = svd.singularValues();
        std::cout << "Singular values of S are:\n" << singular_values << std::endl;

        std::ofstream file("fullmat.csv");
        for (int i = 0; i < S.rows(); ++i) {
            for (int j = 0; j < S.cols(); ++j) {
                file << S(i, j);
                if (j < S.cols() - 1)
                    file << ", ";  
            }
            file << "\n";
        }
        file.close();
        std::cout << "Full-rank block dumped to fullmat.csv.\n" << std::endl;
        exit(0);
      }

      // Build the children
      children_[0] = new Node<KernelType>(
          diag_, kernel_, start_, half, min_size, tol, tol_abs, verbose, random, 0, this);
      children_[1] = new Node<KernelType>(
          diag_, kernel_, start_+half, size_-half, min_size, tol, tol_abs, verbose, random, 1, this);

    } else {
      is_leaf_ = true;
    }
  };

  ~Node () {
    if (!is_leaf_) {
      delete children_[0];
      delete children_[1];
    }
  };

  void compute () {
    log_det_ = 0.0;
    if (!is_leaf_) {
      children_[0]->compute();
      children_[1]->compute();
      log_det_ = children_[0]->log_det_ + children_[1]->log_det_;
    }

    // Compute a factorize the inner matrix S
    factorize();

    // Compute the determinant
    if (is_leaf_) {
      Eigen::VectorXd diag = ldlt_.vectorD();
      for (int n = 0; n < diag.rows(); ++n) log_det_ += log(std::abs(diag(n)));
    } else {
      Eigen::MatrixXd lu = lu_.matrixLU();
      for (int n = 0; n < lu.rows(); ++n) log_det_ += log(std::abs(lu(n, n)));
    }

    Node<KernelType>* node = parent_;
    int start = start_, ind = direction_;
    while (node) {
      apply_inverse(node->U_[ind], start);
      start = node->start_;
      ind = node->direction_;
      node = node->parent_;
    }
  };

  double log_determinant () const { return log_det_; };

  template <typename Derived>
  void solve (Eigen::MatrixBase<Derived>& x) const {
    if (!is_leaf_) {
      children_[0]->solve(x);
      children_[1]->solve(x);
    }
    apply_inverse(x, 0);
  };

  Eigen::VectorXd dot_solve (Eigen::MatrixXd& x) const {
    Eigen::MatrixXd b = x;
    solve(b);
    return x.transpose() * b;
  };

  Eigen::MatrixXd get_exact_matrix () const {
    Eigen::MatrixXd K(size_, size_);
    for (int n = 0; n < size_; ++n) {
      K(n, n) = diag_(start_ + n) + kernel_->get_value(start_ + n, start_ + n);
      for (int m = n+1; m < size_; ++m) {
        double value = kernel_->get_value(start_ + m, start_ + n);
        K(m, n) = value;
        K(n, m) = value;
      }
    }
    return K;
  };

private:
  int low_rank_approx (int start_row,
                       int n_rows,
                       int start_col,
                       int n_cols,
                       double tol,
                       double tol_abs,
                       std::mt19937& random,
                       Eigen::MatrixXd& U_out,
                       Eigen::MatrixXd& V_out) const
  {

    // Allocate all the memory that we'll need.
    int max_rank = std::min(n_rows, n_cols);
    Eigen::MatrixXd U(n_rows, max_rank),
                    V(n_cols, max_rank);

    // Setup
    int rank = 0;
    int ntry = 0;
    int qrsvd = 0;
    double norm = 0.0, tol2 = tol * tol, tol_abs2 = tol_abs * tol_abs;
    double smallval = 1e-14;
    std::vector<int> index(n_rows);
    for (int n = 0; n < n_rows; ++n) index[n] = n;

    while (1) {
      int i, j, k;
      do {
        // If we run out of rows to try, just return the trivial factorization
        if (index.empty()) {
          U_out.resize(n_rows, max_rank);
          V_out.resize(n_cols, max_rank);
          if (n_cols <= n_rows) {
            V_out.setIdentity();
            for (int m = 0; m < n_cols; ++m)
              for (int n = 0; n < n_rows; ++n)
                U_out(n, m) = kernel_->get_value(start_row + n, start_col + m);
          } else {
            U_out.setIdentity();
            for (int n = 0; n < n_rows; ++n)
              for (int m = 0; m < n_cols; ++m)
                V_out(m, n) = kernel_->get_value(start_row + n, start_col + m);
          }
          return max_rank;
        }

        // Choose a random row
        std::uniform_int_distribution<int> uniform_dist(0, index.size()-1);
        k = uniform_dist(random);
        i = index[k];
        index[k] = index.back();
        index.pop_back();

        // Compute the residual and choose the pivot
        for (int n = 0; n < n_cols; ++n)
          V(n, rank) = kernel_->get_value(start_row + i, start_col + n);
        V.col(rank) -= U.row(i).head(rank) * V.block(0, 0, n_cols, rank).transpose();
        V.col(rank).cwiseAbs().maxCoeff(&j);

      // } while (std::abs(V(j, rank)) < smallval);
      } while (std::abs(V(j, rank)) < smallval && ++ntry<10);
      if(std::abs(V(j, rank)) < smallval) break;

      // Normalize
      V.col(rank) /= V(j, rank);

      // Compute the U factorization
      for (int n = 0; n < n_rows; ++n)
        U(n, rank) = kernel_->get_value(start_row + n, start_col + j);
      U.col(rank) -= V.row(j).head(rank) * U.block(0, 0, n_rows, rank).transpose();

      // Update the rank
      rank++;
      if (rank >= max_rank) break;

      // Only update if this is a substantial change
      double rowcol_norm = U.col(rank-1).squaredNorm() * V.col(rank-1).squaredNorm();
      // std::cout<< rank << " "<< rowcol_norm << " "<< tol2 * norm  << " "<< tol_abs2<<std::endl;
      if (rowcol_norm < tol2 * norm || rowcol_norm < tol_abs2) break;

      // Update the estimate of the norm
      norm += rowcol_norm;
      if (rank > 1) {
        norm += 2.0 * (U.block(0, 0, n_rows, rank-1).transpose() * U.col(rank-1)).cwiseAbs().maxCoeff();
        norm += 2.0 * (V.block(0, 0, n_cols, rank-1).transpose() * V.col(rank-1)).cwiseAbs().maxCoeff();
      }
    }

    if(rank==0)rank=1;

    if(qrsvd==1){
      // QR-SVD
      Eigen::MatrixXd Utmp = U.block(0, 0, n_rows, rank);
      Eigen::MatrixXd Vtmp = V.block(0, 0, n_cols, rank);
      Eigen::HouseholderQR<Eigen::MatrixXd> qru(Utmp);
      Eigen::MatrixXd Qu = qru.householderQ() * Eigen::MatrixXd::Identity(Utmp.rows(), Utmp.cols());
      Eigen::MatrixXd Ru = qru.matrixQR().triangularView<Eigen::Upper>();
      Eigen::HouseholderQR<Eigen::MatrixXd> qrv(Vtmp);
      Eigen::MatrixXd Qv = qrv.householderQ() * Eigen::MatrixXd::Identity(Vtmp.rows(), Vtmp.cols());
      Eigen::MatrixXd Rv = qrv.matrixQR().triangularView<Eigen::Upper>();
      Eigen::MatrixXd RuRv = Ru * Rv.transpose();

      // Compute the full SVD of RuRv
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(RuRv, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::VectorXd singularValues = svd.singularValues();
      // Determine the truncation threshold
      double maxSingularValue = singularValues(0);  // Assuming the singular values are sorted in descending order
      double threshold = std::max(tol * maxSingularValue, tol_abs);
      // Count significant singular values
      int ranknew = 0;
      for (int i = 0; i < singularValues.size(); ++i) {
          if (singularValues(i) > threshold)
              ranknew++;
          else
              break;
      }
      ranknew = std::max(ranknew,1);
      rank = ranknew;
      // Construct the truncated matrices
      Eigen::MatrixXd U_truncated = svd.matrixU().leftCols(rank);
      Eigen::MatrixXd V_truncated = svd.matrixV().leftCols(rank);
      Eigen::VectorXd Sigma_truncated = singularValues.head(rank);

      // Optionally, construct the diagonal matrix for Sigma
      Eigen::MatrixXd Sigma_diag = Sigma_truncated.asDiagonal();

      U_out = Utmp* U_truncated* Sigma_diag;
      V_out = Vtmp * V_truncated.transpose();
    }else{
      U_out = U.block(0, 0, n_rows, rank);
      V_out = V.block(0, 0, n_cols, rank);
    }

    return rank;
  };

  void factorize () {
    Eigen::MatrixXd S;
    if (is_leaf_) {
      S = get_exact_matrix();
      ldlt_.compute(S);
    } else {
      S.resize(2*rank_, 2*rank_);
      S.setIdentity();
      S.block(0, rank_, rank_, rank_) = V_[1].transpose() * U_[1];
      S.block(rank_, 0, rank_, rank_) = V_[0].transpose() * U_[0];
      lu_.compute(S);
    }
  };

  template <typename Derived>
  void apply_inverse (Eigen::MatrixBase<Derived>& x, int start) const {
    int nrhs = x.cols();
    start = start_ - start;
    if (is_leaf_) {
      x.block(start, 0, size_, nrhs) = ldlt_.solve(x.block(start, 0, size_, nrhs));
      return;
    }

    int s1 = size_ / 2, s2 = size_ - s1;
    Eigen::MatrixXd temp(2*rank_, nrhs);
    temp.block(0, 0, rank_, nrhs)     = V_[1].transpose() * x.block(start+s1, 0, s2, nrhs);
    temp.block(rank_, 0, rank_, nrhs) = V_[0].transpose() * x.block(start, 0, s1, nrhs);
    temp = lu_.solve(temp);

    x.block(start, 0, s1, nrhs)    -= U_[0] * temp.block(0, 0, rank_, nrhs);
    x.block(start+s1, 0, s2, nrhs) -= U_[1] * temp.block(rank_, 0, rank_, nrhs);
  };

};

  } // namespace hodlr
}   // namespace george

#endif
