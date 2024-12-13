#ifndef _GEORGE_HODLR_H_
#define _GEORGE_HODLR_H_

#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <Eigen/SVD>

using RowMatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace george {
  namespace hodlr {

template <typename KernelType>
class Node {
private:
  const Eigen::VectorXd& diag_;
  KernelType* kernel_;
  Node<KernelType>* parent_;
  std::vector<Node<KernelType>*> children_;
  int start_, size_, direction_, rank_, sym_;
  bool is_leaf_;
  std::vector<Eigen::MatrixXd> U_, V_, Qfactor;
  Eigen::MatrixXd fullmat_,K;
  Eigen::FullPivLU<Eigen::MatrixXd> lu_;
  Eigen::LDLT<Eigen::MatrixXd> ldlt_;
  Eigen::LLT<Eigen::MatrixXd> llt_;
  double log_det_;
  int grad_;

public:
  Eigen::MatrixXd fullmat_nxn_;
  Node (const Eigen::VectorXd& diag,
        KernelType* kernel,
        int start,
        int size,
        int min_size,
        double tol,
        double tol_abs,
        int verbose,
        int debug,
        int sym,
        std::mt19937& random,
        int direction = 0,
        int grad = 0,
        Node<KernelType>* parent = NULL)
    : diag_(diag)
    , kernel_(kernel)
    , parent_(parent)
    , children_(2)
    , start_(start)
    , size_(size)
    , sym_(sym)
    , direction_(direction)
    , grad_(grad)
    , U_(2)
    , V_(2)
    , Qfactor(2)
  {
    if(debug==1 && parent_==NULL)
      fullmat_nxn_=get_exact_matrix();

    int half = size_ / 2;
    if (half >= min_size) {
      is_leaf_ = false;

      // std::cout<<kernel_->nns_(0,0)<<" "<<kernel_->nns_(0,1)<<" "<<kernel_->nns_(0,2)<<std::endl;

      // Low-rank approximation
      rank_ = low_rank_approx(start+half, size-half, start, half, tol, tol_abs, kernel_->nns_, random, U_[1], V_[0]);
      U_[0] = V_[0];
      V_[1] = U_[1];
      Qfactor[0]=U_[0];
      Qfactor[1]=U_[1];
      K = Eigen::MatrixXd::Identity(rank_, rank_);
      if(parent_==NULL && verbose==1)
          if(grad_==0)
            std::cout<<"top-level rank of K: "<<rank_<<std::endl;
          else
            std::cout<<"top-level rank of pK: "<<rank_<<" for hyperparameter "<< grad_-1 <<std::endl;

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
          diag_, kernel_, start_, half, min_size, tol, tol_abs, verbose, debug, sym, random, 0, grad_, this);
      children_[1] = new Node<KernelType>(
          diag_, kernel_, start_+half, size_-half, min_size, tol, tol_abs, verbose, debug, sym, random, 1, grad_, this);

    } else {
      is_leaf_ = true;
      fullmat_=get_exact_matrix();
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

    // Compute and factorize the inner matrix S
    factorize();

    // Compute the determinant
    if(sym_==0)
      if (is_leaf_) {
        Eigen::VectorXd diag = ldlt_.vectorD();
        for (int n = 0; n < diag.rows(); ++n) log_det_ += log(std::abs(diag(n)));
      } else {
        Eigen::MatrixXd lu = lu_.matrixLU();
        for (int n = 0; n < lu.rows(); ++n) log_det_ += log(std::abs(lu(n, n)));
      }
    else{
      if (is_leaf_) {
        // bool use_ldlt = (ldlt_.vectorD().array() > 0).all();
        bool use_ldlt = 1;
        if(use_ldlt){
          Eigen::VectorXd diag = ldlt_.vectorD();
          for (int n = 0; n < diag.rows(); ++n) log_det_ += log(std::abs(diag(n)));  
        }else{
          for (int n=0; n<llt_.matrixL().rows(); ++n)log_det_ += 2.0*log(std::abs(llt_.matrixL()(n,n)));
        }
      } else {
        // bool use_ldlt = (ldlt_.vectorD().array() > 0).all();
        bool use_ldlt = 1;
        if(use_ldlt){
          Eigen::VectorXd diag = ldlt_.vectorD();
          for (int n = 0; n < diag.rows(); ++n) log_det_ += log(std::abs(diag(n)));  
        }else{
          for (int n=0; n<llt_.matrixL().rows(); ++n)log_det_ += 2.0*log(std::abs(llt_.matrixL()(n,n)));
        }
      }        
		}  

    Node<KernelType>* node = parent_;
    int start = start_, ind = direction_;
    while (node) {
      if(sym_==0)
        apply_inverse_normal(node->U_[ind], start);
      else
        apply_inverse_sym(node->Qfactor[ind], start);

      start = node->start_;
      ind = node->direction_;
      node = node->parent_;
    }
  };

  double log_determinant () const { return log_det_; };

  template <typename Derived>
  void solve_sym (Eigen::MatrixBase<Derived>& x) const {
    if (!is_leaf_) {
      children_[0]->solve_sym(x);
      children_[1]->solve_sym(x);
    }
    apply_inverse_sym(x, 0);
  };

  template <typename Derived>
  void solve_sym_transpose (Eigen::MatrixBase<Derived>& x) const {
    apply_inverse_sym_transpose(x, 0);
    if (!is_leaf_) {
      children_[0]->solve_sym_transpose(x);
      children_[1]->solve_sym_transpose(x);
    }
  };


  template <typename Derived>
  void solve_normal (Eigen::MatrixBase<Derived>& x) const {
    if (!is_leaf_) {
      children_[0]->solve_normal(x);
      children_[1]->solve_normal(x);
    }
    apply_inverse_normal(x, 0);
  };


  template <typename Derived>
  void solve (Eigen::MatrixBase<Derived>& x) const {
    if(sym_==0){
      solve_normal(x);
    }else{
      solve_sym(x);      
      solve_sym_transpose(x);  
    }
  };

  Eigen::VectorXd dot_solve (Eigen::MatrixXd& x) const {
    Eigen::MatrixXd b = x;
    if(sym_==0){
      solve_normal(b);
    }else{
      solve_sym(b);      
      solve_sym_transpose(b);      
    }
    return x.transpose() * b;
  };

  Eigen::MatrixXd get_exact_matrix () const {
    Eigen::MatrixXd K(size_, size_);
    for (int n = 0; n < size_; ++n) {
      if(grad_==0)
        K(n, n) = diag_(start_ + n) + kernel_->get_value(start_ + n, start_ + n);
      else
        K(n, n) = kernel_->get_gradient(start_ + n, start_ + n, grad_-1);
    
      double value;
      for (int m = n+1; m < size_; ++m) {
        if(grad_==0)
          value = kernel_->get_value(start_ + m, start_ + n);
        else 
          value = kernel_->get_gradient(start_ + m, start_ + n, grad_-1);

        K(m, n) = value;
        K(n, m) = value;
      }
    }
    return K;
  };


  template <typename Derived>
  void apply_forward(Eigen::MatrixBase<Derived>& x, Eigen::MatrixBase<Derived>& y) const {
    int nrhs = x.cols();
    int start = start_;
    if (is_leaf_) {
      y.block(start, 0, size_, nrhs) +=  fullmat_ * x.block(start, 0, size_, nrhs);
      return;
    }
    int s1 = size_ / 2, s2 = size_ - s1;
    Eigen::MatrixXd temp(2*rank_, nrhs);
    temp.block(0, 0, rank_, nrhs)     = V_[1].transpose() * x.block(start+s1, 0, s2, nrhs);
    temp.block(rank_, 0, rank_, nrhs) = V_[0].transpose() * x.block(start, 0, s1, nrhs);
    y.block(start, 0, s1, nrhs)    += U_[0] * temp.block(0, 0, rank_, nrhs);
    y.block(start+s1, 0, s2, nrhs) += U_[1] * temp.block(rank_, 0, rank_, nrhs);

    children_[0]->apply_forward(x,y);
    children_[1]->apply_forward(x,y);
  };



private:


  Eigen::MatrixXd pseudoInverse(Eigen::MatrixXd &A, double tol_rel = 1e-6, double tol_abs = 1e-30) const {
      // Compute SVD
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
      const auto &U = svd.matrixU();
      const auto &V = svd.matrixV();
      const auto &S = svd.singularValues();

      // Compute the reciprocal of singular values
      // If the singular value is smaller than tolerance, treat it as zero.
      Eigen::VectorXd S_inv = S;
      for (int i = 0; i < S.size(); i++) {
          if (S(i) > S(0)*tol_rel && S(i) >tol_abs) {
              S_inv(i) = 1.0 / S(i);
          } else {
              S_inv(i) = 0.0;
          }
      }

      // Construct the diagonal matrix S_plus from S_inv
      Eigen::MatrixXd S_plus = S_inv.asDiagonal();

      // Compute A+ = V * S_plus * U^T
      return V * S_plus * U.transpose();
  }

  void removeRedundantElements(std::vector<int>& ridxs) const {
      // Step 1: Sort the vector
      std::sort(ridxs.begin(), ridxs.end());

      // Step 2: Remove duplicates using std::unique
      auto newEnd = std::unique(ridxs.begin(), ridxs.end());

      // Step 3: Erase the redundant elements
      ridxs.erase(newEnd, ridxs.end());
  }


  int low_rank_approx (int start_row,
                       int n_rows,
                       int start_col,
                       int n_cols,
                       double tol,
                       double tol_abs,
                       RowMatrixXi& nns,
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
    int qrsvd = 1;
    int fullsvd = 0;
    int skeleton = 0;
    double norm = 0.0, tol2 = tol * tol, tol_abs2 = tol_abs * tol_abs;
    double smallval = 1e-14;
    std::vector<int> index(n_rows);
    std::vector<int> ridxs;
    std::vector<int> cidxs;
    for (int n = 0; n < n_rows; ++n) index[n] = n;

    if(fullsvd==1){
      Eigen::MatrixXd full = Eigen::MatrixXd::Zero(n_rows, n_cols);

      for (int m = 0; m < n_cols; ++m)
        for (int n = 0; n < n_rows; ++n)
          if(grad_==0)
            full(n, m) = kernel_->get_value(start_row + n, start_col + m);
          else
            full(n, m) = kernel_->get_gradient(start_row + n, start_col + m, grad_-1);

      // Compute the full SVD 
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(full, Eigen::ComputeFullU | Eigen::ComputeFullV);
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
      U_out.resize(n_rows, rank);
      U_out = U_truncated* Sigma_diag;
      V_out.resize(n_cols, rank);
      V_out = V_truncated;
      return rank;
    }


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
                if(grad_==0)
                  U_out(n, m) = kernel_->get_value(start_row + n, start_col + m);
                else
                  U_out(n, m) = kernel_->get_gradient(start_row + n, start_col + m, grad_-1);
          } else {
            U_out.setIdentity();
            for (int n = 0; n < n_rows; ++n)
              for (int m = 0; m < n_cols; ++m)
                if(grad_==0)
                  V_out(m, n) = kernel_->get_value(start_row + n, start_col + m);
                else
                  V_out(m, n) = kernel_->get_gradient(start_row + n, start_col + m, grad_-1);
          }
          return max_rank;
        }

        // Choose a random row
        std::uniform_int_distribution<int> uniform_dist(0, index.size()-1);
        k = uniform_dist(random);
        i = index[k];
        ridxs.push_back(i);
        index[k] = index.back();
        index.pop_back();

        // Compute the residual and choose the pivot
        for (int n = 0; n < n_cols; ++n)
          if(grad_==0)
            V(n, rank) = kernel_->get_value(start_row + i, start_col + n);
          else
            V(n, rank) = kernel_->get_gradient(start_row + i, start_col + n, grad_-1);
        V.col(rank) -= U.row(i).head(rank) * V.block(0, 0, n_cols, rank).transpose();
        V.col(rank).cwiseAbs().maxCoeff(&j);
        cidxs.push_back(j);

      // } while (std::abs(V(j, rank)) < smallval);
      } while (std::abs(V(j, rank)) < smallval && ++ntry<10);
      if(std::abs(V(j, rank)) < smallval) break;

      // Normalize
      V.col(rank) /= V(j, rank);

      // Compute the U factorization
      for (int n = 0; n < n_rows; ++n)
        if(grad_==0)
          U(n, rank) = kernel_->get_value(start_row + n, start_col + j);
        else 
          U(n, rank) = kernel_->get_gradient(start_row + n, start_col + j, grad_-1);
          
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

    if(rank==0){
      U.col(0).setZero();
      V.col(0).setZero();
      ridxs.push_back(0);
      cidxs.push_back(0);
      rank=1;
      }



  if(skeleton==1){
   for (int n = 0; n < n_rows; ++n){
    for (int k=0; k<nns.cols();++k){
      if(nns(start_row + n,k)>=start_col && nns(start_row + n,k)<start_col+n_cols){
        cidxs.push_back(nns(start_row + n,k) - start_col);
      }
    }
   }
   removeRedundantElements(cidxs);

   for (int m = 0; m < n_cols; ++m){
    for (int k=0; k<nns.cols();++k){
      if(nns(start_col + m,k)>=start_row && nns(start_col + m,k)<start_row+n_rows){
        ridxs.push_back(nns(start_col + m,k) - start_row);
      }
    }
   }
   removeRedundantElements(ridxs);





    rank = ridxs.size();
    Eigen::MatrixXd U1(n_rows, cidxs.size()),
                    V1(n_cols, ridxs.size()),
                    K1(ridxs.size(), cidxs.size());
    
      for (int m = 0; m < cidxs.size(); ++m)
        for (int n = 0; n < n_rows; ++n)
          if(grad_==0)
            U1(n, m) = kernel_->get_value(start_row + n, start_col + cidxs[m]);
          else
            U1(n, m) = kernel_->get_gradient(start_row + n, start_col + cidxs[m], grad_-1);

      for (int m = 0; m < n_cols; ++m)
        for (int n = 0; n < ridxs.size(); ++n)
          if(grad_==0)
            V1(m, n) = kernel_->get_value(start_row + ridxs[n], start_col + m);
          else
            V1(m, n) = kernel_->get_gradient(start_row + ridxs[n], start_col + m, grad_-1);

      for (int m = 0; m < cidxs.size(); ++m)
        for (int n = 0; n < ridxs.size(); ++n)
          if(grad_==0)
            K1(n, m) = kernel_->get_value(start_row + ridxs[n], start_col + cidxs[m]);
          else
            K1(n, m) = kernel_->get_gradient(start_row + ridxs[n], start_col + cidxs[m], grad_-1);
      

      Eigen::MatrixXd U1tmp = U1*pseudoInverse(K1, tol, tol_abs);
      U1.resize(n_rows,ridxs.size());
      U1 = U1tmp;

      // Eigen::MatrixXd A1 = U1*V1;
      // Eigen::MatrixXd A = U.block(0, 0, n_rows, rank)*V.block(0, 0, n_cols, rank);
      // std::cout<<" hh "<<A1.norm()<<" "<<A.norm()<<" "<<(A-A1).norm()<<std::endl;
      U.block(0, 0, n_rows, rank)=U1;
      V.block(0, 0, n_rows, rank)=V1;
    }


    if(qrsvd==1){
      // QR-SVD
      Eigen::MatrixXd Utmp = U.block(0, 0, n_rows, rank);
      Eigen::MatrixXd Vtmp = V.block(0, 0, n_cols, rank);
      Eigen::HouseholderQR<Eigen::MatrixXd> qru(Utmp);
      Eigen::MatrixXd Qu = qru.householderQ() * Eigen::MatrixXd::Identity(Utmp.rows(), Utmp.cols());
      Eigen::MatrixXd Ru = qru.matrixQR().topLeftCorner(Utmp.cols(), Utmp.cols()).triangularView<Eigen::Upper>();
      Eigen::HouseholderQR<Eigen::MatrixXd> qrv(Vtmp);
      Eigen::MatrixXd Qv = qrv.householderQ() * Eigen::MatrixXd::Identity(Vtmp.rows(), Vtmp.cols());
      Eigen::MatrixXd Rv = qrv.matrixQR().topLeftCorner(Vtmp.cols(), Vtmp.cols()).triangularView<Eigen::Upper>();
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
      U_out = Qu* U_truncated* Sigma_diag;
      V_out = Qv * V_truncated;
    }else{
      U_out = U.block(0, 0, n_rows, rank);
      V_out = V.block(0, 0, n_cols, rank);
    }

    return rank;
  };

  void factorize () {
    if(sym_==0){
      Eigen::MatrixXd S;
      if (is_leaf_) {

        double jitter=1e-10;
        ldlt_.compute(fullmat_);
        // bool is_spd = (ldlt_.vectorD().array() > 0).all();
        // int iter=0;
        // while(!is_spd){
        //   ldlt_.compute(Eigen::MatrixXd::Identity(fullmat_.rows(), fullmat_.rows())*jitter + fullmat_);
        //   is_spd = (ldlt_.vectorD().array() > 0).all();
        //   iter = iter + 1;
        //   // std::cout<<"Jitter for fullmat_ "<<iter<<" "<<jitter<<std::endl;
        //   // std::cout<<ldlt_.vectorD().array()<<std::endl;
        //   jitter=jitter*10;
        // }
        // if(iter>0)std::cout<<"Jitter for fullmat_ set to "<<jitter<<std::endl;
        // if(!is_spd){
        //   std::cerr << "matrix not spd to use ldlt"<<std::endl;
        //   std::cout<<ldlt_.vectorD().array()<<std::endl;
        //   std::abort();
        // }

      } else {
        S.resize(2*rank_, 2*rank_);
        S.setIdentity();
        S.block(0, rank_, rank_, rank_) = V_[1].transpose() * U_[1];
        S.block(rank_, 0, rank_, rank_) = V_[0].transpose() * U_[0];
        lu_.compute(S);
      }
    }else{
      if (is_leaf_) {
        llt_.compute(fullmat_);
        ldlt_.compute(fullmat_);

        bool is_spd = (ldlt_.vectorD().array() > 0).all();
        double jitter=1e-10;
        int iter=0;
        while(!is_spd){
          ldlt_.compute(Eigen::MatrixXd::Identity(fullmat_.rows(), fullmat_.rows())*jitter + fullmat_);
          is_spd = (ldlt_.vectorD().array() > 0).all();
          iter = iter + 1;
          // std::cout<<"Jitter for fullmat_ "<<iter<<" "<<jitter<<std::endl;
          // std::cout<<ldlt_.vectorD().array()<<std::endl;
          jitter=jitter*10;
        }
        if(iter>0)std::cout<<"Jitter for fullmat_ set to "<<jitter<<std::endl;
        if(!is_spd){
          std::cerr << "matrix not spd to use ldlt"<<std::endl;
          std::abort();
        }

      } else {
        int min0 = std::min(Qfactor[0].rows(), Qfactor[0].cols());
        int min1 = std::min(Qfactor[1].rows(), Qfactor[1].cols());
        int rank = Qfactor[0].cols();

        // Update K and Qfactor to make sure Qfactor is unitary via QR
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(Qfactor[0]);
        Qfactor[0] = qr.householderQ()*(Eigen::MatrixXd::Identity(Qfactor[0].rows(), min0));
        K = qr.matrixQR().block(0,0,min0,Qfactor[0].cols()).triangularView<Eigen::Upper>()*K;

        Eigen::HouseholderQR<Eigen::MatrixXd> qr1(Qfactor[1]);
        Qfactor[1] = qr1.householderQ()*(Eigen::MatrixXd::Identity(Qfactor[1].rows(), min1));
        K *= qr1.matrixQR().block(0,0,min1,Qfactor[1].cols()).triangularView<Eigen::Upper>().transpose();
       
       
        // Compute the LL^T of I-K^T*K
      	llt_.compute(Eigen::MatrixXd::Identity(rank, rank) - K.transpose()*K);
        
      	ldlt_.compute(Eigen::MatrixXd::Identity(rank, rank) - K.transpose()*K);

        double jitter=1e-10;
        bool is_spd = (ldlt_.vectorD().array() > 0).all();
        int iter=0;
        if(ldlt_.vectorD().array()(0) < 0){
          ldlt_.compute(Eigen::MatrixXd::Identity(rank, rank));
          is_spd = (ldlt_.vectorD().array() > 0).all();
        }
        while(!is_spd){
          ldlt_.compute(Eigen::MatrixXd::Identity(rank, rank)*(1+jitter) - K.transpose()*K);
          is_spd = (ldlt_.vectorD().array() > 0).all();
          iter = iter + 1;
          // std::cout<<"Jitter for I-K^T*K "<<iter<<" "<<jitter<<std::endl;
          jitter=jitter*10;
        }
        if(iter>0)std::cout<<"Jitter for I-K^T*K set to "<<jitter<< " rank "<<rank<<" Knorm "<<K.norm()<<std::endl;
        if(!is_spd){
          std::cout<<ldlt_.vectorD().array()<<std::endl;
          std::cerr << "I-K^T*K not spd to use ldlt"<<std::endl;
          std::abort();
        }        
        
        
        // ldlt_.compute(Eigen::MatrixXd::Identity(rank, rank) - K.transpose()*K);
        // bool is_spd = (ldlt_.vectorD().array().abs() > 1e-14).all();
        // if(!(ldlt_.vectorD().array() > 0).all())std::cout<<ldlt_.vectorD().array()<<std::endl;
        // if(!is_spd){
        //   std::cout<<ldlt_.vectorD().array()<<std::endl;
        //   std::cerr << "D of ldlt of I-K^T*K contains small entries"<<std::endl;
        //   std::abort();
        // }        
      }      
    }
  };



  template <typename Derived>
  void apply_inverse_normal (Eigen::MatrixBase<Derived>& x, int start) const {
    int nrhs = x.cols();
    start = start_ - start;
    if (is_leaf_) {


      // x.block(start, 0, size_, nrhs) = ldlt_.solve(x.block(start, 0, size_, nrhs));
      
      
      Eigen::MatrixXd tmpX = ldlt_.solve(x.block(start, 0, size_, nrhs));

    // Step 1: Permute B
    Eigen::MatrixXd PB = ldlt_.transpositionsP() * x.block(start, 0, size_, nrhs);

    // Step 2: Forward solve L * Y = PB
    Eigen::MatrixXd Y = ldlt_.matrixL().solve(PB);

    // Step 3: Solve D * Z = Y (row-wise division by diagonal entries of D)
      double smallval = fabs(ldlt_.vectorD()(0))*1e-13;
      for (int i = 0; i < ldlt_.vectorD().size(); ++i){
        if(fabs(ldlt_.vectorD()(i))<smallval || ldlt_.vectorD()(i)<0){
            Y.row(i).setZero();
        }else{
          Y.row(i) /= ldlt_.vectorD()(i);
          // if(ldlt_.vectorD()(i)<0){
          //   Y.row(i) /= -sqrt(fabs(ldlt_.vectorD()(i)));  
          // }else{
          //   Y.row(i) /= sqrt(ldlt_.vectorD()(i));
          // }
        }
      }

    // Step 4: Backward solve L^T * X = Z
    Eigen::MatrixXd X = ldlt_.matrixL().transpose().solve(Y);

    // Step 5: Apply the inverse permutation
    x.block(start, 0, size_, nrhs) = ldlt_.transpositionsP().transpose() * X;

    //  std::cout << "Difference between manual and eigen: " << x.block(start, 0, size_, nrhs).norm()<<" "<<tmpX.norm() << " "<<(tmpX-x.block(start, 0, size_, nrhs)).norm()<< std::endl;

      
      
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


  template <typename Derived>
  void apply_inverse_sym (Eigen::MatrixBase<Derived>& x, int start) const {
    
    int nrhs = x.cols();
    start = start_ - start;
    if (is_leaf_) {
      // bool use_ldlt = (ldlt_.vectorD().array() > 0).all();
      bool use_ldlt = 1;
      if(use_ldlt){
        Eigen::MatrixXd Pb = ldlt_.transpositionsP() * x.block(start, 0, size_, nrhs);
        x.block(start, 0, size_, nrhs) = ldlt_.matrixL().solve(Pb);
        double smallval = fabs(ldlt_.vectorD()(1))*1e-13;
        for (int i = 0; i < ldlt_.vectorD().size(); ++i){
          // if(fabs(ldlt_.vectorD()(i))<smallval || ldlt_.vectorD()(i)<0){
          //   x.block(start, 0, size_, nrhs).row(i).setZero();
          // }else{
            // if(ldlt_.vectorD()(i)<1e-13){
            //   x.block(start, 0, size_, nrhs).row(i).setZero();
            // }else{
              x.block(start, 0, size_, nrhs).row(i) /= sqrt(ldlt_.vectorD()(i));
            // }
          // }
        }
      }else{
        x.block(start, 0, size_, nrhs) = llt_.matrixL().solve(x.block(start, 0, size_, nrhs));  
      }

      return;
    }

    int s1 = size_ / 2, s2 = size_ - s1;
    Eigen::MatrixXd tmp = Qfactor[1].transpose()*x.block(start+s1, 0, s2, nrhs);
    Eigen::MatrixXd tmpb = (K.transpose()*(Qfactor[0].transpose()*x.block(start, 0, s1, nrhs))) - tmp;

     
    // bool use_ldlt = (ldlt_.vectorD().array() > 0).all();
    bool use_ldlt = 1;
    if(use_ldlt){
      Eigen::MatrixXd Pb = ldlt_.transpositionsP() * tmpb;
      Eigen::MatrixXd y = ldlt_.matrixL().solve(Pb);
      double smallval = fabs(ldlt_.vectorD()(1))*1e-13;
      for (int i = 0; i < ldlt_.vectorD().size(); ++i){
        // if(fabs(ldlt_.vectorD()(i))<smallval){
        //     y.row(i).setZero();
        // }else{
          // if(ldlt_.vectorD()(i)<1e-13){
          //   y.row(i).setZero();
          // }else{
            y.row(i) /= sqrt(ldlt_.vectorD()(i));
          // }
        // }
      }  
      x.block(start+s1, 0, s2, nrhs) -= Qfactor[1]*(y +tmp);      
    }else{
      x.block(start+s1, 0, s2, nrhs) -= Qfactor[1]*(llt_.matrixL().solve(tmpb) +tmp);      
    }
  };




  template <typename Derived>
  void apply_inverse_sym_transpose (Eigen::MatrixBase<Derived>& x, int start) const {
    int nrhs = x.cols();
    start = start_ - start;
    if (is_leaf_) {
      // bool use_ldlt = (ldlt_.vectorD().array() > 0).all();
      bool use_ldlt = 1;
      if(use_ldlt){
        // double smallval = fabs(ldlt_.vectorD()(1))*1e-13;
        for (int i = 0; i < ldlt_.vectorD().size(); ++i){
          // if(fabs(ldlt_.vectorD()(i))<smallval || ldlt_.vectorD()(i)<0){
          //   x.block(start, 0, size_, nrhs).row(i).setZero();
          // }else{
            // if(ldlt_.vectorD()(i)<1e-13){
            //   x.block(start, 0, size_, nrhs).row(i).setZero();
            // }else{
              x.block(start, 0, size_, nrhs).row(i) /= sqrt(ldlt_.vectorD()(i));
            // } 
          // }
        }
        x.block(start, 0, size_, nrhs) = ldlt_.matrixL().transpose().solve(x.block(start, 0, size_, nrhs));
        x.block(start, 0, size_, nrhs) = ldlt_.transpositionsP().transpose() *x.block(start, 0, size_, nrhs);
      }else{
        x.block(start, 0, size_, nrhs) = llt_.matrixL().transpose().solve(x.block(start, 0, size_, nrhs));  
      }

      return;
    }

    int s1 = size_ / 2, s2 = size_ - s1;
    Eigen::MatrixXd xtmp = Qfactor[1].transpose()*x.block(start+s1, 0, s2, nrhs);

    // bool use_ldlt = (ldlt_.vectorD().array() > 0).all();
    bool use_ldlt = 1;
    Eigen::MatrixXd ytmp;
    if(use_ldlt){
      Eigen::MatrixXd xtmp1 = xtmp;
      double smallval = fabs(ldlt_.vectorD()(1))*1e-13;
      for (int i = 0; i < ldlt_.vectorD().size(); ++i){
        // if(fabs(ldlt_.vectorD()(i))<smallval){
        //     xtmp1.row(i).setZero();
        // }else{
          // if(ldlt_.vectorD()(i)<1e-13){
          //   xtmp1.row(i).setZero(); 
          // }else{
            xtmp1.row(i) /= sqrt(ldlt_.vectorD()(i));
          // }
        // }
      }
      ytmp = ldlt_.matrixL().transpose().solve(xtmp1);
      ytmp = ldlt_.transpositionsP().transpose() *ytmp;
    }else{
      ytmp = llt_.matrixL().transpose().solve(xtmp);
    }
    
    
    x.block(start, 0, s1, nrhs) -= Qfactor[0]*(K*ytmp);
    x.block(start+s1, 0, s2, nrhs) -= Qfactor[1]*(xtmp - ytmp);

  };



  // template <typename Derived>
  // void apply_inverse (Eigen::MatrixBase<Derived>& x, int start) const {
  //   if(sym_==0){
  //     apply_inverse_normal(x,start);
  //   }else{
  //     apply_inverse_sym(x,start);
  //     apply_inverse_sym_transpose(x,start);
  //   }
  // }



};

  } // namespace hodlr
}   // namespace george

#endif
