#ifndef __utils_hpp__  
#define __utils_hpp__

#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <numeric> // For std::accumulate
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// Write tensor to binary
template <typename Scalar, int Rank>
void tensor_to_bin(std::string& filename, Eigen::Tensor<Scalar, Rank>& tensor)
{
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) 
    {
        file.write(reinterpret_cast<const char*>(tensor.data()), tensor.size() * sizeof(Scalar));
        file.close();
    } 
    else 
    {
        std::cerr << "Could not open file\n";
    }
}

// Truncate singular values after one SVD
void reduce_bond_dim(Eigen::MatrixXd& U, Eigen::MatrixXd& V, Eigen::VectorXd s, const double tol);

// Perform SVD decomposition on tensor once
Eigen::Tensor<double, 2> rank3_tensor_SVD(const Eigen::Tensor<double, 3>& tensor, const double tol, std::vector<Eigen::Tensor<double, 2>>& U_);

#endif
