#include "../include/utils.hpp"


// Truncate singular values after one SVD
void reduce_bond_dim(Eigen::MatrixXd& U, Eigen::MatrixXd& V, Eigen::VectorXd s, const double tol)
{
	int d = s.size();
	int new_d;
    for (int i = 0; i < d; ++i) 
    {
        if (s(i) < tol) 
        {
            new_d = i;
            break;
        }
    }
	U = U(Eigen::all, Eigen::seqN(0,new_d)); // keep new_d first columns in U
	V = V(Eigen::seqN(0,new_d), Eigen::all); // keep new_d first rows in V
	s = s(Eigen::seqN(0, new_d)); // keep first new_d singular values
}

// Perform SVD decomposition on tensor once
Eigen::Tensor<double, 2> rank3_tensor_SVD(const Eigen::Tensor<double, 3>& tensor, const double tol, std::vector<Eigen::Tensor<double, 2>>& U_)
{
	// Reshape tensor by grouping all except first index
	auto tensor_dims = tensor.dimensions(); // array of dimension sizes [x, y, z]
	Eigen::array<Eigen::Index, 2> new_dims{{tensor_dims[0], tensor_dims[1] * tensor_dims[2]}};
	Eigen::Tensor<double, 2> reshaped_tensor = tensor.reshape(new_dims); // define reshaped matrix

	// Convert data type
	Eigen::MatrixXd matrix = Eigen::Map<Eigen::MatrixXd>(reshaped_tensor.data(), reshaped_tensor.dimension(0), reshaped_tensor.dimension(1));

	// Perform SVD
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::VectorXd s = svd.singularValues();
	Eigen::MatrixXd U = svd.matrixU();
	Eigen::MatrixXd V = svd.matrixV();

	// Reduce bond dim by discarding singular values smaller than tol
	reduce_bond_dim(U, V, s, tol);

	// Convert back to Eigen::Tensor datatype
	Eigen::TensorMap<Eigen::Tensor<double, 2>> U_tensor(U.data(), U.rows(), U.cols());
	Eigen::TensorMap<Eigen::Tensor<double, 2>> V_tensor(V.data(), V.rows(), V.cols());
	Eigen::TensorMap<Eigen::Tensor<double, 1>> s_tensor(s.data(), s.size());

	// Save U
	U_.push_back(U_tensor);

	// Return V * s
	// dont know where singular values should actually go
	return V_tensor * s_tensor.reshape(Eigen::array<Eigen::Index, 2>{{1, s_tensor.size()}}).broadcast(Eigen::array<Eigen::Index, 2>{{V_tensor.dimension(0), 1}}); // reshape into diagonal matrix
}

