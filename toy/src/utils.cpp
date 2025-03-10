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
    s = s(Eigen::seqN(0, new_d)); // keep first new_d singular values
	U = U(Eigen::seqN(0,s.size()), Eigen::all); // keep only the first s.size() rows in U
	V = V(Eigen::all, Eigen::seqN(0, s.size())); // keep only the first s.size() columns
}

// Perform SVD decomposition on tensor once
Eigen::Tensor<double, 2> rank3_tensor_SVD(const Eigen::Tensor<double, 3>& tensor, const double tol, std::vector<Eigen::Tensor<double, 3>>& MPS, int it)
{
	// Reshape tensor by grouping all except first index
	auto tensor_dims = tensor.dimensions(); // array of dimension sizes [x, y, z]
	Eigen::array<Eigen::Index, 2> new_dims;

	// First SVD
	if(it == 0)
	{
	    new_dims = {{tensor_dims[0], tensor_dims[1] * tensor_dims[2]}};
	}
	// Other SVD's
	else
	{
		new_dims = {{tensor_dims[0] * tensor_dims[1], tensor_dims[2]}};
	}
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

	// Check dimensions before multiplication
    if (V.cols() != s.rows())
    {
        std::cerr << "Error: Dimension mismatch in matrix multiplication. V.cols() = " << V.cols() 
                  << ", s.rows() = " << s.rows() << std::endl;
        throw std::runtime_error("Dimension mismatch in matrix multiplication");
    }

    // Compute V * s
	// dont know where singular values should actually go
	Eigen::MatrixXd Vs = V * s.asDiagonal();

	// Convert back to Eigen::Tensor datatype
	Eigen::TensorMap<Eigen::Tensor<double, 2>> U_tensor(U.data(), U.rows(), U.cols());
	Eigen::TensorMap<Eigen::Tensor<double, 2>> Vs_tensor(Vs.data(), Vs.rows(), Vs.cols());

	// Reshape and store U 
	auto dims = U_tensor.dimensions();
	// Add dummy index to first matrix
	if(it == 0)
    {
        Eigen::Tensor<double, 3> U_tensor3 = U_tensor.reshape(Eigen::array<Eigen::Index, 3>{{1, dims[0], dims[1]}});
        MPS.push_back(U_tensor3);
    }
    else
    // Split combined index
    {
    	Eigen::Tensor<double, 3> U_tensor3 = U_tensor.reshape(Eigen::array<Eigen::Index, 3>{{MPS[0].dimension(2), MPS[0].dimension(1), dims[1]}}); //only works if x, y, z have same dimensions
        MPS.push_back(U_tensor3);
    }

	// Return remaining tensor
	return Vs_tensor;
}

