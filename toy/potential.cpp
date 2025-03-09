//#include <cmath>
#include "include/utils.hpp"

// Bulding on M1 mac: 
// g++ -std=c++11 potential.cpp -I/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 -o potential

int main(int argc, char* argv[])
{
	// Verify command line input 
    if (argc != 5) 
    {
        std::string executable_name = argv[0];

        std::cerr << "Error: Wrong number of input arguments." << std::endl;
        std::cerr << "Usage: " << executable_name << " <N> <L> <mu> <tol>" << std::endl;

        return 1;   
    }

    // Get command line input parameters
    int N = atof(argv[1]);
    int L = atof(argv[2]);
    double mu = atof(argv[3]);
    const double tol = atof(argv[4]);

    // Other parameters
    double h = 2 * L / (N - 1);
    double c = 0.923 + 1.568 * mu;
    double a = 0.2411 + 1.405 * mu;

    // Define dimensions
    // maybe Eigen::Dimension type instead, but that may not be neccessary here
    // if parametres are not to be taken as input arguments anymore, change the vector data type
    Eigen::VectorXd x(N), y(N), z(N); 
    x.setLinSpaced(N, -L, L);
    y.setLinSpaced(N, -L, L);
    z.setLinSpaced(N, -L, L);

    // Compute tensor elements for potential 
    Eigen::Tensor<double, 3> Verfgau(N, N, N);
    double r_ijk, erf_r;
    for(int i = 0;i < N;i++)
    {
    	for(int j = 0;j < N;j++)
    	{
    		for(int k = 0;k < N;k++)
    		{
    			r_ijk = std::sqrt(pow(x(i), 2) + pow(y(j), 2) + pow(z(k), 2));
    			erf_r = std::erf(r_ijk); // look this up

                Verfgau(i, j, k) = erf_r / r_ijk * c * exp(- a * r_ijk);
    		}
    	}
    }

    // Save for testing
    std::string filename = "output/potential_test.bin";
    tensor_to_bin(filename, Verfgau);

    // Perform TT decomposotion
    std::vector<Eigen::Tensor<double, 3>> MPS; // SJEKK DETTE og gi nytt navn for det er ikke matriser
    Eigen::Tensor<double, 3> remaining_tensor = Verfgau; // Start with original tensor
    for(int mode = 0;mode < 2;mode++)
    {
        std::vector<Eigen::Tensor<double, 2>> U;
        Eigen::Tensor<double, 2> V = rank3_tensor_SVD(remaining_tensor, tol, U);

        // Store tensors
        MPS.push_back(U[0]);

        // Reshape V for next iteration
        auto dims = remaining_tensor.dimensions();
        remaining_tensor = V.reshape(Eigen::array<Eigen::Index, 3>{{V.dimension(0), dims[1], dims[2]}});
    }

    // Store last remaining tensor in the MPS
    MPS.push_back(remaining_tensor);

    // Print dimensions of MPS matrices
    for(size_t i = 0;i < MPS.size();i++)
    {
        std::cout << "MPS core " << i << " dimensions: "
                  << MPS[i].dimension(0) << " x " << MPS[i].dimension(1) << std::endl;
    }

	return 0;
}