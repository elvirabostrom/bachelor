#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "utils.hpp"

// Buliding: g++ -std=c++11 main.cpp src/utils.cpp -I/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 -o program

// Første test for riktig building
int main()
{
	int x = 10;
	int y = 10;
	int z = 10;
	Eigen::Tensor<float, 3> grid(x, y, z);

	// get_potential
	// import it??

	// get_kinetic
	int N = 10
	Eigen::Tensor<double 2> T_x, T_y, T_z;
	get_kinetic_operator(N, T_x);
	get_kinetic_operator(N, T_y);
	get_kinetic_operator(N, T_z);
	// lurt å ha dem hver for seg eller samlet??

	return 0;
}










/* 
notater
OBC eller PBC?
- Kontrollere økt entanglement (bond dim): komprimere ny MPS etter operasjon variational compression(?)
Forventningsverdi
DMRG variasjonell sweep
- Enhetsmatrise
- Gjøres til egenverdiproblem med superblock hamiltonian?
*\


