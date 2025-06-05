# Estimating hydrogen ground state energy with the Alternating linear scheme (ALS)

## Prerequisites
This project requires ttml and scikit tt, which can be installed as described on their GitHub sites. It also requires Matplotlib, NumPy and Scipy. 

## Running project
- `run_sweep.py`: Run an experiment with set bond dimensions (D) and grid resolutions (N). Construct Hamiltonian MPO and estimate ground state using FDM, ALS and the power method. 
- `utils.py`: Contains functions for MPO construction and other functions used in early version of the project. 
- `test.py`: Tests performed for some of the early code developed for this project, by solving for an electron in the harmonic oscillator potential. 

## Plotting
- `results.py`: Plots final results used in BSc.
- `test_potential.py`: Plots for testing the TT-cross approximated potential.
