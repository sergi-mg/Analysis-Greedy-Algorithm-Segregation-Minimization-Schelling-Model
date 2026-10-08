# Analysis-Greedy-Algorithm-Segregation-Minimization-Schelling-Model
This repository contains the an implementation of a greedy algorithm to minimize segregation in the Schelling's model [1,2]. It is based on the content of my Bachelor's degree in Physics thesis, at Universitat de Barcelona.

## Requirements (Python)

The code was developed using Python 3.13.5 and requires the following Python libraries:

- Matplotlib
- Numba
- NumPy
- SciPy

## Repository Structure

This repository contains a single folder with the three scripts developed. 
Additionally, other folders may be generated when running the code, as described below. These folders are included in the `.gitignore` file, so they will only appear locally.

## codes 

This folder contains the different scripts used to simulate the studied system and create the plots for analysing the results. 

### simulation.py

This code contains all the necessary functions to simulate the model and save the final state data, along with the script needed to execute the simulations both in Python and in Fortran (see Schelling_fortran.f90). 


## References
[1] Schelling, T. C. (1971). Dynamic models of segregation. Journal of mathematical sociology, 1(2), 143-186.
[2] Olivé, A. N., Prignano, L., Marinelli, D., & Cozzo, E. (2025). Using Gamified Experiments to Tame Complexity: the case of the Schelling Model of Segregation. arXiv preprint arXiv:2501.08280.
