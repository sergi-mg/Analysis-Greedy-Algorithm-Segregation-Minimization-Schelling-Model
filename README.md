# Analysis-Greedy-Algorithm-Segregation-Minimization-Schelling-Model
This repository contains the an implementation of a greedy algorithm to minimize segregation in the Schelling's model [1,2]. It is based on the content of my Bachelor's degree in Physics thesis, at Universitat de Barcelona. Advisor: Dr. Emanuele Cozzo.

## Requirements (Python)

The code was developed using Python 3.13.5 and requires the following Python libraries:

- matplotlib
- numba
- numpy
- scipy
- random
- subprocess
- os

## Repository Structure

This repository contains a single folder with the three scripts developed. 
Additionally, other folders may be generated when running the code, as described below. These folders are included in the `.gitignore` file, so they will only appear locally.

## codes 

This folder contains the different scripts used to simulate the studied system and create the plots for analysing the results. 

### simulation.py

This code contains all the necessary functions to simulate the model and save the final state data, along with the script needed to execute the simulations both in Python and in Fortran (see Schelling_fortran.f90). 

#### Grredy Algorithm 

Minimization in each step of $G=\alpha\mathcal{S}-(1-\alpha)\mathcal{H}$ where $\mathcal{S}$ is the global segregation, $\mathcal{H}$ is the global happines and $\alpha\in[0,1]$.  

When simulating the greedy algorithm, the code can be excuted to only simulate the greedy algorithm in Python or in both Python and Fortran; inidcations are stated in the script with comments when corresponding. WARNING: The code is not intended to simulate only in Fortran, some modifications in this script and in plots.py functions would be needed.

- **Only Python:** `data` folder is created containing a file for each `(alpha, rho_0, L)` condition. Each file contains the final values of the global segregation $\mathcal{S}$, the global happines $\mathcal{H}$ and the number of movements $\mathcal{T}$. The names of the file are given by `"alpha_"+str(round(alpha,4))+"_rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+"_L_"+str(L)+".dat"`. 

    **WARNING**: When computing the first simulations with $L=20$, the name did not contain the `"_L_"+str(L)` characters. Apart from that, a second file for each `(alpha, rho_0, L)` condition was created with a `"_2"` at the end of the file's name. This is only needed if you aim to reproduce exactly the same simulations needed for excuting the `plots.py` file without modifying it, and indications to do so are stated in the script. This simulations where excuted with the non-improved version of the functions. For new simulations the only modifications needed are the `L` and the `N_sim` values, which have to be modified by hand.

- **Python and Fortran:** `data` folder and file names are created as stated before since the simulation in both programs was intended to check the consistency of the codes. In this project all figures correspond to Python simulated data. Nevertheless, when working with the two programs, files contain 6 columns the first three correspond to ($\mathcal{S}$, $\mathcal{H}$, $\mathcal{T}$) from Python simulations and the other three to the same values from Fortran simulations. In this case two aditional folders will be created:
    * `matrix`: contains the initial condition lattice of the current simulation that will be read by the fortran program.
    * `data_results_fortran`: contains the results obtained in a single Fortran simulation that will be read by the Python code.
All of the files appearing in these folders are temporary for internal data sharing between Python and Fortran codes, the final results are saved in the `data` folder as previously explained.
WARNING: The fortran script `schelling_fortran.f90` must have been previously compiled with the name `schelling.exe` ìn the same folder.

#### Classic Algorithm

This model is only simulated in Python and results for each `(alpha, rho_0, L)` condition are saved in a folder called `data_classic`. Each file contains the final values of the global segregation $\mathcal{S}$, the global happines $\mathcal{H}$ and the number of movements $\mathcal{T}$. The names of the file are given by `"rho_"+str(round(rho_0,4))+"_Nsim_"+str(N_sim)+"_L_"+str(L)+".dat"`. 

### schelling_fortran.f90

This script contains the functions and main program needed to simulate the greedy algorithm a single time given the values of `L`, `alpha`, `tau`, `rho_0` and `seed`. The system is simulated considering a Moore Neighbourhood as its kernel, since is the one implemented in Python. However, it can be changed to another kernel by modifying the `funcio` input of the `kernel_creation` subroutine.

### plots.py

This script contains the code used to generate the different plots that were needed to analyse the obtained results. The plots are saved in `.pdf` format in a folder called `plots`. The script (without modifications) expect the following files regarding the greedy algorithm (for the classic algorithm, all needed files will be generated when considering the greedy algorithm cases).
- **Initial scenario: $L=20$** - Files indicated in the WARNING from the `simulations.py` Only Python description. 
- **L dependence: $L\in{10,20,30,40}$** - Files simulated with the current version of the code, a single file for each $(\alpha, \rho_0, L)$ indicating the $L$ value in its name.

## References
[1] Schelling, T. C. (1971). Dynamic models of segregation. Journal of mathematical sociology, 1(2), 143-186.  
[2] Olivé, A. N., Prignano, L., Marinelli, D., & Cozzo, E. (2025). Using Gamified Experiments to Tame Complexity: the case of the Schelling Model of Segregation. arXiv preprint arXiv:2501.08280.
