# Parareal-dynamo
Implementation of kinematic dynamo using parareal and Dedalus

This project uses the open source [Dedalus project](http://dedalus-project.org) spectral solver code.
The source code can be found in the Astrophysical Source Code Library [Dedalus code](http://ascl.net/1603.015).

The code implements the Parareal algorithm to parallelise simulations of the kinematic dynamo in time 
as well as in space. 

## How to use
1. A set of initial conditions can be created by running `python3 generate_initial_conditions.py`. This will create a set random numpy arrays of magnitude 1e-5 at high resolution, and a set of initial conditions of lower resolution generated by fourier resampling.

2. `Galloway_proctor_serial.py` and `Roberts_serial.py` will run serial or spatial parallelised simulations of the Galloway Proctor and Roberts dynamos respectively. These scripts can be run in serial with `python3 script.py ...`, or in parallel by using `mpiexec -n X python3 script.py ...`.

3. Typing any script as `python3 script.py -h` will show the parameters required by the script to run. 

4. `Galloway_proctor_parareal.py` and `Roberts_parareal.py` will run simulations of the dynamos with Parareal algorithm included. These must be run using `mpiexec -n X python3 script.py ...`, with at least 5 processors (for 5 time slices.). 

5. Parallel in time and space simulations can be carried out by using the `-px` option with the parareal scripts, with a number greater than 1. Best performance is found from numbers like 2<sup>n</sup>, for FFTW performance. The script should be executed using (N<sub>space</sub> x N<sub>time</sub>) processors.

6. Simulations should be merged after completion using either `merge_serial.py` or `merge_paraeal.py`.

7. Multiple serial simulations with different dt or different resolution can be used to estimate convergence properties. After completing the simulations in the same folder, use the `Serial_dt_converge.py` or `Serial_space_converge.py` scripts.

8. Scaling results can be estimated using `Parareal_plot_scaling.py`. At least one serial run using the same conditions as the fine solver should be carried out as the base measure of run time. 

9. Growth rate and time series data can be found by using the `Serial_growth_rate.py` script. 

10. Parareal simulations do not stop once converged. Convergence testing and runtime estimates are carried out after simulations are complete. Automatic stopping of simulations is set to be added in a future version. 
