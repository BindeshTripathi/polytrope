# polytrope

The file named "mathematica_notebook_calc.nb" is the Mathematica notebook where analytical manipulations of lengthy terms are carried out.

The script entitled "polytrope_run10.py" numerically evaluates, using the spectral solver Dedalus (https://dedalus-project.org/) and MPI-parallelization, eigenmodes for a wide range of $(k_x, k_y)$ for the linear waves in a polytropic atmosphere. An inhomogeneous magnetic field, directed orthogonal to a constant vertical gravity, varies as $\mathbf{B_0}(z) \sim z^{(n+1)/2} \hat{\mathbf{e}}_{x}$ ; gas pressure and magnetic pressure both vary as $\sim z^{n+1}$, and density varies as $\sim z^n$ where n is the polytropic index.

