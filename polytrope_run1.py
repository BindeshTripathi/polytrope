"""
Dedalus script for numerically evaluating the eigenfunctions and eigenvalues for the waves
in a 3D magnetized polytropic atmosphere.
To run using 4 processes, for instance, the following is useful:
    $ mpiexec -n 4 python3 polytrope_run1.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as de
from mpi4py import MPI
CW = MPI.COMM_WORLD
import shutil, os
import h5py

import logging
logger = logging.getLogger(__name__)



#==============================================================================================================
#==============================================================================================================
Lx                   = 10.0*np.pi
Ly                   = 10.0*np.pi
Lz                   = 10.0*np.pi
nx                   = 3
ny                   = 3
nz                   = 32  #512 #96 #1024  #256 #96   
n                    = 1.50
gamma                = 1.66666666667
MA                   = (10.0)**(-2)
G                    = (n+1)/gamma * ( 1 + gamma/(2*(MA**2)) )
run_number           = 'polytrope_run1'  #run1 #used to store files for separate runs in separate directory
month_of_run         = 'nov2020'
script_version_number= 'polytrope_run1'
submit_version_number= script_version_number
modifiedleft         = True #Do you want to solve for modified left eigenvectors? Yes=True, No=False.
left                 = False #This one finds the left eigenvectors

######################################
#The options below are for scipy sparse eigenvalue solver only.
guess                = -0.18*1.0j #guess value for the eigenmode if using sparse solver. If using dense solver, this guess is not used at all.
which                = 'LI' #for scipy sparse solver only. largest imaginar y= 'LI'
######################################

#Now, choose which kx and ky should be solved in parallel.
kx_ky_global = np.array([ [2*np.pi/Lx*1, 2*np.pi/Ly*0],
                          [2*np.pi/Lx*2, 2*np.pi/Ly*0],
                          [2*np.pi/Lx*3, 2*np.pi/Ly*0],
                          [2*np.pi/Lx*4, 2*np.pi/Ly*0],
                          [2*np.pi/Lx*5, 2*np.pi/Ly*0],
                          [2*np.pi/Lx*6, 2*np.pi/Ly*0],
                          [2*np.pi/Lx*7, 2*np.pi/Ly*0],
                          [2*np.pi/Lx*8, 2*np.pi/Ly*0],
                          [2*np.pi/Lx*0, 2*np.pi/Ly*1],
                          [2*np.pi/Lx*0, 2*np.pi/Ly*2],
                          [2*np.pi/Lx*0, 2*np.pi/Ly*3],
                          [2*np.pi/Lx*0, 2*np.pi/Ly*4],
                          [2*np.pi/Lx*0, 2*np.pi/Ly*5],
                          [2*np.pi/Lx*0, 2*np.pi/Ly*6],
                          [2*np.pi/Lx*0, 2*np.pi/Ly*7],
                          [2*np.pi/Lx*0, 2*np.pi/Ly*8],
                        ])


#kx_global = np.linspace(1/10*2*np.pi/Lx, (2*5/nx)*2*np.pi/Lx*nx, int(nx)) #It somehow appears that np.linspace(start, stop, no_of_processes) should match with mpiexec -n no_of_processes python3 hydro.py
#kx_global = np.linspace(2*np.pi/Lx, 2*np.pi/Lx*nx, int(nx)) #It somehow appears that np.linspace(start, stop, no_of_processes) should match with mpiexec -n 



#==============================================================================================================
#==============================================================================================================


#maintain your directories here:
parent_dir = "/scratch/07443/bindesh/EVP_runs"
parent_dir2 = "/work2/07443/bindesh/stampede2/phd2020/polytrope"




########################################
##Creating directories for each run  ###
########################################
start_time = time.time()

file_dir_1 = '%s/%s' %(month_of_run, run_number)
path_1 = os.path.join(parent_dir, file_dir_1)
path_2 = '%s/eigenmodes' %path_1   
path_3 = '%s/post_processing' %path_1

if CW.rank == 0:
    access_rights = 0o755

    try:
        os.mkdir(path_1, access_rights)
    except OSError:
        print ("")
    else:
        print ("Successfully created the directory %s" % path_1)
    
    
    
    try:
        os.mkdir(path_2, access_rights)
    except OSError:
        print ("")
    else:
        print ("Successfully created the directory %s" % path_2)
        
    
    try:
        os.mkdir(path_3, access_rights)
    except OSError:
        print ("")
    else:
        print ("Successfully created the directory %s" % path_3)
    

    ###################################################
    ##Copying files from WORK to SCRATCH in each run###
    ###################################################
    files_to_copy = ['%s/%s.py' %(parent_dir2, script_version_number), '%s/submit_%s.cmd' %(parent_dir2, submit_version_number)]
    for f1 in files_to_copy:
        shutil.copy(f1, '%s' %path_1)

        

if CW.rank == 0:
    import json

    person_dict = {        
    "Lx": Lx,
    "Ly": Ly, 
    "Lz": Lz, 
    "nx": nx,
    "ny": ny,
    "nz": nz,
    "run_number": run_number,
    "month_of_run": month_of_run,          
    "script_version_number": script_version_number,
    "submit_version_number": submit_version_number
    }


    filename_param = '%s/eig_2modes_parameters.txt' %path_1
    json_file = open(filename_param, "w")
    json.dump(person_dict, json_file)
    json_file.close()
    
    logger.info('Initial json file created!')
    


# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes

#z_basis = de.Chebyshev('z', nz, interval=(-Lz/2, Lz/2))

z_basis = de.Chebyshev('z', nz, interval = (0, Lz))
domain = de.Domain([z_basis],  grid_dtype=np.complex128, comm=MPI.COMM_SELF)
z = domain.grid(0)


# Problem
#I have used exp(i*omg*t+i*k*x) here in the code
problem_vars = ['u', 'v', 'w', 'wz']


problem = de.EVP(domain, variables = problem_vars, eigenvalue = 'omgsqrd')
problem.meta[:]['z']['dirichlet'] = True

problem.parameters['Lz']          = Lz
problem.parameters['kx']          = 0.2000
problem.parameters['ky']          = 0.0
problem.parameters['G']           = G
problem.parameters['gamma']       = gamma
problem.parameters['MA']          = MA
problem.parameters['n']           = n

#________________________________________________________________________________________________________________________________________________________________



problem.substitutions['dx(A)'] = "1.0j*kx*A"
problem.substitutions['dy(A)'] = "1.0j*ky*A"
problem.substitutions['dtsqrd(A)'] = "-1.0*omgsqrd*A"

problem.substitutions['chi']     = "dx(u)+dy(v)+wz"


problem.add_equation("dtsqrd(u) - z*dx(chi) - G*dx(w) - gamma/(MA**2) * ( dx(wz) + dx(dx(u)) + dy(dy(u)) )          = 0")
problem.add_equation("dtsqrd(v) - z*dy(chi) - G*dy(w)                                                               = 0")
problem.add_equation("dtsqrd(w) - z*dz(chi) - gamma*G*chi + G*( dx(u)+dy(v) ) - 1/MA**2 * ( gamma*G*( wz+dx(u) ) - (gamma**2) *G/2*chi + z*( dy(dy(w)) +dz(wz) + dx(dz(u)) ) )  = 0")
problem.add_equation("wz - dz(w)  = 0")

problem.add_bc("left((z**(n+1)) * chi ) = 0")
problem.add_bc("right(w)  = 0")
#problem.add_bc("left(ux)  = 0")
#problem.add_bc("right(ux) = 0")

print("done with BCs")

# Solver
solver = problem.build_solver()


# Create function to compute max growth rate for given kx
def max_growth_rate(kx_ky):
    logger.info('Computing max growth rate for kx = %f, ky = %f' %(kx_ky[0], kx_ky[1]))
    # Change kx parameter
    problem.namespace['kx'].value = kx_ky[0]
    problem.namespace['ky'].value = kx_ky[1]
    # Solve for eigenvalues with sparse search near zero, rebuilding NCCs
    #solver.solve_sparse(solver.pencils[0], N=3, target=guess, which=which, modifiedleft=modifiedleft, rebuild_coeffs=False)
    #solver.solve_sparse(solver.pencils[0], N=3, target=guess, rebuild_coeffs=True)    
    solver.solve_dense(solver.pencils[0], left=left, modifiedleft=modifiedleft, rebuild_coeffs=True)    
    
    
    # Filter infinite/nan eigenmodes
    finite = np.isfinite(solver.eigenvalues)
    solver.eigenvalues = solver.eigenvalues[finite]
    solver.eigenvectors = solver.eigenvectors[:, finite]
 
    if int(kx_ky[1]/(2*np.pi/Ly))<0:
        hf = h5py.File('%s/eig_2modes_kxmode_%i_kymode_minus%i.h5' %(path_2, np.abs(int(kx_ky[0]/(2*np.pi/Lx))),  np.abs(int(kx_ky[1]/(2*np.pi/Ly)))), 'w')  
    else:
        hf = h5py.File('%s/eig_2modes_kxmode_%i_kymode_plus%i.h5'  %(path_2, np.abs(int(kx_ky[0]/(2*np.pi/Lx))),  np.abs(int(kx_ky[1]/(2*np.pi/Ly)))), 'w')

    #hf = h5py.File('%s/eig_2modes_%i.h5' %(path_2, CW.rank), 'w')
    
    
    g3_kx = hf.create_group('kx')
    g31_kx = g3_kx.create_dataset('kx', data=kx_ky[0])
    
    g3_ky = hf.create_group('ky')
    g31_ky = g3_ky.create_dataset('ky', data=kx_ky[1])
    
    g4 = hf.create_group('full_eigenvectors')
    g41 = g4.create_dataset('full_eigenvectors',data=solver.eigenvectors)
                
    if left == True:
        g4left = hf.create_group('full_lefteigenvectors')
        g41left = g4left.create_dataset('full_lefteigenvectors',data=solver.left_eigenvectors)

    g5 = hf.create_group('full_eigenvalues')
    g51 = g5.create_dataset('full_eigenvalues',data=solver.eigenvalues)
    
    if modifiedleft == True:
        g6 = hf.create_group('full_lefts')
        g61 = g6.create_dataset('full_lefts',data=solver.lefts)

        g7 = hf.create_group('full_eigenvaluesleft')
        g71 = g7.create_dataset('full_eigenvaluesleft',data=solver.eigenvaluesleft)

        g8 = hf.create_group('full_leftsb')
        g81 = g8.create_dataset('full_leftsb',data=solver.leftsb)
    
    hf.close()
    
    logger.info(np.min(solver.eigenvalues.imag))
    logger.info(np.max(solver.eigenvalues.imag)) 
    # Return largest imaginary part
    #return np.max(solver.eigenvalues.imag)
    return np.max(solver.eigenvalues.imag) #because exp(i*omg*t)--> growth rate is abs(i*(i*imaginary_part_of_omg))


# Compute growth rate over local wavenumbers
kx_ky_local = kx_ky_global[CW.rank::CW.size]
t1 = time.time()
growth_local = np.array([max_growth_rate(kx_ky) for kx_ky in kx_ky_local])
t2 = time.time()
logger.info('Elapsed solve time: %f' %(t2-t1))
logger.info('Campaign successful!')

