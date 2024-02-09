"""
Dedalus script for numerically evaluating the eigenfunctions and eigenvalues for the waves
in a 3D magnetized polytropic atmosphere.
To run using 4 processes, for instance, the following is useful:
    $ mpiexec -n 4 python3 polytrope_run7.py
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

#For efficient parallel computation, the total number of processors to employ may be taken as (2*nx+1)*(2*ny+1).

Lx                   = 10.0*np.pi
Ly                   = Lx
Lz                   = 60.0*np.pi #Make Lz much larger than Lx to obtain well-resolved eigenmodes for the kx/ky values.
kspacing             = 0.1 #kspacing is the spacing between consecutive kx or ky values. For a reference value, it may be considered ~2*np.pi/Lx or some factor of this.
nx                   = 20 #no. of kx to scan in: -nx*kspacing, ..., -2*kspacing, -1*kspacing, 0*kspacing, 1*kspacing, 2*kspacing, ..., nx*kspacing.
ny                   = 20 #no. of ky to scan in: -ny*kspacing, ..., -2*kspacing, -1*kspacing, 0*kspacing, 1*kspacing, 2*kspacing, ..., ny*kspacing.
nz                   = 256  #512 #96 #1024  #256 #96   
n                    = 2.50 #Polytropic index
gamma                = 1.66666666667 #Adiabatic index
epsilon              = (10.0)**(-1) #Epsilon here is 1/MA in this code. This is chosen so that hydro limit can be recovered perturbatively.
run_number           = 'polytrope_run10'  #run1 #used to store files for separate runs in separate directory
month_of_run         = 'nov2020'
script_version_number= run_number 
submit_version_number= run_number 
modifiedleft         = False #Do you want to solve for modified left eigenvectors? Yes=True, No=False.
left                 = False #This one finds the left eigenvectors


######################################
#The options below are for scipy sparse eigenvalue solver only.
guess                = -0.18*1.0j #guess value for the eigenmode if using sparse solver. If using dense solver, this guess is not used at all.
which                = 'LI' #for scipy sparse solver only. largest imaginar y= 'LI'
######################################

#Now, choose which kx and ky should be solved in parallel.
kx_ky_global = np.zeros(((2*nx+1)*(2*ny+1), 2), dtype=np.float64)
counting = 0
for jj in range(-ny, ny+1):
    for ii in range(-nx, nx+1):
        kx_ky_global[counting] = np.array([ii, jj]) * kspacing
        counting += 1


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
problem_vars = ['ux', 'uy', 'uz', 'uz_z']


problem = de.EVP(domain, variables = problem_vars, eigenvalue = 'Omegasqrd')
problem.meta[:]['z']['dirichlet'] = True

problem.parameters['Lz']          = Lz
problem.parameters['kx']          = 0.2000
problem.parameters['ky']          = 0.0
problem.parameters['gamma']       = gamma
problem.parameters['epsilonsqrd'] = epsilon**2
problem.parameters['n']           = n

#________________________________________________________________________________________________________________________________________________________________



problem.substitutions['dx(A)'] = "1.0j*kx*A"
problem.substitutions['dy(A)'] = "1.0j*ky*A"
problem.substitutions['fac']   = "(1.0 + gamma*(epsilonsqrd)/2)"
problem.substitutions['chi']     = "dx(ux)+dy(uy)+uz_z"


#These equations are copied from p. 30 of the notes, written with hand; find the document in the overleaf "Notes" folder.
problem.add_equation("-Omegasqrd*gamma*fac*ux + 1.0j*kx*(-1)*(n+1)*fac*uz - 1.0j*kx*gamma*z*chi   = 0")
problem.add_equation("epsilonsqrd*kx*ky*z*gamma*ux + (-Omegasqrd*gamma*fac + epsilonsqrd*(kx**2)*gamma*z)*uy + (-1.0j*ky)*(n+1)*fac*uz - 1.0j*ky*gamma*z*(1+epsilonsqrd)*chi   = 0")
problem.add_equation("1.0j*kx*(fac/gamma+epsilonsqrd)*ux + 1.0j*kx*epsilonsqrd*z/(n+1)*dz(ux) + 1.0j*ky*fac/gamma*uy + ((-Omegasqrd*fac+epsilonsqrd*z*(kx**2))/(n+1))*uz - (1+epsilonsqrd)*chi - (1+epsilonsqrd)*z/(n+1)*dz(chi) = 0")
problem.add_equation("uz_z - dz(uz)  = 0")


problem.add_bc("left((z**(n+1)) * chi) = 0")
problem.add_bc("right(uz)  = 0")

print("done with BCs")

# Building a solver
solver = problem.build_solver()


# Create function to compute max growth rate for given (kx, ky)
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
 
    if int(kx_ky[1]/kspacing)<0:
        if int(kx_ky[0]/kspacing)<0:
            hf = h5py.File('%s/eigmodes_kxindex_minus%i_kyindex_minus%i.h5' %(path_2, np.abs(int(kx_ky[0]/kspacing)),  np.abs(int(kx_ky[1]/kspacing))), 'w') 
        else:
            hf = h5py.File('%s/eigmodes_kxindex_plus%i_kyindex_minus%i.h5' %(path_2, np.abs(int(kx_ky[0]/kspacing)),  np.abs(int(kx_ky[1]/kspacing))), 'w')  
    else:
        if int(kx_ky[0]/kspacing)<0:
            hf = h5py.File('%s/eigmodes_kxindex_minus%i_kyindex_plus%i.h5' %(path_2, np.abs(int(kx_ky[0]/kspacing)),  np.abs(int(kx_ky[1]/kspacing))), 'w') 
        else:
            hf = h5py.File('%s/eigmodes_kxindex_plus%i_kyindex_plus%i.h5' %(path_2, np.abs(int(kx_ky[0]/kspacing)),  np.abs(int(kx_ky[1]/kspacing))), 'w') 
    
    
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

