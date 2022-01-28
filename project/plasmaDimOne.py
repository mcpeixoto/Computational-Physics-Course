
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt
import time
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from banded import banded

########## DEFINING PARAMETERS ##########

side = 100          # Box side - width = height, its a box
n_particles = 100000 # Number of particles
grid_cells  = 1000   # Number of cells in the grid (to be used in the Poisson solver)
n0 = 1              # average electron density TODO
dx = dy = side/grid_cells   # Distance between grid points, its a box so dx=dy


##### Beam inicialization #####

positions = np.random.rand( n_particles) * side   # Shape de 2, Nparticulas para x e y

# Adicionar-lhes perturbação com um sin + espessura
positions += 0.5*np.sin(2*np.pi*positions[0]/side) + np.random.rand(n_particles)*5

## Definir as velocidades de cada particula
velocities = np.ones(n_particles)

## Velocidade dos beams em x
top_beam_velocity = 2
low_beam_velocity = -2

# Configurar as velocidades em x dos beams
velocities[:int(n_particles/2)] = top_beam_velocity
velocities[int(n_particles/2):] = low_beam_velocity

# No que toca as velocidades em y, vamos dar-lhes uma
# pequena peturbação com um sin
velocities += 0.1*np.sin(2*np.pi*positions[0]/side) + 0.1*np.random.rand(n_particles)*5



# Defining potential matrix system
# for being used by banded
e = np.ones(grid_cells, dtype=np.float32)
potential_vals  = np.vstack((e,-2*e,e))
potential_vals[1][0] = 1
potential_vals[0][0] = 0
potential_vals[1][-1] = 1
potential_vals[2][-1] = 0
potential_vals  /= dx**2



# Defining the matrix for the 1st derivative
# computation of the eletric field
e = np.ones(grid_cells)
diags = np.array([-1,1])
vals  = np.vstack((-e,e))
first_derivative = sp.spdiags(vals, diags, grid_cells, grid_cells)
first_derivative = sp.lil_matrix(first_derivative)
first_derivative[0,-1] = -1
first_derivative[-1,0] = 1
first_derivative /= (2*dx)
first_derivative = sp.csr_matrix(first_derivative)


def calcular_aceleração(positions):
    # ### O 1º passo será calcular a densidade, para dps calcular o potencial, para calcular o campo eletrico, para calcular a força

    # Make bins and count
    density, _ = np.histogram(positions, bins=grid_cells, range=(0,side))
    density = density.astype(float)

    # De modo a obter a densidade temos de 
    # dividir o numero de particulas em cada bin pelo
    # tamanho do bin



    # Normalize
    density *= n0 * side / n_particles / dx


    # Solve Poisson's Equation: laplacian(phi) = n-n0
    phi_grid = banded(potential_vals, density-n0, 1, 1)


    # Apply Derivative to get the Electric field
    E_grid = - first_derivative @ phi_grid

    
    # Interpolate grid value onto particle locations
    xp = np.linspace(0, side, num=grid_cells)
    E = np.interp(positions, xp, E_grid)


    a = - E

    return a



########## SIMULATION START ##########

acc = calcular_aceleração(positions)


fig = plt.figure(figsize=(10,8), dpi=80)
dt = 1
t = 0


velocities -= acc * dt

# Simulation Main Loop
for i in range(100):
    if i % 2 == 0:
        
        plt.cla()
        plt.title("Speed / Position")
        plt.scatter(positions[:int(n_particles/2)], velocities[:int(n_particles/2)], s=.4,color='blue', alpha=0.5)
        plt.scatter(positions[int(n_particles/2):], velocities[int(n_particles/2):], s=.4,color='red', alpha=0.5)
        plt.xlabel("X Position")
        plt.ylabel("X Speed")
        plt.pause(0.01)


    # update accelerations
    acc = calcular_aceleração(positions)

    # (1/2) kick
    velocities += acc * dt
    positions += velocities * dt


    positions = np.mod(positions, side)


    # update time
    t += dt
    





