
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt
import time
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

########## DEFINING PARAMETERS ##########

side = 100          # Box side - width = height, its a box
n_particles = 50000 # Number of particles
grid_cells  = 500   # Number of cells in the grid (to be used in the Poisson solver)
n0 = 1              # average electron density TODO
dx = dy = side/grid_cells   # Distance between grid points, its a box so dx=dy


##### Beam inicialization #####

positions = np.random.rand(2, n_particles) * side   # Shape de 2, Nparticulas para x e y

## Posição dos beams em y
top_beam = side*0.6
low_beam = side*0.4

# Configurar as posições em y dos beams
positions[1, :int(n_particles/2)] = top_beam 
positions[1, int(n_particles/2):] = low_beam

# Adicionar-lhes perturbação com um sin + espessura
positions[1] += 0.5*np.sin(2*np.pi*positions[0]/side) + np.random.rand(n_particles)*5

## Definir as velocidades de cada particula
velocities = np.ones((2,n_particles))

## Velocidade dos beams em x
top_beam_velocity = 2
low_beam_velocity = -2

# Configurar as velocidades em x dos beams
velocities[0,:int(n_particles/2)] = top_beam_velocity
velocities[0,int(n_particles/2):] = low_beam_velocity

# No que toca as velocidades em y, vamos dar-lhes uma
# pequena peturbação com um sin
velocities[1] = 0.1*np.sin(2*np.pi*positions[0]/side)




e = np.ones(grid_cells)
diags = np.array([-1,0,1])
vals  = np.vstack((e,-2*e,e))
Lmtx = sp.spdiags(vals, diags, grid_cells, grid_cells)
Lmtx = sp.lil_matrix(Lmtx)
Lmtx[0,grid_cells-1] = 1
Lmtx[grid_cells-1,0] = 1
Lmtx /= dx**2
Lmtx = sp.csr_matrix(Lmtx)


# Construct matrix G to computer Gradient  (1st derivative)
e = np.ones(grid_cells)
diags = np.array([-1,1])
vals  = np.vstack((-e,e))
Gmtx = sp.spdiags(vals, diags, grid_cells, grid_cells)
Gmtx = sp.lil_matrix(Gmtx)
Gmtx[0,grid_cells-1] = -1
Gmtx[grid_cells-1,0] = 1
Gmtx /= (2*dx)
Gmtx = sp.csr_matrix(Gmtx)


def calcular_aceleração(positions):
    # ### O 1º passo será calcular a densidade, para dps calcular o potencial, para calcular o campo eletrico, para calcular a força

    """ Why can't I calculate the density with this?
    # O lado da nossa cell vai ser o tamanho
    # da box a dividir pelo número de células
    cell_side = box_side/grid_cells

    # Initialize grid
    density = np.zeros((int(box_side/cell_side),int(box_side/cell_side)))


    for particle in positions:
        x = particle[0]
        y = particle[1]

        # Calculate the cell idx
        cell_x = np.floor(x/cell_side).astype(int)
        cell_y = np.floor(y/cell_side).astype(int)

        # Add the particle to the grid
        # acording to its weights
        hx = (x - cell_x*cell_side)/cell_side
        hy = (y - cell_y*cell_side)/cell_side

        density[cell_x,cell_y]     += hx*hy             #(1-hx)*(1-hy)
        density[cell_x+1,cell_y]   += (1-hx)*hy
        density[cell_x,cell_y+1]   += hx*(1-hy)         #(1-hx)*hy
        density[cell_x+1,cell_y+1] += (1-hx)*(1-hy)
    """

    x = positions[0]
    y = positions[1]

    # Make bins and count
    hist_x, _ = np.histogram(x, bins=grid_cells, range=(0,side))
    hist_y, _ = np.histogram(y, bins=grid_cells, range=(0,side))

    # De modo a obter a densidade temos de 
    # dividir o numero de particulas em cada bin pelo
    # tamanho do bin
    density = np.array([hist_x, hist_y], dtype=float)


    # Normalize
    density[0] *= n0 * side / n_particles / dx
    density[1] *= n0 * side / n_particles / dy

    # Solve Poisson's Equation: laplacian(phi) = n-n0
    phi_grid_x = spsolve(Lmtx, density[0]-n0, permc_spec="MMD_AT_PLUS_A")
    phi_grid_y = spsolve(Lmtx, density[1]-n0, permc_spec="MMD_AT_PLUS_A")


    # Apply Derivative to get the Electric field
    E_grid_x = - Gmtx @ phi_grid_x
    E_grid_y = - Gmtx @ phi_grid_y

    
    # Interpolate grid value onto particle locations
    xp = np.linspace(0, side, num=grid_cells)
    E_x = np.interp(positions[0], xp, E_grid_x)
    E_y = np.interp(positions[1], xp, E_grid_y)
    # Ex 50k
    # edges_x.shape (501,)

    a = - np.array([E_x, E_y], dtype=float)

    return a



########## SIMULATION START ##########

acc = calcular_aceleração(positions)


fig = plt.figure(figsize=(15,10), dpi=80)
dt = 1
t = 0

# Simulation Main Loop
for i in range(50):
    if i % 2 == 0:
        plt.cla()
        fig.suptitle('t = %.2f' % t)

        plt.subplot(2, 2, 1)
        plt.cla()
        plt.title("Positions")
        plt.scatter(positions[0,:int(n_particles/2)], positions[1,:int(n_particles/2)], s=.4,color='blue', alpha=0.5)
        plt.scatter(positions[0,int(n_particles/2):], positions[1,int(n_particles/2):], s=.4,color='red',  alpha=0.5)

        plt.subplot(2, 2, 3)
        plt.cla()
        plt.title("Speed / Position")
        plt.scatter(positions[0, :int(n_particles/2)],velocities[0 ,:int(n_particles/2)], s=.4,color='blue', alpha=0.5)
        plt.scatter(positions[0,int(n_particles/2):], velocities[0,int(n_particles/2):], s=.4,color='red', alpha=0.5)
        plt.xlabel("X Position")
        plt.ylabel("X Speed")

        plt.subplot(2, 2, 4)
        plt.cla()
        plt.title("Speed / Position")
        plt.scatter(positions[1, :int(n_particles/2)],velocities[1 ,:int(n_particles/2)], s=.4,color='blue', alpha=0.5)
        plt.scatter(positions[1,int(n_particles/2):], velocities[1,int(n_particles/2):], s=.4,color='red', alpha=0.5)
        plt.xlabel("Y Position")
        plt.ylabel("Y Speed")
        
        plt.pause(0.01)
        #time.sleep(0.2)


    # (1/2) kick
    velocities += acc * dt

    # velocities 2, 50k
    # acc = velocity
    
    # drift (and apply periodic boundary conditions)
    positions += velocities * dt
    positions[0] = np.mod(positions[0], side)
    positions[1] = np.mod(positions[1], side)

    # update accelerations
    acc = calcular_aceleração(positions)
    
    # (1/2) kick
    velocities += acc * dt/2.0
    
    # update time
    t += dt
    





