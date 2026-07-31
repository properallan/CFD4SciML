#===============================================================
# Some libraries
#===============================================================
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# To make your graphics nicer
plt.rc('font', family='serif', size=16)
plt.rc('lines', linewidth=1.5)
plt.rc('legend', fontsize=12)

#===============================================================
# Define parameters
#===============================================================
Nx = 101                          
xmax = 2.                         
xmin = -2.                        
Lx = xmax-xmin                    
dx = Lx/(Nx-1)                    
x = np.linspace(xmin, xmax, Nx)     
dt = 0.06                         # unica coisa que precisa mudar
t_end = 5.                        
Nt = int(t_end/dt)                
a = 0.8                           
CFL = a*dt/dx                     
U = np.zeros((Nt+1, Nx))           
U[0,:] = np.exp(-0.5*(x/0.4)**2)  
Uex = U[0,:]                      

#===============================================================
# Create folder for images based on problem state
#===============================================================
folder_name = f"LFF_advection_Nx{Nx}_CFL{CFL:.2f}_T{t_end}"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

#===============================================================
# Setup the figure ONCE outside the loop
#===============================================================
fig, ax = plt.subplots(figsize=(5.5, 4))
image_files = []

#===============================================================
# Solve equation using the Lax-Friedrichs scheme
#===============================================================
for n in range(0, Nt):
    # Laço para os nós internos (1 até Nx-2)
    for i in range(1, Nx-1):
        U[n+1, i] = 0.5 * (U[n, i+1] + U[n, i-1]) - 0.5 * CFL * (U[n, i+1] - U[n, i-1])
    
    # Condições de contorno periódicas nas bordas
    # O nó da esquerda (0) "enxerga" o penúltimo nó da direita (Nx-2)
    U[n+1, 0] = 0.5 * (U[n, 1] + U[n, Nx-2]) - 0.5 * CFL * (U[n, 1] - U[n, Nx-2])
    
    # O nó da extremidade direita (Nx-1) é uma cópia exata do nó da esquerda (0)
    U[n+1, Nx-1] = U[n+1, 0]
      
    # Compute exact solution
    d = a * (n+1) * dt
    Uex = np.exp(-0.5 * (np.mod(x-d+xmax, 4) - xmax)**2 / 0.4**2)
  
    #===============================================================
    # Plot and save solution
    #===============================================================
    ax.clear() # Clear the axis instead of the whole figure
    
    ax.plot(x, U[n+1, :], label=f'Lax-Friedrichs scheme (CFL={CFL:.2f})')
    ax.scatter(x, Uex, marker='o', facecolors='white', color='k', label='Exact solution')
    
    # Set fixed limits to prevent axis from moving
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0, 1.4])
    
    ax.legend()
    ax.set_title(f't={round(dt*(n+1), 3)}', fontsize=16)
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('u', fontsize=18)
    
    # Apply tight layout to the figure before saving
    fig.tight_layout()
    
    # Save each frame WITHOUT bbox_inches='tight' to keep pixel size constant
    filename = f"{folder_name}/frame_{n+1:04d}.png"
    fig.savefig(filename, dpi=100)
    image_files.append(filename)

# Close the figure only after all frames are generated
plt.close(fig)

#===============================================================
# Create GIF from saved images
#===============================================================
print("Creating GIF...")
images = [Image.open(filename) for filename in image_files]

# Save as GIF
gif_path = f"{folder_name}/simulation.gif"
images[0].save(
    gif_path,
    save_all=True,
    append_images=images[1:],
    duration=100,  
    loop=0  
)
print(f"GIF saved as: {gif_path}")