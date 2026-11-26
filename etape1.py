import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from fonction import *


L = 0.33                 # Longueur vibrante [m]
x_bow = 0.03             # Position de l'archet [m]
n_elem = 330             # Nombre d'elements
T = 50                   # Tension de la corde [N]
rho = 3.1e-3             # Masse volumique [kg/m]
S = 1                    # Surface


nodes, connect = mesh_1d(L, n_elem)

n_nodes = len(nodes)
h = L / n_elem


# Asssemblage des matrices de raideur et de masse
K=assemblage_K(nodes,connect,T)
M=assemblage_M(nodes,connect,rho,S,h)

    
# Elimination des degres de liberte bloques aux extremites
K_red = tronque(K)
M_red = tronque(M)



# Détermination de la solution statique de la corde sous l'action de la force F
F = np.zeros(n_nodes - 2)
F[int(x_bow/h)] = 1   # Force du pincé aplliqué au noeud le plus proche de la position de l'archet

solu_u = np.pad(np.linalg.solve(tronque(assemblage_K(nodes, connect, T)),F),(1, 1),mode='constant',constant_values=0)

# Tracé de la corde à l'instant initial sous la force F
plt.plot(nodes, solu_u, "o-")
plt.xlabel("Position x (m)")
plt.ylabel("Déplacement u selon y (m)")
plt.title("Solution statique de la corde à l'instant initial")
plt.grid()
plt.show()



# Ici on récupère les fréquences propres wn(vals propres) et les modes phi_n (vecteurs propres)
val_p2 ,vect_p =linalg.eigh(tronque(K),tronque(M)) 

print("La taille de vect_p/phi est:", np.shape(vect_p),"(n_nodes=)",n_nodes)




# Tracé des premiers modes propres
for i in range(3):# a modifier
    plt.plot(nodes,ajout0(vect_p[:,i]),label=f"mode propre: {i}")
    plt.xlabel("Position x (m)")
    plt.ylabel("Déplacement u")
    plt.grid()
    plt.legend()
plt.title("Modes propres de vibration de la corde")
plt.show()

