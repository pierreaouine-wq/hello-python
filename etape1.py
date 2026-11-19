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

K=assemblage_K(n_nodes,connect,T)
M=assemblage_M(n_nodes,connect,rho,S,h)

# Elimination des degres de liberte bloques aux extremites
K_red = tronque(K)
M_red = tronque(M)

solu_u = np.pad(np.linalg.solve(
        tronque(assemblage_K(n_elem + 1, connect, T)),F),(1, 1),mode='constant',constant_values=0)

plt.plot(nodes, solu_u, marker='o')

# Ici on récupère les fréquences propres wn(vals propres) et les modes phi_n (vecteurs propres)
val_p2 ,vect_p =linalg.eigh(tronque(K),tronque(M)) 

print("La taille de vect_p/phi:", np.shape(vect_p),"(n_nodes=)",n_nodes)
#print(val_p2**(1/2))

for i in range(5):# a modifier
    plt.plot(nodes,ajout0(vect_p[:,i]),label=f"vp={i}")
    plt.xlabel("Position x (m)")
    plt.ylabel("Déplacement u")
    plt.grid()
    plt.legend()
plt.show()


T_var=np.arange(10,100,5)
for i in range(len(T_var)):
    K=assemblage_K(n_nodes,connect,T_var[i])
    val_p2 ,vect_p =linalg.eigh(tronque(K),tronque(M))
    plt.plot(T,val_p2)
    plt.xlabel("T")
    plt.ylabel("f")
    plt.grid()
    plt.legend()
   
plt.show()