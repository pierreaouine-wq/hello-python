import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.linalg as linalg
from fonction import *
import scipy.io.wavfile


####################Données nécessaires a l'étape 2

L = 0.33                 # Longueur vibrante [m]
x_bow = 0.03             # Position de l'archet [m]
n_elem = 330             # Nombre d'elements
T = 50                   # Tension de la corde [N]
rho = 3.1e-3             # Masse volumique [kg/m]
S = 1                    # Surface
nodes, connect = mesh_1d(L, n_elem)
n_nodes = len(nodes)
h = L / n_elem

F = np.zeros(n_nodes - 2)
F[int(x_bow/h)] = 1

K=assemblage_K(nodes,connect,T)
M=assemblage_M(nodes,connect,rho,S,h)



# Elimination des degres de liberte bloques aux extremites
K_red = tronque(K)
M_red = tronque(M)

solu_u = np.pad(np.linalg.solve(tronque(assemblage_K(nodes, connect, T)),F),(1, 1),mode='constant',constant_values=0)

val_p2 ,vect_p =linalg.eigh(K_red,M_red) 


##################################etape 2


fs=44100         # Fréquence d'échantillonnage [Hz]
t0 = 0           # Temps initial [s]
t1 = 2           # Temps final [s]
t = np.linspace(t0, t1, int((t1-t0)*fs) + 1)


################### Méthode d'intégration classique

u0 = solu_u
v0 = n_nodes * [0]
Y0 = np.concatenate([u0[1:-1], v0[1:-1]])


"""
def rhs(t, Y):
    n = len(Y) // 2
    y1, y2 = Y[:n], Y[n:]
    return np.concatenate([y2, linalg.solve(M_red, -K_red @ y1)])

sol = solve_ivp(rhs, (t0, t1), Y0, method="RK45",t_eval=t, rtol=1e-3, atol=1e-9)

# Reconstruction des déplacements de tous les nœuds
sol_U = np.vstack([
    np.zeros((1, len(sol.t))),
    sol.y[:n_nodes-2, :],
    np.zeros((1, len(sol.t)))
])

sol_V = sol.y[n_nodes-2:, :]

# Tracé de la corde à plusieurs instants
for k in [0,1,2,3,4,5,6,7,8,9,10]:
    plt.plot(nodes, sol_U[:, k*10], label=f"t = {sol.t[k*10]:.3f} s")


plt.xlabel("Position x (m)")
plt.ylabel("Déplacement u(x,t) (m)")
plt.title("Déformation de la corde à différents instants")
plt.legend()
plt.grid()
plt.show()
"""



##################### AVEC LA BASE MODALE

n_modes = 20 # nombre de modes propres à prendre en compte
# Pourquoi ne pas tous les prendre

omega=np.diag(val_p2[:n_modes]**0.5)
omega2=np.diag(val_p2[:n_modes]) # matrice diag des carrés des fréquences propres au carré
phi=vect_p[:, :n_modes] # matrice rectangle dont les colonnes sont les modes propres (vecteurs propres)
# On ne prend que les n_modes premiers modes

q0= np.transpose(phi)@M_red@u0[1:-1]
p0=np.zeros(np.shape(q0)) #on initialise q point 0
Z0=np.concatenate([q0,p0]) # colonne de conditions initiales : déplacements modaux q0 et vitesses modales p0



def rhs_modal(t,Y,xi=0.001):
    """
    Fonction qui retourne le vecteur dérivé de Y=[q,p] pour l'intégration temporelle
    """
    n = len(Y)//2
    q = Y[:n]
    p = Y[n:]
    return np.concatenate([p,-2*xi*omega@p -omega2@q])



# Résolution numérique des équations différentielles régit par q(t)
# rtol et atol définissent les tolérances d'erreur de l'intégration

sol = solve_ivp(rhs_modal, (t0, t1), Z0, method="RK45",t_eval=t, rtol=1e-3, atol=1e-9)
 
sol_q= sol.y[:n_modes,:]     # Prend que les q(t): deplacements modaux
sol_Umoda_red=phi@sol_q      # Reconstruction des déplacements de la corde à partir des modes propres
# Les déplacements des extrémités ne sont pas inclus

sol_Umoda=np.vstack([
    np.zeros((1, len(sol.t))),
    sol_Umoda_red,
    np.zeros((1, len(sol.t)))
])
# ajout d'une premiere et d'une derniere ligne de 0 representant le fait que le premier et le dernier point sont fixes a tt instant t
# np.vstack() empile verticalement les matrices et vecteurs.



### Tracé de la corde à plusieurs instants

for i in range(11):
    plt.plot(nodes, sol_Umoda[:, i*10], label=f"t = {sol.t[i*10]:.3f} s")

plt.xlabel("Position x (m)")
plt.ylabel("Déplacement u(x,t) (m)")
plt.title("Déformation de la corde à différents instants")
plt.legend()
plt.grid()
plt.show()



### Tracé du déplacement en un noeud en fonction du temps

plt.plot(sol.t,sol_Umoda[int(x_bow/h),:])
plt.xlabel("Temps t (s)")
plt.ylabel("Déplacement u(t) (m)")
plt.title("Déplacement en un noeud en fonction du temps")
plt.grid()
plt.show()



### Tracé de la Fchev en fonction du temps
# Calcul de la force transversale au chevalet.
# Celle-ci est liée à la pente de la corde à son extrémité :
# F_chev = T * du/dx (x=0,t).
# Ici T = 1, donc la force est numériquement égale à la pente.

F_chev=(sol_Umoda[1,:]-sol_Umoda[0,:])/h
plt.plot(sol.t, F_chev)
plt.xlabel("Temps t (s)")
plt.ylabel("Force chevalet (N)")
plt.title("Force chevalet en fonction du temps")
plt.grid()
plt.show()



##### Création du son

signal=(F_chev/np.max(np.abs(F_chev))*0.9*32767).astype(np.int16)

fichier_audio=scipy.io.wavfile.write("pincement.wav", fs, signal)
