import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import scipy.integrate
import scipy.linalg as linalg
from scipy.integrate import solve_ivp


def mesh_1d(L,n_elem):
    """
    Parameters
    L : longueur
    n_elem : nombre d'éléments
    ------
    Returns
    nodes : liste d'indice des noeuds
    connect : liste des segments de la corde

    """
    nodes = np.linspace(0.0, L, n_elem + 1)   # coordonnées des noeuds
    n_nodes = len(nodes)
    connect = np.column_stack((np.arange(0,n_nodes-1),np.arange(1,n_nodes)))
    return nodes,connect

L = 0.33        # longueur vibrante [m]
x_bow=0.03      #
n_elem = 330   # nombre d'éléments
T=50
rho=3.1*10**(-3)
S=1
h=L/n_elem
nodes, connect = mesh_1d(L,n_elem)
n_nodes=len(nodes)



def string_elem_stiffness(h:float, T):
    """
    h : longueur
    T : tension
    """
    return T/h*np.array([[1,-1],[-1,1]])



def string_element_mass(rho,S,h):
    """
    rho : masse voluumique
    h:longueur
    S : surface
    """
    return rho*S*h/6*np.array([[2,1],[1,2]])



def assemblage_K(n_nodes,connect,T):
    """
    connect : liste des segments de la corde
    n_nodes : nombre de noeuds
    T : tension
    """
    ndof_per_node = 1                    # 1 ddl (u) par noeud pour la corde
    n_dof = n_nodes * ndof_per_node
    K = np.zeros((n_dof, n_dof))
    for elem_nodes in connect:           # elem_nodes = [i, i+1]
        h = nodes[elem_nodes[1]] - nodes[elem_nodes[0]]
        Ke = string_elem_stiffness(h, T)
        # numéros globaux des ddl de l'élément
        dofs = [i * ndof_per_node + k for i in elem_nodes for k in range(ndof_per_node)]
        for a in range(len(dofs)):
            for b in range(len(dofs)):
                K[dofs[a], dofs[b]] += Ke[a, b]
    return K



def assemblage_M(n_nodes,connect,rho,S,h):
    """
    connect : liste des segments de la corde
    n_nodes : nombre de noeuds
    rho : masse volumique
    S : surface fil
    h : longueur fil elementaire
    """
    ndof_per_node = 1                    # 1 ddl (u) par noeud pour la corde
    n_dof = n_nodes * ndof_per_node
    M = np.zeros((n_dof, n_dof))
    for elem_nodes in connect:           # elem_nodes = [i, i+1]
        h = nodes[elem_nodes[1]] - nodes[elem_nodes[0]]
        Ke = string_element_mass(rho,S,h)
        # numéros globaux des ddl de l'élément
        dofs = [i * ndof_per_node + k for i in elem_nodes for k in range(ndof_per_node)]
        for a in range(len(dofs)):
            for b in range(len(dofs)):
                M[dofs[a], dofs[b]] += Ke[a, b]
    return M

#plt.spy(assemblage_K(n_elem+1,mesh_1d(L,n_elem)[1],T))
#plt.show()


def tronque(A):
    """
    Retourne M tronquée des premieres et dernieres lignes/colonnes
    """
    return A[1:-1,1:-1]

F = np.zeros(n_nodes - 2)
F[int(x_bow/h)] = 1


solu_u = np.pad(np.linalg.solve(
        tronque(assemblage_K(n_elem + 1, connect, T)),F),(1, 1),mode='constant',constant_values=0)

#plt.plot(nodes, solu_u, marker='o')



K=assemblage_K(n_nodes,connect,T)
M=assemblage_M(n_nodes,connect,rho,S,h)
M_inv=np.linalg.inv(M)

# Ici on récupère les fréquences propres wn(vals propres) et les modes phi_n (vecteurs propres)
val_p2 ,vect_p =linalg.eigh(tronque(K),tronque(M)) 

print("La taille de vect_p/phi:", np.shape(vect_p),"(n_nodes=)",n_nodes)
#print(val_p2**(1/2))



def ajout0(V:list):
    """
    Retourne le vecteur V avec 2 dim en plus, un 0 avant et à la fin
    """
    return np.pad(V,(1, 1),mode='constant',constant_values=0)


def ajout0matrice(M):
    """
    Retourne la matrice M avec 2 lignes et 2 colonnes en plus, des 0 sur les bords
    """
    return np.pad(M,(1, 1),mode='constant',constant_values=0)


#for i in range(5):# a modifier
    plt.plot(nodes,ajout0(vect_p[:,i]),label=f"vp={i}")
    plt.xlabel("Position x (m)")
    plt.ylabel("Déplacement u")
    plt.grid()
    plt.legend()
#plt.show()


#T_var=np.arange(10,100,5)
#for i in range(len(T_var)):
    #K=assemblage_K(n_nodes,connect,T_var[i])
    #val_p2 ,vect_p =linalg.eigh(tronque(K),tronque(M))
    #plt.plot(T,val_p2)
    #plt.xlabel("T")
    #plt.ylabel("f")
    #plt.grid()
    #plt.legend()
   
#plt.show()





fs=44100
t0 = 0
t1 = 4
t = np.linspace(t0, t1, fs)

u0 = solu_u
v0 = n_nodes * [0]

# Élimination des degrés de liberté bloqués aux extrémités
K_red = tronque(K)
M_red = tronque(M)

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

omega=np.diag(val_p2[:n_modes]**0.5)
omega2=np.diag(val_p2[:n_modes]) # matrice diag des carrés des fréquences propres au carré
phi=vect_p[:, :n_modes] # matrice rectangle dont les colonnes sont les modes propres (vecteurs propres)
# On ne prend que les n_modes premiers modes

q0= np.transpose(phi)@M_red@u0[1:-1]
p0=np.zeros(np.shape(q0))
Z0=np.concatenate([q0,p0]) 



def rhs_modal(t,Y,xi=0.001):
    """
    Fonction qui retourne le vecteur dérivé de Y=[q,p] pour l'intégration temporelle
    """
    n = len(Y)//2
    q = Y[:n]
    p = Y[n:]
    return np.concatenate([p,-2*xi*omega@p -omega2@q])


sol = solve_ivp(rhs_modal, (t0, t1), Z0, method="RK45",t_eval=t, rtol=1e-3, atol=1e-9)

sol_q= sol.y[:n_modes,:] # On prend que les q(t) et pas les p(t)
sol_Umoda_red=phi@sol_q

sol_Umoda=np.vstack([
    np.zeros((1, len(sol.t))),
    sol_Umoda_red,
    np.zeros((1, len(sol.t)))
])




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

