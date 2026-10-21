import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg


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
n_elem = 11*5    # nombre d'éléments
T=50
rho=3.1*10**(-3)
S=1
h=L/n_elem
nodes, connect = mesh_1d(L,n_elem)
#print(nodes)
#print(mesh)
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

print(linalg.eigh(tronque(K),tronque(M)))
val_p2 ,vect_p =linalg.eigh(tronque(K),tronque(M))


print(val_p2**(1/2))

def ajout0(V:list):
    """
    Retourne le vecteur V avec 2 dim en plus, un 0 avant et à la fin
    """
    return np.pad(V,(1, 1),mode='constant',constant_values=0)

for i in range(5):# a modifier
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
   
plt.show()