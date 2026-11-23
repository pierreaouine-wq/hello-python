import numpy as np


def mesh_1d(L,n_elem):
    """
    Parameters
    L : longueur
    n_elem : nombre d'éléments
    ------
    Returns
    nodes : liste des positions des noeuds
    connect : liste des segments de la corde

    """
    nodes = np.linspace(0.0, L, n_elem + 1)   # coordonnées des noeuds
    n_nodes = len(nodes)
    connect = np.column_stack((np.arange(0,n_nodes-1),np.arange(1,n_nodes)))
    return nodes,connect



def string_elem_stiffness(h:float, T):
    """
    h : longueur
    T : tension
    Return
    Matrice de raideur élémentaire
    """
    return T/h*np.array([[1,-1],[-1,1]])



def string_element_mass(rho,S,h):
    """
    rho : masse voluumique
    h:longueur
    S : surface
    Return
    Matrice de masse élémentaire
    """
    return rho*S*h/6*np.array([[2,1],[1,2]])



def assemblage_K(nodes,connect,T):
    """
    connect : liste des segments de la corde
    n_nodes : nombre de noeuds
    T : tension
    Assemblage de la matrice globale de raideur K
    """
    ndof_per_node = 1  
    n_nodes = len(nodes)                  # 1 ddl (u) par noeud pour la corde
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



def assemblage_M(nodes,connect,rho,S,h):
    """
    connect : liste des segments de la corde
    n_nodes : nombre de noeuds
    rho : masse volumique
    S : surface fil
    h : longueur fil elementaire

    Return 
    Assemblage de la matrice globale de masse M
    """
    ndof_per_node = 1  
    n_nodes = len(nodes)                   # 1 ddl (u) par noeud pour la corde
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



def tronque(A):
    """
    Retourne M tronquée des premieres et dernieres lignes/colonnes
    """
    return A[1:-1,1:-1]



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