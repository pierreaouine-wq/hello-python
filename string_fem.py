import numpy as np


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
n_elem = 11     # nombre d'éléments
nodes, mesh = mesh_1d(L,n_elem)
print(nodes)
print(mesh)
