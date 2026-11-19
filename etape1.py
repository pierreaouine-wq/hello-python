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