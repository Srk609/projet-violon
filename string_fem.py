import numpy as np
import matplotlib.pyplot as plt   

def mesh_1d(L, n_elem):
    """
    Entrées :
    - L : longueur totale [m] (float)
    - n_elem : nombre d'éléments (int)

    Retour :
    - nodes : coordonnées des nœuds (tableau 1D)
    - connect : table de connectivité (tableau de taille (n_elem, 2))
    """
    nodes = np.linspace(0.0, L, n_elem + 1) #crée un tableau qui va de 0 à la longueur L avec n_elements+1 
    n_nodes=len(nodes)
    connect = np.column_stack((np.arange(n_elem), np.arange(1, n_elem + 1)))
    return nodes,connect


def string_elem_stiffness(h, T):
    return (T / h) * np.array([
        [ 1.0, -1.0],
        [-1.0,  1.0]
    ])
def string_elem_mass(h, rho, S):
    return rho * S * h / 6 * np.array([[2.0, 1.0],
                                       [1.0, 2.0]])


def string_elem_mass_lumped(h, rho, S):
    return rho * S * h / 2 * np.array([[1.0, 0.0],
                                       [0.0, 1.0]])


def string_elem_mass(h, rho, S):
    """Masse cohérente : Me = rho*S*h/6 * [[2, 1], [1, 2]]"""
    return rho * S * h / 6 * np.array([[2.0, 1.0],
                                       [1.0, 2.0]])


def string_elem_mass_lumped(h, rho, S):
    """Masse concentrée : Me = rho*S*h/2 * [[1, 0], [0, 1]]"""
    return rho * S * h / 2 * np.eye(2)

def assemble(nodes, connect, matrice_elementaire, ndof_per_node=1, **params):
    """
    Assemble une matrice globale à partir d'une fonction élémentaire.
 
    matrice_elementaire : fonction (h, **params) -> matrice élémentaire
    params              : paramètres passés à la fonction (T=..., ou rho=..., S=...)
    """
    n_nodes = len(nodes)
    n_dof = n_nodes * ndof_per_node
    A = np.zeros((n_dof, n_dof))
    for elem_nodes in connect:
        h = nodes[elem_nodes[1]] - nodes[elem_nodes[0]]
        Ae = matrice_elementaire(h, **params)
        dofs = [i * ndof_per_node + k
                for i in elem_nodes for k in range(ndof_per_node)]
        for a in range(len(dofs)):
            for b in range(len(dofs)):
                A[dofs[a], dofs[b]] += Ae[a, b]
    return A
 
def free_dofs(n_dof, fixed):
    """Liste des ddl libres : tous les ddl sauf ceux bloqués."""
    return np.setdiff1d(np.arange(n_dof), fixed)


def reduire_matrice(A, free):
    """Matrice réduite : lignes libres, puis colonnes libres."""
    return A[free][:, free]
 
 
def etendre_vecteur(U_free, free, n_dof):
    """Reconstruit le vecteur complet (zéros aux ddl bloqués)."""
    U = np.zeros(n_dof)
    U[free] = U_free
    return U

def force_ponctuelle(nodes, x_load, F0, ndof_per_node=1):
    """
    Force ponctuelle F0 en x_load, répartie sur les deux noeuds de l'élément
    qui le contient : F_i = F0 * N_i(x_load).
    """
    F = np.zeros(len(nodes) * ndof_per_node)
    e = np.clip(np.searchsorted(nodes, x_load, side="right") - 1, 0, len(nodes) - 2)
    h = nodes[e + 1] - nodes[e]
    xi = (x_load - nodes[e]) / h              # coordonnée locale dans [0, 1]
    F[e * ndof_per_node] = F0 * (1 - xi)      # N_1(x_load)
    F[(e + 1) * ndof_per_node] = F0 * xi      # N_2(x_load)
    return F
 

def u_exact(x, F0, T, L, x_bow):
    x = np.asarray(x)
    return F0 / (T * L) * np.where(x <= x_bow,
                                   x * (L - x_bow),
                                   x_bow * (L - x))


