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
    Vecteur force pour une force ponctuelle F0 en x_load.

    Hypothèse simplificatrice : la force de l'archet est supposée appliquée
    directement sur un noeud du maillage (cas F_i = F0, autres composantes nulles).
    On prend le noeud le plus proche de x_load. Cette hypothèse est exacte si
    x_load coïncide avec un noeud ; sinon, la position de la force est décalée
    d'au plus h/2 (h = longueur d'un élément).

    Entrées :
    - nodes : positions des noeuds
    - x_load : position de la force [m]
    - F0 : intensité de la force [N]
    - ndof_per_node : nombre de ddl par noeud

    Sortie : vecteur F de taille n_dof, nul sauf au noeud chargé
    """
    F = np.zeros(len(nodes) * ndof_per_node)

    # recherche du noeud i le plus proche de x_load
    i = 0
    for j in range(len(nodes)):
        if abs(nodes[j] - x_load) < abs(nodes[i] - x_load):
            i = j

    F[i * ndof_per_node] = F0     # toute la force sur le noeud i
    return F

def u_exact(x, F0, T, L, x_bow):
    x = np.asarray(x)
    return F0 / (T * L) * np.where(x <= x_bow,
                                   x * (L - x_bow),
                                   x_bow * (L - x))

from scipy import linalg
def build_string(L, rho, T, n_elem):
    """Construit le maillage et assemble les matrices globales K et M."""
    nodes, connect = mesh_1d(L, n_elem)
    ndof_per_node = 1
    K = assemble(nodes, connect, string_elem_stiffness, ndof_per_node,T)
    M = assemble(nodes, connect, string_elem_mass, ndof_per_node, rho, S=1.0)
    
    n_dof = K.shape[0]
    fixed = [0, n_dof - 1] 
    free = free_dofs(n_dof, fixed)
    return nodes, K, M, free
def solve_modal(K, M, free, n_modes=10):
    K_free = reduire_matrice(K, free)
    M_free = reduire_matrice(M, free)
    omega2, phi_free = linalg.eigh(K_free, M_free, subset_by_index=(0, n_modes - 1))
    omega = np.sqrt(omega2)
    frequencies = omega / (2 * np.pi)
    n_dof = K.shape[0]
    phi = np.zeros((n_dof, n_modes))
    for i in range(n_modes):
        phi[:, i] = etendre_vecteur(phi_free[:, i], free, n_dof)
        
    return frequencies, phi