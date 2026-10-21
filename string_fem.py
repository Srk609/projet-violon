import numpy as np

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