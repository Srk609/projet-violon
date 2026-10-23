import numpy as np
import matplotlib.pyplot as plt   
import scipy

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


def string_elem_stiffness(h, T):# Matrice de la raideur élémentaire
    return (T / h) * np.array([
        [ 1.0, -1.0],
        [-1.0,  1.0]
    ])
def string_elem_mass(h, rho, S):#Matrice de la masse élémentaire
    return rho * S * h / 6 * np.array([[2.0, 1.0],
                                       [1.0, 2.0]])


def string_elem_mass_lumped(h, rho, S):#matrice de la masse "concentrée"
    return rho * S * h / 2 * np.array([[1.0, 0.0],
                                       [0.0, 1.0]])


def assemble(nodes, connect, matrice_elementaire, ndof_per_node=1, **params):
    """
    Assemble une matrice globale à partir d'une fonction élémentaire.
 
    matrice_elementaire : fonction (h, **params) -> matrice élémentaire
    params              : paramètres passés à la fonction (T=..., ou rho=..., S=...)
    """
    n_nodes = len(nodes)
    n_dof = n_nodes * ndof_per_node
    A = np.zeros((n_dof, n_dof)) # Initialisation de la matrice globale vide
    for elem_nodes in connect: #on parcourt chaque élément du maillage
        h = nodes[elem_nodes[1]] - nodes[elem_nodes[0]] #Longueur de l'élément
        Ae = matrice_elementaire(h, **params) # calcul de la matrice élémentaire à partir des deux fonctions définis ci-dessus
        dofs = [i * ndof_per_node + k
                for i in elem_nodes for k in range(ndof_per_node)]
        for a in range(len(dofs)):
            for b in range(len(dofs)):
                A[dofs[a], dofs[b]] += Ae[a, b] # Addition dans la matrice globale
    return A
 
def free_dofs(n_dof, fixed):
    """Liste des ddl libres : tous les ddl sauf ceux bloqués."""
    return np.setdiff1d(np.arange(n_dof), fixed)


def reduire_matrice(A, free):
    """Matrice réduite : lignes libres, puis colonnes libres."""
    return A[free][:, free]# Extrait la sous-matrice correspondant uniquement aux DDL libres
 
 
def etendre_vecteur(U_free, free, n_dof):
    """Reconstruit le vecteur complet (zéros aux ddl bloqués)."""
    U = np.zeros(n_dof)
    U[free] = U_free
    return U # Replace les valeurs libres et met des 0 aux extrémités bloquées


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
def build_string(L, rhoS, T, n_elem):
    """Construit le maillage et assemble les matrices globales K et M."""
    nodes, connect = mesh_1d(L, n_elem)
    ndof_per_node = 1
    K = assemble(nodes, connect, string_elem_stiffness, ndof_per_node, T=T) #utilisation de l'outil d'assemblage en associant la fonction de raideur élémentaire pour ensuite calculer la matrice de raideur globale K de toute la corde
    M = assemble(nodes, connect, string_elem_mass, ndof_per_node, rho=rhoS, S=1.0)#utilisation de l'outil d'assemblage en associant la fonction de masse élémentaire pour ensuite calculer la matrice de masse globale M de toute la corde
    
    n_dof = K.shape[0]
    fixed = [0, n_dof - 1] # Extrémités encastrées
    free = free_dofs(n_dof, fixed)# permet d'identifier les noeuds qui ont le droit de bouger(les noeuds intérieurs),en retirant les noeuds bloqués de la liste.
    return nodes, K, M, free
def solve_modal(K, M, free, n_modes=10):
    K_free = reduire_matrice(K, free)#permet d'éliminer les lignes et colonnes correspondant aux extrémités bloquées
    M_free = reduire_matrice(M, free)
    omega2, phi_free = linalg.eigh(K_free, M_free, subset_by_index=(0, n_modes - 1)) # Permet d'obtenir les n_modes premiers modes
    omega = np.sqrt(omega2)
    frequencies = omega / (2 * np.pi)
    n_dof = K.shape[0]
    phi = np.zeros((n_dof, n_modes)) #création d'un tableau vide qui permettra de stocker les vecteurs propres.
    for i in range(n_modes):
        phi[:, i] = etendre_vecteur(phi_free[:, i], free, n_dof)# on extrait la ième colonne qui correspond au ième vecteur propre réduit d'un mode,puis on lui rajoute des zéros aux extrémités bloquées puis on injecte ce vecteur complet dans la matrice phi
        
    return frequencies, phi
L = 0.33            # longueur [m]
x_bow = 0.03        # position de l'archet [m]
rhoS = 3.1e-3       # masse linéique [kg/m]
f1 = 196.0          # fréquence fondamentale [Hz]
T = rhoS * (2 * L * f1) ** 2     # tension déduite [N]
n_elem = 330        # Maillage raffiné pour capturer proprement les 10 modes

nodes, K, M, free = build_string(L, rhoS, T, n_elem)


n_modes = 4
frequencies, phi = solve_modal(K, M, free, n_modes=n_modes)

n_elem = 110
x_pluck = 0.03          # Position du pincement [m]
u_pluck = 1e-3          # Déplacement au point de pincement [m]
duree = 0.2             # Durée simulée [s]
fs=44100                #Fréquence [Hz]
nodes, K, M, free = build_string(L, rhoS, T, n_elem)

K_libre = reduire_matrice(K, free)
M_libre = reduire_matrice(M, free)

n_libres = len(free)

#construction du vecteur initial avec le déplacement et la vitesse
U0_initial=u_exact(nodes,1.0,T,L,x_pluck)
# Conservation des seuls degrés de liberté libres
U0 = U0_initial[free]

#la vitesse initiale est supposée nulle
V0=np.zeros(n_libres)

Y0=np.concatenate((U0,V0))
facteur_M = scipy.linalg.cho_factor(M_libre)
def second_membre(t,Y):
    U=Y[n_libres:]
    V=Y[:n_libres]
    acceleration = scipy.linalg.cho_solve(facteur_M, -K_libre @ U)
    return np.concatenate((V, acceleration))
t_span=(0.0,duree)
t_eval = np.linspace(*t_span)

solution = scipy.integrate.solve_ivp( second_membre,t_span,Y0, method="RK45",t_eval=t_eval,rtol=1e-3,atol=1e-6)

#On récupère le déplacement en fonction du temps 
U_libre_t = solution.y[:n_libres, :]

#Reconstruction des déplacements sur tous les noeuds
U_t = np.zeros((len(nodes), len(solution.t)))
U_t[free, :] = U_libre_t
jj



