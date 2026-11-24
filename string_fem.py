import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.integrate
from scipy.io import wavfile
from scipy import signal as sig  

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
    omega2, phi_free = scipy.linalg.eigh(K_free, M_free, subset_by_index=(0, n_modes - 1)) # Permet d'obtenir les n_modes premiers modes
    omega = np.sqrt(omega2)
    frequencies = omega / (2 * np.pi)
    n_dof = K.shape[0]
    phi = np.zeros((n_dof, n_modes)) #création d'un tableau vide qui permettra de stocker les vecteurs propres.
    for i in range(n_modes):
        phi[:, i] = etendre_vecteur(phi_free[:, i], free, n_dof)# on extrait la ième colonne qui correspond au ième vecteur propre réduit d'un mode,puis on lui rajoute des zéros aux extrémités bloquées puis on injecte ce vecteur complet dans la matrice phi
        
    return frequencies, phi
# Calcul des spectres

# Spectre de la force au chevalet (avec et sans amortissement)

f_max = 4000.0                                            # fréquence maximale à tracer [Hz] (demandée par l'énoncé)

def spectre(F, fs):                                       # calcule le spectre d'amplitude d'un signal F échantillonné à fs
    w = np.hanning(len(F))                                # fenêtre de Hann : courbe en cloche qui vaut 0 aux bords et 1 au milieu
    F_fenetre = (F - np.mean(F)) * w                      # on retire la valeur moyenne (pic parasite à 0 Hz) puis on multiplie par la fenêtre (limite la fuite spectrale)
    X = np.fft.rfft(F_fenetre)                            # FFT pour un signal réel : coefficients complexes pour les fréquences de 0 à fs/2
    amplitude = 2 * np.abs(X) / np.sum(w)                 # module = amplitude ; x2 pour les fréquences négatives ignorées ; /sum(w) corrige l'effet de la fenêtre -> amplitude en N
    f = np.fft.rfftfreq(len(F), d=1/fs)                   # fréquence associée à chaque case de X : 0, df, 2df, ... jusqu'à fs/2
    return f, amplitude                                   # on renvoie l'axe des fréquences et l'amplitude


# 1. Motif de la matrice de raideur globale K
L, x_bow, rho_S, f1, F0 = 0.33, 0.03, 3.1e-3, 196.0, 1.0  # paramètres physiques de la corde
T = rho_S * (2 * L * f1) ** 2  # tension déduite de la corde [N]
n_elem = 11  # maillage grossier à 11 éléments

nodes, connect = mesh_1d(L, n_elem)  # création des nœuds et de la connectivité
K = assemble(nodes, connect, string_elem_stiffness, T=T)  # assemblage de la matrice de raideur

plt.figure()
plt.spy(K, marker="s", markersize=14)  # affichage des termes non nuls de K
plt.title(f"corde : {K.shape[0]} ddl, nnz = {np.count_nonzero(K)}")  # titre avec ddl et nnz
plt.show()

# 2. Déformée statique : comparaison éléments finis / solution exacte
n_dof = len(nodes)  # nombre total de degrés de liberté
free = free_dofs(n_dof, [0, n_dof - 1])  # ddl libres (extrémités encastrées bloquées)

F = force_ponctuelle(nodes, x_bow, F0)  # vecteur force ponctuelle en x_bow
K_red = reduire_matrice(K, free)  # réduction de K aux ddl libres
U_free = np.linalg.solve(K_red, F[free])  # résolution du système statique
U = etendre_vecteur(U_free, free, n_dof)  # reconstruction sur tous les nœuds

x_fin = np.linspace(0, L, 500)  
u_ex = u_exact(x_fin, F0, T, L, x_bow)  # solution analytique exacte

plt.figure()
plt.plot(x_fin, u_ex * 1e3, "k", label="exact")  # tracé exact en mm
plt.plot(nodes, U * 1e3, "o--", label=f"EF, {n_elem} éléments")  # tracé éléments finis en mm
plt.xlabel("x [m]")  # abscisse en mètres
plt.ylabel("u [mm]")  # déplacement en millimètres
plt.title(f"Déformée statique sous F = {F0} N en x = {x_bow} m (T = {T:.1f} N)")  # titre de la figure
plt.legend()
plt.show()

# 3. Premières déformées modales de la corde
n_elem_mod = 330  # maillage raffiné pour capturer les modes
n_modes = 10  # nombre de modes propres à tracer

nodes_mod, K_mod, M_mod, free_mod = build_string(L, rho_S, T, n_elem_mod)  # construction K et M
frequencies, phi = solve_modal(K_mod, M_mod, free_mod, n_modes=n_modes)  # fréquences et modes

plt.figure(figsize=(10, 6))
for m in range(n_modes):
    plt.plot(nodes_mod, phi[:, m], label=f"mode {m + 1} : {frequencies[m]:.1f} Hz")  # déformée du mode m+1

plt.axvline(x_bow, color="k", linestyle=":", label="archet")  # repère vertical de la position d'archet
plt.xlabel("Position sur la corde [m]")  # axe des positions
plt.ylabel("Déformée modale (normée)")  # axe des déformées modales
plt.title("Premières déformées modales de la corde")  # titre de la figure
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()


# Paramètres physiques et géométriques
L = 0.33               # longueur [m]
x_bow = 0.03           # position de l'archet [m]
rhoS = 3.1e-3          # masse linéique [kg/m]
f1 = 196.0             # fréquence fondamentale [Hz]
T = rhoS * (2 * L * f1) ** 2  # tension déduite [N]

# Paramètres numériques
n_elem = 110
x_pluck = 0.03
u_pluck = 1e-3
duree = 0.2
fs = 44100
xi = 0.001             # taux d'amortissement modal (0.1 %)
N = 25                 # nombre de modes conservés
f_max = 4000.0         # fréquence maximale d'intérêt [Hz]

t_span = (0.0, duree)
t_eval_complet = np.linspace(0, duree, int(duree * fs) + 1)
t_eval_modal = np.linspace(t_span[0], t_span[1], 1000)

# Construction du maillage et des matrices
nodes, K, M, free = build_string(L, rhoS, T, n_elem)

K_libre = reduire_matrice(K, free)
M_libre = reduire_matrice(M, free)

n_libres = len(free)

# Conditions initiales
U0_initial = u_exact(nodes, 1.0, T, L, x_pluck)
U0 = U0_initial[free]

V0 = np.zeros(n_libres)
Y0 = np.concatenate((U0, V0))

# Factorisation de la matrice de masse
facteur_M = scipy.linalg.cho_factor(M_libre)

# Second membre du système
def second_membre(t, Y):
    U = Y[:n_libres]
    V = Y[n_libres:]
    acceleration = scipy.linalg.cho_solve(
        facteur_M, -K_libre @ U
    )
    return np.concatenate((V, acceleration))

# Début du chronométrage
debut = time.perf_counter()

# Résolution temporelle avec RK45
solution = scipy.integrate.solve_ivp(
    second_membre,
    t_span,
    Y0,
    method="RK45",
    t_eval=t_eval_complet,
    rtol=1e-3,
    atol=1e-6
)
# Fin du chronométrage
fin = time.perf_counter()

temps_calcul = fin - debut

print(f"Temps de calcul RK45 : {temps_calcul:.6f} secondes")

# Reconstruction du déplacement sur tous les nœuds
U_libre_t = solution.y[:n_libres, :]

U_t = np.zeros((len(nodes), len(solution.t)))
U_t[free, :] = U_libre_t

# Affichage du déplacement au nœud choisi
i = 10

plt.figure()
plt.plot(solution.t * 1e3, U_t[i] * 1e3)
plt.xlabel("Temps [ms]")
plt.ylabel("Déplacement [mm]")
plt.title("Déplacement en x = 30 mm")
plt.grid()
plt.show()

# Début du chronométrage
debut = time.perf_counter()

# Calcul des modes propres
frequencies, phi = solve_modal(K, M, free, n_modes=N)

phi_libre = phi[free, :]

# Projection des conditions initiales dans la base modale
q0 = phi_libre.T @ (M_libre @ U0)

q_point0 = np.zeros(N)

Y0_modal = np.concatenate((q0, q_point0))


# Second membre du système modal
def second_membre_modal(t, y):
    q = y[:N]
    q_point = y[N:]

    acceleration_modal = -(2 * np.pi * frequencies)**2 * q

    return np.concatenate((q_point, acceleration_modal))


# Résolution temporelle
solution_modal = scipy.integrate.solve_ivp(
    second_membre_modal,
    t_span,
    Y0_modal,
    method="DOP853",
    t_eval=t_eval_modal,
    rtol=1e-6,
    atol=1e-9
)

# Fin du chronométrage
fin = time.perf_counter()

temps_calcul = fin - debut

print(f"Temps de calcul modal : {temps_calcul:.6f} secondes")

# Reconstruction du déplacement
q_t = solution_modal.y[:N, :]

U_modal = phi @ q_t

# Visualisation de la corde à différents instants
T1 = 1 / f1

fig, ax = plt.subplots()

for idx in [0,int(0.125 * T1 * fs), int(0.25 * T1 * fs),int(0.375 * T1 * fs),int(0.5 * T1 * fs)]:
    ax.plot(nodes,
        U_modal[:, idx],
        label=f"t = {solution_modal.t[idx]:.5f} s"
    )

ax.set_xlabel("Position x (m)")
ax.set_ylabel("Déplacement u_modal(x,t) (m)")
ax.set_title("Évolution de la corde en fonction de x pour différents instants")
ax.legend()
ax.grid()

plt.show()

# Construction EF et base modale
x_pluck = L / 5
duree_sim = 1.0
t_eval = np.arange(0, duree_sim, 1 / fs)

# Condition initiale modale
F0 = u_pluck * T * L / (x_pluck * (L - x_pluck))
U0_sim = u_exact(nodes, F0, T, L, x_pluck)[free]

q0_sim = phi_libre.T @ (M_libre @ U0_sim)
Y0_sim = np.concatenate((q0_sim, np.zeros(N)))

omega = 2 * np.pi * frequencies


def second_membre_non_amorti(t, y):
    q = y[:N]
    q_point = y[N:]
    acceleration = -(omega ** 2) * q
    return np.concatenate((q_point, acceleration))


def second_membre_amorti(t, y):
    q = y[:N]
    q_point = y[N:]
    acceleration_amorti = -2 * xi * omega * q_point - (omega ** 2) * q
    return np.concatenate((q_point, acceleration_amorti))


# Intégration temporelle
solution_non_amortie = scipy.integrate.solve_ivp(
    second_membre_non_amorti,
    (0.0, duree_sim),
    Y0_sim,
    method="DOP853",
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-9,
)

solution_amortie = scipy.integrate.solve_ivp(
    second_membre_amorti,
    (0.0, duree_sim),
    Y0_sim,
    method="DOP853",
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-9,
)

# Récupération des coordonnées modales
q_sans = solution_non_amortie.y[:N, :]
q_avec = solution_amortie.y[:N, :]
q_point_sans = solution_non_amortie.y[N:, :]
q_point_avec = solution_amortie.y[N:, :]

omega_col = omega.reshape(N, 1)
q_seconde_sans = -(omega_col ** 2) * q_sans
q_seconde_avec = -2 * xi * omega_col * q_point_avec - (omega_col ** 2) * q_avec

# Reconstruction physique et force au chevalet (nœud 0)
U_sans = phi @ q_sans
U_avec = phi @ q_avec
U_seconde_sans = phi @ q_seconde_sans
U_seconde_avec = phi @ q_seconde_avec

force_sans = -(K @ U_sans + M @ U_seconde_sans)[0, :]
force_avec = -(K @ U_avec + M @ U_seconde_avec)[0, :]

# Export audio (.wav)
signal = force_avec / np.max(np.abs(force_avec))
data = np.iinfo(np.int16).max * signal
wavfile.write("pincement1.wav", fs, data.astype(np.int16))

# Figure 1 : signal complet
plt.figure(figsize=(10, 4))
plt.plot(t_eval, force_sans, label="sans amortissement")
plt.plot(t_eval, force_avec, label="xi = 0.1% pour tous les modes")
plt.xlabel("t [s]")
plt.ylabel("Force au chevalet [N]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Figure 2 : zoom sur les premiers instants
masque_debut = t_eval <= 0.015
plt.figure(figsize=(8, 4))
plt.plot(t_eval[masque_debut], force_sans[masque_debut], color="lightgray", label="sans amortissement")
plt.plot(t_eval[masque_debut], force_avec[masque_debut], color="C1", label="avec amortissement")
plt.title("premiers instants")
plt.xlabel("t [s]")
plt.ylabel("force au chevalet [N]")
plt.grid()
plt.tight_layout()
plt.show()

# Figure 3 : zoom sur les derniers instants
masque_fin = t_eval >= (duree_sim - 0.015)
plt.figure(figsize=(8, 4))
plt.plot(t_eval[masque_fin], force_sans[masque_fin], color="lightgray", label="sans amortissement")
plt.plot(t_eval[masque_fin], force_avec[masque_fin], color="C1", label="avec amortissement")
plt.title("derniers instants")
plt.xlabel("t [s]")
plt.ylabel("force au chevalet [N]")
plt.grid()
plt.tight_layout()
plt.show()

# Calcul des spectres
f, A_sans = spectre(force_sans, fs)                       # spectre de la force au chevalet sans amortissement
f, A_avec = spectre(force_avec, fs)                       # spectre de la force au chevalet avec amortissement (même axe f car même nombre de points)

A_ref = np.max(A_sans)                                    # amplitude de référence : le plus grand pic sans amortissement (il sera à 0 dB)
dB_sans = 20 * np.log10(A_sans / A_ref + 1e-12)           # passage en décibels : 20*log10 car c'est une amplitude ; +1e-12 évite log10(0)
dB_avec = 20 * np.log10(A_avec / A_ref + 1e-12)           # même chose avec amortissement, même référence pour pouvoir comparer

df = f[1] - f[0]                                          # résolution fréquentielle = écart entre deux cases = 1/durée
print("Résolution fréquentielle df =", df, "Hz")          # 1 Hz pour une durée de 1 s
print("Fréquence maximale représentée =", f[-1], "Hz")    # fréquence de Nyquist fs/2 = 22050 Hz

fig, ax = plt.subplots(figsize=(10, 4))                   # figure large avec un seul graphique
for n in range(1, int(f_max / f1) + 1):                   # boucle sur les harmoniques n = 1, 2, ... tant que n*f1 <= 4000 Hz
    ax.axvline(n * f1, color="grey", linestyle=":", linewidth=0.8)   # trait vertical en pointillés à la fréquence n*f1
ax.plot(f, dB_sans, linewidth=0.8, label="sans amortissement")       # spectre sans amortissement
ax.plot(f, dB_avec, linewidth=0.8, label="avec amortissement")       # spectre avec amortissement
ax.set_xlim(0, f_max)                                     # on n'affiche que de 0 à 4 kHz
ax.set_ylim(-100, 5)                                      # axe vertical entre -100 dB et +5 dB
ax.set_xlabel("fréquence [Hz]")                           # titre de l'axe horizontal
ax.set_ylabel("amplitude [dB]")                           # titre de l'axe vertical
ax.set_title(r"spectre de la force au chevalet (pointillés : $nf_1$)")   # titre ($nf_1$ écrit en LaTeX)
ax.legend()                                               # légende
plt.tight_layout()                                        # ajuste les marges pour que rien ne soit coupé
plt.show()                                                # affiche la figure

# Spectrogramme de la force au chevalet amortie

n_fenetre = 4096                                          # nombre de points de chaque tranche (puissance de 2 : FFT rapide)
duree_fenetre = n_fenetre / fs                            # durée d'une tranche en secondes : 4096 / 44100 = 93 ms
n_recouvrement = 3 * n_fenetre // 4                       # deux tranches successives se recouvrent de 75 % (la fenêtre avance de 1024 points = 23 ms)

f_spec, t_spec, Sxx = sig.spectrogram(                    # calcul du spectrogramme (transformée de Fourier à court terme) :
    force_avec,                                           #   signal analysé : force au chevalet avec amortissement
    fs=fs,                                                #   fréquence d'échantillonnage [Hz] pour graduer l'axe des fréquences
    window="hann",                                        #   chaque tranche est pondérée par une fenêtre de Hann (limite la fuite spectrale)
    nperseg=n_fenetre,                                    #   longueur de chaque tranche
    noverlap=n_recouvrement,                              #   nombre de points communs entre deux tranches voisines
    detrend=False,                                        #   on ne retire rien au signal dans chaque tranche
    mode="magnitude")                                     #   on veut l'amplitude |X| (et non la puissance |X|^2)
# f_spec : fréquences (lignes), t_spec : instants au centre de chaque tranche (colonnes), Sxx : amplitude pour chaque (f, t)

Sxx_dB = 20 * np.log10(Sxx / np.max(Sxx) + 1e-12)         # passage en dB, 0 dB = point le plus intense ; +1e-12 évite log10(0)

fig, ax = plt.subplots(figsize=(10, 4))                   # création d'une figure large avec un seul graphique
carte = ax.pcolormesh(t_spec, f_spec, Sxx_dB,             # carte temps-fréquence : la couleur de chaque case donne l'amplitude en dB
                      cmap="magma",                       #   palette de couleurs (noir = faible, jaune clair = fort)
                      vmin=-80, vmax=0,                   #   échelle des couleurs de -80 dB à 0 dB (en dessous de -80 dB : noir)
                      shading="auto")                     #   placement automatique des cases autour des points (t, f)
ax.set_ylim(0, 4000)                                      # on n'affiche que les fréquences de 0 à 4 kHz
ax.set_xlabel("t [s]")                                    # titre de l'axe horizontal
ax.set_ylabel("fréquence [Hz]")                           # titre de l'axe vertical
ax.set_title(f"spectrogramme de la force au chevalet (amortie), fenêtre de {duree_fenetre*1e3:.0f} ms")   # titre avec la durée de la fenêtre en ms
fig.colorbar(carte, ax=ax, label="amplitude [dB]")        # barre de couleurs à droite pour lire les dB
plt.tight_layout()                                        # ajuste les marges pour que rien ne soit coupé
plt.show()  

# =============================================================
# Données de l'étape 3 (Table 3)
# =============================================================
MU_S = 0.4        # coefficient de frottement statique (corde qui colle à l'archet)
MU_D = 0.2        # coefficient de frottement dynamique (corde qui glisse vite)
C = 5.0           # décroissance de l'effet Stribeck [s/m] : vitesse du passage de mu_s à mu_d
V_LIM = 1e-3      # vitesse de régularisation [m/s] : largeur de la transition autour de v = 0


# =============================================================
# Définitions
# =============================================================
def mu_stribeck(v, mu_s=MU_S, mu_d=MU_D, C=C):
    """Loi de Coulomb avec effet Stribeck, NON régularisée (discontinue en v = 0)."""
    return np.sign(v) * (mu_d + (mu_s - mu_d) * np.exp(-C * np.abs(v)))   # sign(v) donne le sens, le crochet l'intensité


def mu(v, v_lim=V_LIM, mu_s=MU_S, mu_d=MU_D, C=C):
    """Loi régularisée (équation 22) : sign(v) est remplacé par tanh(v / v_lim), continue et très raide."""
    return np.tanh(v / v_lim) * (mu_d + (mu_s - mu_d) * np.exp(-C * np.abs(v)))   # tanh vaut ±1 dès que |v| >> v_lim


def force_frottement(v_corde, V_archet, F_archet, v_lim=V_LIM):
    """Force de l'archet sur la corde (équation 21) : F = -F_b mu(v_b - V_b)."""
    v_glissement = v_corde - V_archet                     # vitesse relative corde/archet (0 pendant l'adhérence)
    return -F_archet * mu(v_glissement, v_lim)            # le signe - : le frottement s'oppose au glissement
v = np.linspace(-0.6, 0.6, 20001)                     # vitesses de glissement sur toute la plage [m/s]v_zoom = np.linspace(-0.01, 0.01, 4001)               # zoom autour de v = 0 pour voir la régularisation
v_zoom = np.linspace(-0.01, 0.01, 4001)               # zoom autour de v = 0 pour voir la régularisation    

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4)) # deux graphiques côte à côte
ax1.plot(v, mu_stribeck(v), color="lightgrey", lw=3, label="non régularisée")   # loi exacte en gris épais
for v_lim in [1e-2, 1e-3, 1e-4]:                      # les trois valeurs demandées par l'énoncé
        ax1.plot(v, mu(v, v_lim), label=f"v_lim = {v_lim:g} m/s")            # vue d'ensemble
        ax2.plot(v_zoom, mu(v_zoom, v_lim), label=f"v_lim = {v_lim:g} m/s")  # zoom sur la transition

ax1.set_title("vue d'ensemble")
ax2.set_title("zoom autour de v = 0")
for ax in (ax1, ax2):                                 # même mise en forme pour les deux graphiques
        ax.set_xlabel("vitesse de glissement v = v_b - V_b [m/s]")
        ax.set_ylabel("μ")
        ax.grid()
        ax.legend()
plt.tight_layout()
plt.show()