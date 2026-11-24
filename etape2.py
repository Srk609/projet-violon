import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.integrate
from scipy.io import wavfile

from string_fem import (
    build_string,
    solve_modal,
    reduire_matrice,
    u_exact,
    spectre,
)

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

for idx in [
    0,
    int(0.125 * T1 * fs),
    int(0.25 * T1 * fs),
    int(0.375 * T1 * fs),
    int(0.5 * T1 * fs)
]:
    ax.plot(
        nodes,
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

# Figure complète : signal complet + zooms début et fin
plt.figure(figsize=(12, 7))

plt.subplot(2, 2, (1, 2))
plt.plot(t_eval, force_sans, label="sans amortissement")
plt.plot(t_eval, force_avec, label="xi = 0.1% pour tous les modes")
plt.xlabel("t [s]")
plt.ylabel("Force au chevalet [N]")
plt.legend()
plt.grid()

masque_debut = t_eval <= 0.015
plt.subplot(2, 2, 3)
plt.plot(t_eval[masque_debut], force_sans[masque_debut], color="lightgray", label="sans amortissement")
plt.plot(t_eval[masque_debut], force_avec[masque_debut], color="C1", label="avec amortissement")
plt.title("premiers instants")
plt.xlabel("t [s]")
plt.ylabel("force au chevalet [N]")
plt.grid()

masque_fin = t_eval >= (duree_sim - 0.015)
plt.subplot(2, 2, 4)
plt.plot(t_eval[masque_fin], force_sans[masque_fin], color="lightgray", label="sans amortissement")
plt.plot(t_eval[masque_fin], force_avec[masque_fin], color="C1", label="avec amortissement")
plt.title("derniers instants")
plt.xlabel("t [s]")
plt.ylabel("force au chevalet [N]")
plt.grid()

plt.tight_layout()
plt.show()

# Calcul des spectres
f, A_sans = spectre(force_sans, fs)
f, A_avec = spectre(force_avec, fs)