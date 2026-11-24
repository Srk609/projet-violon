
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.integrate

from string_fem import build_string, reduire_matrice, u_exact

L = 0.33
x_bow = 0.03
rhoS = 3.1e-3
f1 = 196.0
T = rhoS * (2 * L * f1) ** 2

n_elem = 110
x_pluck = 0.03
u_pluck = 1e-3
duree = 0.2
fs = 44100

nodes, K, M, free = build_string(L, rhoS, T, n_elem)

t_span = (0.0, duree)
t_eval = np.linspace(0, duree, int(duree * fs) + 1)

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
    t_eval=t_eval,
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

plt.plot(solution.t * 1e3, U_t[i] * 1e3)
plt.xlabel("Temps [ms]")
plt.ylabel("Déplacement [mm]")
plt.title("Déplacement en x = 30 mm")
plt.grid()
plt.show()
