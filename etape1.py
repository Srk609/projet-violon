import numpy as np
import matplotlib.pyplot as plt
from string_fem import (
    mesh_1d,
    string_elem_stiffness,
    assemble,
    free_dofs,
    reduire_matrice,
    etendre_vecteur,
    force_ponctuelle,
    u_exact,
    build_string,
    solve_modal,
)

# 1. Motif creux de la matrice de raideur globale K
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

x_fin = np.linspace(0, L, 500)  # grille fine pour la courbe exacte
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
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()