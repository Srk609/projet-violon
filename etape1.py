import numpy as np
import matplotlib.pyplot as plt   
import scipy
import time
from string_fem import mesh_1d,string_elem_stiffness,string_elem_mass_lumped,string_elem_mass,reduire_matrice,etendre_vecteur,force_ponctuelle,assemble,free_dofs,u_exact

import numpy as np
import matplotlib.pyplot as plt

# Données
L, x_bow, rho_S, f1, F0 = 0.33, 0.03, 3.1e-3, 196, 1.0
T = rho_S * (2 * L * f1)**2          # ≈ 51.9 N
n_elem = 11

# Maillage et raideur
nodes, connect = mesh_1d(L, n_elem)
K = assemble(nodes, connect, string_elem_stiffness, T=T)
n_dof = len(nodes)

# Conditions aux limites : u = 0 en x = 0 et x = L
fixed = [0, n_dof - 1]
free = free_dofs(n_dof, fixed)

# Force et résolution sur les ddl libres
F = force_ponctuelle(nodes, x_bow, F0)
K_red = reduire_matrice(K, free)
U_free = np.linalg.solve(K_red, F[free])
U = etendre_vecteur(U_free, free, n_dof)

# Solution exacte sur une grille fine
x_fin = np.linspace(0, L, 500)
u_ex = u_exact(x_fin, F0, T, L, x_bow)

# Figure : superposition exact / EF
plt.plot(x_fin, u_ex * 1e3, "k", label="exact")
plt.plot(nodes, U * 1e3, "o--", label=f"EF, {n_elem} éléments")
plt.xlabel("x [m]")
plt.ylabel("u [mm]")
plt.title(f"Déformée statique sous F = {F0} N en x = {x_bow} m (T = {T:.1f} N)")
plt.legend()
plt.show()