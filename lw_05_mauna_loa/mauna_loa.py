#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:52:50 2026

@author: redabens
"""

# %% Chargement des données
from pathlib import Path
from scipy.io import loadmat

data_path = Path(__file__).parent / "Keeling.mat"   # chemin relatif au script
data = loadmat(data_path)                           # dict {nom: tableau}

# Les clés qui commencent par "__" sont des métadonnées, on les masque
print([k for k in data if not k.startswith("__")])


# %% Extraction et contrôle
import numpy as np

t = data["t"].ravel()   # années
y = data["y"].ravel()   # CO2 en ppm

print("formes :", t.shape, y.shape)
print("t de", t.min(), "à", t.max())
print("y de", y.min(), "à", y.max())


# %% Tracé des données brutes
import matplotlib.pyplot as plt

print("pas moyen (mois) :", np.diff(t).mean() * 12)   # attendu ≈ 1

fig, ax = plt.subplots(figsize=(8, 4))   # ≈ figure en MATLAB
ax.plot(t, y, lw=0.8)
ax.set_xlabel("Année")
ax.set_ylabel("CO$_2$ (ppm)")
ax.set_title("Courbe de Keeling — Mauna Loa")
ax.grid(alpha=0.3)
plt.show()

# %% I. Ajustement parabolique (moindres carrés)
def approx_parabole(t, y):
    """Ajuste y ≈ a·t² + b·t + c au sens des moindres carrés.
    Équivalent de approx_parabole.m : renvoie (coefficients, courbe ajustée)."""
    M = np.column_stack([t**2, t, np.ones_like(t)])   # ≈ [t.^2, t, ones(...)]
    A_hat, *_ = np.linalg.lstsq(M, y, rcond=None)      # ≈ A_hat = M \ y
    return A_hat, M @ A_hat                            # @ = produit matriciel

A_hat, y_fit = approx_parabole(t, y)
a, b, c = A_hat
print(f"a = {a:.6g}   b = {b:.6g}   c = {c:.6g}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, y, "k-", lw=0.8, label="Données réelles")
ax.plot(t, y_fit, "r-", label="Modèle ajusté")
ax.set_xlabel("Année")
ax.set_ylabel("CO$_2$ (ppm)")
ax.set_title("Courbe de Keeling et ajustement parabolique")
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# %% Prédiction jusqu'en 2100
t_2100 = np.arange(2024, 2101)    # ≈ 2024:2100 — la borne de fin est EXCLUE en Python
M_2100 = np.column_stack([t_2100**2, t_2100, np.ones_like(t_2100)])
y_2100 = M_2100 @ A_hat

print(f"CO2 prévu en 2100 : {y_2100[-1]:.4f} ppm   (MATLAB : 689.1932)")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, y, "k-", lw=0.8, label="Données réelles")
ax.plot(t, y_fit, "r-", label="Modèle parabolique")
ax.plot(t_2100, y_2100, "b--", label="Projection")
ax.set_xlabel("Année")
ax.set_ylabel("CO$_2$ (ppm)")
ax.set_title("Projection parabolique jusqu'en 2100")
ax.legend()
ax.grid(alpha=0.3)
plt.show()