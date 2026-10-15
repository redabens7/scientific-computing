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


# %% Résidus de l'ajustement parabolique
res_abs = y - y_fit
res_rel = res_abs / y        # en NumPy, / est déjà terme à terme (≈ ./)
print(f"RMSE parabole : {np.sqrt(np.mean(res_abs**2)):.3f} ppm")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)   # ≈ subplot(2,1,k)
ax1.plot(t, res_abs, "b-", lw=0.8)
ax1.set_ylabel("Résidus (ppm)")
ax1.set_title("Résidus de l'ajustement parabolique")
ax2.plot(t, res_rel, "g-", lw=0.8)
ax2.set_ylabel("Résidus relatifs")
ax2.set_xlabel("Année")
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
plt.show()




# %% II. Ajustement parabole + variation saisonnière
def approx_parab_sin(t, y, T=1.0):
    """Ajuste y ≈ a·t² + b·t + c + d·cos(2πt/T) + e·sin(2πt/T).
    Équivalent de approx_parab_sin.m (T = période, en années)."""
    w = 2 * np.pi * t / T
    M = np.column_stack([t**2, t, np.ones_like(t), np.cos(w), np.sin(w)])
    A_hat, *_ = np.linalg.lstsq(M, y, rcond=None)
    return A_hat, M @ A_hat

A_hat2, y_fit2 = approx_parab_sin(t, y)
a2, b2, c2, d, e = A_hat2
res2 = y - y_fit2
print(f"d = {d:.4f}   e = {e:.4f}   amplitude = {np.hypot(d, e):.3f} ppm")
print(f"RMSE parabole + sinus : {np.sqrt(np.mean(res2**2)):.3f} ppm")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(t, y, "k-", lw=0.8, label="Données réelles")
ax1.plot(t, y_fit2, "r-", lw=0.8, label="Modèle ajusté")
ax1.set_ylabel("CO$_2$ (ppm)")
ax1.set_title("Ajustement parabolique avec variation saisonnière")
ax1.legend()
ax2.plot(t, res2, "b-", lw=0.8)
ax2.set_ylabel("Résidus (ppm)")
ax2.set_xlabel("Année")
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
plt.show()


# %% Composante saisonnière sur un an
t1 = np.linspace(0, 1, 366)            # ≈ 0:1/365:1  (366 points, bornes incluses)
y_sin = d * np.cos(2 * np.pi * t1) + e * np.sin(2 * np.pi * t1)
mois = 12 * t1
print(f"Maximum saisonnier vers le mois {mois[np.argmax(y_sin)]:.1f} (0 = 1er janvier)")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(mois, y_sin, "m-")
ax.set_xlabel("Temps (mois)")
ax.set_ylabel("Variation saisonnière (ppm)")
ax.set_title("Composante saisonnière du modèle")
ax.grid(alpha=0.3)
plt.show()

# %% Validation hors échantillon
def rmse(r):
    return np.sqrt(np.mean(r**2))

train = t < 2016                      # apprentissage : jusqu'en 2015
test = ~train                         # test : 2016 → fin  (≈ t >= 2016)

A_train, _ = approx_parab_sin(t[train], y[train])   # _ = sortie ignorée (≈ ~)
w = 2 * np.pi * t
M_all = np.column_stack([t**2, t, np.ones_like(t), np.cos(w), np.sin(w)])
y_hat = M_all @ A_train

print(f"RMSE apprentissage : {rmse(y[train] - y_hat[train]):.3f} ppm")
print(f"RMSE test          : {rmse(y[test] - y_hat[test]):.3f} ppm")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, y, "k-", lw=0.8, label="Données réelles")
ax.plot(t, y_hat, "r-", lw=0.8, label="Modèle (appris < 2016)")
ax.axvline(2016, color="gray", ls="--", label="Début du test")
ax.set_xlabel("Année")
ax.set_ylabel("CO$_2$ (ppm)")
ax.set_title("Validation hors échantillon")
ax.legend()
ax.grid(alpha=0.3)
plt.show()