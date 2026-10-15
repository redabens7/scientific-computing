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