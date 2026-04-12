# P2 — Geometric Resolvability Score

**Statut** : métrique définie AVANT de regarder les courbes.
**But** : prédire le β optimal pour la détection topologique à partir de la géométrie du GMM fitté.

---

## Définitions

**Séparation** = min_{j≠k} ||μ̂_j - μ̂_k||

Distance minimale entre centroïdes du GMM-K fitté sur les latents VAE (condition "full").

**Spread** = (1/K) Σ_k σ̂_k

Std effective moyenne des composantes, avec σ̂_k = sqrt(trace(Σ̂_k) / d) où d = dimension latente.

**Geometric resolvability score** = séparation / spread

Ratio sans dimension. Monte quand les modes sont compacts et bien séparés, descend quand ils se chevauchent (β faible, spread grand) ou collapsent (β fort, séparation chute).

**β\*_GMM** = argmax_β (resolvability score)

Le β qui maximise la résolvabilité géométrique du GMM.

---

## Protocole

**Conditionné par gap** (Option B — compare des objets cohérents).

Pour chaque condition C ∈ {minus_7, minus_3_7, minus_cluster} :
  Pour chaque β ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0} :
    a. Encoder le dataset ablaté (classes restantes) à travers le VAE entraîné à ce β sous condition C
    b. Fitter un GMM-(K-M) sur les latents (M = nb classes retirées)
    c. Calculer séparation, spread, score

  Identifier β*_GMM(C) = argmax du score pour cette condition
  Comparer à β*_TDA(C) = argmax du Wasserstein H1 (V1 beta sweep)

**Pourquoi conditionné** : le β*_TDA dépend de la condition (minus_7 → β*=5.0, minus_cluster → β*=2.0). Un score GMM sur le full ignorerait cette dépendance. Fitter sur l'ablaté capture la résolvabilité de la structure résiduelle, qui est ce que le TDA mesure.

---

## β*_TDA observé (V1 beta sweep)

| Condition | β=0.1 | β=0.5 | β=1.0 | β=2.0 | β=5.0 | β=10.0 | β*_TDA |
|-----------|-------|-------|-------|-------|-------|--------|--------|
| minus_cluster | 74.1 | 89.4 | 105.0 | **118.1** | 54.4 | 30.9 | **2.0** |
| minus_3_7 | 61.6 | 66.1 | 95.2 | **125.7** | 36.4 | 19.5 | **2.0** |
| minus_7 | 61.3 | 63.0 | 56.8 | 50.4 | **83.1** | 18.5 | **5.0** |

Note : minus_7 (1 seule classe retirée) a son pic à β=5.0, pas β=2.0. Signal plus faible → besoin de plus de régularisation.

---

## Critère PASS/FAIL

- **PASS** : β*_GMM dans un facteur 2× de β*_TDA (pour les conditions multi-classes)
- **MARGINAL** : β*_GMM dans un facteur 3× ou dans le même régime qualitatif
- **FAIL** : β*_GMM et β*_TDA dans des régimes différents

---

## Variante robuste (appendice, pas au centre)

- Médiane des distances inter-centroïdes au lieu de min
- 10e percentile inter-centroïdes

---

*Défini le 11 avril 2026, AVANT calcul du score GMM.*
