# Prédictions falsifiables — Gap Detection V2

**Statut** : formulées, à tester expérimentalement.
**Principe** : chaque prédiction découle de la théorie (Monde A, GMM analytique) et est testable sur le VAE réel (Monde B).

---

## P1 — Monotonie de l'aspiration

### Prédiction

Les classes survivantes plus proches du vide (classe retirée) subissent une aspiration plus forte que les classes éloignées.

### Dérivation théorique

Proposition 1 : ||Δ_k|| ∝ exp(-||μ_k - μ_{c*}||² / (2σ²)), qui est strictement décroissante en distance.

### Test

- **Données** : ghost centroid analysis de V1 (5 conditions × 3 seeds × ~8 centroids = ~234 points)
- **Métrique** : corrélation de Spearman entre distance au ghost centroid et magnitude d'aspiration α
- **Seuil de rejet** : Spearman ρ > 0 (pas de monotonie → théorie rejetée)
- **Résultat attendu** : ρ < 0, significatif (p < 0.01)
- **Résultat V1 existant** : Pearson r = -0.28, p < 10^-5. Spearman à calculer.
- **Choix de Spearman** : invariant aux transformations monotones — teste la monotonie sans supposer une forme fonctionnelle (contrairement à Pearson qui suppose linéarité).

### Critère PASS/FAIL

- **PASS** : Spearman ρ < -0.15, p < 0.01 (monotonie faible mais significative)
- **FAIL** : ρ ≥ 0 ou p ≥ 0.05

### Note

Le r = -0.28 (8% variance expliquée) est attendu en haute dimension. La théorie prédit un signal bruité :

    α = f(1/d) + ε

où ε vient de : overlap entre classes, variance d'initialisation, approximation Procrustes, haute dimensionnalité (16D → dispersion).

Estimer Var(ε) apporterait un niveau supplémentaire. Approche : bootstrap sur les 3 seeds, mesurer la variance inter-seeds de α par condition.

---

## P2 — Beta optimal prédictible depuis la géométrie latente

### Prédiction

Le β qui maximise la détection topologique (Wasserstein distance between ablated/full persistence diagrams) est prédictible à partir de la géométrie du GMM ajusté sur l'espace latent.

### Dérivation théorique

Dans le toy model GMM :
- β petit → bassins peu profonds → cycle H1 non persistant → faible signal TDA
- β grand → collapse (tous les modes fusionnent) → un seul bassin → pas de cycle
- β optimal ≈ argmax_β [ profondeur_bassin(β) · séparation_modes(β) ]

**Définitions explicites** (pour éviter la critique "unclear metric, post-hoc fitting") :
- **Séparation** = min_{j≠k} ||μ̂_j - μ̂_k||, distance inter-centroïdes minimale du GMM fitté.
- **Spread** = (1/K) Σ_k σ̂_k, avec σ̂_k = sqrt(trace(Σ̂_k)/d), std effective moyenne des composantes.
- **Geometric resolvability score** = séparation / spread (sans dimension).

Pour le GMM isotrope :
- Spread dépend de σ̂_k (varie avec β : β grand → σ̂ petit mais collapse)
- Séparation décroît avec β (KL compresse les centroïdes vers l'origine)
- Le ratio séparation / spread a un maximum à un β* intermédiaire

### Test

1. Entraîner des VAE avec β ∈ {0.1, 0.5, 1, 2, 5, 10} (V1 a déjà β sweep)
2. Pour chaque β, fitter un GMM-K sur les encodages latents
3. Calculer : profondeur_bassin × séparation_modes pour le GMM fitté
4. Calculer : signal TDA (Wasserstein) pour chaque β
5. Comparer : le β qui maximise le produit GMM vs le β qui maximise le signal TDA

### Critère PASS/FAIL

- **PASS** : le β optimal GMM est dans un facteur 2× du β optimal TDA (même ordre de grandeur)
- **FAIL** : les deux optima sont dans des régimes différents (β_GMM < 1 et β_TDA > 5, ou inverse)

### C'est la killer prediction

Si ça marche, le papier passe de "on observe un inverted-U" à "on PRÉDIT le pic de l'inverted-U à partir de la géométrie". C'est le saut qualitatif.

---

## P3 — Persistence croissante avec le nombre de classes retirées

### Prédiction

Retirer plus de classes crée un signal TDA plus fort (cycles H1 plus persistants ou plus nombreux).

### Dérivation théorique

Chaque classe retirée = un bassin supprimé = un nouveau cycle H1 potentiel. Plus de bassins supprimés → plus de volume libéré → déformation plus forte.

Quantitativement : si on retire M classes, la Wasserstein distance entre les persistence diagrams (full vs ablated) devrait croître avec M.

### Test

- **Données V1** : conditions minus_7 (1 classe), minus_3_7 et minus_2_4 et minus_4_9 (2 classes), minus_cluster (3 classes)
- **Métrique** : Wasserstein distance en H1 entre persistence diagrams (ablated vs full reference)
- **Test** : corrélation rang entre M (nombre de classes retirées) et Wasserstein

### Critère PASS/FAIL

- **PASS** : corrélation positive significative (les conditions 2-classes > 1-classe, 3-classes > 2-classes)
- **FAIL** : pas de tendance ou tendance inverse

### Note

Le V1 a les résultats pour 5 conditions mais le test monotonie en M n'a pas été fait explicitement. C'est une ré-analyse des données existantes — pas besoin de nouvelle expérience.

---

## Résumé

| Prédiction | Source théorique | Données requises | Nouvelle expérience ? |
|-----------|-----------------|-----------------|----------------------|
| P1 Monotonie aspiration | Proposition 1 | V1 ghost centroid | Non (ré-analyse) |
| P2 Beta optimal | Toy model inverted-U | V1 beta sweep + GMM fit | Partiellement (GMM fit) |
| P3 Persistence vs M | Proposition 2 (extension) | V1 TDA results | Non (ré-analyse) |

**Note stratégique** : P1 et P3 sont testables immédiatement sur les données V1. P2 nécessite un petit travail supplémentaire (fitter des GMM). Si P1 et P3 passent sur V1, on lance P2 + multi-architecture.

---

*Rédigé le 11 avril 2026*
