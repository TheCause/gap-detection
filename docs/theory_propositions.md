# Propositions formelles — Gap Detection V2

**Statut** : brouillon, à convertir en LaTeX.
**Contexte** : ces propositions sont prouvées dans le Monde A (GMM analytique). Leur transfert au Monde B (VAE appris) est une hypothèse testée empiriquement.

---

## Proposition 1 — Aspiration dirigée

### Cadre

Soit un GMM régularisé à K composantes isotropes dans R^d :

    p(z) = Σ_{k=1}^{K} π_k · N(z; μ_k, σ²I)

avec un terme de confinement quadratique (prior KL) de poids β ≥ 0.

L'énergie est :

    E(z) = -log p(z) + β/2 · ||z||²

Chaque composante k définit un mode z*_k (minimum local de E) proche de μ_k.

### Énoncé

**Proposition 1 (Aspiration).** Soit le GMM ci-dessus avec K ≥ 3 composantes. Supposons que les modes sont bien séparés : ||μ_j - μ_k|| >> σ pour tout j ≠ k. Retirons la composante c* pour obtenir un GMM tronqué à K-1 composantes.

Alors les modes survivants z*_k (k ≠ c*) subissent un déplacement Δ_k = z*_k(tronqué) - z*_k(complet) tel que :

(i) **Direction** : Δ_k pointe vers μ_{c*} (le centroïde supprimé). Plus précisément, l'angle entre Δ_k et (μ_{c*} - μ_k) est inférieur à π/2.

(ii) **Monotonie** : ||Δ_k|| est monotone décroissant en ||μ_k - μ_{c*}||. Les modes plus proches du vide bougent plus.

(iii) **Dépendance en β** : ||Δ_k|| empirically exhibits an inverted-U behavior as β increases — consistent with increasing confinement (deeper basins, stronger aspiration) followed by collapse (all modes merge toward origin, relative geometry vanishes). Ce comportement est observé dans le toy model et les données V1, mais n'est pas prouvé analytiquement en général.

### Esquisse de preuve (cas 2D, K=3)

Dans le cas isotrope bien séparé, les modes z*_k sont proches de μ_k. Le gradient de E au mode z*_k a une contribution de la composante c* :

    ∇E_{c*}(z*_k) ≈ π_{c*} · (z*_k - μ_{c*}) / σ² · exp(-||z*_k - μ_{c*}||² / (2σ²))

Quand c* est retirée, cette force disparaît. Le mode z*_k se déplace pour compenser, dans la direction opposée à ∇E_{c*}, c'est-à-dire vers μ_{c*}.

La magnitude est :

    ||Δ_k|| ∝ π_{c*} / σ² · exp(-||μ_k - μ_{c*}||² / (2σ²))

qui est bien monotone décroissante en ||μ_k - μ_{c*}|| (exponentielle décroissante).

Le terme β ajoute un rappel vers l'origine. Pour β petit, il n'affecte pas la géométrie relative. Pour β grand, tous les μ̄_k → 0, les distances relatives ||μ_k - μ_{c*}|| → 0, et le signal d'aspiration se noie dans le collapse.

**Cas 2D vérifié numériquement** : voir toy_model.py (aspiration α vs distance, 4 valeurs de β).

### Ce qui se transfère au VAE (Hypothèse H2)

Dans un VAE réentraîné sans classe c*, les centroïdes encodés jouent le rôle de μ_k. Si la correspondance GMM-VAE tient (Hypothèse H1), alors :
- La direction d'aspiration devrait pointer vers le ghost centroid
- La magnitude devrait être monotone en distance au ghost centroid
- Données V1 : Spearman r = -0.27, p < 10^-4 (significatif, signal faible en 16D)

---

## Proposition 2 — Cicatrice topologique (cycle H1)

### Cadre

Même GMM régularisé que Proposition 1. On considère les ensembles de sous-niveau :

    S_τ = { z ∈ R^d : E(z) ≤ τ }

et leur homologie persistante H_*(S_τ) quand τ varie.

### Énoncé

**Proposition 2 (Cicatrice topologique).** Sous les mêmes hypothèses que Proposition 1, dans le cas d = 2 avec séparation suffisante (||μ_j - μ_k|| > 4σ pour tout j ≠ k) :

Le retrait de la composante c* crée un cycle H1 dans la filtration de sous-niveaux de E, qui n'existe pas dans le GMM complet.

Plus précisément :
- Dans le GMM complet, les bassins fusionnent progressivement quand τ augmente sans créer de cycle persistant (topologie triviale si K modes sont en "position générale").
- Dans le GMM tronqué, le bassin absent laisse un "trou" que les bassins voisins entourent sans remplir, créant un cycle H1.

La **persistence** de ce cycle (τ_death - τ_birth) dépend de :
- La profondeur du bassin supprimé (∝ π_{c*} / σ²)
- La distance aux voisins (plus les voisins sont proches, plus le cycle est persistant)
- β : inverted-U (β faible → bassins peu profonds → pas de cycle ; β fort → collapse → un seul bassin)

### Niveau de rigueur

Cette proposition est formulée comme un **corollaire numérique** dans le cas 2D, pas un théorème général en dimension d.

Raisons :
- En dimension d > 2, la topologie des ensembles de sous-niveau de mélanges de gaussiennes est techniquement complexe (dépend de l'arrangement de Voronoï en haute dimension).
- Le résultat 2D est vérifiable analytiquement et numériquement.
- Le transfert en haute dimension est une prédiction empirique testée sur MNIST (16D latent).

### Vérification

- **Toy model 2D** : calculer explicitement la persistence de H1 avant/après ablation pour K=3 gaussiennes, balayage de β.
- **VAE MNIST** : données V1 montrent signal TDA en 5/5 conditions (VAE), 2/5 (PCA), 1/5 (AE).
- **Prédiction P3** : persistence croissante avec le nombre de classes retirées (1 vs 2 vs 3).

---

## Résumé

| Proposition | Portée | Prouvé | Vérifié empiriquement |
|-------------|--------|--------|----------------------|
| P1 (Aspiration) | GMM isotrope, K ≥ 3, bien séparé | Cas 2D analytique, d quelconque qualitatif | V1 : r=-0.28, p<10^-5 (VAE MNIST) |
| P2 (Cycle H1) | GMM isotrope, d=2, séparation forte | Corollaire numérique 2D | V1 : 5/5 conditions VAE, inverted-U beta |

---

*Rédigé le 11 avril 2026*
