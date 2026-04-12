# Section 3.0 — From ELBO to Energy Landscape

**Statut** : brouillon brut, pas encore LaTeX.
**But** : dériver formellement que le paysage latent d'un VAE entraîné a la forme d'une énergie de mélange, fonctionnellement équivalente à une énergie Modern Hopfield.

---

## 3.0.1 — Aggregated posterior

Un VAE entraîne un encodeur q(z|x) = N(z; μ(x), σ²(x)I) et un décodeur p(x|z).

Étant donné un dataset D = {x_1, ..., x_N}, la **posterior agrégée** est :

    q_agg(z) = (1/N) Σ_{i=1}^{N} q(z|x_i)

C'est la distribution marginale de z induite par le dataset passé à travers l'encodeur. C'est ce qu'on observe si on échantillonne un x uniformément dans D puis on encode.

**Point clé** : q_agg(z) est un mélange de gaussiennes à N composantes (une par datapoint). En pratique, les N composantes ne sont pas toutes distinctes — les points d'une même classe se regroupent.

## 3.0.2 — Approximation par classe

Si D contient K classes avec N_k points chacune, et si les encodages intra-classe sont suffisamment concentrés (hypothèse raisonnable pour un VAE bien entraîné sur des données séparables), alors :

    q_agg(z) ≈ Σ_{k=1}^{K} π_k · N(z; μ̄_k, Σ̄_k)

où :
- π_k = N_k / N (proportion de la classe k)
- μ̄_k = (1/N_k) Σ_{i ∈ class k} μ(x_i)  (centroïde encodé de la classe k)
- Σ̄_k ≈ σ̄_k² I  (variance moyenne intra-classe, approx isotrope)

C'est un **Gaussian Mixture Model à K composantes**.

**Hypothèse H1** : Pour un VAE suffisamment bien entraîné sur des données à K classes séparables, q_agg est bien approximé par un GMM à K composantes.

*Vérifiable empiriquement* : fitter un GMM-K sur les z encodés, comparer la log-likelihood vs un GMM à N composantes. Si le GMM-K explique >95% de la variance, H1 tient.

## 3.0.3 — Paysage d'énergie induit

Définissons l'énergie comme la log-densité négative de la posterior agrégée :

    E(z) = -log q_agg(z)

En substituant l'approximation GMM (cas isotrope Σ̄_k = σ² I pour simplifier) :

    E(z) = -log [ Σ_{k=1}^{K} π_k · (2πσ²)^{-d/2} · exp(-||z - μ̄_k||² / (2σ²)) ]

En factorisant la constante de normalisation :

    E(z) = d/2 · log(2πσ²) - log Σ_{k=1}^{K} π_k · exp(-||z - μ̄_k||² / (2σ²))

Le terme intéressant est le **logsumexp négatif** :

    E(z) = const - logsumexp_k [ log π_k - ||z - μ̄_k||² / (2σ²) ]

C'est une énergie avec :
- Un **minimum local** (attracteur) près de chaque μ̄_k
- La **profondeur** du bassin de μ̄_k dépend de π_k et σ²
- La **largeur** du bassin dépend de σ²

## 3.0.4 — Effet de la régularisation KL

L'entraînement du VAE minimise :

    L = E_q[log p(x|z)] - β · KL(q(z|x) || p(z))

Le terme KL pousse q(z|x) vers le prior p(z) = N(0, I). Effet sur q_agg :
- Les centroïdes μ̄_k sont tirés vers l'origine
- Les variances σ̄_k² sont tirées vers 1
- Plus β est grand, plus les classes sont compactées vers l'origine

Dans le paysage d'énergie, cela induit un **effet de confinement approximativement quadratique**. L'énergie totale "vue" par un point z est approximativement :

    E_eff(z) ≈ -logsumexp_k [ log π_k - ||z - μ̄_k||² / (2σ²) ] + β/2 · ||z||²

**Attention** : le terme β/2 · ||z||² n'est PAS ajouté littéralement à E(z). Il modélise l'effet indirect du KL sur la géométrie de q_agg : le KL comprime les centroïdes vers l'origine et les variances vers 1, ce qui produit un confinement effectif *approximativement* quadratique près de l'origine.

This approximation holds near convergence under isotropic priors and separable clusters. Le vrai effet du KL est indirect (via les poids du réseau). L'approximation se dégrade si les classes se chevauchent fortement ou si le VAE est loin de son optimum.

## 3.0.5 — Équivalence fonctionnelle avec Modern Hopfield

L'énergie Modern Hopfield (Ramsauer et al. 2020) pour des patterns stockés {ξ_1, ..., ξ_K} est :

    E_MH(z) = -logsumexp_k [ β_H · ξ_k^T z ] + β_H/2 · ||z||² + const

Notre énergie VAE agrégée a la forme :

    E_eff(z) = -logsumexp_k [ log π_k - ||z - μ̄_k||² / (2σ²) ] + β/2 · ||z||²

En développant ||z - μ̄_k||² = ||z||² - 2z^T μ̄_k + ||μ̄_k||² :

    E_eff(z) = -logsumexp_k [ log π_k - ||z||²/(2σ²) + z^T μ̄_k / σ² - ||μ̄_k||²/(2σ²) ] + β/2 · ||z||²

Le terme -||z||²/(2σ²) est commun à toutes les composantes et sort du logsumexp :

    E_eff(z) = ||z||²/(2σ²) - logsumexp_k [ log π_k + z^T μ̄_k / σ² - ||μ̄_k||²/(2σ²) ] + β/2 · ||z||²

    E_eff(z) = -logsumexp_k [ log π_k + μ̄_k^T z / σ² - ||μ̄_k||²/(2σ²) ] + (1/(2σ²) + β/2) · ||z||²

En identifiant :
- **Patterns** : ξ_k = μ̄_k / σ² (centroïdes encodés, normalisés par la variance)
- **Inverse température** : β_H = 1 (absorbé dans ξ_k)
- **Biais** : b_k = log π_k - ||μ̄_k||²/(2σ²) (combine fréquence de classe et norme du centroïde)
- **Confinement** : (1/(2σ²) + β/2) · ||z||² (combine dispersion intra-classe et régularisation KL)

**L'énergie du VAE agrégé est équivalente à une énergie Modern Hopfield biaisée, à transformations affines et biais par composante près.**

Under sufficient separation (||μ̄_j - μ̄_k|| >> σ for all j ≠ k), each component mean μ̄_k corresponds to a local minimum of E(z). Les propriétés des attracteurs Hopfield (bassins, convergence) s'importent donc dans ce régime.

## 3.0.6 — Ce que cela implique (et ce que cela N'implique PAS)

**Ce que ça implique :**
- Chaque classe apprise est un attracteur dans le paysage d'énergie latent
- La profondeur du bassin dépend de π_k (fréquence), σ² (dispersion), et β (régularisation)
- Retirer une classe du GMM = retirer un attracteur → les propriétés Hopfield prédisent la déformation

**Ce que ça N'implique PAS :**
- On n'utilise pas de réseau de Hopfield. On observe que le VAE *induit* un paysage d'énergie de même forme.
- Le VAE n'est pas "un" Hopfield. L'analogie porte sur la forme de l'énergie agrégée, pas sur la dynamique d'apprentissage.
- L'approximation repose sur H1 (GMM à K composantes). Si les classes ne sont pas séparables, l'approximation se dégrade.

---

## Résumé en une phrase

> L'aggregated posterior d'un VAE entraîné sur K classes séparables induit un paysage d'énergie fonctionnellement équivalent à une énergie Modern Hopfield à K attracteurs, ce qui permet d'importer les propriétés de déformation des attracteurs pour prédire les signatures topologiques des classes manquantes.
