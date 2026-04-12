# Claim Audit — Gap Detection V2

**Règle** : si une ligne de ce tableau te gêne ou sonne trop forte, elle doit être réduite avant soumission.

---

## Tableau principal

| # | Claim | Status | Evidence type | Failure condition |
|---|-------|--------|---------------|-------------------|
| H1 | q_agg d'un VAE entraîné sur K classes séparables ≈ GMM-K | **À tester** | Empirique (GMM fit sur latents) | BIC(GMM-K) > BIC(GMM-2K) ou held-out LL du GMM-K < 90% du GMM-N |
| H2 | Le paysage d'un VAE réentraîné sans c* ≈ GMM tronqué | **À tester** | Empirique (comparaison structurelle) | Distance centroïdes GMM vs VAE > 2σ ou Wasserstein PD > seuil |
| F1 | E(z) = -log q_agg a la forme -logsumexp + confinement | **Acquis** | Dérivation analytique | Erreur dans la dérivation (vérifiable) |
| F2 | Cette forme est fonctionnellement équivalente à Modern Hopfield (avec biais) | **Acquis** | Dérivation analytique | Erreur d'identification des termes (vérifiable) |
| F3 | Sous séparation suffisante, chaque μ̄_k est un minimum local de E(z) | **Acquis** | Analytique (propriété standard des GMM) | Classes non séparées → modes fusionnent |
| P1a | L'aspiration pointe vers le vide (direction) | **PASS** | Empirique (ghost centroid V1) | Fraction aspirée ≤ 50% (chance) |
| P1b | L'aspiration est monotone en distance au vide | **PASS** | Empirique (per-seed Spearman sur V1 artifacts) | Spearman ρ ≥ 0 ou p ≥ 0.05 |
| P2a | Le β optimal TDA est prédictible via resolvability des classes survivantes | **FAIL** | ○ test empirique négatif (1 PASS, 1 MARGINAL, 1 FAIL sur 3 conditions) | β* GMM et β* TDA dans des régimes différents |
| P3 | La persistence H1 croît avec le nombre de classes retirées | **MARGINAL** | Empirique (5 conditions V1, ρ=0.35, p~0.08) | Pas de tendance monotone ou tendance inverse |
| T1 | Le phénomène est ELBO-driven, pas architecture-dependent | **À tester** | Empirique (VAE MLP) | VAE MLP ne reproduit pas P1/P3 |
| T2 | PCA montre un signal préexistant faible | **V1 acquis** | Empirique (V1 : 2/5 conditions) | — (déjà mesuré) |
| T3 | AE sans prior ne montre pas d'amplification | **V1 acquis** | Empirique (V1 : 1/5 conditions) | — (déjà mesuré) |
| T4 | Le KL amplifie le signal topologique | **V1 acquis** | Empirique (V1 : 5/5 conditions VAE) | — (déjà mesuré) |

---

## Niveaux d'évidence (à séparer visuellement dans le papier)

| Niveau | Symbole | Signification | Exemples |
|--------|---------|---------------|----------|
| **Démontré** | ◼ | Dérivation analytique, vérifiable | F1, F2, F3, Prop 1 (2D), Prop 2 (2D) |
| **Illustré numériquement** | ◧ | Toy model 2D, calcul exact mais cas simplifié | Aspiration toy model, cycle H1 toy model, inverted-U toy model |
| **Observé empiriquement** | ○ | Mesuré sur VAE réel (MNIST/Fashion-MNIST) | P1a, P1b, P3, T2, T3, T4 (données V1) |
| **Hypothèse de transfert** | △ | Affirmé sous conditions, testé mais pas prouvé | H1, H2, P2, T1 |

**Règle d'écriture** : chaque claim dans le texte doit utiliser le wording correspondant à son niveau :
- ◼ "we show that..." / "it follows that..."
- ◧ "the toy model confirms..." / "numerical computation shows..."
- ○ "we observe..." / "empirical results are consistent with..."
- △ "we hypothesize that..." / "under hypothesis H1/H2..." / "we provide evidence that..."

---

## Opérationnalisation de H1

**Claim** : q_agg ≈ GMM-K

**Protocole de test** :

1. Encoder tout le dataset d'entraînement (ablated) à travers le VAE → obtenir {μ(x_i)} pour tout i
2. Fitter un GMM-K sur ces points (K = nombre de classes présentes)
3. Fitter un GMM-2K et un GMM-K/2 comme contrôles

**Métriques** :
- **BIC** : BIC(GMM-K) < BIC(GMM-2K) ET BIC(GMM-K) < BIC(GMM-K/2)
  → Le bon nombre de composantes est bien K
- **Held-out LL** : split 80/20, mesurer log-likelihood du GMM-K sur le held-out
  → Rapporter la valeur absolue + la comparer au GMM-N (upper bound)
- **ARI** : Adjusted Rand Index entre clusters GMM-K et classes vraies
  → ARI > 0.80 = bon mapping classes-composantes
- **Séparation** : min_{j≠k} ||μ̄_j - μ̄_k|| / σ̄
  → Rapporter la valeur. Si < 2, signaler que l'hypothèse de séparation est marginale.

**Seuils PASS/FAIL** :
- PASS : BIC favorise K, ARI > 0.80, séparation > 2
- MARGINAL : BIC favorise K mais ARI 0.60-0.80 ou séparation 1-2
- FAIL : BIC ne favorise pas K ou ARI < 0.60

---

## Opérationnalisation de H2

**Claim** : VAE réentraîné sans c* ≈ GMM tronqué (GMM complet moins composante c*)

**Protocole de test** :

1. Entraîner VAE complet → fitter GMM-K → obtenir {μ̄_k, σ̄_k}
2. Tronquer le GMM : retirer composante c* → GMM-(K-1) prédit
3. Entraîner VAE sans classe c* → fitter GMM-(K-1) sur ses latents → GMM-(K-1) observé
4. Comparer prédit vs observé

**Métriques** :
- **Distance centroïdes** : pour chaque classe survivante, ||μ̄_k(prédit) - μ̄_k(observé)||
  → Rapporter en unités de σ̄. Si < 1σ = bon, 1-2σ = acceptable, > 2σ = H2 ne tient pas.
- **Wasserstein PD** : distance entre persistence diagrams (sublevel filtration) du GMM prédit vs VAE observé
  → Pas de seuil absolu, mais rapporter la valeur + la comparer au Wasserstein entre le GMM complet et le GMM tronqué (reference scale).
- **Overlap des bassins** : pour chaque mode, comparer le rayon du bassin de Voronoï (prédit vs observé)
  → Corrélation rang entre les rayons prédits et observés.

**Seuils PASS/FAIL** :
- PASS : distance centroïdes < 1σ en moyenne, corrélation bassins > 0.80
- MARGINAL : distance < 2σ, corrélation > 0.60
- FAIL : distance > 2σ ou corrélation < 0.60

**Note** : H2 est le test le plus critique du papier. Si H2 échoue, la théorie GMM reste vraie mais ne se transfère pas au VAE, et le papier devient "théorie + observation empirique séparée" plutôt que "théorie prédictive vérifiée".

---

## Claims à surveiller (tendance à survendre)

| Risque | Ce qu'on pourrait écrire | Ce qu'il faut écrire |
|--------|-------------------------|---------------------|
| P2 concordance grossière | "our theory predicts the optimal β" | "the predicted regime is consistent with the empirically observed optimum" |
| r = -0.28 | "confirming the theoretical prediction" | "consistent with the predicted monotonic relationship; the moderate effect size is expected in 16D" |
| Multi-architecture | "the phenomenon is universal" | "the phenomenon reproduces across two architectures sharing the same ELBO objective" |
| Hopfield | "we establish a Hopfield theory of VAE gaps" | "the induced energy landscape admits a Hopfield interpretation" |
| H1 | "the aggregated posterior IS a GMM" | "the aggregated posterior is well-approximated by a K-component GMM under the conditions tested" |

---

## Résultats acquis

### P1 — Monotonie aspiration (11 avril 2026)

**Status** : PASS
**Source** : re-analysis of archived V1 protocol outputs on M4
**Script** : `perseed_spearman_p1.py` → `p1_perseed_results.json`

**Evidence** :
- Consistent negative rank correlation across seeds (3/3 V1, 5/5 total)
- Per-seed V1 rhos: -0.31, -0.36, -0.07 (mean = -0.25 +/- 0.13)
- Per-seed all rhos: -0.31, -0.36, -0.07, -0.31, -0.13 (mean = -0.24 +/- 0.11)
- Aggregate V1: ρ = -0.2539, p = 8.57e-05, n = 234
- Aggregate all: ρ = -0.2531, p = 4.09e-07, n = 390
- Bootstrap (point-level): -0.2546 +/- 0.0512

**Control** : AE shows no systematic correlation (ρ ≈ 0, p > 0.78)

**Limitation** : one seed (456) non-significant (ρ = -0.07, p = 0.56) but in predicted direction

**Wording for paper** : "consistent negative rank correlation across seeds (mean ρ = -0.25, std = 0.13), with significant aggregate effect (p < 10^-4). Weak per-seed signal under high-dimensional noise, absent in AE control."

### P3 — Persistence vs M (11 avril 2026)

**Status** : MARGINAL
**Source** : re-analysis of V1 TDA results (mnist_vae_results.json)

**Evidence** :
- Order holds: M=1 (89.6) < M=2 (106.7) < M=3 (109.5)
- Spearman ρ = 0.35, p ~ 0.08 (borderline, n=25)
- All above null (55.1)

**Wording for paper** : "topological signal increases with number of removed classes in the expected order (M=1 < 2 < 3), with a positive rank correlation (ρ = 0.35), although significance remains borderline (p ≈ 0.08) given the small number of conditions."

### P2a — Geometric resolvability proxy (11 avril 2026)

**Status** : FAIL (résultat négatif pré-enregistré)
**Source** : archived V1 beta sweep latents on M4
**Script** : `p2_resolvability_score.py` → `p2_resolvability_results.json`
**Metric** : separation / weighted_spread (empirical class-conditional, label-based, no EM)

**Results** :

| Condition | β*_GMM | β*_TDA | Ratio | Verdict |
|-----------|--------|--------|-------|---------|
| minus_cluster | 0.1 | 2.0 | 20.0 | FAIL |
| minus_3_7 | 5.0 | 2.0 | 2.5 | MARGINAL |
| minus_7 | 5.0 | 5.0 | 1.0 | PASS |
| full (diag) | 5.0 | — | — | — (geometric reference only, no TDA optimum) |

**Diagnostic** : le score est quasi-plat de β=0.1 à β=5.0 pour minus_cluster (2.69–2.97) — séparation et spread se compressent ensemble, le ratio reste stable. The score remains nearly flat across β, indicating that the failure is due to lack of sensitivity rather than estimation noise. Le score ne capture pas l'inverted-U car il mesure la résolvabilité des classes *présentes*, pas la géométrie du *vide*.

**Interpretation** : the TDA signal is sensitive to missing-volume geometry, not only to cluster resolvability. A surviving-class proxy is insufficient. This is itself informative — it supports the claim that the topological scar is a proper geometric object, not a byproduct of cluster compactness.

**Wording for paper** : "A pre-registered geometric resolvability score based on surviving-class separation and spread does not reliably predict the TDA optimum across ablation conditions (1 PASS, 1 MARGINAL, 1 FAIL). This suggests that the topological signal captures missing-volume geometry beyond cluster separability — the scar left by a removed attractor is not reducible to the resolution of remaining attractors."

---

*Rédigé le 11 avril 2026 — mis à jour avec résultats P1/P3/P2a*
