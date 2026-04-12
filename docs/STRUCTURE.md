# Gap Detection V2 — Structure du papier

## Titre provisoire

**"Missing Attractors: How Variational Regularization Creates Detectable Topological Scars"**

Alternative :
**"The Energy Landscape of Missing Knowledge: Connecting Mixture Models, Variational Inference, and Topological Gap Detection"**

Note : "Hopfield" retiré du titre. Le lien Hopfield est une équivalence fonctionnelle, pas le sujet du papier.

---

## Positionnement

V1 = phénomène empirique ("on observe que...")
V2 = théorie + prédictions + validation élargie ("on dérive que... et on vérifie")

La V2 n'est PAS une réécriture de la V1. C'est un nouveau papier qui :
1. Propose un cadre théorique (énergie de mélange, analogie Hopfield)
2. Dérive des prédictions qualitatives et quantitatives
3. Valide sur plus d'architectures et de datasets
4. Subsume les résultats de V1 comme cas particulier

---

## Décisions structurantes (session 11 avril 2026)

### Hopfield = analogie fonctionnelle, pas colonne vertébrale
- L'énergie du VAE agrégé a la MÊME FORME que l'énergie Modern Hopfield
- On IMPORTE les propriétés des attracteurs, on n'UTILISE PAS de réseau Hopfield
- Formulation correcte : "The induced energy landscape is equivalent in form to modern Hopfield energies, allowing us to interpret mixture modes as attractors."

### Deux mondes
- **Monde A (GMM analytique)** : théorèmes exacts. Retirer une composante = retirer un terme.
- **Monde B (VAE appris)** : hypothèse de transfert. Le VAE réentraîné ≈ GMM tronqué.
- Test de fidélité structurelle : comparer GMM tronqué vs VAE réentraîné (distance modes, persistence diagram similarity).

### Claims calibrés
- α (aspiration) : relation de MONOTONIE, pas loi fonctionnelle. α = f(1/d) + ε.
- PCA > AE < VAE : Hopfield explique UNIQUEMENT l'amplification VAE. PCA = structure préexistante, AE = pas de mécanisme d'amplification.
- Inverted-U beta : qualitatif (faible = pas de bassin, fort = collapse), pas de seuil exact.

### Prédictions falsifiables (3 max)
1. **Monotonie aspiration** : classes plus proches du vide → plus d'aspiration (test de rang Spearman)
2. **Beta optimal prédictible** : le beta maximisant la détection est dérivable de la géométrie latente (profondeur bassins / collapse)
3. **Persistence H1 croissante** : plus de classes retirées → cycle H1 plus persistant

---

## Structure

### 1. Introduction (~1.5 pages)

**Hook** : un modèle ne sait pas ce qu'il ne sait pas. Comment détecter les absences structurelles ?

**Gap in literature** :
- V1 (notre preprint, DOI Zenodo) : phénomène empirique, une seule architecture, pas de prédiction
- Yadav et al. 2025 : trous dans embeddings, post-hoc, sans théorie
- Personne n'a formalisé POURQUOI la régularisation KL amplifie les signatures topologiques

**Contributions** :
1. Formalisation : l'aggregated posterior d'un VAE forme un GMM dont l'énergie a la forme d'un paysage d'attracteurs. Retrait de classe = retrait d'attracteur → déformation prédictible.
2. Prédictions testables : monotonie de l'aspiration, beta optimal, persistence croissante.
3. Validation multi-architecture : VAE conv + VAE MLP (+ optionnel CIFAR-10).
4. Cadre "deux mondes" : ce qui est prouvé (GMM) vs ce qui est vérifié (VAE).

### 2. Background (~1.5 pages)

**2.1 VAE and the ELBO**
- ELBO = reconstruction - β · KL
- Prior holes (littérature existante)
- Dai & Wipf 2019 : diagnostic des modes de défaillance

**2.2 Gaussian Mixture Models and Energy Landscapes**
- GMM comme paysage d'énergie : E(z) = -log p(z)
- Modes = minima d'énergie, bassins de Voronoï
- Propriétés de déformation quand on retire une composante

**2.3 Modern Hopfield Networks (bref, pas central)**
- Ramsauer et al. 2020 : attracteurs continus
- Forme de l'énergie : -logsumexp + confinement
- Remarque : même forme fonctionnelle que notre énergie GMM (Section 3)

**2.4 Topological Data Analysis**
- Persistent homology, Wasserstein distance
- V1 (notre preprint) : résultats empiriques clés (résumé)

### 3. Theory: Mixture Energy and Missing Attractors (~3.5 pages) [SECTION CLÉ]

**3.0 From ELBO to energy landscape** [NOUVELLE — voir theory_section3_0.md]
- Aggregated posterior q_agg(z) = 1/N Σ q(z|x_i)
- Hypothèse H1 : q_agg ≈ GMM à K composantes (vérifiable)
- Énergie E(z) = -log q_agg(z) = -logsumexp + confinement
- Équivalence fonctionnelle avec énergie Modern Hopfield (dérivation complète)
- Ce que ça implique / ce que ça n'implique pas

**3.1 Monde A : GMM analytique — retrait d'un attracteur**
- GMM isotrope à K composantes, prior N(0,I)
- Retirer composante c* : GMM tronqué à K-1 composantes
- Dérivation exacte du déplacement des modes survivants Δ_k
- **Proposition 1 (Aspiration)** : énoncé formel avec hypothèses

**3.2 Conséquences topologiques**
- Retrait d'un attracteur → nouveau cycle H1 (bassin vide)
- Persistence du cycle dépend de la profondeur du bassin (= f(β, σ, π_k))
- **Proposition 2 (Cycle H1)** : restreint au cas 2D bien séparé, ou corollaire numérique
- Prédiction de l'inverted-U qualitativement

**3.3 Prédictions quantitatives**
- P1 : monotonie de l'aspiration en fonction de la distance au vide
- P2 : beta optimal ≈ argmax(profondeur bassins / collapse) — dérivable du GMM
- P3 : persistence H1 croissante avec le nombre de classes retirées

**3.4 Toy model analytique (2D, 3 gaussiennes)**
- Illustration complète des Propositions 1-2 et Prédictions P1-P3
- Figures : paysage d'énergie, aspiration, persistence
- Déjà codé (toy_model.py), à enrichir avec P2 et P3

### 4. Monde B : Du GMM au VAE (~2.5 pages)

**4.1 Hypothèse de transfert**
- H2 : le paysage d'énergie d'un VAE réentraîné sans classe c* est approximativement le GMM tronqué correspondant
- Test de fidélité structurelle : comparer modes, bassins, persistence diagrams
- Métriques : distance entre centroïdes (GMM vs VAE), Wasserstein entre persistence diagrams

**4.2 Validation sur VAE conv (MNIST) — données V1**
- Rappel résultats V1 : gradient PCA > AE < VAE, ghost centroid (r=-0.28)
- Test P1 (monotonie aspiration) : Spearman sur les données existantes
- Test P2 (beta optimal) : comparer prédiction GMM vs beta sweep V1
- Test P3 (persistence vs nb classes) : 5 conditions V1

**4.3 VAE MLP (2e architecture)**
- Architecture : FC 784 → 256 → 16 → 256 → 784
- Mêmes conditions, mêmes seeds, même TDA
- Si P1-P3 tiennent → le phénomène est ELBO-driven, pas architecture-dependent

**4.4 Explication du gradient PCA > AE < VAE** (scope réduit)
- PCA : structure linéaire préexistante (signal faible mais réel)
- AE : compression non-linéaire sans prior → pas de mécanisme d'amplification
- VAE : KL crée des bassins → amplifie
- Le cadre théorique explique l'amplification VAE, pas le signal PCA

### 5. Discussion (~1.5 pages)

**5.1 Le paysage d'énergie comme outil de diagnostic**
- Audit de couverture, détection de drift catégoriel
- Au-delà de "manque/pas manque" : profondeur du bassin = confiance

**5.2 Lien avec Modern Hopfield et Transformers** (remarque, pas claim)
- Ramsauer et al. : attention = rappel Hopfield
- Si les transformers induisent un paysage d'énergie similaire, nos résultats pourraient s'étendre
- Future work, pas claim

**5.3 Limites**
- Toy model 2D gaussien → gap avec haute dimension
- Hypothèse H1 (GMM) : dépend de la séparabilité des classes
- Hypothèse H2 (transfert) : testée empiriquement, pas prouvée
- MNIST/Fashion-MNIST restent simples
- α : monotonie vérifiée, mais r=-0.28 (8% variance) — signal bruité en haute dim

**5.4 De la détection à la reconstruction** (future work)
- Si on connaît la forme du bassin manquant → décoder le centroïde fantôme
- Pipeline : detect → localize → reconstruct

### 6. Conclusion (~0.5 page)

Claim central recalibré :

"The aggregated posterior of a variational autoencoder trained on K separable classes induces an energy landscape with K attractors. Removing a class removes an attractor, creating a topological scar — a detectable, directionally predictable deformation whose magnitude scales monotonically with proximity to the void."

### References (~40-50 refs)

### Appendices
A. Dérivation complète Section 3.0 (ELBO → énergie)
B. Preuves Propositions 1-2
C. Toy model : code et figures supplémentaires
D. Tables détaillées multi-architecture
E. Test de fidélité H2 (GMM vs VAE)

---

## Papiers clés à citer

| Papier | Rôle |
|--------|------|
| Kingma & Welling 2014 | VAE fondation |
| Higgins et al. 2017 | β-VAE |
| Dai & Wipf 2019 | Diagnostic VAE, prior holes |
| Ramsauer et al. 2020 | Modern Hopfield — équivalence fonctionnelle |
| Moor et al. 2020 | Topological Autoencoders |
| Burns & Fukai 2023 | Simplicial Hopfield |
| Van Der Meersch et al. 2023 | Hopfield-VAE (contraster) |
| Fay et al. 2025 | TDA + LLM latents |
| Yadav et al. 2025 | Missing data + PH (contraster) |
| Rigaud 2026 (V1 DOI) | Nos résultats empiriques |

---

## Estimation de longueur

~12-14 pages (format ICLR/NeurIPS) + appendices
Section théorie (3+3.5) = cœur du papier (~3.5 pages)
Section validation = Monde B (~2.5 pages, dont V1 résumée en 0.5)

---

## Cible

La science, pas une deadline. NeurIPS 2026 ou ICLR 2027 si le timing le permet, TMLR sinon.

---

*Structure créée le 25 mars 2026 — révisée le 11 avril 2026 (formalisation corrigée, deux mondes, claims calibrés)*
