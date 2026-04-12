# Missing Attractors — Gap Detection V2

Code and artifacts for **"Missing Attractors: An Energy-Landscape View of Detectable Topological Scars"**.

This is the V2 paper. For V1 (Zenodo DOI `10.5281/zenodo.19309874`, "Topological Gaps Exist Before Learning and Are Amplified by Regularization"), see the [`main`](https://github.com/TheCause/gap-detection/tree/main) branch of this repository (tag [`v1-zenodo`](https://github.com/TheCause/gap-detection/releases/tag/v1-zenodo)).

V2 **does not replace** V1. It is a distinct theoretical contribution that:

1. Formalises the aggregated posterior of a VAE as a GMM whose energy has the form of a Modern Hopfield attractor landscape.
2. Derives three falsifiable predictions (aspiration monotonicity, β-optimal, persistence growth).
3. Re-analyses V1 archived latents per-seed (P1 Spearman, P2 resolvability) to validate those predictions.

## Contents

```
.
├── README.md                  — this file
├── ABSTRACT.txt               — paper abstract
├── LICENSE                    — CC-BY-4.0
├── requirements.txt           — Python dependencies
├── toy_model.py               — 2D GMM toy model (Figures 1–2)
├── perseed_spearman_p1.py     — P1 reanalysis: Spearman monotonicity per seed
├── p2_resolvability_score.py  — P2 reanalysis: resolvability score per seed
├── search_beta_latents.py     — helper: locate β-sweep latents in V1 artifacts
├── search_gap_latents.py      — helper: locate ablation latents in V1 artifacts
├── results/
│   ├── p1_perseed_results.json       — pre-computed P1 results
│   └── p2_resolvability_results.json — pre-computed P2 results
├── arxiv_source/              — LaTeX sources (main.tex, bib, figures, compiled PDF)
└── docs/                      — design notes, theory propositions, audits
    ├── theory_propositions.md
    ├── theory_section3_0.md
    ├── predictions_falsifiables.md
    ├── claim_audit.md
    ├── p2_metric_definition.md
    └── STRUCTURE.md
```

## Reproducing Figures

```bash
pip install -r requirements.txt
python toy_model.py
```

Outputs:
- `figures/toy_model_energy.png` — energy landscape before/after ablation (Figure 1)
- `figures/toy_model_aspiration.png` — aspiration vs distance to ghost centroid (Figure 2)

## Reproducing Empirical Results (P1, P2)

The empirical results re-use V1 archived latents (see Zenodo DOI above). Pre-computed
per-seed results are in `results/`:

- `p1_perseed_results.json` — output of `perseed_spearman_p1.py`
- `p2_resolvability_results.json` — output of `p2_resolvability_score.py`

To re-run from V1 latents:

```bash
# After retrieving V1 latents from Zenodo and locating them with the search_*.py helpers:
python perseed_spearman_p1.py
python p2_resolvability_score.py
```

## Hardware

- Toy model: any machine with Python 3.8+.
- Empirical re-analysis (P1, P2): Apple M4 (CPU).

## License

CC-BY-4.0
