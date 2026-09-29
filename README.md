# Solar Grand Minima — HMM Pipeline (Supplementary Material)

## What this is

Supplementary material for a study of Grand Solar Minima detected from the
annual number of sunspot groups, denoted **GN** (Vaquero et al. 2016,
*Solar Physics*, based on Hoyt & Schatten 1998). One self-contained Jupyter
notebook, [`solar_grand_minima_hmm.ipynb`](solar_grand_minima_hmm.ipynb),
reproduces the full analysis end to end, starting only from the raw daily
observation file
[`GNobservations_JV_V1-12.csv`](GNobservations_JV_V1-12.csv) included in
this repository:

- Preprocessing and annual aggregation of the raw daily observations.
- A 2-state Gaussian Hidden Markov Model (proof of concept).
- A 3-state Gaussian HMM with BIC-based model selection, Shannon-entropy
  analysis, and independent validation — the model used in the paper.
- A battery of robustness checks: sensitivity to the coverage-gap
  treatment, posterior (soft) state probabilities, sensitivity of the
  Grand-Minimum boundary to the classification threshold, a
  bootstrap-calibrated likelihood-ratio test comparing 3 vs. 4 states,
  within-state independence and missingness diagnostics, and a check of
  how plausible the longest observed episode is under the fitted model's
  own dynamics.
- Every figure and results table referenced in the manuscript, plus
  several supplementary diagnostic figures not in the main text.

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`
- The raw file `GNobservations_JV_V1-12.csv` must sit in the same directory
  as the notebook — paths are resolved via `Path.cwd()`, so the notebook
  must be launched with this repository's root as the working directory.

## How to run

Cells run top-to-bottom exactly once — there is no hidden cross-cell state.
Restart-and-run-all is the supported way to execute it:

```bash
jupyter nbconvert --to notebook --execute --inplace solar_grand_minima_hmm.ipynb
```

or open it in Jupyter Lab / VS Code and choose "Run All Cells".

## Runtime

Verified end-to-end run time: **~6 minutes** on a standard laptop CPU. The
dominant costs are the 3-state model-selection step (3 candidate model
sizes × 50 random restarts × up to 200 EM iterations each) and the
bootstrap-calibrated likelihood-ratio test (200 simulated replicates × 2
models × 10 restarts each, ~4–5 minutes on its own); every other section
runs in a couple of seconds.

## Outputs

Everything the notebook produces is written under `./outputs/` (`data/`,
`params/`, `figures/`), using relative paths only, and is included in this
repository pre-generated so the results and figures can be inspected
without running anything:

- `outputs/data/annual_series.csv`, `coverage.csv` — preprocessed annual
  series and observation-coverage table.
- `outputs/data/bic_table.csv` — BIC model-selection table (n = 2, 3, 4
  states).
- `outputs/data/results_q2.csv` — annual series with the decoded 3-state
  HMM labels.
- `outputs/params/hmm2_params.txt`, `hmm3_params.txt` — full parameter
  reports (emission parameters, transition matrices, entropies,
  Viterbi-decoded state intervals) for the 2-state and 3-state models.
- `outputs/figures/fig01`–`fig09` — all figures referenced in the
  manuscript:
  - `fig01` — dataset overview (GN and active fraction vs. year, plus
    observational coverage).
  - `fig02` — 3-state Viterbi sequence with state shading and the smoothed
    posterior probability of the grand-minimum state.
  - `fig03` — transition-matrix heatmap and Shannon entropy per state.
  - `fig04` — Maunder Minimum onset zoom (Eddy 1976 vs. HMM-detected
    onset), with posterior-probability overlay.
  - `fig05` — effect of the `log(1+GN)` transform on skewness.
  - `fig06` — fitted state-dependent Gaussian densities vs. the observed
    histogram (goodness of fit).
  - `fig07` — independent validation: active fraction vs. GN, colored by
    state.
  - `fig08` — bootstrap-calibrated likelihood-ratio test (3 vs. 4 states).
  - `fig09` — plausibility of the observed episode length under the fitted
    model's own simulated dynamics.
  - `figure_rnaas`, `inspection_plot` — supplementary/exploratory figures
    (2-state proof-of-concept figure and a plain diagnostic plot,
    respectively; not manuscript figures).
  - `restart_stability`, `model_selection_summary`,
    `missingness_diagnostic` — additional robustness-check figures
    referenced in the notebook's markdown discussion but not included as
    numbered manuscript figures.

Re-running the notebook regenerates all of the above in place; the values
were verified to reproduce byte-for-byte-equivalent tables (matching to
6-decimal precision) and the same winning random-restart seeds (24 for the
2-state model, 6 for the 3-state model) across independent runs.

## Reproducibility notes

- All random seeds are fixed and the full 50-restart search is genuinely
  re-run each time (nothing is hard-coded to the known-good result) — see
  the notebook's Setup and Section 3/4 markdown cells for details.
- Model selection for the 3-state model is a **deliberate departure from
  the raw BIC optimum** (which favors 4 states), justified on physical and
  numerical-stability grounds. A bootstrap-calibrated likelihood-ratio test
  (`fig08`) confirms the 4-state model's likelihood gain is real (not
  sampling noise), so the departure is framed as a preference for
  interpretability over a genuine — but not physically meaningful —
  statistical improvement, not as evidence that the 4-state model is
  spurious.
- The primary analysis trains on the 339 valid years as one concatenated
  sequence. A documented sensitivity check refits the model treating each
  of the 23 contiguous coverage blocks as an independent sequence
  (`lengths=` in `hmmlearn`); the resulting State-C interval structure is
  unchanged except for two additional isolated years.
- `hmmlearn`/`numpy`/`scipy` version drift can, in principle, shift which
  restart wins the multi-start search; the notebook warns (never raises) if
  installed versions differ from those pinned in `requirements.txt`.
