# CIRN Supplementary and Reproducibility Materials

## Overview

This archive contains the supplementary information, analysis data, complete analytical outputs, figures, diagnostics, configuration records, and software-environment information supporting the study:

**Classification-Based Inference of Regulatory Networks (CIRN): Bayesian Logistic Regression for Directional and Signed Regulatory Reconstruction from Time-Series Data**

CIRN is an R-based framework for generating directed, signed, temporally ordered, and uncertainty-aware regulatory hypotheses from single or multivariate time-series data. This archive complements the manuscript, the four-part Supplementary Information, the script-based CIRN implementation, and CIRN Studio.

## Current access status

This Open Science Framework project is maintained privately during dissertation-committee and institutional review. Access may be provided to authorized reviewers through OSF contributor permissions or a controlled view-only link.

The active CIRN source code and CIRN Studio files are maintained at:

https://github.com/gioentero/CIRN

The interactive CIRN Studio application is available at:

https://jfrabajanteuplb.shinyapps.io/CIRN/

A versioned public archival record and persistent identifier will be prepared after the required review and clearance procedures are completed.

## Archive organization

| Component | Contents |
|---|---|
| `00_README_and_Manifest` | Root documentation, archive manifest, provenance notes, and redistribution guidance |
| `01_BIB_Supplementary_Information` | Four-part BIB Supplementary Information and the complete CIRN user manual |
| `02_Classical_Lotka_Volterra` | Deterministic classical Lotka–Volterra analysis and complete CIRN outputs |
| `03_Predator_Prey_With_Lags` | Lag-dependent predator–prey analysis and supporting outputs |
| `04_Stochastic_Predator_Prey` | Stochastic predator–prey analysis with demographic noise |
| `05_Basic_Network` | Representative two-variable and basic dynamical-system analyses |
| `06_Repressilator` | Synthetic repressilator analysis |
| `07_GeneNetWeaver_Ecoli_Network` | GeneNetWeaver five-node *Escherichia coli* network analysis |
| `08_Cardiovascular_Disease_Air_Pollutants` | Observational cardiovascular disease occurrence and air-pollution analysis |
| `09_Damped_Oscillator` | Damped harmonic oscillator analysis |

The analytical result folders are archived substantially as generated during the study. Their existing filenames and internal organization are retained to preserve consistency with the original CIRN outputs and manuscript-development records.

## Relationship to the manuscript

The four formal journal supplementary files are stored under `01_BIB_Supplementary_Information`:

1. Part 1 of 4 — Theory, Methods, and Diagnostics
2. Part 2 of 4 — Implementation and Practical Workflow
3. Part 3 of 4 — CIRN Studio Shiny App
4. Part 4 of 4 — Detailed Results and References

The remaining components provide the full result folders supporting the figures, tables, diagnostics, sensitivity analyses, comparator summaries, and numerical claims reported in the manuscript and Supplementary Information.

## Reproducibility notes

Each analysis component may contain some or all of the following:

- analysis-ready input data;
- CIRN result compendia in Excel format;
- machine-readable CSV exports;
- publication-oriented tables;
- source-quality figures;
- posterior and MCMC diagnostics;
- class-balance and data-sufficiency summaries;
- collinearity diagnostics;
- cross-mode consistency outputs;
- sensitivity or validation results;
- comparator-method outputs;
- `sessionInfo_CIRN.txt`;
- archived analysis settings; and
- supporting notes or documentation.

The complete CIRN configuration is ordinarily recorded in the `settings` sheet of the corresponding result compendium.

## Interpretation boundary

A retained CIRN edge is a posterior-supported temporal-response regulatory hypothesis under the declared data, preprocessing, lag, predictor library, inference mode, prior, response construction, and diagnostic context.

CIRN results do not, by themselves, establish interventional causality, direct mechanism, complete governing-equation recovery, or unconditional regulatory structure.

## Data provenance and redistribution

Simulated datasets generated for this study may be redistributed with their generation procedures and parameter records.

Third-party benchmark or observational datasets must be interpreted and redistributed according to their original source terms, licenses, data-use conditions, and citation requirements. Where redistribution is not permitted or remains uncertain, the archive should provide the source citation, access location, retrieval information, and preprocessing procedure without redistributing restricted raw data.

See `DATA_PROVENANCE_AND_LICENSES.md` for the current review record.

## Authors

1. Giovannie M. Entero
2. Jomar F. Rabajante
3. Neil Jerome A. Egarguin
4. Mark Jayson V. Cortez
5. Maica Krizna A. Gavina
6. Patricia Ann J. Sanchez

## Contacts

- Giovannie M. Entero: gmentero@up.edu.ph
- Jomar F. Rabajante: jfrabajante@up.edu.ph

## Citation status

The formal article citation, versioned software citation, archived research-record citation, and persistent identifiers will be added when the first approved public release is finalized.

Until then, refer to this archive as:

> Entero, G. M., Rabajante, J. F., Egarguin, N. J. A., Cortez, M. J. V., Gavina, M. K. A., and Sanchez, P. A. J. (2026). *CIRN Supplementary and Reproducibility Materials*. Private Open Science Framework project.

## Version and review record

- Archive status: Private committee-review version
- Software status: Pre-release
- Repository: https://github.com/gioentero/CIRN
- Date prepared: 2026-07-30
- Public release DOI: Pending
