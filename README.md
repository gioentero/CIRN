# Classification-Based Inference of Regulatory Networks (CIRN)

### Bayesian logistic regression for directional and signed regulatory reconstruction from time-series data

![R](https://img.shields.io/badge/R-%E2%89%A54.3-276DC3?logo=r&logoColor=white)
![Shiny](https://img.shields.io/badge/Shiny-interactive-0099F7?logo=posit&logoColor=white)
![Bayesian inference](https://img.shields.io/badge/Bayesian-brms%20%2B%20CmdStan-6C5CE7)
![Project status](https://img.shields.io/badge/status-research%20software-orange)

**CIRN** is an R-based framework for generating **directed, signed, temporally ordered, and uncertainty-aware regulatory hypotheses** from single or multivariate time-series data. It reformulates regulatory reconstruction as a target-wise Bayesian classification problem: rather than estimating the full magnitude or functional form of a dynamical system, CIRN models whether each target is locally **increasing or decreasing** from earlier state-, first-derivative-, and second-derivative-based predictor representations.

This repository provides two complementary implementations:

- a fully configurable **script-based R workflow** for reproducible and publication-oriented analyses; and
- **CIRN Studio**, a guided R Shiny application designed to make the same core methodology more accessible to specialists and non-specialists.

> [!IMPORTANT]
> A retained CIRN edge is a posterior-supported temporal-response regulatory hypothesis under the declared data, preprocessing, lag, predictor library, inference mode, prior, response construction, and diagnostic context. It does **not**, by itself, establish interventional causality, a direct biochemical or physical mechanism, or complete governing-equation recovery.

---

## Why CIRN?

Many network-reconstruction approaches identify dependence, predictive direction, or dynamical coupling, but these outputs do not always provide regulatory direction, sign, uncertainty, and representation-level provenance in one interpretable workflow. CIRN is designed for the intermediate setting between static association screening and full mechanistic equation discovery.

### Core capabilities

- **Directed and signed inference** through temporally lagged predictor-to-target-response models
- **State, velocity, and curvature representations** from smoothed trajectories and their first and second derivatives
- **Bayesian uncertainty quantification** using `brms`, `cmdstanr`, and Stan
- **Posterior edge retention** when a coefficient's 95% highest-density interval excludes zero
- **Multiple inferential resolutions**
  - pairwise: marginal temporal-response evidence
  - sublevel: representation-specific multivariable evidence
  - all-predictors: joint conditional evidence
- **Response-leakage prevention** by excluding the target first derivative that defines the binary response from its own predictor matrix
- **Adaptive response jitter** for one-class or nearly one-class target responses, with explicit diagnostic and reporting metadata
- **Model diagnostics** for MCMC behavior, class balance, usable sample size, collinearity, rank deficiency, aliasing, and predictor correlations
- **Optional sensitivity analysis** for lag, noise, downsampling, trajectory length, and missingness
- **Optional signed ground-truth benchmarking** for simulated or curated benchmark systems
- **Publication-oriented exports**, reproducibility records, figures, and result compendia

---

## Method at a glance

For target variable $x_i$, CIRN defines the directional response from the sign of its unstandardized estimated first derivative:

$$
Y_i(t_k)=
\begin{cases}
1, & x_i'(t_k)>\varepsilon,\\
0, & x_i'(t_k)<-\varepsilon,
\end{cases}
$$

while observations satisfying $|x_i'(t_k)|\leq\varepsilon$ are removed as near-stationary or numerically ambiguous.

CIRN then:

1. smooths the selected trajectories;
2. estimates state, first-derivative, and second-derivative representations;
3. aligns predictors at $t_{k-\ell}$ with the target response at $t_k$;
4. excludes response-defining quantities to prevent leakage;
5. standardizes retained predictors after lagging and filtering;
6. fits target-wise Bayesian Bernoulli-logit models;
7. retains coefficients whose 95% HDI excludes zero; and
8. qualifies retained edges using diagnostics, cross-mode consistency, sensitivity analysis, and domain knowledge.

A positive posterior mean coefficient indicates that a larger lagged predictor value supports a higher posterior probability that the target is increasing. A negative coefficient indicates support for a lower posterior probability of target increase. These signs are defined on the **target-response scale**.

---

## Repository structure

| Path | Contents |
|---|---|
| [`CIRN_Script_Based/`](./CIRN_Script_Based/) | Standalone R implementation, script-specific documentation, supporting files, and distribution archive |
| [`CIRN_Shiny_App/`](./CIRN_Shiny_App/) | CIRN Studio, the core CIRN algorithm used by the app, bundled data, interface assets, documentation, and distribution archive |
| [`Sample_Results_Script_Based/`](./Sample_Results_Script_Based/) | Example outputs generated through the scripted workflow |
| [`Sample_Results_Shiny_App/`](./Sample_Results_Shiny_App/) | Example outputs generated through CIRN Studio |

Each implementation folder contains its own documentation. The complete scientific and computational manual is distributed with the repository.

---

## Choose a workflow

### Script-based CIRN

Use the scripted implementation when you need:

- direct control over the full `cirn_config` specification;
- explicit, repeatable batch analyses;
- publication-ready result tables and figures;
- detailed sensitivity or benchmark runs; or
- an auditable analysis folder containing the exact code, data, settings, outputs, and session information.

Main entry point:

```text
CIRN_Script_Based/CIRN_Algorithm.R
```

### CIRN Studio

Use CIRN Studio when you prefer:

- a guided data-to-results workflow;
- built-in exploratory data analysis and simulations;
- interactive preprocessing and model configuration;
- integrated results, diagnostics, sensitivity, and benchmark views; or
- one-click export of tables, figures, settings, session information, reports, and reproducibility bundles.

Main entry point:

```text
CIRN_Shiny_App/app.R
```

CIRN Studio is an interface to the same underlying CIRN methodology; it is not a separate inference method.

---

## System requirements

Recommended local environment:

- **R 4.3 or later**
- latest stable **RStudio Desktop**
- a working **C++ toolchain**
  - macOS: Xcode Command Line Tools
  - Windows: the Rtools version compatible with the installed R version
- **CmdStan**
- at least **8 GB RAM** for small to moderate analyses
- a multicore processor; modest values such as one or two model cores are often safer for long pairwise or sensitivity runs

The first Bayesian run may take longer because Stan models may need to compile.

---

## Installation

Clone or download the repository:

```bash
git clone https://github.com/gioentero/CIRN.git
cd CIRN
```

Install the main R dependencies:

```r
install.packages(c(
  "shiny", "bslib", "DT",
  "dplyr", "tidyr", "purrr", "tibble", "readr", "readxl", "broom",
  "ggplot2", "scales", "igraph", "visNetwork",
  "htmltools", "htmlwidgets", "jsonlite",
  "deSolve", "randomForest",
  "brms", "posterior", "bayestestR", "bayesplot",
  "car", "pROC", "openxlsx"
))
```

Install `cmdstanr` and CmdStan:

```r
install.packages(
  "cmdstanr",
  repos = c("https://mc-stan.org/r-packages/", getOption("repos"))
)

cmdstanr::install_cmdstan()
cmdstanr::cmdstan_version()
```

### macOS toolchain

```bash
xcode-select --install
```

### Windows toolchain check

After installing the compatible Rtools version:

```r
install.packages("pkgbuild")
pkgbuild::has_build_tools(debug = TRUE)
```

---

## Quick start

### 1. Run CIRN Studio

From the repository root:

```r
shiny::runApp("CIRN_Shiny_App")
```

Alternatively:

1. open `CIRN_Shiny_App/app.R` in RStudio;
2. click **Run App**; and
3. begin with the bundled predator-prey example.

A console message such as `Listening on http://127.0.0.1:PORT` means that the application is running locally.

### 2. Run the script-based workflow

1. Open `CIRN_Script_Based/CIRN_Algorithm.R` in a fresh R session.
2. Set the intended input file and time column.
3. Edit the `cirn_config` block, especially:
   - targets and predictors;
   - lag;
   - representation mode;
   - optional pairwise mode;
   - smoothing, outlier, response-tolerance, and jitter settings;
   - prior and MCMC settings;
   - optional validation and sensitivity settings.
4. Run the script from top to bottom.
5. Inspect the retained edges, all coefficients, diagnostics, effective-sample-size audit, collinearity outputs, edge consistency, figures, and reproducibility records.

When `input_file = NULL`, the script may also be run from the command line:

```bash
Rscript CIRN_Algorithm.R path/to/data.csv
```

An explicitly defined input path inside the script takes priority over a command-line path.

---

## Input data

CIRN expects **wide-format time-series data** with one time column and one or more numeric state-variable columns.

```csv
t,X,Y,Z
0,1.20,2.30,0.50
1,1.35,2.10,0.62
2,1.42,1.95,0.71
```

Requirements and recommendations:

- one row per observed time point;
- one time column is required;
- naming the time column `t` or `time` is recommended, but another column may be selected;
- state-variable columns must be numeric;
- short, unique variable names improve figures and output tables;
- data should be ordered by time or be sortable by the time column;
- the scripted workflow reads CSV files;
- CIRN Studio supports CSV and Excel inputs;
- long-format data should be converted to wide format before analysis.

A one-variable dataset can be used to inspect self-dynamics and derivative-response behavior, but a multivariable regulatory network requires at least two state variables.

---

## Main outputs

Depending on the selected options, a CIRN run can generate:

- retained multivariable and pairwise edge tables;
- summaries of all fitted coefficients, including non-retained terms;
- posterior means, odds ratios, 95% HDIs, and auxiliary interval summaries;
- MCMC, class-balance, data-sufficiency, and collinearity diagnostics;
- cross-mode edge-consistency summaries;
- optional signed ground-truth recovery metrics;
- optional state-level and feature-level sensitivity-stability summaries;
- static and interactive regulatory-network figures;
- posterior and trace plots;
- adaptive-jitter diagnostic figures;
- `CIRN Results Compendium.xlsx`;
- `CIRN Publication Tables.xlsx`;
- publication-oriented CSV and LaTeX outputs; and
- `sessionInfo_CIRN.txt`.

> [!CAUTION]
> Retained-edge tables, coefficient tables, model diagnostics, effective-sample-size summaries, and VIF/correlation outputs have different units of analysis. Their row counts are not expected to match. Link them using the shared target, representation, analysis mode, lag, response source, pairwise predictor, and model identifiers.

---

## Reproducible reporting

For every analysis intended for formal reporting, preserve:

- the exact input dataset or data-generation procedure;
- the exact CIRN script and application version;
- time column, targets, predictors, and admissible exclusions;
- preprocessing and derivative settings;
- response tolerance and adaptive-jitter settings;
- lag and inference modes;
- prior and MCMC settings;
- random seed;
- optional LOO, validation, and sensitivity settings;
- complete result compendium and figure files; and
- R, package, operating-system, compiler, and CmdStan information.

Numerical agreement between the scripted workflow and CIRN Studio requires matching data values and row order, variable selections, preprocessing, response construction, representations, pairwise settings, lag, priors, MCMC controls, seed, and software environment. A fixed seed improves reproducibility within a matched environment but does not guarantee bit-for-bit equality across platforms or software versions.

---

## Scientific interpretation

CIRN is intended for **transparent hypothesis generation and scientific evaluation**.

A retained edge may be reported as:

> a directed, signed, posterior-supported temporal-response regulatory hypothesis.

It should not be interpreted, without independent evidence, as:

- proof of interventional causality;
- a direct molecular, ecological, environmental, or physical mechanism;
- an estimated kinetic or physical parameter;
- a complete governing ODE or SDE term;
- an unconditional regulatory relationship; or
- evidence that an absent edge is biologically absent.

Derivative-supported edges should retain their representation in reporting. For example, prefer:

> The lagged first derivative of $X$ supported a higher posterior probability that $Y$ was increasing.

rather than the unqualified statement:

> $X$ activates $Y$.

---

## Troubleshooting

<details>
<summary><strong>Missing R package</strong></summary>

Install the package reported in the error:

```r
install.packages("package_name")
```

</details>

<details>
<summary><strong>CmdStan or brms cannot compile</strong></summary>

- macOS: install the Xcode Command Line Tools, restart RStudio, and reinstall CmdStan.
- Windows: install the compatible Rtools version, restart RStudio, and reinstall CmdStan.

</details>

<details>
<summary><strong>CIRN Studio cannot be found</strong></summary>

Run the application folder rather than an unrelated file:

```r
shiny::runApp("/path/to/CIRN/CIRN_Shiny_App")
```

Use quotation marks around paths containing spaces.

</details>

<details>
<summary><strong>The analysis is slow</strong></summary>

For an exploratory run, reduce the number of targets or predictors, disable pairwise mode, use fewer iterations, disable LOO, postpone sensitivity analysis, or use a smaller sensitivity plan. Final reported analyses should be rerun with the intended settings and complete diagnostic review.

</details>

<details>
<summary><strong>No edges were retained</strong></summary>

This can be a valid result under the 95% HDI rule. Review the input trajectories, preprocessing, derivative quality, response tolerance, class balance, lag, candidate predictors, MCMC diagnostics, collinearity, and sensitivity results.

</details>

<details>
<summary><strong>Script and app results differ</strong></summary>

Confirm that the data values and row order, time column, targets, predictors, preprocessing, representation mode, pairwise mode, lag, priors, iterations, warmup, chains, cores, seed, and sensitivity scope are identical.

</details>

---

## Authors

- **Giovannie M. Entero**¹˒²
- **Jomar F. Rabajante**¹˒²
- **Neil Jerome A. Egarguin**¹˒²
- **Mark Jayson V. Cortez**¹˒²
- **Maica Krizna A. Gavina**¹˒²
- **Patricia Ann J. Sanchez**¹˒³

¹ Graduate School, University of the Philippines Los Baños  
² Institute of Mathematical Sciences, University of the Philippines Los Baños  
³ School of Environmental Science and Management, University of the Philippines Los Baños

### Contacts

- Giovannie M. Entero: [gmentero@up.edu.ph](mailto:gmentero@up.edu.ph)
- Jomar F. Rabajante: [jfrabajante@up.edu.ph](mailto:jfrabajante@up.edu.ph)

---

## Citation

The formal article citation, versioned software citation, `CITATION.cff`, and archived release DOI will be added when the first public release is finalized.

Until then, please use this provisional software citation:

> Entero, G. M., Rabajante, J. F., Egarguin, N. J. A., Cortez, M. J. V., Gavina, M. K. A., & Sanchez, P. A. J. (2026). *Classification-Based Inference of Regulatory Networks (CIRN)* [Computer software]. GitHub.

---

## Project status and license

This repository is research software under active preparation for its first versioned public release. Interfaces, documentation, example outputs, and file organization may be refined before release.

A formal software license has not yet been added. Until a license is selected and included in the repository, no permission for reuse, modification, or redistribution should be assumed.

---

## Responsible use

Users are responsible for:

- checking that the target and predictor sets are scientifically admissible;
- reviewing derivative quality, model-ready class counts, MCMC diagnostics, and collinearity;
- disclosing adaptive response jitter whenever used;
- preserving representation-specific provenance;
- avoiding causal or mechanistic overstatement; and
- validating important findings using domain expertise, complementary methods, experiments, or independent data whenever feasible.
