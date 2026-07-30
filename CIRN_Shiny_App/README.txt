===============================================================================
                         C I R N   S T U D I O
===============================================================================

Classification-Based Inference of Regulatory Networks (CIRN)
Shiny application for exploratory signed regulatory network reconstruction
from multivariate time-series data.

Authors:
  Giovannie M. Entero
  Jomar F. Rabajante
  Institute of Mathematical Sciences, Graduate School,
  University of the Philippines Los Banos (UPLB)

Contact:
  gmentero@up.edu.ph
  jfrabajante@up.edu.ph

===============================================================================
1. FILES IN THIS APP FOLDER
===============================================================================

Important files:

  app.R
    Main Shiny app implementation. This is the file RStudio and Shiny expect
    when running the whole app folder. Open this file and click Run App, or run
    shiny::runApp() from this folder.

  CIRN_Algorithm.R
    Main CIRN algorithm implementation used by the Shiny app.

  www/cirn-studio.css
    Styling file for the app interface.

  data/Predator_Prey.csv
    Built-in example data.

  CIRN_User_Manual.pdf
    Detailed scientific and algorithm documentation, if included in your copy.

  README.txt
    This plain-text guide.

Recommended clean user-facing folder:

  CIRN_Studio/
    app.R
    CIRN_Algorithm.R
    CIRN_User_Manual.pdf
    README.txt
    data/
      Predator_Prey.csv
    www/
      cirn-studio.css

Only app.R, CIRN_Algorithm.R, data/, and www/ are required to run the app.
The manual and README are documentation files.

===============================================================================
2. REQUIRED R PACKAGES
===============================================================================

Install these packages before running the app:

  shiny
  bslib
  DT
  ggplot2
  dplyr
  tidyr
  purrr
  tibble
  readr
  readxl
  openxlsx
  visNetwork
  igraph
  jsonlite
  deSolve
  randomForest
  brms
  cmdstanr
  posterior
  bayestestR
  bayesplot
  car
  broom
  scales
  htmltools
  htmlwidgets

Recommended install command:

  install.packages(c(
    "shiny", "bslib", "DT", "ggplot2", "dplyr", "tidyr", "purrr",
    "tibble", "readr", "readxl", "openxlsx", "visNetwork", "igraph",
    "jsonlite", "deSolve", "randomForest", "brms", "cmdstanr",
    "posterior", "bayestestR", "bayesplot", "car", "broom", "scales",
    "htmltools", "htmlwidgets"
  ))

If cmdstanr is not available from your default CRAN mirror, install it using:

  install.packages(
    "cmdstanr",
    repos = c("https://stan-dev.r-universe.dev", getOption("repos"))
  )

Then install CmdStan:

  cmdstanr::install_cmdstan()

Check that CmdStan works:

  cmdstanr::cmdstan_version()

===============================================================================
3. SYSTEM REQUIREMENTS
===============================================================================

Recommended:

  R version:
    R 4.3 or newer is recommended.

  RStudio:
    Latest stable RStudio Desktop is recommended for local use.

  Memory:
    At least 8 GB RAM for small to moderate examples.
    More RAM is helpful for pairwise CIRN, sensitivity analysis, and large data.

  Processor:
    Multicore CPU is useful, but do not set model_cores too high.
    For routine use, 1 or 2 cores is often safer than using all available cores.

Important:

  CIRN uses Bayesian logistic regression through brms and CmdStan.
  This means the first run may take longer because Stan models may compile.

===============================================================================
4. MACOS SETUP
===============================================================================

Step 1. Install R
  Download and install R from:
    https://cran.r-project.org/

Step 2. Install RStudio
  Download and install RStudio Desktop from:
    https://posit.co/download/rstudio-desktop/

Step 3. Install Apple command line tools
  Open Terminal and run:

    xcode-select --install

  Follow the installation prompt. This helps CmdStan compile Bayesian models.

Step 4. Open RStudio and install packages
  Run the package installation command shown in Section 2.

Step 5. Install CmdStan
  In RStudio, run:

    cmdstanr::install_cmdstan()
    cmdstanr::cmdstan_version()

Step 6. Run the app
  Open the CIRN_Studio folder in RStudio.
  Open app.R.
  Click "Run App".

  Or run this in the R console:

    shiny::runApp("/Users/gio/Desktop/My Dissertation/Dissertation Focus/Main Version/Shiny App/CIRN/CIRN_Studio")

If your path contains spaces, always keep quotation marks around the path.

===============================================================================
5. WINDOWS SETUP
===============================================================================

Step 1. Install R
  Download and install R from:
    https://cran.r-project.org/

Step 2. Install RStudio
  Download and install RStudio Desktop from:
    https://posit.co/download/rstudio-desktop/

Step 3. Install Rtools
  CmdStan needs a working C++ toolchain.
  Install the Rtools version that matches your R version:
    https://cran.r-project.org/bin/windows/Rtools/

  After installation, restart RStudio.

Step 4. Check Rtools
  In RStudio, run:

    pkgbuild::has_build_tools(debug = TRUE)

  If pkgbuild is not installed:

    install.packages("pkgbuild")

Step 5. Install packages
  Run the package installation command shown in Section 2.

Step 6. Install CmdStan
  Run:

    cmdstanr::install_cmdstan()
    cmdstanr::cmdstan_version()

Step 7. Run the app
  Open the CIRN_Studio folder in RStudio.
  Open app.R.
  Click "Run App".

  Or run:

    shiny::runApp("C:/path/to/CIRN_Studio")

Use forward slashes in R paths on Windows, for example:

    "C:/Users/YourName/Desktop/CIRN_Studio"

===============================================================================
6. HOW TO RUN THE APP
===============================================================================

Recommended local run:

  1. Open RStudio.
  2. Open the CIRN_Studio folder or set the working directory to it.
  3. Open app.R.
  4. Click "Run App".

Console run:

  setwd("/path/to/CIRN_Studio")
  shiny::runApp()

Direct folder run:

  shiny::runApp("/path/to/CIRN_Studio")

Why the app file is named app.R:

  app.R is the standard Shiny filename. Keeping the main app code in app.R makes
  the folder easier for non-specialist users because RStudio and shiny::runApp()
  can run the app without extra instructions.

===============================================================================
7. INPUT DATA FORMAT
===============================================================================

The app expects wide-format time-series data.

Minimum format:

  t, X
  0, 1.20
  1, 1.35
  2, 1.42

Typical multivariable format:

  t, X, Y, Z
  0, 1.20, 2.30, 0.50
  1, 1.35, 2.10, 0.62
  2, 1.42, 1.95, 0.71

Requirements:

  - The time column should be named t.
  - State variables should be numeric.
  - Each row should represent one time point.
  - Data should be ordered by time, or at least sortable by the time column.
  - CSV and Excel inputs are supported.
  - Avoid duplicated column names.
  - Avoid nonnumeric symbols inside state-variable columns.

One-variable data:

  A dataset with only t and one variable, such as X, is still wide-format data.
  The app can inspect and preprocess it. CIRN can examine self-related
  derivative-response patterns, but it cannot reconstruct a multi-node
  regulatory network from only one state variable.

Long-format data:

  Long format usually means rows like:

    t, variable, value
    0, X, 1.20
    0, Y, 2.30
    1, X, 1.35
    1, Y, 2.10

  Convert this to wide format before using CIRN Studio:

    t, X, Y
    0, 1.20, 2.30
    1, 1.35, 2.10

===============================================================================
8. BASIC WORKFLOW
===============================================================================

Recommended first run:

  Before starting, open What Is CIRN? for the scientific overview or User
  Guide for detailed, non-specialist instructions for every app stage.

  1. Start tab
     Load the built-in predator-prey example first.

  2. Data Source > Example or Uploaded Data
     Confirm the active data source, time column, targets, and predictors.

  3. Data Source > Simulation Lab
     Optional. Generate synthetic data from built-in systems or user-defined
     equations. After simulating, confirm that the simulated output becomes the
     active data source.

  4. EDA tab
     Inspect trajectories, missingness, correlations, phase plots, derivative
     views, outliers, time gaps, and basic statistical summaries.

  5. Preprocess tab
     Set normalization, smoothing, derivative grid, outlier action, epsilon,
     and adaptive jitter options. Check derivative previews and response class
     balance before fitting CIRN.

  6. Run CIRN Algorithm tab
     Choose representation mode, lag, Bayesian settings, and optional pairwise
     CIRN. For a first serious run, use the default or script-matched settings.

  7. Results and CIRN Figures tabs
     Read retained edges, coefficient HDIs, mode-specific networks, consistency
     plots, and publication-oriented CIRN figures.

  8. Diagnostics tab
     Review model diagnostics, class balance, VIF/collinearity, RF support,
     latent-Z screening, and adaptive jitter information.

  9. Sensitivity tab
     Test whether retained edges remain stable under perturbations such as lag,
     noise, downsampling, missing rows, and sample-size changes.

 10. Benchmark tab
     Optional. Upload a ground-truth signed adjacency matrix or use built-in
     truth from simulation systems to compute benchmark metrics.

 11. Export tab
     Export CSV files, workbook, figure pack, settings JSON, session info,
     HTML report, and ZIP bundle.

 12. Feedback tab
     Report bugs, confusing behavior, documentation questions, and suggestions
     by email, or download a feedback report or structured CSV record. CIRN
     Studio prepares an addressed email draft but never sends it automatically;
     review the draft and click Send in your own email application.

===============================================================================
9. SCIENTIFIC INTERPRETATION
===============================================================================

What CIRN reports:

  CIRN reports signed, directed, uncertainty-aware candidate regulatory
  hypotheses from multivariate time-series data.

Core idea:

  CIRN models whether a target variable's first derivative is increasing or
  decreasing using earlier state and derivative-based features.

Green and red:

  Green means activation or positive association with the target increasing.
  Red means inhibition or negative association with the target increasing.

Edge retention:

  A CIRN edge is retained when the Bayesian logistic coefficient's 95 percent
  highest-density interval excludes zero.

Edge thickness:

  Thicker edges indicate larger absolute posterior mean coefficient magnitude,
  |omega|. Thickness is not proof of stronger mechanism by itself; it should be
  read together with uncertainty, diagnostics, sensitivity, and benchmarks.

What CIRN does not prove by itself:

  - It does not prove interventional causality.
  - It does not identify exact ODE/SDE equations.
  - It does not guarantee final biological, ecological, or physical mechanism.
  - It does not replace domain validation or experimental evidence.

Best practice:

  Treat CIRN output as evidence for signed regulatory hypotheses, then qualify
  those hypotheses using diagnostics, sensitivity analysis, benchmark checks,
  and domain knowledge.

===============================================================================
10. SENSITIVITY ANALYSIS NOTES
===============================================================================

If you want to focus only on lag sensitivity:

  - Put the lag values in "Lag units", for example: 1,2,3.
  - Leave noise fractions blank if you do not want noise perturbations.
  - Leave downsample intervals blank if you do not want downsampling.
  - Leave target sample sizes blank if you do not want sample-size perturbation.
  - Leave missing row fractions blank if you do not want missing-row stress tests.
  - Use enough stochastic replicates for stability. For quick checks, 1 or 2 is
    fine. For serious checks, use more.

Sensitivity inference scope:

  use run settings:
    Use the representation and pairwise choices currently selected in the
    Run CIRN Algorithm tab.

  sublevel:
    Rerun only sublevel multivariable CIRN.

  all predictors:
    Rerun only all-predictors CIRN.

  sublevel + all predictors:
    Rerun both multivariable modes, without pairwise CIRN.

  pairwise only:
    Focus displayed stability summaries on pairwise retained edges.

  everything:
    Rerun sublevel, all-predictors, pairwise sublevel, and pairwise
    all-predictors. This is the broadest and most expensive option.

State-level versus feature-level sensitivity:

  State-level stability asks whether a base regulator-to-target relation is
  stable after collapsing derivative representations.

  Feature-level stability asks whether a specific representation, such as X,
  dX, or d2X, is stable across perturbation scenarios.

===============================================================================
11. GROUND-TRUTH ADJACENCY FOR BENCHMARKING
===============================================================================

Optional benchmark matrices should be square signed adjacency matrices.

Example:

      X   Y
  X   0   1
  Y  -1   0

Interpretation:

  1  = activation
 -1  = inhibition
  0  = no known edge

Rows are source/regulator variables.
Columns are target variables.

Use base variable names such as X and Y, not dX or dY, for the ground-truth
matrix. CIRN will map derivative-response targets back to base target names
for benchmark summaries.

===============================================================================
12. EXPORTS AND REPRODUCIBILITY
===============================================================================

For every analysis you plan to cite, export and keep:

  - Edge table
  - Coefficient HDI table
  - Diagnostics table
  - Sensitivity tables, when used
  - Benchmark metrics, when ground truth is available
  - Workbook
  - Figure pack
  - Settings JSON
  - Session info
  - HTML report
  - ZIP bundle

Always report:

  - Data source
  - Time column
  - Targets and predictors
  - Preprocessing settings
  - Representation mode
  - Pairwise setting
  - Lag units
  - Bayesian iterations, warmup, chains, cores, prior mean, prior SD
  - HDI retention rule
  - Sensitivity inference scope, if sensitivity analysis was used
  - Whether ground truth was used for benchmarking

===============================================================================
13. TROUBLESHOOTING
===============================================================================

Problem: "there is no package called ..."
  Install the missing package with install.packages("package_name").

Problem: CmdStan or brms cannot compile.
  macOS: run xcode-select --install, restart RStudio, then reinstall CmdStan.
  Windows: install the correct Rtools version, restart RStudio, then reinstall
  CmdStan.

Problem: runApp cannot find the app.
  Run the folder, not only a random file. Use:
    shiny::runApp("/path/to/CIRN_Studio")

Problem: path has spaces.
  Put quotation marks around the path.

Problem: browser says "Listening on http://127.0.0.1:xxxx".
  This is normal. It means Shiny is running locally on your computer.
  Open the displayed address in a browser if it does not open automatically.

Problem: the app is slow.
  Reduce pairwise mode, use fewer targets/predictors, reduce iterations for
  exploratory runs, reduce chains/cores, avoid "everything" sensitivity mode
  until needed, or use a smaller sensitivity plan.

Problem: plots show no retained edges.
  This can be a valid result under the 95 percent HDI rule. Check preprocessing,
  class balance, epsilon, lag, selected predictors, Bayesian settings,
  diagnostics, and sensitivity results.

Problem: results differ from a script-based run.
  Confirm that the data source, preprocessing, targets, predictors,
  representation mode, pairwise setting, lag units, prior settings, iterations,
  warmup, chains, seed, and sensitivity scope are identical.

Problem: a one-variable dataset is used.
  The app can inspect and model derivative-response behavior, but a full
  multivariable regulatory network requires at least two state variables.

===============================================================================
14. FINAL PRACTICAL ADVICE
===============================================================================

For a first demonstration:

  Load the built-in example, keep defaults, run CIRN, inspect Results and CIRN
  Figures, then export the workbook and report.

For a serious analysis:

  Use careful preprocessing, adequate Bayesian sampling settings, diagnostics,
  sensitivity analysis, and benchmark validation when ground truth is available.

For a publication or dissertation result:

  Never cite only the network figure. Cite the retained-edge table, HDI
  evidence, diagnostics, sensitivity stability, benchmark results when
  available, settings file, and session info.

===============================================================================
End of README.txt
===============================================================================
