================================================================================
   CCCCC  IIIII  RRRR   N   N
  C        I    R   R  NN  N
  C        I    RRRR   N N N
  C        I    R R    N  NN
   CCCCC  IIIII R  RR  N   N

  CLASSIFICATION-BASED INFERENCE OF REGULATORY NETWORKS
  CIRN ALGORITHM INSTALLATION AND USER README
================================================================================

Authors
-------
Giovannie M. Entero
Jomar F. Rabajante
Neil Jerome A. Egarguin
Mark Jayson V. Cortez
Maica Krizna A. Gavina
Patricia Ann J. Sanchez

Contact
-------
gmentero@up.edu.ph
jfrabajante@up.edu.ph


1. PURPOSE OF THIS FILE
=======================

This README explains how to install, configure, run, verify, transfer, and
troubleshoot CIRN_Algorithm.R on macOS and Windows.

CIRN is not restricted to macOS. Its R code is platform-independent, but its
Bayesian models use Stan through brms and CmdStanR. Each computer therefore
needs a working C++ toolchain and its own CmdStan installation:

  - macOS uses Apple's Command Line Tools.
  - Windows uses Rtools.

The R package "cmdstanr" and the CmdStan toolchain are related but different:

  - cmdstanr is the R interface.
  - CmdStan is the external Stan compiler and runtime installed separately.


2. PROJECT FILES
================

Keep the following items together when sharing or moving the project:

  CIRN_Algorithm.R
      Standalone implementation of the CIRN analysis workflow.

  README_CIRN_Algorithm.txt
      This installation and usage guide.

  CIRN_User_Manual.pdf
      Detailed scientific definitions, configuration guidance, interpretation,
      algorithm documentation, diagnostics, sensitivity analysis, and outputs.

  data/
      Bundled input data. The distributed example is:
      data/Predator_Prey.csv

The Shiny application has its own README.txt. This file is specifically for
running CIRN_Algorithm.R directly.


3. MINIMUM SYSTEM REQUIREMENTS
==============================

Recommended:

  - A current 64-bit release of R
  - RStudio Desktop or another R development environment
  - At least 8 GB RAM; more memory is helpful for large Bayesian runs
  - Several GB of free disk space for R packages, CmdStan, and output files
  - Internet access during first-time installation
  - A working C++17 compilation toolchain

The script may take substantial time when many targets, predictors, inference
modes, chains, iterations, pairwise fits, or sensitivity scenarios are used.


4. REQUIRED AND OPTIONAL R PACKAGES
===================================

Required by the main workflow:

  dplyr
  tidyr
  purrr
  broom
  ggplot2
  igraph
  visNetwork
  scales
  htmltools
  htmlwidgets
  brms
  cmdstanr
  bayestestR
  bayesplot
  car
  tibble
  posterior

Optional but recommended:

  pROC       Benchmark ROC/AUC calculations when applicable
  openxlsx   Excel workbook export

The base-R package stats is included with R and should not be installed
separately.


5. ONE-TIME R PACKAGE INSTALLATION
==================================

Open R or RStudio and run:

install.packages(c(
  "dplyr", "tidyr", "purrr", "broom", "ggplot2", "igraph",
  "visNetwork", "scales", "htmltools", "htmlwidgets", "brms",
  "bayestestR", "bayesplot", "car", "tibble", "posterior",
  "pROC", "openxlsx"
))

Install CmdStanR from the official Stan package repository:

install.packages(
  "cmdstanr",
  repos = c("https://stan-dev.r-universe.dev", getOption("repos"))
)

Do not run renv::snapshot() until the packages needed by the project are
installed. A snapshot records installed dependencies; it does not install
missing packages.


6. macOS SETUP
==============

Step 1: Install R and RStudio
-----------------------------

Install R from:

  https://cran.r-project.org/

Install RStudio Desktop from:

  https://posit.co/download/rstudio-desktop/

Use the build matching the Mac processor where possible:

  - Apple silicon: M1, M2, M3, M4, or later
  - Intel: older Intel-based Macs


Step 2: Install Apple's Command Line Tools
------------------------------------------

Open the Terminal application and run:

  xcode-select --install

Complete the installer and restart RStudio.

To verify the active developer tools, run in Terminal:

  xcode-select -p
  xcrun --show-sdk-path


Step 3: Install the R packages
------------------------------

Run the commands in Section 5 from the R console.


Step 4: Verify the Stan toolchain
---------------------------------

Run in R:

library(cmdstanr)
cmdstanr::check_cmdstan_toolchain(fix = TRUE)


Step 5: Install CmdStan
-----------------------

Run once in R:

cmdstanr::install_cmdstan()

Compilation can take several minutes. Verify the installation:

cmdstanr::cmdstan_version()

If RStudio cannot see the macOS SDK, close and reopen RStudio after installing
the Command Line Tools. The CIRN script also attempts to detect the active SDK
automatically with xcrun.


7. WINDOWS SETUP
================

Step 1: Install R and RStudio
-----------------------------

Install R from:

  https://cran.r-project.org/

Install RStudio Desktop from:

  https://posit.co/download/rstudio-desktop/


Step 2: Install the matching Rtools release
-------------------------------------------

Download Rtools from:

  https://cran.r-project.org/bin/windows/Rtools/

Install the Rtools version that matches the installed major R release. Use the
installer defaults unless your institution requires another location. Restart
RStudio after installation.

Optional diagnostic:

install.packages("pkgbuild")
pkgbuild::has_build_tools(debug = TRUE)

A result of TRUE indicates that R can find the required build tools.


Step 3: Install the R packages
------------------------------

Run the commands in Section 5 from the R console.


Step 4: Verify the Stan toolchain
---------------------------------

Run in R:

library(cmdstanr)
cmdstanr::check_cmdstan_toolchain(fix = TRUE)

If the check requests a restart, restart RStudio and run it again.


Step 5: Install CmdStan
-----------------------

Run once in R:

cmdstanr::install_cmdstan()

Verify:

cmdstanr::cmdstan_version()

Windows path examples in R:

  "C:/Users/YourName/Documents/CIRN/my_data.csv"

or:

  "C:\\Users\\YourName\\Documents\\CIRN\\my_data.csv"

Forward slashes are usually simpler and are recommended.


8. FINAL INSTALLATION CHECK
===========================

Run this in R on either platform:

required_packages <- c(
  "dplyr", "tidyr", "purrr", "broom", "ggplot2", "igraph",
  "visNetwork", "scales", "htmltools", "htmlwidgets", "brms",
  "cmdstanr", "bayestestR", "bayesplot", "car", "tibble",
  "posterior"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages)) {
  message("Missing: ", paste(missing_packages, collapse = ", "))
} else {
  message("All required R packages are installed.")
}

cmdstanr::check_cmdstan_toolchain(fix = TRUE)
cmdstanr::cmdstan_version()


9. INPUT DATA FORMAT
====================

CIRN expects wide-format time-series data:

  - One row represents one observation time.
  - The default time-column name is t, using lowercase.
  - Every other selected state-variable column must be numeric.
  - Time values should be ordered, finite, and unique.
  - Variable names should be unique.

Example:

t,X,Y
0,2.0000,2.0000
1,1.4859,2.4852
2,1.0032,2.7155

A dataset with only t and one state variable, such as X, is still valid
wide-format data. CIRN can analyze its temporal and derivative representations,
but cross-variable regulatory relations cannot be estimated because there is
only one state variable.

Inspect missingness, duplicates, time spacing, outliers, and measurement quality
before interpreting any inferred edge.


10. SELECTING THE INPUT FILE
============================

The script searches for input in this order:

  1. A path assigned directly to input_file in CIRN_Algorithm.R
  2. The first command-line argument
  3. The bundled data/Predator_Prey.csv example
  4. The only CSV file located beside CIRN_Algorithm.R

Method A: Use the bundled demonstration
---------------------------------------

Leave:

  input_file <- NULL

Keep data/Predator_Prey.csv with the script, then run the script.


Method B: Assign a file in the script
-------------------------------------

macOS example:

  input_file <- "/Users/yourname/Documents/CIRN/my_data.csv"

Windows example:

  input_file <- "C:/Users/yourname/Documents/CIRN/my_data.csv"


Method C: Supply the file from a terminal
-----------------------------------------

macOS:

  cd "/path/to/the/CIRN/folder"
  Rscript CIRN_Algorithm.R "/path/to/my_data.csv"

Windows Command Prompt:

  cd /d "C:\path\to\the\CIRN\folder"
  Rscript CIRN_Algorithm.R "C:\path\to\my_data.csv"

Windows PowerShell:

  Set-Location "C:\path\to\the\CIRN\folder"
  Rscript .\CIRN_Algorithm.R "C:\path\to\my_data.csv"

Quote every path containing spaces.


11. KEY CIRN CONFIGURATION
==========================

The main settings are stored in the cirn_config list near the end of
CIRN_Algorithm.R. Review that block before a formal analysis.

Important fields include:

  time_col
      Name of the time column. The default is "t".

  targets
      State variables whose first-derivative direction will be modeled.
      NULL uses all eligible non-time numeric variables.

  predictors
      Base state variables allowed to contribute predictor representations.
      NULL uses all eligible non-time numeric variables.

  representation
      Selects sublevel, all-predictor, or both multivariable configurations,
      according to the values documented in the user manual.

  lag_units
      Number of observation steps separating predictors from target responses.

  run_pairwise
      Controls whether pairwise CIRN analyses are also fitted.

  pairwise_representation
      Controls which feature representations are used for pairwise fits.

  points_per_interval
      Derivative interpolation density. Higher values can change the effective
      sample size and should not be adjusted casually.

  spar / GCV settings
      Control spline smoothing and derivative estimation.

  response_epsilon
      Defines the near-zero derivative blank region excluded from binary
      increase/decrease response construction.

  prior_mean and prior_sd
      Set the Bayesian coefficient prior.

  model_iter, model_warmup, model_chains, model_cores
      Control MCMC sampling. Warm-up must be smaller than total iterations.

  adapt_delta
      Stan sampler control. Higher values may reduce divergent transitions but
      can increase runtime.

  seed
      Supports reproducibility on the same data, settings, package versions,
      Stan version, and computational environment.

  sensitivity_inference_scope
      Controls which CIRN modes are rerun in sensitivity analysis.

Use the CIRN user manual for exact definitions and scientific interpretation.
For comparisons with earlier script outputs, match the input data,
preprocessing, representation, lag, priors, MCMC settings, seed, package
versions, and CmdStan version.


12. RUNNING CIRN IN RSTUDIO
===========================

1. Open RStudio.
2. Open CIRN_Algorithm.R.
3. Confirm that the project folder contains data/Predator_Prey.csv, or set
   input_file to your own CSV.
4. Review cirn_config.
5. Click Source, or run:

   source("CIRN_Algorithm.R")

6. Keep the R console open while models are compiling and sampling.
7. Read warnings and diagnostics before interpreting retained edges.
8. Inspect the output folder created beside the selected input file.

The first Bayesian run can take longer because Stan compiles model code.


13. OUTPUTS
===========

Depending on the selected configuration and installed optional packages, CIRN
can produce:

  - CIRN summary records
  - All coefficient summaries and 95% HDIs
  - Retained signed edges
  - Pairwise edges and diagnostics
  - Edge-consistency summaries
  - MCMC and model diagnostics
  - Effective-sample-size and data-sufficiency summaries
  - VIF and collinearity summaries
  - Random Forest support summaries
  - Latent-Z screening summaries
  - Sensitivity plans, runs, retained edges, and stability summaries
  - Ground-truth benchmark metrics when a valid adjacency is supplied
  - CSV tables
  - Excel workbook when openxlsx is installed
  - Publication-oriented figures
  - Settings and reproducibility records

Exact files depend on which optional analyses are enabled and completed.


14. SCIENTIFIC INTERPRETATION
=============================

A retained positive coefficient indicates that the earlier predictor feature is
associated with a higher probability that the target's first derivative is
positive. A retained negative coefficient indicates association with a higher
probability that the target's first derivative is negative.

An edge is retained when its posterior 95% highest-density interval excludes
zero under the configured model. Green denotes activation or positive
association with target increase; red denotes inhibition or negative
association.

These are exploratory, signed, directed, uncertainty-aware regulatory
hypotheses. They do not, by themselves, establish:

  - interventionally verified causality
  - a biochemical mechanism
  - an exact differential equation
  - a uniquely identifiable regulatory network

Interpret edges together with class balance, temporal ordering, coefficient
HDIs, convergence diagnostics, collinearity, pairwise and multivariable mode
agreement, sensitivity stability, domain knowledge, benchmarks, and
complementary methods.


15. USING renv FOR REPRODUCIBILITY
==================================

renv is optional but strongly recommended when sharing the project.

What renv::dependencies() means
-------------------------------

renv::dependencies() scans project files and reports which R packages appear to
be used. It does not install those packages and does not install CmdStan.

If renv reports:

  "Packages must first be installed before renv can snapshot them"

install the named packages first, then create the snapshot.


Create a reproducible environment
---------------------------------

From the project folder:

install.packages("renv")
renv::init()

Install all required packages from Sections 4 and 5, verify that CIRN runs, and
then execute:

renv::dependencies()
renv::snapshot()

Share these with the recipient:

  - CIRN_Algorithm.R
  - README_CIRN_Algorithm.txt
  - data files or documented data-access instructions
  - renv.lock
  - renv/activate.R and the small renv infrastructure files

The recipient runs:

install.packages("renv")
renv::restore()

After restoring R packages, the recipient must still install the platform
compiler and CmdStan as described in Sections 6 or 7.


16. TRANSFER CHECKLIST
======================

Before sending the project to another user:

  [ ] Remove private or machine-specific absolute paths.
  [ ] Keep input_file as NULL for the bundled example, or document the data path.
  [ ] Include CIRN_Algorithm.R and this README.
  [ ] Include the data folder when redistribution is permitted.
  [ ] Include the user manual.
  [ ] Record the R version.
  [ ] Record package versions and the CmdStan version.
  [ ] Include renv.lock if renv is used.
  [ ] Document the exact cirn_config settings.
  [ ] Document the seed.
  [ ] Confirm that the recipient installed Command Line Tools or Rtools.
  [ ] Confirm that cmdstanr::check_cmdstan_toolchain() succeeds.
  [ ] Confirm that cmdstanr::cmdstan_version() returns an installed version.
  [ ] Run the bundled example before attempting a large analysis.


17. TROUBLESHOOTING
===================

Problem: "there is no package called ..."
-----------------------------------------

Install the named package:

  install.packages("packageName")

For CmdStanR, use:

  install.packages(
    "cmdstanr",
    repos = c("https://stan-dev.r-universe.dev", getOption("repos"))
  )


Problem: bayestestR or cmdstanr is missing during renv::snapshot()
-----------------------------------------------------------------

Install the packages first:

  install.packages("bayestestR")
  install.packages(
    "cmdstanr",
    repos = c("https://stan-dev.r-universe.dev", getOption("repos"))
  )

Then:

  renv::dependencies()
  renv::snapshot()


Problem: CmdStan is not installed
---------------------------------

Run:

  cmdstanr::check_cmdstan_toolchain(fix = TRUE)
  cmdstanr::install_cmdstan()
  cmdstanr::cmdstan_version()


Problem: compiler or build-tool error on Windows
------------------------------------------------

Install the Rtools release matching R, restart RStudio, and run:

  pkgbuild::has_build_tools(debug = TRUE)
  cmdstanr::check_cmdstan_toolchain(fix = TRUE)


Problem: SDK, clang, or xcrun error on macOS
--------------------------------------------

In Terminal:

  xcode-select --install
  xcode-select -p
  xcrun --show-sdk-path

Restart RStudio, then run:

  cmdstanr::check_cmdstan_toolchain(fix = TRUE)


Problem: input file not found
-----------------------------

Use an absolute path, quote paths containing spaces, and verify:

  file.exists(input_file)

On Windows, prefer forward slashes.


Problem: expected columns are missing
-------------------------------------

Check:

  names(read.csv(input_file, check.names = FALSE))

Confirm that time_col, targets, predictors, and adjacency labels match the data
exactly, including capitalization.


Problem: warm-up must be smaller than total iterations
-------------------------------------------------------

Set:

  model_warmup < model_iter


Problem: divergent transitions or poor convergence
---------------------------------------------------

Do not interpret unstable fits as reliable edges. Inspect R-hat, effective
sample size, trace plots, divergences, class balance, and collinearity. Consider
more iterations, higher adapt_delta, scientifically justified predictor
reduction, better data, or revised preprocessing.


Problem: very slow execution or high memory use
------------------------------------------------

Begin with the bundled example and a smaller exploratory configuration. Reduce
the number of targets, predictors, modes, pairwise fits, chains, sensitivity
scenarios, or replicates. Use larger runs only after the workflow is verified.


Problem: Excel output is absent
-------------------------------

Install:

  install.packages("openxlsx")


Problem: ROC/AUC benchmark output is absent
-------------------------------------------

Install:

  install.packages("pROC")

ROC/AUC also requires a suitable ground-truth benchmark and both relevant
classes.


18. RECOMMENDED FIRST RUN
=========================

For the safest first test:

  1. Keep input_file <- NULL.
  2. Keep data/Predator_Prey.csv in the project folder.
  3. Use the documented default configuration.
  4. Verify the Stan toolchain.
  5. Source CIRN_Algorithm.R.
  6. Confirm that models complete and outputs are created.
  7. Inspect diagnostics before changing settings.
  8. Only then replace the example with your own data.


19. CITATION AND RESPONSIBLE USE
================================

When reporting a CIRN analysis, record:

  - data source and preprocessing
  - time column, targets, and predictors
  - feature representation and lag
  - prior and MCMC settings
  - edge-retention rule
  - pairwise settings
  - diagnostic results
  - sensitivity scope and scenarios
  - benchmark definition, when used
  - R, package, and CmdStan versions
  - random seed

Consult CIRN_User_Manual.pdf for the complete methodological and reporting
guidance. CIRN results should be presented as exploratory regulatory evidence
unless independently validated.

================================================================================
END OF README
================================================================================
