################################################################################
############# CLASSIFICATION-BASED INFERENCE OF REGULATORY NETWORKS ############
#############                      ~ CIRN ~                         ############
################################################################################
#
# Main Author:
#   Giovannie M. Entero (Gio)
#
# Affiliation:
#   PhD in Applied Mathematics
#   University of the Philippines Los Baños (UPLB)
#
# Dissertation Advisory Committee:
#   Chair/Supervisor:
#     Prof. Jomar F. Rabajante, D.Sc.
#     Full Professor, Institute of Mathematical Sciences (IMS)
#     Dean of UPLB Graduate School
#     University of the Philippines Los Baños (UPLB)
#   Co-Chair:
#     Neil Jerome A. Egarguin, Ph.D.
#     Full Professor, Institute of Mathematical Sciences (IMS)
#     University of the Philippines Los Baños (UPLB)
#   Members:
#     Mark Jayson V. Cortez, Ph.D.
#     Associate Professor, Institute of Mathematical Sciences (IMS)
#     University of the Philippines Los Baños (UPLB)
#     Maica Krizna A. Gavina, D.Sc.
#     Full Professor, Institute of Mathematical Sciences (IMS)
#     University of the Philippines Los Baños (UPLB)
#     Patricia Ann J. Sanchez, Ph.D.
#     Full Professor, School of Environmental Science and Management (SESAM)
#     University of the Philippines Los Baños (UPLB)
#
# Research Area:
#   Data-driven inference of regulatory networks
#   with applications to ecological and environmental systems
#
# Dissertation:
#   Classification-Based Inference of Regulatory Networks (CIRN): Bayesian
#   Logistic Regression for Reconstructing Directional and Signed Interactions
#   from Time-Series Data
#
# Description:
#   This script implements the CIRN framework, a Bayesian,
#   classification-based methodology for inferring signed and
#   directed regulatory interactions from multivariate time-series
#   data using state variables and their temporal derivatives.
#
# Intended Use:
#   • Reproducible research and dissertation support
#   • Simulation studies and methodological validation
#   • Exploratory and diagnostic analysis of real-world systems
#
# Notes:
#   • Inference is uncertainty-aware and posterior-based
#   • Diagnostic plots are provided for transparency and validation
#   • The script is modular to support extension and reuse
#
################################################################################


# ==============================================================
# CIRN Descriptions
# ==============================================================
# 
# This script implements the Classification-Based Inference of
# Regulatory Networks (CIRN) framework. CIRN is a data-driven
# methodology for inferring directed and signed interaction
# networks from multivariate time-series data.
#
# The core idea of CIRN is to recast local dynamical interactions
# as a classification problem: for each target variable, the
# direction of its temporal change is modeled as a function of
# lagged system states and their temporal derivatives. Bayesian
# logistic regression is then used to estimate posterior
# distributions of interaction coefficients, enabling probabilistic
# inference of activation and inhibition relationships.
#
# By combining temporal derivatives, temporal ordering through
# lagging, and Bayesian uncertainty quantification, CIRN provides
# a flexible framework that is applicable across domains,
# including gene regulatory networks, ecological systems, and
# environmental dynamics.
#
# The CIRN framework is implemented in this script as a structured,
# end-to-end computational pipeline that transforms multivariate
# observational time-series data into a probabilistically inferred,
# signed, and directed regulatory network.
#
# Each step below corresponds directly to a concrete component of
# the implementation.
#
# --------------------------------------------------------------
# (1) Data acquisition and temporal organization
# --------------------------------------------------------------

#   • Input data consist of multivariate time-series with a designated
#     temporal index; by default, the first column is used unless
#     time_col is set explicitly.
#   • All variables except the time column are treated as potentially
#     dynamical system components.
#
#   Implemented in:
#     infer_network(df, time_col)
#
#
# --------------------------------------------------------------
# (2) Data preprocessing and numerical stabilization
# --------------------------------------------------------------

#   • Missing values are removed prior to derivative computation to
#     prevent propagation of numerical artifacts.
#   • Spline smoothing is applied to reduce noise amplification during
#     numerical differentiation; spar = NULL uses generalized cross-validation.
#
#   Implemented in:
#     infer_network()  → complete-case filtering for analysis variables
#     compute_derivatives(df, time_col, spar = NULL)
#
#
# --------------------------------------------------------------
# (3) Derivative-based representation of system dynamics
# --------------------------------------------------------------

#   • First- and second-order temporal derivatives are approximated
#     using spline smoothing and centered finite differences.
#   • These derivatives encode local trends and accelerations while
#     remaining agnostic to explicit functional forms.
#
#   Implemented in:
#     compute_derivatives()
#       → X_smooth
#       → X_d1
#       → X_d2
#
#
# --------------------------------------------------------------
# (4) Binary response encoding
# --------------------------------------------------------------

#   • For each target variable x_j, the sign of its first temporal
#     derivative dx_j/dt is first encoded as a binary response variable:
#         Class = 1  (increasing, dx_j/dt > 0)
#         Class = 0  (decreasing, dx_j/dt < 0)
#   • Near-zero derivatives are excluded to avoid numerical ambiguity.
#   • Class labels are constructed from the raw, unstandardized target
#     derivative before predictor standardization.
#   • If this directional encoding produces one class or a nearly
#     one-class response, CIRN can apply adaptive minimal jitter to
#     the target trajectory and recompute the target derivative.
#   • The smallest jitter scale that yields a usable two-class
#     derivative response is retained and explicitly reported.
#   • Jitter is never added directly to class labels.
#
#   Implemented in:
#     infer_for_target()
#       → construction of Class from target_d1
#       → optional adaptive minimal-jitter derivative response
#
#
# --------------------------------------------------------------
# (5) Temporally ordered feature construction
# --------------------------------------------------------------

#   • Predictor variables (state variables and their derivatives)
#     are temporally lagged by one time step and then standardized
#     for Bayesian model fitting.
#   • This enforces temporal precedence by ensuring predictors
#     occur before the observed target response.
#   • The target first derivative is excluded from its own predictor set
#     because dX is represented by X_d1; lagged target state and
#     second derivative predictors may still be used when declared.
#   • Jittered response-construction columns, when created, are excluded
#     from the predictor set and used only to define Class.
#
#   Implemented in:
#     infer_for_target()
#       → dplyr::lag() applied to predictor_cols
#
#
# --------------------------------------------------------------
# (6) Bayesian classification-based regulatory inference
# --------------------------------------------------------------

#   • For each target variable, Bayesian logistic regression models
#     the probability of the active target-response class given
#     lagged predictors.
#   • Weakly informative priors regularize inference under noise
#     and collinearity.
#   • Full posterior distributions quantify uncertainty in each
#     regulatory effect.
#
#   Implemented in:
#     run_bayes_logit()
#       → brm(..., family = bernoulli(link = "logit"))
#
#
# --------------------------------------------------------------
# (7) Multi-representation sensitivity analysis
# --------------------------------------------------------------

#   • Sublevel Bayesian models may be fitted using three predictor
#     levels:
#         (i)   state level,
#         (ii)  first-derivative level,
#         (iii) second-derivative level.
#   • Optionally, one combined all-predictors model fits original,
#     first-derivative, and second-derivative predictors together
#     for d[target], excluding d[target] itself.
#   • The sublevel outputs form the primary CIRN regulatory network.
#   • The combined all-predictors model is used as a robustness and
#     consistency check because it asks whether retained effects persist
#     when all predictor representations compete in one joint model.
#
#   Implemented in:
#     infer_for_target()
#       → results$original
#       → results$first_derivative
#       → results$second_derivative
#       → results$all_predictors
#
#
# --------------------------------------------------------------
# (8) Posterior-based edge definition and network construction
# --------------------------------------------------------------

#   • Regulatory edges are inferred using a posterior credibility
#     criterion: an edge is retained if the 95% HDI of the regression
#     coefficient excludes zero.
#   • The sign of the posterior mean determines activation or
#     inhibition.
#   • Edges from all representations are aggregated into a single
#     regulatory network.
#
#   Implemented in:
#     run_bayes_logit()
#       → HDI computation and filtering
#     infer_for_target()
#       → bind_rows(...) into edges_tbl
#
#
# --------------------------------------------------------------
# (9) Uncertainty synthesis and stability assessment
# --------------------------------------------------------------

#   • Convergence diagnostics (R-hat, ESS) and optional LOO measures
#     are collected for each model.
#   • Posterior uncertainty is used to distinguish strong, stable
#     regulatory relationships from weak or non-significant ones.
#
#   Implemented in:
#     run_bayes_logit()
#       → diagnostics list
#     infer_network()
#       → aggregation of diagnostics across targets
#
#
# --------------------------------------------------------------
# (10) Interpretative synthesis and decision-relevant reporting
# --------------------------------------------------------------

#   • The final CIRN network is interpreted as a probabilistic
#     regulatory structure rather than a deterministic causal model.
#   • Results are reported as signed, directed networks suitable
#     for scientific interpretation, scenario analysis, and
#     policy-relevant hypothesis generation.
#
#   Implemented in:
#     infer_network()   → final edges object
#     plot_network_main(), plot_cirn_panels(), etc.
#
# ==============================================================


################################################################################
#############                       ~ START ~                        ###########
################################################################################


# ==============================================================
# Working directory
# ==============================================================

# Resolve the script directory without changing the user's working
# directory. Relative input files are interpreted relative to the script
# directory when available; otherwise, they are interpreted relative to
# the current R working directory.

resolve_script_dir <- function() {
  frames <- sys.frames()
  candidates <- vapply(
    frames,
    function(frame) {
      ofile <- frame$ofile
      if (is.character(ofile) && length(ofile) == 1 && nzchar(ofile)) {
        return(ofile)
      }
      NA_character_
    },
    character(1)
  )
  candidates <- candidates[!is.na(candidates)]
  
  if (length(candidates) == 0) {
    return(NA_character_)
  }
  
  dirname(normalizePath(tail(candidates, 1), mustWork = FALSE))
}

script_dir <- resolve_script_dir()
analysis_dir <- if (!is.na(script_dir) && dir.exists(script_dir)) {
  script_dir
} else {
  getwd()
}

# Temporary default. After input_file is resolved, output_dir is reset to the
# folder containing the input data so each dataset keeps its own CIRN outputs.
output_dir <- analysis_dir
message("CIRN analysis directory: ", analysis_dir)


################################################################################
################################################################################


# ==============================================================
# 0. Clean session & reproducibility
# ==============================================================

# This section sets reproducibility options without destroying the
# user's workspace by default. Set clear_workspace = TRUE only when
# running this script in a dedicated analysis session.

clear_workspace <- FALSE
if (isTRUE(clear_workspace)) {
  rm(list = setdiff(
    ls(),
    c("clear_workspace", "analysis_dir", "output_dir", "script_dir", "resolve_script_dir")
  ))
}

# Run garbage collection to free memory from any previously loaded objects

gc()

# Set a fixed random seed to ensure reproducibility of all stochastic 
# components (e.g., Bayesian sampling)

set.seed(143)

# Close existing graphics devices only when requested.

close_existing_graphics <- FALSE
if (isTRUE(close_existing_graphics) && !is.null(dev.list())) {
  dev.off()
}

# Limit native thread pools before Stan is loaded. This avoids OpenMP
# pthread resource errors during long CIRN runs with many repeated
# brms/CmdStan fits, especially when run_pairwise = TRUE.

set_cirn_thread_limits <- function() {
  Sys.setenv(
    OMP_NUM_THREADS = "1",
    STAN_NUM_THREADS = "1",
    TBB_NUM_THREADS = "1",
    VECLIB_MAXIMUM_THREADS = "1",
    MKL_NUM_THREADS = "1"
  )
  invisible(NULL)
}

set_cirn_thread_limits()


################################################################################
################################################################################


# ==============================================================
# 1. Required Libraries
# ==============================================================

# CIRN relies on a combination of packages for:
#   (i) data manipulation,
#   (ii) Bayesian modeling via Stan,
#   (iii) network inference and visualization, and
#   (iv) diagnostic and uncertainty assessment.
#
# Before running this script for the first time, ensure that
# all required packages are installed.
#
# You may install missing packages using:
#
# install.packages(c(
#   "dplyr", "tidyr", "purrr", "broom", "ggplot2", "igraph",
#   "visNetwork", "scales", "htmltools", "htmlwidgets", "brms",
#   "bayestestR", "bayesplot", "car", "tibble", "posterior",
#   "pROC", "openxlsx"
# ))
# install.packages(
#   "cmdstanr",
#   repos = c("https://stan-dev.r-universe.dev", getOption("repos"))
# )
# cmdstanr::check_cmdstan_toolchain(fix = TRUE)
# cmdstanr::install_cmdstan()
#
# IMPORTANT (Stan toolchain):
#   - macOS: install Apple's Command Line Tools (`xcode-select --install`)
#   - Windows: install the Rtools release matching the installed R version
#   - CmdStan must be installed once on each computer
#
# ==============================================================

# --------------------------------------------------------------
# 1.1 Core data manipulation & utilities
# --------------------------------------------------------------

library(stats)     # Base statistical functions (models, distributions)
library(dplyr)     # Data manipulation (filter, mutate, summarise)
library(tidyr)     # Data reshaping (pivot_longer, pivot_wider)
library(purrr)     # Functional programming (map, imap)
library(broom)     # Tidy summaries of model outputs

# --------------------------------------------------------------
# 1.2 Visualization
# --------------------------------------------------------------

library(ggplot2)   # Grammar of graphics (publication-quality plots)

# --------------------------------------------------------------
# 1.3 Network analysis & Graphing
# --------------------------------------------------------------

library(igraph)        # Graph objects, layouts, and network visualization
library(visNetwork)    # Interactive network visualization (browser-based)
library(scales)        # Scaling functions (e.g., rescaling, color mapping)
library(htmltools)     # HTML tag generation and customization
library(htmlwidgets)   # Create and export interactive web widgets

# --------------------------------------------------------------
# 1.4 Bayesian modeling (CIRN core)
# --------------------------------------------------------------

library(brms)        # Bayesian regression modeling (Stan interface)
library(cmdstanr)    # CmdStan backend (faster, more stable than rstan)

# --------------------------------------------------------------
# 1.5 Bayesian diagnostics & uncertainty quantification
# --------------------------------------------------------------

library(bayestestR)  # Highest Density Intervals (HDI), posterior summaries
library(bayesplot)   # MCMC diagnostics (trace, density, R-hat visuals)

# --------------------------------------------------------------
# 1.6 Classical diagnostics (auxiliary)
# --------------------------------------------------------------

library(car)         # Variance Inflation Factor (VIF) for collinearity checks

# --------------------------------------------------------------
# 1.7 brms configuration
# --------------------------------------------------------------

# Explicitly enforce the CmdStan backend for all brms models.
# This improves numerical stability, sampling speed, and
# reproducibility, especially for repeated model fitting
# in the CIRN framework.

options(brms.backend = "cmdstanr")

message("Using brms backend: ", getOption("brms.backend"))

# --------------------------------------------------------------
# 1.8 System sanity check (macOS + Stan toolchain)
# --------------------------------------------------------------

# Stan requires a valid macOS SDK path for C++ compilation.
# --------------------------------------------------------------
# 1.8 System sanity check (platform-aware Stan toolchain)
# --------------------------------------------------------------
# On macOS, expose the active Apple SDK to Stan when RStudio did not
# inherit SDKROOT. Windows and Linux do not use this setting.

if (identical(Sys.info()[["sysname"]], "Darwin") &&
    !nzchar(Sys.getenv("SDKROOT"))) {
  sdk_path <- tryCatch(
    system2("xcrun", "--show-sdk-path", stdout = TRUE, stderr = FALSE),
    error = function(e) character()
  )
  sdk_path <- sdk_path[nzchar(sdk_path)]
  if (length(sdk_path) > 0L && dir.exists(sdk_path[[1L]])) {
    Sys.setenv(SDKROOT = sdk_path[[1L]])
    message("Detected macOS SDK: ", sdk_path[[1L]])
  } else {
    warning(
      "macOS SDK could not be detected. Install Apple's Command Line Tools ",
      "with `xcode-select --install`, restart R/RStudio, and rerun. ",
      "Verify the Stan toolchain with ",
      "`cmdstanr::check_cmdstan_toolchain(fix = TRUE)`."
    )
  }
}

# --------------------------------------------------------------
# 1.9 Optional packages
# --------------------------------------------------------------

# pROC is used only for optional AUC computation during
# evaluation. The analysis proceeds safely if it is absent.

use_auc <- requireNamespace("pROC", quietly = TRUE)
if (use_auc) library(pROC)


################################################################################
################################################################################


# ==============================================================
# 2. Regularized Central-Difference Derivative Engine (CIRN)
# ==============================================================

# Purpose:
#   • Regularize irregular time sampling
#   • Smooth state variables via cubic smoothing spline
#   • Compute first and second derivatives using
#     centered finite differences
#
# Design:
#   1. Time is sorted and regularized
#   2. Smoothed trajectory evaluated on regular grid
#   3. Central difference applied with constant Δt
#
# Arguments:
#   df                   : data frame
#   time_col             : name of time column
#   points_per_interval  : number of points per original interval
#   spar                 : smoothing parameter
#
# Output:
#   Expanded data frame with:
#     • Regularized time
#     • Smoothed state variables
#     • First derivatives (_d1)
#     • Second derivatives (_d2)
# ==============================================================

compute_derivatives <- function(df, time_col, points_per_interval = 1, spar = NULL, outlier_method = "MAD", outlier_thresh = 3.5, outlier_action = "winsorize") {
  
  if (!time_col %in% names(df)) {
    stop("time_col is not present in df.")
  }
  
  if (length(points_per_interval) != 1 ||
      is.na(points_per_interval) ||
      points_per_interval < 1 ||
      points_per_interval != floor(points_per_interval)) {
    stop("points_per_interval must be a single positive integer.")
  }
  points_per_interval <- as.integer(points_per_interval)
  
  outlier_method <- match.arg(outlier_method, choices = c("MAD", "none"))
  outlier_action <- match.arg(outlier_action, choices = c("winsorize", "remove", "keep"))
  
  if (!is.null(spar) &&
      (length(spar) != 1 || is.na(spar) || spar < 0 || spar > 1)) {
    stop("spar must be NULL or a single value between 0 and 1.")
  }
  
  # --------------------------------------------------------------
  # 2.1 Ensure numeric, sorted time
  # --------------------------------------------------------------
  
  time_raw <- df[[time_col]]
  
  if (inherits(time_raw, "Date") || inherits(time_raw, "POSIXt")) {
    time_raw <- as.numeric(time_raw)
  } else if (!is.numeric(time_raw)) {
    stop("time_col must be numeric, Date, or POSIXt.")
  }
  
  if (any(!is.finite(time_raw))) {
    stop("time_col contains missing or non-finite values.")
  }
  
  var_names <- setdiff(names(df), time_col)
  
  non_numeric_vars <- var_names[!vapply(df[var_names], is.numeric, logical(1))]
  if (length(non_numeric_vars) > 0) {
    stop(
      "All CIRN state and predictor variables must be numeric. Non-numeric columns: ",
      paste(non_numeric_vars, collapse = ", ")
    )
  }
  
  if (anyDuplicated(time_raw)) {
    warning(
      "Duplicate time values detected; numeric observations with the same time ",
      "are averaged before derivative computation.",
      call. = FALSE
    )
    
    df_for_aggregation <- df
    df_for_aggregation[[time_col]] <- time_raw
    
    df <- stats::aggregate(
      df_for_aggregation[var_names],
      by = stats::setNames(list(time_raw), time_col),
      FUN = function(x) mean(x, na.rm = TRUE)
    )
    
    time_raw <- df[[time_col]]
  }
  
  ord <- order(time_raw)
  df <- df[ord, ]
  time_raw <- time_raw[ord]
  
  # --------------------------------------------------------------
  # 2.2 Construct globally regularized time grid
  # --------------------------------------------------------------
  
  if (length(time_raw) < 3) {
    stop("At least 3 observed time points are required.")
  }
  
  t_min <- min(time_raw)
  t_max <- max(time_raw)
  
  if (!is.finite(t_min) || !is.finite(t_max) || t_min == t_max) {
    stop("time_col must span a nonzero finite time interval.")
  }
  
  n_intervals <- length(time_raw) - 1
  n_points <- n_intervals * points_per_interval + 1
  
  time_regular <- seq(
    from = t_min,
    to   = t_max,
    length.out = n_points
  )
  
  if (length(time_regular) < 3) {
    stop("At least 3 regularized time points are required.")
  }
  
  df_out <- data.frame(time = time_regular)
  names(df_out)[1] <- time_col
  
  dt <- time_regular[2] - time_regular[1]
  
  
  # --------------------------------------------------------------
  # 2.3 Loop through variables
  # --------------------------------------------------------------
  
  for (v in var_names) {
    
    y_raw <- df[[v]]
    
    if (length(unique(na.omit(y_raw))) < 3) {
      df_out[[v]] <- NA_real_
      df_out[[paste0(v, "_d1")]] <- NA_real_
      df_out[[paste0(v, "_d2")]] <- NA_real_
      next
    }
    
    # ----------------------------------------------------------
    # 2.3.1 Robust Outlier Detection (MAD-based)
    # ----------------------------------------------------------
    
    if (outlier_method == "MAD") {
      
      med <- median(y_raw, na.rm = TRUE)
      mad_val <- mad(y_raw, constant = 1, na.rm = TRUE)
      
      if (mad_val > 0) {
        
        mod_z <- 0.6745 * (y_raw - med) / mad_val
        outliers <- abs(mod_z) > outlier_thresh
        
        n_out <- sum(outliers, na.rm = TRUE)
        
        if (n_out > 0) {
          
          message(
            sprintf(
              "Potential outliers detected in variable '%s': %d observation(s). Action = %s",
              v,
              n_out,
              outlier_action
            )
          )
          
          if (outlier_action == "remove") {
            y_raw[outliers] <- NA
          }
          
          if (outlier_action == "winsorize") {
            upper_cap <- quantile(y_raw, 0.95, na.rm = TRUE)
            lower_cap <- quantile(y_raw, 0.05, na.rm = TRUE)
            
            y_raw[outliers & y_raw > med] <- upper_cap
            y_raw[outliers & y_raw < med] <- lower_cap
          }
        }
      }
    }
    
    # ----------------------------------------------------------
    # 2.3.2 Fit smoothing spline
    # ----------------------------------------------------------
    
    valid <- !is.na(y_raw) & !is.na(time_raw)
    
    if (sum(valid) < 3) {
      df_out[[v]] <- NA_real_
      df_out[[paste0(v, "_d1")]] <- NA_real_
      df_out[[paste0(v, "_d2")]] <- NA_real_
      next
    }
    
    if (is.null(spar)) {
      spline_fit <- smooth.spline(
        x = time_raw[valid],
        y = y_raw[valid]
      )
    } else {
      spline_fit <- smooth.spline(
        x = time_raw[valid],
        y = y_raw[valid],
        spar = spar
      )
    }
    
    # ---- GCV diagnostics ----
    
    message(sprintf(
      "Variable %s | GCV spar = %.3f | df = %.2f | GCV = %.5f",
      v,
      spline_fit$spar,
      spline_fit$df,
      spline_fit$cv.crit
    ))
    
    y_smooth <- predict(spline_fit, time_regular)$y
    n_reg <- length(y_smooth)
    
    # ----------------------------------------------------------
    # 2.4 Central Difference: First Derivative (vectorized)
    # ----------------------------------------------------------
    
    if (n_reg >= 3) {
      y_d1 <- c(
        NA,
        (y_smooth[3:n_reg] - y_smooth[1:(n_reg - 2)]) / (2 * dt),
        NA
      )
    } else {
      y_d1 <- rep(NA_real_, n_reg)
    }
    
    # ----------------------------------------------------------
    # 2.5 Central Difference: Second Derivative (vectorized)
    # ----------------------------------------------------------
    
    if (n_reg >= 3) {
      y_d2 <- c(
        NA,
        (y_smooth[3:n_reg] - 2 * y_smooth[2:(n_reg - 1)] + y_smooth[1:(n_reg - 2)]) / (dt^2),
        NA
      )
    } else {
      y_d2 <- rep(NA_real_, n_reg)
    }
    
    df_out[[v]] <- y_smooth
    df_out[[paste0(v, "_d1")]] <- y_d1
    df_out[[paste0(v, "_d2")]] <- y_d2
  }
  
  return(df_out)
}



################################################################################
################################################################################


# ==============================================================
# 3. Predictor Standardization
# ==============================================================

# CIRN constructs Class from the raw target derivative first. Only
# model predictors are standardized afterward, once temporal lagging
# has aligned predictors at t_{k-lag} with the target response at t_k.
#
# This prevents the class-label error in which standardized derivatives
# define "above/below mean derivative" instead of true increasing versus
# decreasing dynamics.
# ==============================================================

standardize_selected_numeric_columns <- function(df, cols) {
  
  cols <- intersect(cols, names(df))
  if (length(cols) == 0) {
    return(df)
  }
  
  numeric_cols <- cols[vapply(df[cols], is.numeric, logical(1))]
  if (length(numeric_cols) == 0) {
    return(df)
  }
  
  valid_cols <- numeric_cols[vapply(
    df[numeric_cols],
    function(x) {
      s <- stats::sd(x, na.rm = TRUE)
      is.finite(s) && s > 0
    },
    logical(1)
  )]
  
  if (length(valid_cols) > 0) {
    df <- df %>%
      dplyr::mutate(
        dplyr::across(
          dplyr::all_of(valid_cols),
          ~ as.numeric(scale(.x))
        )
      )
  }
  
  df
}


response_class_counts <- function(class_vector) {
  
  class_vector <- class_vector[!is.na(class_vector)]
  
  class_0_count <- sum(class_vector == 0, na.rm = TRUE)
  class_1_count <- sum(class_vector == 1, na.rm = TRUE)
  n_classes <- sum(c(class_0_count, class_1_count) > 0)
  min_class_count <- if (n_classes == 2) {
    min(class_0_count, class_1_count)
  } else {
    0L
  }
  
  list(
    class_0_count = as.integer(class_0_count),
    class_1_count = as.integer(class_1_count),
    n_classes = as.integer(n_classes),
    min_class_count = as.integer(min_class_count)
  )
}

classify_directional_response <- function(df,
                                          derivative_col,
                                          response_eps,
                                          response_name = "Class") {
  
  df %>%
    dplyr::mutate(
      "{response_name}" := dplyr::case_when(
        .data[[derivative_col]] >  response_eps ~ 1,
        .data[[derivative_col]] < -response_eps ~ 0,
        TRUE ~ NA_real_
      )
    ) %>%
    dplyr::filter(!is.na(.data[[response_name]]))
}

restore_random_seed <- function(seed_exists, old_seed) {
  if (isTRUE(seed_exists)) {
    assign(".Random.seed", old_seed, envir = .GlobalEnv)
  } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    rm(".Random.seed", envir = .GlobalEnv)
  }
  invisible(NULL)
}

make_target_specific_seed <- function(seed, target) {
  target_code <- sum(utf8ToInt(as.character(target)))
  as.integer((as.numeric(seed) + target_code) %% .Machine$integer.max)
}

adaptive_jitter_directional_response <- function(df_source,
                                                 target,
                                                 time_col,
                                                 target_d1,
                                                 response_eps,
                                                 min_class_count = 5,
                                                 jitter_scale_grid = c(
                                                   1e-8, 3e-8,
                                                   1e-7, 3e-7,
                                                   1e-6, 3e-6,
                                                   1e-5, 3e-5,
                                                   1e-4, 3e-4,
                                                   1e-3, 3e-3,
                                                   1e-2
                                                 ),
                                                 jitter_scale_basis = c(
                                                   "state_sd",
                                                   "state_range",
                                                   "derivative_sd",
                                                   "derivative_max_abs",
                                                   "absolute"
                                                 ),
                                                 seed = 123) {
  
  jitter_scale_basis <- match.arg(jitter_scale_basis)
  
  if (!target %in% names(df_source)) {
    stop("Target state column not found for adaptive jitter: ", target)
  }
  if (!target_d1 %in% names(df_source)) {
    stop("Target derivative column not found for adaptive jitter: ", target_d1)
  }
  if (!time_col %in% names(df_source)) {
    stop("time_col not found for adaptive jitter: ", time_col)
  }
  
  if (length(jitter_scale_grid) == 0 ||
      any(!is.finite(jitter_scale_grid)) ||
      any(jitter_scale_grid <= 0)) {
    stop("jitter_scale_grid must contain positive finite values.")
  }
  
  jitter_scale_grid <- sort(unique(as.numeric(jitter_scale_grid)))
  min_class_count <- as.integer(min_class_count)
  
  time_values <- df_source[[time_col]]
  target_values <- df_source[[target]]
  derivative_values <- df_source[[target_d1]]
  
  finite_state <- target_values[is.finite(target_values)]
  finite_derivative <- derivative_values[is.finite(derivative_values)]
  
  basis_value <- switch(
    jitter_scale_basis,
    state_sd = stats::sd(finite_state, na.rm = TRUE),
    state_range = diff(range(finite_state, na.rm = TRUE)),
    derivative_sd = stats::sd(finite_derivative, na.rm = TRUE),
    derivative_max_abs = max(abs(finite_derivative), na.rm = TRUE),
    absolute = 1
  )
  
  if (!is.finite(basis_value) || basis_value <= 0) {
    backup_basis <- suppressWarnings(max(abs(finite_state), na.rm = TRUE))
    basis_value <- if (is.finite(backup_basis) && backup_basis > 0) {
      backup_basis
    } else {
      1
    }
  }
  
  seed_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed <- if (seed_exists) {
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  } else {
    NULL
  }
  on.exit(restore_random_seed(seed_exists, old_seed), add = TRUE)
  
  set.seed(make_target_specific_seed(seed, target))
  
  n <- nrow(df_source)
  
  for (attempt in seq_along(jitter_scale_grid)) {
    
    scale_factor <- jitter_scale_grid[[attempt]]
    jitter_sd <- basis_value * scale_factor
    jitter <- stats::rnorm(n, mean = 0, sd = jitter_sd)
    
    jittered_state <- target_values + jitter
    jittered_d1 <- rep(NA_real_, n)
    
    if (n >= 3) {
      denominator <- time_values[3:n] - time_values[1:(n - 2)]
      valid_denominator <- is.finite(denominator) & denominator != 0
      derivative_inner <- rep(NA_real_, length(denominator))
      derivative_inner[valid_denominator] <-
        (jittered_state[3:n][valid_denominator] -
           jittered_state[1:(n - 2)][valid_denominator]) /
        denominator[valid_denominator]
      jittered_d1[2:(n - 1)] <- derivative_inner
    }
    
    jittered_d1_col <- paste0(target, "_d1_jittered")
    jittered_state_col <- paste0(target, "_jittered")
    
    candidate <- df_source
    candidate[[jittered_state_col]] <- jittered_state
    candidate[[jittered_d1_col]] <- jittered_d1
    candidate_full <- candidate
    
    candidate <- classify_directional_response(
      df = candidate,
      derivative_col = jittered_d1_col,
      response_eps = response_eps
    )
    
    counts <- response_class_counts(candidate$Class)
    
    if (counts$n_classes >= 2 && counts$min_class_count >= min_class_count) {
      return(
        list(
          success = TRUE,
          data = candidate,
          data_full = candidate_full,
          response_source = jittered_d1_col,
          jitter_scale = jitter_sd,
          jitter_scale_factor = scale_factor,
          jitter_basis = jitter_scale_basis,
          jitter_basis_value = basis_value,
          jitter_attempts = as.integer(attempt),
          class_0_count = counts$class_0_count,
          class_1_count = counts$class_1_count,
          min_class_count = counts$min_class_count
        )
      )
    }
  }
  
  list(
    success = FALSE,
    data = NULL,
    response_source = NA_character_,
    jitter_scale = NA_real_,
    jitter_scale_factor = NA_real_,
    jitter_basis = jitter_scale_basis,
    jitter_basis_value = basis_value,
    jitter_attempts = as.integer(length(jitter_scale_grid)),
    class_0_count = NA_integer_,
    class_1_count = NA_integer_,
    min_class_count = NA_integer_
  )
}

compute_grid_derivatives_from_state <- function(time_values, state_values) {
  n <- length(state_values)
  d1 <- rep(NA_real_, n)
  d2 <- rep(NA_real_, n)
  
  if (n >= 3) {
    first_denominator <- time_values[3:n] - time_values[1:(n - 2)]
    valid_first <- is.finite(first_denominator) & first_denominator != 0
    first_inner <- rep(NA_real_, length(first_denominator))
    first_inner[valid_first] <-
      (state_values[3:n][valid_first] -
         state_values[1:(n - 2)][valid_first]) /
      first_denominator[valid_first]
    d1[2:(n - 1)] <- first_inner
    
    dt_prev <- time_values[2:(n - 1)] - time_values[1:(n - 2)]
    dt_next <- time_values[3:n] - time_values[2:(n - 1)]
    valid_second <- is.finite(dt_prev) &
      is.finite(dt_next) &
      dt_prev != 0 &
      dt_next != 0
    second_inner <- rep(NA_real_, length(dt_prev))
    second_inner[valid_second] <-
      2 * (
        ((state_values[3:n][valid_second] - state_values[2:(n - 1)][valid_second]) /
           dt_next[valid_second]) -
          ((state_values[2:(n - 1)][valid_second] - state_values[1:(n - 2)][valid_second]) /
             dt_prev[valid_second])
      ) /
      (dt_prev[valid_second] + dt_next[valid_second])
    d2[2:(n - 1)] <- second_inner
  }
  
  list(d1 = d1, d2 = d2)
}

apply_predictor_jitter_sensitivity <- function(df_model,
                                               df_source,
                                               time_col,
                                               predictor_vars,
                                               jitter_scale_factor,
                                               jitter_scale_basis = c(
                                                 "state_sd",
                                                 "state_range",
                                                 "derivative_sd",
                                                 "derivative_max_abs",
                                                 "absolute"
                                               ),
                                               seed = 123,
                                               target = "target") {
  
  jitter_scale_basis <- match.arg(jitter_scale_basis)
  predictor_vars <- unique(as.character(unlist(predictor_vars)))
  predictor_vars <- predictor_vars[nzchar(predictor_vars)]
  predictor_vars <- setdiff(predictor_vars, time_col)
  predictor_vars <- predictor_vars[predictor_vars %in% names(df_source)]
  
  if (length(predictor_vars) == 0 ||
      !is.finite(jitter_scale_factor) ||
      jitter_scale_factor <= 0) {
    return(
      list(
        data = df_model,
        predictor_jitter_used = FALSE,
        predictor_jitter_scale_factor = NA_real_,
        predictor_jitter_variables = NA_character_
      )
    )
  }
  
  time_values <- df_source[[time_col]]
  seed_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  old_seed <- if (seed_exists) {
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  } else {
    NULL
  }
  on.exit(restore_random_seed(seed_exists, old_seed), add = TRUE)
  
  set.seed(make_target_specific_seed(seed + 100000L, paste0(target, "_predictor_jitter")))
  
  jittered_source <- df_source
  
  for (v in predictor_vars) {
    state_values <- df_source[[v]]
    derivative_values <- df_source[[paste0(v, "_d1")]]
    
    finite_state <- state_values[is.finite(state_values)]
    finite_derivative <- derivative_values[is.finite(derivative_values)]
    
    basis_value <- switch(
      jitter_scale_basis,
      state_sd = stats::sd(finite_state, na.rm = TRUE),
      state_range = diff(range(finite_state, na.rm = TRUE)),
      derivative_sd = stats::sd(finite_derivative, na.rm = TRUE),
      derivative_max_abs = max(abs(finite_derivative), na.rm = TRUE),
      absolute = 1
    )
    
    if (!is.finite(basis_value) || basis_value <= 0) {
      backup_basis <- suppressWarnings(max(abs(finite_state), na.rm = TRUE))
      basis_value <- if (is.finite(backup_basis) && backup_basis > 0) {
        backup_basis
      } else {
        1
      }
    }
    
    jitter_sd <- basis_value * jitter_scale_factor
    jitter <- stats::rnorm(length(state_values), mean = 0, sd = jitter_sd)
    jittered_state <- state_values + jitter
    jittered_derivs <- compute_grid_derivatives_from_state(
      time_values = time_values,
      state_values = jittered_state
    )
    
    jittered_source[[v]] <- jittered_state
    jittered_source[[paste0(v, "_d1")]] <- jittered_derivs$d1
    jittered_source[[paste0(v, "_d2")]] <- jittered_derivs$d2
  }
  
  row_index <- match(df_model[[time_col]], jittered_source[[time_col]])
  replacement_cols <- unique(c(
    predictor_vars,
    paste0(predictor_vars, "_d1"),
    paste0(predictor_vars, "_d2")
  ))
  replacement_cols <- intersect(replacement_cols, names(df_model))
  
  for (col in replacement_cols) {
    df_model[[col]] <- jittered_source[[col]][row_index]
  }
  
  list(
    data = df_model,
    predictor_jitter_used = TRUE,
    predictor_jitter_scale_factor = jitter_scale_factor,
    predictor_jitter_variables = paste(predictor_vars, collapse = ",")
  )
}


################################################################################
################################################################################


# ==============================================================
# 4. CIRN Inference for a Single Target Variable
# ==============================================================

# This function implements the core CIRN inference procedure
# for one target variable x_j.
#
# Conceptually, CIRN estimates:
#   P( Class_j(t_k) | x_i(t), dx_i/dt, d²x_i/dt² )
#
# via Bayesian logistic regression, where:
#   • The primary response is the direction of change of x_j
#   • Degenerate or nearly degenerate derivative responses may use
#     adaptive minimal jitter when enabled
#   • Predictors are lagged state variables and derivatives
#   • Temporal precedence is explicitly encoded
#
# The output consists of:
#   • Inferred directed, signed edges
#   • Posterior diagnostics
#   • Fitted Bayesian models (for visualization and diagnostics)
#
# DESIGN CHOICE:
# CIRN employs a single-step temporal lag to encode temporal
# precedence while maintaining interpretability and tractability.
# Multi-lag, multi-scale, or memory-based extensions are
# conceptually compatible but left for future work.
# ==============================================================

# --------------------------------------------------------------
# Whenever model fails or has no predictors
# --------------------------------------------------------------

empty_bayes_result_tibble <- tibble(
  term = character(),
  target = character(),
  omega = numeric(),
  odds_ratio = numeric(),
  sign = numeric(),
  regulation_type = character(),
  rel_strength = character(),
  hdi_lower95 = numeric(),
  hdi_upper95 = numeric(),
  eti_lower95 = numeric(),
  eti_upper95 = numeric(),
  response_mode = character(),
  response_label = character(),
  response_source = character(),
  response_trigger = character(),
  dominant_direction = numeric(),
  class_0_count = integer(),
  class_1_count = integer(),
  min_class_count = integer(),
  jitter_used = logical(),
  jitter_scale = numeric(),
  jitter_scale_factor = numeric(),
  jitter_basis = character(),
  jitter_basis_value = numeric(),
  jitter_attempts = integer(),
  predictor_jitter_used = logical(),
  predictor_jitter_scale_factor = numeric(),
  predictor_jitter_variables = character(),
  effect_interpretation = character(),
  method = character()
)

empty_bayes_all_coefficients_tibble <- empty_bayes_result_tibble %>%
  dplyr::mutate(
    retained = logical(),
    credibility = character()
  )

empty_vif_group_tibble <- tibble(
  target = character(),
  predictor_set = character(),
  term = character(),
  vif = numeric(),
  status = character(),
  aliased = logical(),
  model_rank = integer(),
  model_columns = integer(),
  rank_deficient = logical(),
  note = character()
)

empty_vif_pair_tibble <- tibble(
  target = character(),
  predictor_set = character(),
  predictor_1 = character(),
  predictor_2 = character(),
  correlation = numeric(),
  abs_correlation = numeric(),
  status = character()
)

cirn_sets_for_mode <- function(representation_mode) {
  switch(
    as.character(representation_mode),
    sublevel = c("original", "first_derivative", "second_derivative"),
    all_predictors = "all_predictors",
    both = c("original", "first_derivative", "second_derivative", "all_predictors")
  )
}

make_empty_bayes_result <- function(target,
                                    status = "skipped",
                                    response_mode = NA_character_,
                                    response_label = NA_character_,
                                    response_source = NA_character_,
                                    dominant_direction = NA_real_,
                                    jitter_used = FALSE,
                                    jitter_scale = NA_real_,
                                    jitter_scale_factor = NA_real_,
                                    jitter_basis = NA_character_,
                                    jitter_basis_value = NA_real_,
                                    jitter_attempts = NA_integer_,
                                    predictor_jitter_used = FALSE,
                                    predictor_jitter_scale_factor = NA_real_,
                                    predictor_jitter_variables = NA_character_,
                                    class_0_count = NA_integer_,
                                    class_1_count = NA_integer_,
                                    min_class_count = NA_integer_,
                                    n_predictors = NA_integer_,
                                    response_trigger = NA_character_) {
  list(
    coefficients = empty_bayes_result_tibble %>%
      dplyr::mutate(
        target = paste0("d", target),
        response_mode = response_mode,
        response_label = response_label,
        response_source = response_source,
        dominant_direction = dominant_direction,
        jitter_used = jitter_used,
        jitter_scale = jitter_scale,
        jitter_scale_factor = jitter_scale_factor,
        jitter_basis = jitter_basis,
        jitter_basis_value = jitter_basis_value,
        jitter_attempts = jitter_attempts,
        predictor_jitter_used = predictor_jitter_used,
        predictor_jitter_scale_factor = predictor_jitter_scale_factor,
        predictor_jitter_variables = predictor_jitter_variables,
        class_0_count = class_0_count,
        class_1_count = class_1_count,
        min_class_count = min_class_count,
        n_predictors = n_predictors,
        response_trigger = response_trigger
      ),
    all_coefficients = empty_bayes_all_coefficients_tibble %>%
      dplyr::mutate(
        target = paste0("d", target),
        response_mode = response_mode,
        response_label = response_label,
        response_source = response_source,
        dominant_direction = dominant_direction,
        jitter_used = jitter_used,
        jitter_scale = jitter_scale,
        jitter_scale_factor = jitter_scale_factor,
        jitter_basis = jitter_basis,
        jitter_basis_value = jitter_basis_value,
        jitter_attempts = jitter_attempts,
        predictor_jitter_used = predictor_jitter_used,
        predictor_jitter_scale_factor = predictor_jitter_scale_factor,
        predictor_jitter_variables = predictor_jitter_variables,
        class_0_count = class_0_count,
        class_1_count = class_1_count,
        min_class_count = min_class_count,
        n_predictors = n_predictors,
        response_trigger = response_trigger
      ),
    diagnostics = list(
      rhat = NA_real_,
      neff_ratio = NA_real_,
      loo = NULL,
      status = status,
      response_mode = response_mode,
      response_label = response_label,
      response_source = response_source,
      dominant_direction = dominant_direction,
      jitter_used = jitter_used,
      jitter_scale = jitter_scale,
      jitter_scale_factor = jitter_scale_factor,
      jitter_basis = jitter_basis,
      jitter_basis_value = jitter_basis_value,
      jitter_attempts = jitter_attempts,
      predictor_jitter_used = predictor_jitter_used,
      predictor_jitter_scale_factor = predictor_jitter_scale_factor,
      predictor_jitter_variables = predictor_jitter_variables,
      class_0_count = class_0_count,
      class_1_count = class_1_count,
      min_class_count = min_class_count,
      n_predictors = n_predictors,
      response_trigger = response_trigger
    ),
    model = NULL,
    vif_group = empty_vif_group_tibble,
    vif_pairs = empty_vif_pair_tibble
  )
}

make_empty_target_result <- function(target,
                                     representation_mode,
                                     debug_full = NULL,
                                     status = "skipped",
                                     response_mode = NA_character_,
                                     response_label = NA_character_,
                                     response_source = NA_character_,
                                     dominant_direction = NA_real_,
                                     jitter_used = FALSE,
                                     jitter_scale = NA_real_,
                                     jitter_scale_factor = NA_real_,
                                     jitter_basis = NA_character_,
                                     jitter_basis_value = NA_real_,
                                     jitter_attempts = NA_integer_,
                                     predictor_jitter_used = FALSE,
                                     predictor_jitter_scale_factor = NA_real_,
                                     predictor_jitter_variables = NA_character_,
                                     class_0_count = NA_integer_,
                                     class_1_count = NA_integer_,
                                     min_class_count = NA_integer_,
                                     n_predictors = NA_integer_,
                                     response_trigger = NA_character_) {
  predictor_sets <- cirn_sets_for_mode(representation_mode)
  results <- stats::setNames(
    lapply(
      predictor_sets,
      function(...) {
        make_empty_bayes_result(
          target = target,
          status = status,
          response_mode = response_mode,
          response_label = response_label,
          response_source = response_source,
          dominant_direction = dominant_direction,
          jitter_used = jitter_used,
          jitter_scale = jitter_scale,
          jitter_scale_factor = jitter_scale_factor,
          jitter_basis = jitter_basis,
          jitter_basis_value = jitter_basis_value,
          jitter_attempts = jitter_attempts,
          predictor_jitter_used = predictor_jitter_used,
          predictor_jitter_scale_factor = predictor_jitter_scale_factor,
          predictor_jitter_variables = predictor_jitter_variables,
          class_0_count = class_0_count,
          class_1_count = class_1_count,
          min_class_count = min_class_count,
          n_predictors = n_predictors,
          response_trigger = response_trigger
        )
      }
    ),
    predictor_sets
  )
  
  edges_tbl <- dplyr::bind_rows(
    purrr::map(results, "coefficients"),
    .id = "predictor_set"
  ) %>%
    dplyr::filter(!is.na(.data$omega))
  
  all_coefficients_tbl <- dplyr::bind_rows(
    purrr::map(results, "all_coefficients"),
    .id = "predictor_set"
  ) %>%
    dplyr::filter(!is.na(.data$omega))
  
  list(
    edges = edges_tbl,
    all_coefficients = all_coefficients_tbl,
    diagnostics = results,
    models = lapply(results, function(x) x$model),
    vif_group = purrr::map_df(results, "vif_group"),
    vif_pairs = purrr::map_df(results, "vif_pairs"),
    debug = debug_full
  )
}

classify_vif <- function(vif) {
  dplyr::case_when(
    is.na(vif) ~ "not_available",
    is.infinite(vif) ~ "exact_collinearity",
    vif >= 10 ~ "severe_collinearity",
    vif >= 5 ~ "high_collinearity",
    TRUE ~ "ok"
  )
}

classify_abs_correlation <- function(abs_cor) {
  dplyr::case_when(
    is.na(abs_cor) ~ "not_available",
    abs_cor >= 0.999999 ~ "exact_or_near_exact_collinearity",
    abs_cor >= 0.9 ~ "high_pairwise_collinearity",
    TRUE ~ "ok"
  )
}

compute_collinearity_diagnostics <- function(df_model,
                                             form,
                                             pred_set,
                                             target,
                                             predictor_set) {
  target_label <- paste0("d", target)
  
  if (length(pred_set) == 0) {
    return(list(
      vif_group = empty_vif_group_tibble,
      vif_pairs = empty_vif_pair_tibble
    ))
  }
  
  mm <- model.matrix(form, data = df_model)
  model_rank <- qr(mm)$rank
  model_columns <- ncol(mm)
  rank_deficient <- model_rank < model_columns
  
  vif_pairs <- empty_vif_pair_tibble
  
  if (length(pred_set) > 1) {
    pair_mat <- utils::combn(pred_set, 2)
    
    vif_pairs <- purrr::map_dfr(seq_len(ncol(pair_mat)), function(k) {
      p1 <- pair_mat[1, k]
      p2 <- pair_mat[2, k]
      
      complete <- stats::complete.cases(df_model[[p1]], df_model[[p2]])
      enough_data <- sum(complete) >= 3
      non_constant <- enough_data &&
        stats::sd(df_model[[p1]][complete]) > 0 &&
        stats::sd(df_model[[p2]][complete]) > 0
      
      corr <- if (non_constant) {
        stats::cor(df_model[[p1]][complete], df_model[[p2]][complete])
      } else {
        NA_real_
      }
      
      tibble(
        target = target_label,
        predictor_set = predictor_set,
        predictor_1 = p1,
        predictor_2 = p2,
        correlation = corr,
        abs_correlation = abs(corr),
        status = classify_abs_correlation(abs(corr))
      )
    })
  }
  
  if (length(pred_set) == 1) {
    vif_group <- tibble(
      target = target_label,
      predictor_set = predictor_set,
      term = pred_set,
      vif = NA_real_,
      status = "single_predictor",
      aliased = FALSE,
      model_rank = model_rank,
      model_columns = model_columns,
      rank_deficient = rank_deficient,
      note = "VIF requires at least two predictors."
    )
    
    return(list(
      vif_group = vif_group,
      vif_pairs = vif_pairs
    ))
  }
  
  vif_model <- lm(form, data = df_model)
  aliased_terms <- names(coef(vif_model))[is.na(coef(vif_model))]
  aliased_terms <- setdiff(aliased_terms, "(Intercept)")
  
  vif_values <- tryCatch(
    car::vif(vif_model),
    error = function(e) e
  )
  
  if (inherits(vif_values, "error")) {
    note <- paste("VIF unavailable:", conditionMessage(vif_values))
    
    vif_group <- tibble(
      target = target_label,
      predictor_set = predictor_set,
      term = pred_set,
      vif = NA_real_,
      status = ifelse(
        pred_set %in% aliased_terms,
        "aliased_coefficient",
        "rank_deficient_model"
      ),
      aliased = pred_set %in% aliased_terms,
      model_rank = model_rank,
      model_columns = model_columns,
      rank_deficient = rank_deficient,
      note = note
    )
    
    warning(
      "VIF diagnostic is non-fatal for target = ",
      target_label,
      ", predictor_set = ",
      predictor_set,
      ". ",
      conditionMessage(vif_values)
    )
    
    return(list(
      vif_group = vif_group,
      vif_pairs = vif_pairs
    ))
  }
  
  if (is.matrix(vif_values)) {
    vif_tbl <- as.data.frame(vif_values) %>%
      tibble::rownames_to_column("term")
    
    if ("GVIF^(1/(2*Df))" %in% names(vif_tbl)) {
      vif_tbl <- vif_tbl %>%
        transmute(term, vif = .data[["GVIF^(1/(2*Df))"]])
    } else {
      first_vif_col <- names(vif_tbl)[2]
      vif_tbl <- vif_tbl %>%
        transmute(term, vif = .data[[first_vif_col]])
    }
  } else {
    vif_tbl <- tibble(
      term = names(vif_values),
      vif = as.numeric(vif_values)
    )
  }
  
  vif_group <- vif_tbl %>%
    mutate(
      target = target_label,
      predictor_set = predictor_set,
      status = classify_vif(vif),
      aliased = term %in% aliased_terms,
      model_rank = model_rank,
      model_columns = model_columns,
      rank_deficient = rank_deficient,
      note = ifelse(
        rank_deficient,
        "Model matrix is rank-deficient; interpret coefficients cautiously.",
        "VIF computed successfully."
      )
    ) %>%
    dplyr::select(
      target,
      predictor_set,
      term,
      vif,
      status,
      aliased,
      model_rank,
      model_columns,
      rank_deficient,
      note
    )
  
  list(
    vif_group = vif_group,
    vif_pairs = vif_pairs
  )
}

call_cirn_progress <- function(progress_callback, ...) {
  
  if (!is.function(progress_callback)) {
    return(invisible(NULL))
  }
  
  progress_args <- list(...)
  callback_formals <- names(formals(progress_callback))
  
  if (!is.null(callback_formals) && !("..." %in% callback_formals)) {
    progress_args <- progress_args[intersect(names(progress_args), callback_formals)]
  }
  
  tryCatch(
    do.call(progress_callback, progress_args),
    error = function(e) {
      warning("progress_callback failed: ", conditionMessage(e), call. = FALSE)
    }
  )
  
  invisible(NULL)
}

infer_for_target <- function(df_derivs,
                             target,
                             time_col,
                             predictors = NULL,
                             representation_mode = c("sublevel", "all_predictors", "both"),
                             lag_units = 1,
                             response_eps = 1e-6,
                             adaptive_jitter = TRUE,
                             jitter_predictors = FALSE,
                             jitter_min_class_count = 5,
                             jitter_scale_grid = c(
                               1e-8, 3e-8,
                               1e-7, 3e-7,
                               1e-6, 3e-6,
                               1e-5, 3e-5,
                               1e-4, 3e-4,
                               1e-3, 3e-3,
                               1e-2
                             ),
                             jitter_scale_basis = c(
                               "state_sd",
                               "state_range",
                               "derivative_sd",
                               "derivative_max_abs",
                               "absolute"
                             ),
                             model_iter = 3000,
                             model_warmup = 1000,
                             model_chains = 4,
                             model_cores = model_chains,
                             prior_mean = 0,
                             prior_sd = 2,
                             adapt_delta = 0.95,
                             compute_loo = FALSE,
                             loo_moment_match = FALSE,
                             loo_reloo = NULL,
                             loo_k_threshold = 0.7,
                             seed = 123,
                             progress_callback = NULL,
                             debug = FALSE,
                             assign_debug_to_global = FALSE) { 
  
  representation_mode <- match.arg(representation_mode)
  jitter_scale_basis <- match.arg(jitter_scale_basis)
  if (is.null(loo_reloo)) {
    loo_reloo <- isTRUE(compute_loo) && isTRUE(loo_moment_match)
  }
  
  if (length(lag_units) != 1 || is.na(lag_units) || lag_units < 1 || lag_units != floor(lag_units)) {
    stop("lag_units must be a single positive integer.")
  }
  lag_units <- as.integer(lag_units)
  
  if (!is.logical(jitter_predictors) ||
      length(jitter_predictors) != 1 ||
      is.na(jitter_predictors)) {
    stop("jitter_predictors must be TRUE or FALSE.")
  }
  
  if (length(jitter_min_class_count) != 1 ||
      is.na(jitter_min_class_count) ||
      jitter_min_class_count < 1 ||
      jitter_min_class_count != floor(jitter_min_class_count)) {
    stop("jitter_min_class_count must be a single positive integer.")
  }
  jitter_min_class_count <- as.integer(jitter_min_class_count)
  
  # --------------------------------------------------------------
  # 4.1 Data preparation
  # --------------------------------------------------------------
  
  # df_derivs is prepared upstream by infer_network():
  #   • analysis variables are selected,
  #   • complete cases are retained for those variables,
  #   • derivatives are computed on the regularized time grid.
  
  # Name of the first derivative of the target variable
  
  target_d1 <- paste0(target, "_d1")
  if (!target_d1 %in% names(df_derivs)) {
    stop("Target derivative column not found: ", target_d1)
  }
  
  debug_full <- NULL
  
  # --------------------------------------------------------------
  # 4.2 Binary response construction (CIRN core idea)
  # --------------------------------------------------------------
  
  # The primary response variable encodes the LOCAL DIRECTION of change
  # of the target variable x_j.
  #
  # Class = 1  → x_j is increasing
  # Class = 0  → x_j is decreasing
  #
  # Near-zero derivatives are discarded to avoid numerical noise.
  # IMPORTANT: Class must be constructed from the raw, unstandardized
  # target derivative so that Class = 1 means increasing and Class = 0
  # means decreasing.
  #
  # Adaptive minimal jitter:
  # If the raw directional response contains one class or too few
  # observations in one derivative class, CIRN can add zero-mean jitter
  # to the target trajectory, recompute the target derivative, and retain
  # the smallest jitter scale that creates a usable two-class response.
  # Jitter is applied to the target trajectory used for response
  # construction only. It is not applied directly to Class labels.
  
  eps <- response_eps  # tolerance for numerical stability
  response_mode <- "directional"
  response_label <- "raw_first_derivative_direction"
  response_source <- target_d1
  dominant_direction <- NA_real_
  jitter_used <- FALSE
  jitter_scale <- NA_real_
  jitter_scale_factor <- NA_real_
  jitter_basis <- NA_character_
  jitter_basis_value <- NA_real_
  jitter_attempts <- NA_integer_
  predictor_jitter_used <- FALSE
  predictor_jitter_scale_factor <- NA_real_
  predictor_jitter_variables <- NA_character_
  response_trigger <- "raw_directional"
  
  df_response_source <- df_derivs
  
  finite_d1 <- df_response_source[[target_d1]]
  finite_d1 <- finite_d1[is.finite(finite_d1) & abs(finite_d1) > eps]
  dominant_direction <- if (length(finite_d1) > 0) {
    sign(stats::median(finite_d1, na.rm = TRUE))
  } else {
    NA_real_
  }
  
  if (!is.finite(dominant_direction) || dominant_direction == 0) {
    mean_d1 <- mean(df_response_source[[target_d1]], na.rm = TRUE)
    dominant_direction <- if (is.finite(mean_d1) && mean_d1 != 0) {
      sign(mean_d1)
    } else {
      NA_real_
    }
  }
  
  df_derivs <- classify_directional_response(
    df = df_response_source,
    derivative_col = target_d1,
    response_eps = eps
  )
  
  raw_counts <- response_class_counts(df_derivs$Class)
  class_0_count <- raw_counts$class_0_count
  class_1_count <- raw_counts$class_1_count
  min_class_count <- raw_counts$min_class_count
  
  raw_directional_has_two_classes <- raw_counts$n_classes >= 2
  raw_directional_is_sparse <- raw_counts$min_class_count < jitter_min_class_count
  
  if (!raw_directional_has_two_classes) {
    response_trigger <- "one_class_raw_derivative"
  } else if (raw_directional_is_sparse) {
    response_trigger <- "near_one_class_raw_derivative"
  }
  
  if (isTRUE(adaptive_jitter) &&
      (!raw_directional_has_two_classes || raw_directional_is_sparse)) {
    
    jitter_result <- adaptive_jitter_directional_response(
      df_source = df_response_source,
      target = target,
      time_col = time_col,
      target_d1 = target_d1,
      response_eps = eps,
      min_class_count = jitter_min_class_count,
      jitter_scale_grid = jitter_scale_grid,
      jitter_scale_basis = jitter_scale_basis,
      seed = seed
    )
    
    jitter_basis <- jitter_result$jitter_basis
    jitter_basis_value <- jitter_result$jitter_basis_value
    jitter_attempts <- jitter_result$jitter_attempts
    
    if (isTRUE(jitter_result$success)) {
      df_derivs <- jitter_result$data
      response_mode <- "adaptive_jitter_directional"
      response_label <- "jittered_first_derivative_direction"
      response_source <- jitter_result$response_source
      jitter_used <- TRUE
      jitter_scale <- jitter_result$jitter_scale
      jitter_scale_factor <- jitter_result$jitter_scale_factor
      class_0_count <- jitter_result$class_0_count
      class_1_count <- jitter_result$class_1_count
      min_class_count <- jitter_result$min_class_count
      
      warning(
        "Target = ", target,
        " required adaptive minimal jitter for derivative-response encoding. ",
        "Scale factor = ", signif(jitter_scale_factor, 4),
        "; jitter SD = ", signif(jitter_scale, 4),
        "; class counts: 0 = ", class_0_count,
        ", 1 = ", class_1_count,
        ".",
        call. = FALSE
      )
    } else if (raw_directional_has_two_classes) {
      response_trigger <- paste0(response_trigger, "_jitter_failed_raw_retained")
      warning(
        "Target = ", target,
        " had a sparse raw derivative response, but adaptive jitter did not ",
        "reach the requested minimum class count. Retaining the raw response.",
        call. = FALSE
      )
    } else {
      response_trigger <- paste0(response_trigger, "_jitter_failed")
    }
  }
  
  if (nrow(df_derivs) < 2 || dplyr::n_distinct(df_derivs$Class) < 2) {
    warning(
      "Skipping target = ", target,
      " because response encoding produced fewer than two classes.",
      call. = FALSE
    )
    
    if (debug) {
      debug_full <- df_derivs
    }
    
    return(
      make_empty_target_result(
        target = target,
        representation_mode = representation_mode,
        debug_full = debug_full,
        status = "skipped_single_class",
        response_mode = response_mode,
        response_label = response_label,
        response_source = response_source,
        dominant_direction = dominant_direction,
        jitter_used = jitter_used,
        jitter_scale = jitter_scale,
        jitter_scale_factor = jitter_scale_factor,
        jitter_basis = jitter_basis,
        jitter_basis_value = jitter_basis_value,
        jitter_attempts = jitter_attempts,
        predictor_jitter_used = predictor_jitter_used,
        predictor_jitter_scale_factor = predictor_jitter_scale_factor,
        predictor_jitter_variables = predictor_jitter_variables,
        class_0_count = class_0_count,
        class_1_count = class_1_count,
        min_class_count = min_class_count,
        n_predictors = 0L,
        response_trigger = response_trigger
      )
    )
  }
  
  effect_interpretation <- if (identical(response_mode, "adaptive_jitter_directional")) {
    "positive_beta_increases_probability_of_target_increase_under_minimal_jitter"
  } else {
    "positive_beta_increases_probability_of_target_increase"
  }
  
  # NOTE (1):
  # CIRN intentionally models a binary target-response state rather
  # than the derivative magnitude. In the primary mode, this response
  # is the direction of temporal change, sign(dx_j/dt). When the raw
  # derivative response is degenerate or nearly degenerate, adaptive
  # minimal jitter preserves this directional target by perturbing the
  # target trajectory only as much as needed to create a valid two-class
  # derivative response. The active response is recorded in response_mode,
  # response_label, response_source, jitter metadata, and
  # effect_interpretation.
  # Quantitative effect sizes and trajectory prediction are outside
  # the scope of the CIRN framework (see Chapter III).
  
  # NOTE (2):
  # Class imbalance is handled implicitly through Bayesian posterior
  # uncertainty rather than explicit resampling or threshold adjustment.
  
  # --------------------------------------------------------------
  # 4.3 Enforcing temporal precedence (CRITICAL STEP)
  # --------------------------------------------------------------
  
  # This step enforces the temporal ordering used by CIRN.
  #
  # In CIRN, we do NOT model the state x_j(t) itself.
  # Instead, we model the target's binary response state:
  #
  #     directional mode:
  #       Class_j(t_k) = sign( d x_j / d t evaluated at t_k )
  #
  #     adaptive minimal-jitter mode:
  #       Class_j(t_k) = sign( d x_j^*(t_k) / d t )
  #       where x_j^*(t) is the minimally jittered target trajectory.
  #
  # To ensure temporally ordered interpretability, predictors must strictly
  # precede the response in time.
  #
  # Therefore, ALL predictor variables are lagged by lag_units time step(s):
  #
  #     x_i(t_{k-lag_units})  →  Class_j(t_k)
  #
  # The default is lag_units = 1, which is the standard CIRN
  # assumption. Larger values allow the analyst to test delayed
  # regulatory effects when the sampling design or domain knowledge
  # suggests that responses occur after more than one observed interval.
  #
  # This construction ensures:
  #
  #   • Temporal precedence:
  #       Predictor values occur BEFORE the observed change
  #
  #   • No future information leakage:
  #       x_i(t_{k+1}) is never used to explain x_j(t_k)
  #
  #   • Directional interpretability of inferred edges:
  #       An edge x_i → x_j means:
  #       "The state of x_i at the previous time step shifts the
  #        probability of the target response class at t_k."
  #
  # Importantly:
  #   • The response variable (Class) is NOT lagged
  #   • Only predictors are lagged
  #   • Rows introduced with missing values due to lagging
  #     are removed to maintain alignment
  #
  # This temporal alignment is what allows CIRN to infer
  # directed regulatory structure rather than mere association.
  #
  # Without this step, the inferred network would be
  # statistically symmetric and temporally uninformative.
  # --------------------------------------------------------------
  
  # Epistemic clarification:
  # Temporal lagging enforces directional asymmetry consistent with
  # regulatory reasoning under observational data. However, CIRN does
  # not claim interventional causality; inferred edges represent
  # probabilistically supported regulatory hypotheses conditional on
  # temporal ordering and model assumptions.
  
  # --------------------------------------------------------------
  # Structural predictor handling (UNIFIED VERSION)
  # --------------------------------------------------------------
  
  # In the structurally constrained version of CIRN:
  #   • 'predictors' are declared in Section 5
  #   • Time variables are automatically excluded
  #   • The target first derivative is excluded to avoid using d[target] as target_d1
  #   • Only declared predictors are lagged
  #
  # If predictors are not explicitly specified,
  # CIRN defaults to the symmetric behavior:
  #   all non-time variables are allowed as predictors.
  # --------------------------------------------------------------
  
  # If predictors not specified → symmetric CIRN behavior
  
  if (is.null(predictors)) {
    predictors <- setdiff(names(df_derivs), c(time_col, "Class"))
    predictors <- predictors[!grepl("_(d1|d2)$", predictors)]
    predictors <- predictors[!grepl("_jittered$", predictors)]
  }
  
  # Structural hygiene:
  # Remove time from predictor space, but keep the target state when declared.
  # Exclude only the target first derivative because d[target] is target_d1.
  
  predictors <- setdiff(predictors, time_col)
  
  # Optional sensitivity mode:
  # By default, adaptive jitter is used only for target-response
  # construction. If jitter_predictors = TRUE, CIRN also jitters the
  # declared predictor state trajectories and recomputes their first
  # and second derivatives. This is intended only as a robustness
  # analysis and is activated only when adaptive response jitter was
  # actually used for the current target.
  
  if (isTRUE(jitter_predictors) && isTRUE(jitter_used)) {
    predictor_jitter_result <- apply_predictor_jitter_sensitivity(
      df_model = df_derivs,
      df_source = df_response_source,
      time_col = time_col,
      predictor_vars = predictors,
      jitter_scale_factor = jitter_scale_factor,
      jitter_scale_basis = jitter_scale_basis,
      seed = seed,
      target = target
    )
    
    df_derivs <- predictor_jitter_result$data
    predictor_jitter_used <- predictor_jitter_result$predictor_jitter_used
    predictor_jitter_scale_factor <- predictor_jitter_result$predictor_jitter_scale_factor
    predictor_jitter_variables <- predictor_jitter_result$predictor_jitter_variables
    
    if (isTRUE(predictor_jitter_used)) {
      warning(
        "Target = ", target,
        " used optional predictor-jitter sensitivity mode for predictor variables: ",
        predictor_jitter_variables,
        ". Interpret these fits as sensitivity analyses, not as the primary CIRN output.",
        call. = FALSE
      )
    }
  }
  
  # Construct full list of predictor columns across representations
  # (state variables + first derivatives + second derivatives)
  
  predictor_cols <- c(
    predictors,
    paste0(predictors, "_d1"),
    paste0(predictors, "_d2")
  )
  
  # Drop response-defining quantities, then keep only columns that exist
  # after derivative computation.
  #
  # Directional CIRN excludes target_d1 because it defines Class.
  # If adaptive jitter is used, the jittered target trajectory and its
  # recomputed first derivative are also excluded. These columns exist
  # only for response construction and must not enter the predictor set.
  
  response_defining_cols <- c(
    target_d1,
    paste0(target, "_jittered"),
    paste0(target, "_d1_jittered")
  )
  
  predictor_cols <- setdiff(predictor_cols, response_defining_cols)
  predictor_cols <- intersect(predictor_cols, names(df_derivs))
  
  # Enforce temporal precedence (lag predictors only)
  
  if (length(predictor_cols) > 0) {
    df_derivs <- df_derivs %>%
      mutate(across(all_of(predictor_cols), ~ dplyr::lag(.x, n = lag_units)))
  }
  
  is_valid_model_predictor <- function(x) {
    if (!is.numeric(x)) {
      return(FALSE)
    }
    
    x <- x[!is.na(x) & is.finite(x)]
    length(x) > 1 &&
      is.finite(stats::sd(x, na.rm = TRUE)) &&
      stats::sd(x, na.rm = TRUE) > 0
  }
  
  valid_predictor_cols <- predictor_cols[
    vapply(df_derivs[predictor_cols], is_valid_model_predictor, logical(1))
  ]
  dropped_predictor_cols <- setdiff(predictor_cols, valid_predictor_cols)
  
  if (length(dropped_predictor_cols) > 0) {
    warning(
      "Dropping invalid, all-missing, or zero-variance predictors for target = ",
      target,
      ": ",
      paste(dropped_predictor_cols, collapse = ", "),
      call. = FALSE
    )
  }
  
  predictor_cols <- valid_predictor_cols
  model_cols <- c("Class", predictor_cols)
  
  df_derivs <- df_derivs %>%
    dplyr::filter(
      dplyr::if_all(
        dplyr::all_of(model_cols),
        ~ !is.na(.x) & if (is.numeric(.x)) is.finite(.x) else TRUE
      )
    )
  
  # Report class counts from the exact model-ready data after temporal
  # lagging and predictor filtering, not from the pre-lag response table.
  
  model_ready_counts <- response_class_counts(df_derivs$Class)
  class_0_count <- model_ready_counts$class_0_count
  class_1_count <- model_ready_counts$class_1_count
  min_class_count <- model_ready_counts$min_class_count
  
  if (nrow(df_derivs) < 2 || dplyr::n_distinct(df_derivs$Class) < 2) {
    warning(
      "Skipping target = ", target,
      " because lagging and predictor filtering left fewer than two classes.",
      call. = FALSE
    )
    
    if (debug) {
      debug_full <- df_derivs
    }
    
    return(
      make_empty_target_result(
        target = target,
        representation_mode = representation_mode,
        debug_full = debug_full,
        status = "skipped_after_lagging",
        response_mode = response_mode,
        response_label = response_label,
        response_source = response_source,
        dominant_direction = dominant_direction,
        jitter_used = jitter_used,
        jitter_scale = jitter_scale,
        jitter_scale_factor = jitter_scale_factor,
        jitter_basis = jitter_basis,
        jitter_basis_value = jitter_basis_value,
        jitter_attempts = jitter_attempts,
        predictor_jitter_used = predictor_jitter_used,
        predictor_jitter_scale_factor = predictor_jitter_scale_factor,
        predictor_jitter_variables = predictor_jitter_variables,
        class_0_count = class_0_count,
        class_1_count = class_1_count,
        min_class_count = min_class_count,
        n_predictors = length(pred_set),
        response_trigger = response_trigger
      )
    )
  }
  
  # Standardize model predictors only AFTER Class has been constructed
  # from the raw target derivative and predictors have been temporally
  # aligned. This preserves the active target-response interpretation
  # while keeping regression coefficients comparable.
  
  df_derivs <- standardize_selected_numeric_columns(
    df_derivs,
    cols = predictor_cols
  )
  
  
  # --------------------------------------------------------------
  # 4.4 Debug snapshot (OPTIONAL -- development & validation only)
  # --------------------------------------------------------------
  
  # By default, CIRN does not write debug tables to the Global
  # Environment. Set debug = TRUE to retain the fully processed
  # model-ready data for each target inside the returned result.
  # Set assign_debug_to_global = TRUE only if you also want objects
  # such as debug_full_<target> to appear in RStudio.
  
  if (debug) {
    debug_full <- df_derivs %>%
      dplyr::select(
        dplyr::all_of(names(df_derivs)),
        dplyr::ends_with("_d1"),
        dplyr::ends_with("_d2"),
        "Class"
      )
    
    if (assign_debug_to_global) {
      assign(paste0("debug_full_", target), debug_full, envir = .GlobalEnv)
    }
  }
  
  
  # --------------------------------------------------------------
  # 4.5 Predictor sublevels (CIRN feature sets)
  # --------------------------------------------------------------
  
  # CIRN explicitly evaluates multiple dynamical sublevels of regulatory
  # influence in order to capture different mechanistic hypotheses
  # about how system components interact over time.
  #
  # Rather than assuming a single functional form, CIRN decomposes
  # potential regulation into three complementary feature sublevels:
  #
  # (1) ORIGINAL STATE VARIABLES
  #     x_i(t)
  #
  # (2) FIRST-ORDER TEMPORAL DERIVATIVES
  #     dx_i/dt
  #
  # (3) SECOND-ORDER TEMPORAL DERIVATIVES
  #     d²x_i/dt²
  #
  #   • The target state and second derivative may be retained when declared
  #     because they do not define the derivative-direction response
  #   • The target first derivative is excluded because it defines the
  #     directional response
  #   • Adaptive jitter response-construction columns are excluded
  #   • Each sublevel is evaluated independently
  #   • Optionally, all sublevels can be fitted together in one
  #     equation-style model for the active target response
  #   • Together these layers form the multi-sublevel
  #     structure of the CIRN framework
  # --------------------------------------------------------------
  
  # Sublevel predictor sets
  # (built from the already cleaned predictor base)
  
  predictors_original <- intersect(predictors, predictor_cols)
  predictors_d1       <- setdiff(
    intersect(paste0(predictors, "_d1"), predictor_cols),
    target_d1
  )
  predictors_d2       <- intersect(paste0(predictors, "_d2"), predictor_cols)
  
  
  # --------------------------------------------------------------
  # 4.6 Bayesian logistic regression (CIRN core engine)
  # --------------------------------------------------------------
  
  # For a fixed target variable x_j, CIRN models the active binary
  # target response using Bayesian logistic regression:
  #
  #   directional mode:
  #     Class_j(t_k) = 1  if  dx_j/dt > 0
  #                  = 0  if  dx_j/dt < 0
  #
  #   adaptive minimal-jitter mode:
  #     Class_j(t_k) = 1  if the minimally jittered target derivative is positive
  #                  = 0  if the minimally jittered target derivative is negative
  #
  # The statistical model is:
  #
  #   logit( P(Class_j(t_k) = 1) )
  #     = β_0 + Σ_i β_ij x_i(t_{k-1})
  #
  # where:
  #   • β_0 is the intercept term
  #   • β_ij is the latent regression coefficient associated with
  #     predictor x_i acting on target x_j
  #   • predictors x_i are temporally lagged to encode
  #     directional asymmetry
  #   • ω_ij := E[β_ij | data] is the posterior mean reported by CIRN
  #
  # Edge inference is based on posterior uncertainty of β_ij,
  # NOT on point estimates alone.
  
  run_bayes_logit <- function(pred_set, predictor_set) {
    
    fit_status <- "failed"
    fit_edges <- NA_integer_
    
    call_cirn_progress(
      progress_callback,
      target = target,
      predictor_set = predictor_set,
      stage = "start",
      status = "running",
      n_predictors = length(pred_set),
      n_edges = NA_integer_
    )
    
    on.exit(
      call_cirn_progress(
        progress_callback,
        target = target,
        predictor_set = predictor_set,
        stage = "finish",
        status = fit_status,
        n_predictors = length(pred_set),
        n_edges = fit_edges
      ),
      add = TRUE
    )
    
    # ----------------------------------------------------------
    # 4.6.1 Guard clause: no predictors available
    # ----------------------------------------------------------
    
    # This situation can arise if:
    #   • All candidate predictors were removed upstream
    #   • The representation yields no valid features
    #
    # Instead of failing, we return a well-defined empty result.
    
    if (length(pred_set) == 0) {
      fit_status <- "skipped_no_predictors"
      fit_edges <- 0L
      return(
        make_empty_bayes_result(
          target = target,
          status = fit_status,
          response_mode = response_mode,
          response_label = response_label,
          response_source = response_source,
          dominant_direction = dominant_direction,
          jitter_used = jitter_used,
          jitter_scale = jitter_scale,
          jitter_scale_factor = jitter_scale_factor,
          jitter_basis = jitter_basis,
          jitter_basis_value = jitter_basis_value,
          jitter_attempts = jitter_attempts,
          predictor_jitter_used = predictor_jitter_used,
          predictor_jitter_scale_factor = predictor_jitter_scale_factor,
          predictor_jitter_variables = predictor_jitter_variables,
          class_0_count = class_0_count,
          class_1_count = class_1_count,
          min_class_count = min_class_count,
          n_predictors = length(pred_set),
          response_trigger = response_trigger
        )
      )
    }
    
    # ----------------------------------------------------------
    # 4.6.2 Model specification
    # ----------------------------------------------------------
    
    # Logistic regression on the active target-response class:
    #
    #   logit( P(Class_j(t_k) = 1) )
    #     = β_0 + Σ_i β_ij x_i(t_{k-1})
    #
    # where:
    #   • Class_j(t_k) is either the raw directional response or the
    #     adaptive minimally jittered directional response
    #   • x_i(t_{k-1}) are temporally lagged predictors
    #   • β_ij are regression coefficients with posterior distributions
    #   • ω_ij := E[β_ij | data] is the posterior mean reported by CIRN
    
    form <- as.formula(
      paste("Class ~", paste(pred_set, collapse = " + "))
    )
    
    collinearity_diagnostics <- compute_collinearity_diagnostics(
      df_model = df_derivs,
      form = form,
      pred_set = pred_set,
      target = target,
      predictor_set = predictor_set
    )
    
    # ----------------------------------------------------------
    # 4.6.3 Multicollinearity diagnostics (VIF)
    # ----------------------------------------------------------
    
    # NOTE: 
    # Variance Inflation Factors (VIF) are computed ONLY as a
    # diagnostic tool to flag potential collinearity.
    # High VIF values may indicate redundant predictors, but
    # Bayesian regularization and posterior uncertainty are
    # relied upon for inference stability.
    #
    # Important:
    #   • No predictors are automatically removed
    #   • Bayesian inference can tolerate moderate collinearity
    
    high_vif_terms <- collinearity_diagnostics$vif_group %>%
      filter(!is.na(vif), vif > 5) %>%
      pull(term)
    
    exact_pair_terms <- collinearity_diagnostics$vif_pairs %>%
      filter(abs_correlation >= 0.999999) %>%
      transmute(pair = paste(predictor_1, predictor_2, sep = " ~ ")) %>%
      pull(pair)
    
    if (length(high_vif_terms) > 0) {
      warning(
        "High multicollinearity detected (VIF > 5) for target = ",
        paste0("d", target),
        ", predictor_set = ",
        predictor_set,
        ": ",
        paste(high_vif_terms, collapse = ", ")
      )
    }
    
    if (length(exact_pair_terms) > 0) {
      warning(
        "Exact or near-exact pairwise collinearity detected for target = ",
        paste0("d", target),
        ", predictor_set = ",
        predictor_set,
        ": ",
        paste(exact_pair_terms, collapse = ", ")
      )
    }
    
    # ----------------------------------------------------------
    # 4.6.4 Bayesian model fitting
    # ----------------------------------------------------------
    
    # Prior specification:
    #
    #   β_ij ~ Normal(0, 2)
    #
    # (applied to all regression coefficients, including the intercept)
    #
    # This weakly informative prior:
    #   • Centers effects at zero (no regulation a priori)
    #   • Places most prior mass on moderate-to-strong log-odds effects
    #     (typically |β_ij| < 4)
    #   • Regularizes extreme estimates without hard truncation
    #
    # Note:
    #   ω_ij (reported by CIRN) is the posterior mean of β_ij
    #   and is NOT a random variable with its own prior.
    
    coef_prior <- brms::set_prior(
      paste0("normal(", prior_mean, ", ", prior_sd, ")"),
      class = "b"
    )
    
    model <- tryCatch(
      brms::brm(                           # Bayesian logistic regression via brms (Stan backend)
        formula = form,                    # Logistic regression formula for CIRN classification
        data    = df_derivs,               # Model-ready target data
        family  = brms::bernoulli(link = "logit"),
        prior   = coef_prior,
        iter    = model_iter,
        warmup  = model_warmup,
        chains  = model_chains,
        cores   = model_cores,
        control = list(adapt_delta = adapt_delta),
        seed    = seed,
        refresh = 0,
        backend = "cmdstanr"
      ),
      error = function(e) e
    )
    
    if (inherits(model, "error")) {
      fit_status <- "failed_model_fit"
      fit_edges <- 0L
      warning(
        "Bayesian model fit failed for target = ",
        target,
        ", predictor_set = ",
        predictor_set,
        ": ",
        conditionMessage(model),
        call. = FALSE
      )
      return(
        make_empty_bayes_result(
          target = target,
          status = fit_status,
          response_mode = response_mode,
          response_label = response_label,
          response_source = response_source,
          dominant_direction = dominant_direction,
          jitter_used = jitter_used,
          jitter_scale = jitter_scale,
          jitter_scale_factor = jitter_scale_factor,
          jitter_basis = jitter_basis,
          jitter_basis_value = jitter_basis_value,
          jitter_attempts = jitter_attempts,
          predictor_jitter_used = predictor_jitter_used,
          predictor_jitter_scale_factor = predictor_jitter_scale_factor,
          predictor_jitter_variables = predictor_jitter_variables,
          class_0_count = class_0_count,
          class_1_count = class_1_count,
          min_class_count = min_class_count,
          n_predictors = length(pred_set),
          response_trigger = response_trigger
        )
      )
    }
    
    # ----------------------------------------------------------
    # 4.6.5 Posterior diagnostics
    # ----------------------------------------------------------
    
    # Convergence and sampling efficiency diagnostics:
    #   • R-hat (Gelman–Rubin diagnostic):
    #       – R-hat ≤ 1.01 : satisfactory convergence
    #       – 1.01 < R-hat ≤ 1.05 : acceptable but warrants inspection
    #       – R-hat > 1.05 : lack of convergence
    #
    #   • Effective sample size (ESS) assesses sampling efficiency
    #     and chain mixing:
    #       – ESS < 100   : poor reliability
    #       – ESS 100–400 : acceptable but low precision
    #       – ESS > 400   : good sampling efficiency
    #
    # Predictive performance assessment (NOT inferential):
    #   • LOO (leave-one-out cross-validation) is used exclusively
    #     to compare alternative model specifications (e.g.,
    #     predictor representations or smoothing choices).
    #   • Differences in expected log predictive density (ELPD):
    #       – |ΔELPD| < 2     : negligible predictive difference
    #       – 2 ≤ |ΔELPD| < 5 : moderate evidence
    #       – |ΔELPD| ≥ 5     : strong evidence
    #
    # IMPORTANT:
    #   LOO plays NO role in edge inclusion, edge sign determination,
    #   or network construction. Regulatory inference in CIRN is based
    #   solely on posterior uncertainty of regression coefficients.
    
    diagnostics <- list(
      rhat = brms::rhat(model),
      neff_ratio = brms::neff_ratio(model),
      loo = if (isTRUE(compute_loo)) {
        tryCatch(
          brms::loo(
            model,
            moment_match = loo_moment_match,
            reloo = loo_reloo,
            k_threshold = loo_k_threshold
          ),
          error = function(e) {
            warning("LOO diagnostic failed for target = ", target,
                    ", predictor_set = ", predictor_set, ": ",
                    conditionMessage(e))
            NULL
          }
        )
      } else {
        NULL
      },
      status = "completed",
      response_mode = response_mode,
      response_label = response_label,
      response_source = response_source,
      dominant_direction = dominant_direction,
      jitter_used = jitter_used,
      jitter_scale = jitter_scale,
      jitter_scale_factor = jitter_scale_factor,
      jitter_basis = jitter_basis,
      jitter_basis_value = jitter_basis_value,
      jitter_attempts = jitter_attempts,
      predictor_jitter_used = predictor_jitter_used,
      predictor_jitter_scale_factor = predictor_jitter_scale_factor,
      predictor_jitter_variables = predictor_jitter_variables,
      class_0_count = class_0_count,
      class_1_count = class_1_count,
      min_class_count = min_class_count,
      n_predictors = length(pred_set),
      response_trigger = response_trigger
    )
    
    # Warn (but do not stop) if convergence is questionable
    
    if (any(diagnostics$rhat > 1.01, na.rm = TRUE)) {
      warning(
        "Potential convergence issues detected for target = ",
        target
      )
    }
    
    # ----------------------------------------------------------
    # 4.6.6 Posterior summarization (CIRN inference rule)
    # ----------------------------------------------------------
    
    # IMPORTANT:
    # CIRN does NOT rely on null-hypothesis significance testing.
    # Regulatory inference is based on posterior credibility,
    # not on p-values.
    #
    # Edge inclusion rule:
    #   • Compute the 95% Highest Density Interval (HDI) of β_ij
    #   • Include edge i → j  ⇔  0 ∉ 95% HDI(β_ij)
    #
    # Edge characterization:
    #   • Direction (activation vs inhibition) is determined by
    #     the sign of ω_ij = E[β_ij | data]
    
    posterior_draws <- as_draws_df(model)
    
    # Helper: compute 95% Highest Density Interval (HDI)
    
    get_hdi <- function(draws, prob = 0.95) {
      h <- bayestestR::hdi(draws, ci = prob)
      c(h$CI_low, h$CI_high)
    }
    
    ps <- as.data.frame(posterior_summary(model))
    ps <- tibble::rownames_to_column(ps, "term")
    
    # Identify CI column names automatically (robust to brms versions)
    
    ci_lower_name <- grep("2.5|l-95", colnames(ps), value = TRUE)
    ci_upper_name <- grep("97.5|u-95", colnames(ps), value = TRUE)
    ci_lower_name <- ci_lower_name[1]
    ci_upper_name <- ci_upper_name[1]
    
    coef_tbl_all <- ps %>%
      filter(grepl("^b_", term) & term != "b_Intercept") %>%
      mutate(
        term = sub("^b_", "", term),
        omega = Estimate,
        odds_ratio = exp(Estimate),
        sign = ifelse(Estimate > 0, 1, -1),
        regulation_type = ifelse(Estimate > 0, "activation", "inhibition"),
        lower95 = .data[[ci_lower_name]],
        upper95 = .data[[ci_upper_name]],
        method = "Bayesian Logistic (brms)",
        target = paste0("d", target),
        response_mode = response_mode,
        response_label = response_label,
        response_source = response_source,
        dominant_direction = dominant_direction,
        jitter_used = jitter_used,
        jitter_scale = jitter_scale,
        jitter_scale_factor = jitter_scale_factor,
        jitter_basis = jitter_basis,
        jitter_basis_value = jitter_basis_value,
        jitter_attempts = jitter_attempts,
        predictor_jitter_used = predictor_jitter_used,
        predictor_jitter_scale_factor = predictor_jitter_scale_factor,
        predictor_jitter_variables = predictor_jitter_variables,
        class_0_count = class_0_count,
        class_1_count = class_1_count,
        min_class_count = min_class_count,
        response_trigger = response_trigger,
        effect_interpretation = effect_interpretation
      ) %>%
      rowwise() %>%
      mutate(
        hdi_bounds  = list(get_hdi(posterior_draws[[paste0("b_", term)]])),
        hdi_lower95 = hdi_bounds[[1]],
        hdi_upper95 = hdi_bounds[[2]],
        retained = hdi_lower95 > 0 | hdi_upper95 < 0,
        credibility = ifelse(
          retained,
          "credible",
          "uncertain"
        )
      ) %>%
      ungroup() %>%
      dplyr::select(
        term,
        target,
        omega,
        odds_ratio,
        sign,
        regulation_type,
        hdi_lower95,
        hdi_upper95,
        eti_lower95 = lower95,
        eti_upper95 = upper95,
        retained,
        credibility,
        response_mode,
        response_label,
        response_source,
        response_trigger,
        dominant_direction,
        class_0_count,
        class_1_count,
        min_class_count,
        jitter_used,
        jitter_scale_factor,
        jitter_scale,
        jitter_basis,
        jitter_basis_value,
        jitter_attempts,
        predictor_jitter_used,
        predictor_jitter_scale_factor,
        predictor_jitter_variables,
        effect_interpretation,
        method
      )
    
    coef_tbl <- coef_tbl_all %>%
      dplyr::filter(.data$retained) %>%
      dplyr::select(
        term,
        target,
        omega,
        odds_ratio,
        sign,
        regulation_type,
        hdi_lower95,
        hdi_upper95,
        eti_lower95,
        eti_upper95,
        response_mode,
        response_label,
        response_source,
        response_trigger,
        dominant_direction,
        class_0_count,
        class_1_count,
        min_class_count,
        jitter_used,
        jitter_scale_factor,
        jitter_scale,
        jitter_basis,
        jitter_basis_value,
        jitter_attempts,
        predictor_jitter_used,
        predictor_jitter_scale_factor,
        predictor_jitter_variables,
        effect_interpretation,
        method
      )
    
    
    # If no coefficients survive the HDI criterion,
    # return a structured empty result instead of NULL
    
    if (nrow(coef_tbl) == 0) {
      coef_tbl <- empty_bayes_result_tibble %>%
        mutate(
          target = paste0("d", target),
          response_mode = response_mode,
          response_label = response_label,
          response_source = response_source,
          dominant_direction = dominant_direction,
          jitter_used = jitter_used,
          jitter_scale = jitter_scale,
          jitter_scale_factor = jitter_scale_factor,
          jitter_basis = jitter_basis,
          jitter_basis_value = jitter_basis_value,
          jitter_attempts = jitter_attempts,
          class_0_count = class_0_count,
          class_1_count = class_1_count,
          min_class_count = min_class_count,
          response_trigger = response_trigger,
          effect_interpretation = effect_interpretation
        )
    }
    fit_status <- "completed"
    fit_edges <- nrow(coef_tbl)
    
    # ----------------------------------------------------------
    # 4.6.7 Return object
    # ----------------------------------------------------------
    
    # Each model fit returns a structured list containing:
    #   • coefficients : inferred CIRN regulatory effects
    #                    (posterior summaries of β_ij and ω_ij)
    #   • diagnostics  : convergence, sampling, and predictive
    #                    performance diagnostics
    #   • model        : full brmsfit object for reproducibility
    #                    and further inspection
    
    list(
      coefficients     = coef_tbl,
      all_coefficients = coef_tbl_all,
      diagnostics  = diagnostics,
      model        = model,
      vif_group    = collinearity_diagnostics$vif_group,
      vif_pairs    = collinearity_diagnostics$vif_pairs
    )
  }
  
  # --------------------------------------------------------------
  # 4.7 Fit requested CIRN representations
  # --------------------------------------------------------------
  
  # CIRN does not assume a single functional form for regulatory
  # influence. Instead, it evaluates multiple representations
  # of predictor variables to capture different dynamical effects:
  #
  #   (i)   Original state variables:
  #         x_i(t)
  #         → captures instantaneous state-dependent regulation
  #
  #   (ii)  First temporal derivatives:
  #         dx_i/dt
  #         → captures rate-based or momentum-driven interactions
  #
  #   (iii) Second temporal derivatives:
  #         d²x_i/dt²
  #         → captures acceleration, curvature, or delayed effects
  #
  #   (iv)  All predictors together:
  #         x_i(t), dx_i/dt, d²x_i/dt²
  #         → approximates an equation-style dependency for d[target]
  #           while still excluding the target first derivative itself
  #
  # For a fixed target variable x_j, each sublevel representation
  # defines a Bayesian logistic regression model of the form:
  #
  #   logit( P(Class_j(t_k) = 1) )
  #     = β_0 + Σ_i β_ij^(r) x_i^(r)(t_{k-1})
  #
  # where:
  #   • r ∈ {original, first_derivative, second_derivative}
  #     denotes a CIRN sublevel representation
  #   • x_i^(r)(t_{k-1}) are temporally lagged predictors constructed
  #     from representation r
  #   • all_predictors uses the union of these predictor columns
  #
  # The representation_mode argument controls which models are fit:
  #   • "sublevel"       : fit the state, first-derivative, and
  #                        second-derivative level models independently
  #   • "all_predictors" : fit one combined model using all state,
  #                        first-derivative, and second-derivative
  #                        predictors together, excluding d[target]
  #   • "both"           : fit the three sublevel representations and
  #                        the combined all-predictors model
  #
  # These models are:
  #   • Fit independently
  #   • Diagnosed independently
  #   • Optionally compared via LOO for predictive adequacy
  #
  # No representation is privileged a priori. The combined model is
  # intended as an optional equation-approximation layer, while the
  # sublevel models remain useful for representation sensitivity.
  
  results <- list()
  
  if (representation_mode %in% c("sublevel", "both")) {
    results <- c(
      results,
      list(
        
        # State-based regulation:
        # x_i(t_k) → active target response
        
        original = run_bayes_logit(predictors_original, "original"),
        
        # Rate-based regulation:
        # dx_i/dt(t_k) → active target response
        
        first_derivative = run_bayes_logit(predictors_d1, "first_derivative"),
        
        # Acceleration-based regulation:
        # d²x_i/dt²(t_k) → active target response
        
        second_derivative = run_bayes_logit(predictors_d2, "second_derivative")
      )
    )
  }
  
  if (representation_mode %in% c("all_predictors", "both")) {
    results$all_predictors <- run_bayes_logit(
      predictor_cols,
      "all_predictors"
    )
  }
  
  # --------------------------------------------------------------
  # 4.8 Combine inferred edges
  # --------------------------------------------------------------
  
  # Each CIRN representation (original state, first derivative,
  # second derivative) yields its own set of inferred regulatory
  # coefficients after posterior filtering.
  #
  # At this stage, CIRN performs "edge aggregation", NOT model
  # averaging. Specifically:
  #
  #   • Each sublevel contributes candidate edges independently
  #   • Only coefficients satisfying the CIRN inference rule
  #     (95% HDI excludes zero) are retained
  #   • No weighting, voting, or averaging is applied across
  #     representations
  #
  # Consequently:
  #   • A regulator–target pair may appear in multiple sublevels
  #   • Each occurrence corresponds to a distinct dynamical hypothesis
  #     (state-based, rate-based, acceleration-based regulation, or
  #      the optional all-predictors equation-style model)
  #
  # The aggregated result constitutes the final CIRN edge list,
  # preserving sublevel-specific mechanistic interpretation.
  
  edges_tbl <- bind_rows(
    purrr::map(results, "coefficients"),
    .id = "predictor_set"
  ) %>%
    
    # Safety filter: remove rows with missing posterior means
    # (can occur if a model returns no significant coefficients)
    
    filter(!is.na(omega))
  
  all_coefficients_tbl <- bind_rows(
    purrr::map(results, "all_coefficients"),
    .id = "predictor_set"
  ) %>%
    
    # Keep the full retained / not-retained posterior summary table
    # available for HDI plots, diagnostics, and app exports.
    
    filter(!is.na(omega))
  
  # INTERPRETATION NOTE:
  # The inferred CIRN network constrains the SIGN and DIRECTION
  # of admissible interaction terms in continuous-time dynamical
  # models (e.g., ODEs, SDEs), but does not identify specific functional
  # forms or parameter values. Multiple mechanistic models may be
  # consistent with the same CIRN-inferred regulatory structure.
  
  # --------------------------------------------------------------
  # 4.8.1 Relative strength classification (REFLECTED IN TABLE)
  # --------------------------------------------------------------
  
  # Strength is defined RELATIVE to other credible effects
  # acting on the SAME TARGET across CIRN representations.
  #
  # This affects TABLE REPORTING ONLY and does NOT influence
  # edge inclusion, direction, or inference.
  
  edges_tbl <- edges_tbl %>%
    group_by(target) %>%
    mutate(
      omega_abs = abs(omega),
      omega_abs_max = if (all(is.na(omega_abs))) NA_real_ else max(omega_abs, na.rm = TRUE),
      omega_abs_q33 = if (all(is.na(omega_abs))) NA_real_ else as.numeric(quantile(omega_abs, 0.33, na.rm = TRUE)),
      omega_abs_q66 = if (all(is.na(omega_abs))) NA_real_ else as.numeric(quantile(omega_abs, 0.66, na.rm = TRUE)),
      rel_strength = case_when(
        is.na(omega_abs) ~ NA_character_,
        n() == 1 ~ "credible",
        n() == 2 & !is.na(omega_abs_max) & omega_abs == omega_abs_max ~ "strong",
        n() == 2 ~ "weak",
        !is.na(omega_abs_q33) & omega_abs <= omega_abs_q33 ~ "weak",
        !is.na(omega_abs_q66) & omega_abs <= omega_abs_q66 ~ "moderate",
        TRUE ~ "strong"
      )
    ) %>%
    ungroup() %>%
    dplyr::select(-omega_abs_max, -omega_abs_q33, -omega_abs_q66) %>%
    dplyr::select(
      predictor_set,
      term,
      target,
      omega,
      odds_ratio,
      sign,
      regulation_type,
      rel_strength,
      hdi_lower95,
      hdi_upper95,
      eti_lower95,
      eti_upper95,
      response_mode,
      response_label,
      response_source,
      response_trigger,
      dominant_direction,
      class_0_count,
      class_1_count,
      min_class_count,
      jitter_used,
      jitter_scale_factor,
      jitter_scale,
      jitter_basis,
      jitter_basis_value,
      jitter_attempts,
      predictor_jitter_used,
      predictor_jitter_scale_factor,
      predictor_jitter_variables,
      effect_interpretation,
      method
    )
  
  # --------------------------------------------------------------
  # 4.9 Output structure
  # --------------------------------------------------------------
  
  # The function returns a structured list containing:
  #
  #   • edges:
  #       Final CIRN edge list comprising:
  #         - regulator variable (term)
  #         - target variable (d[target])
  #         - posterior mean ω_ij
  #         - HDI & CI (credible interval) bounds
  #         - regulation sign (activation / inhibition)
  #         - CIRN representation (predictor_set)
  #         - response mode and response source used for the target
  #         - effect interpretation for the active response mode
  #
  #   • diagnostics:
  #       Convergence, sampling, and predictive diagnostics
  #       for all CIRN representations (reported for assessment
  #       and transparency only)
  #
  #   • models:
  #       Fitted brms model objects retained for reproducibility,
  #       visualization, and detailed MCMC inspection
  
  list(
    edges = edges_tbl,
    all_coefficients = all_coefficients_tbl,
    diagnostics = results,
    models = lapply(results, function(x) x$model),
    vif_group = purrr::map_df(results, "vif_group"),
    vif_pairs = purrr::map_df(results, "vif_pairs"),
    debug = debug_full
  )
}


################################################################################
################################################################################


# ============================================================
# 5. Full Network Inference (STRUCTURALLY CONSTRAINED VERSION)
# ============================================================
#
# This version of infer_network() extends CIRN by allowing the user
# to explicitly specify:
#
#   • targets    : endogenous variables to be modeled dynamically
#   • predictors : variables allowed to act as regulators
#   • representation_mode :
#       - "sublevel"       = state, first-derivative, and second-derivative CIRN models
#       - "all_predictors" = one combined model for d[target]
#       - "both"           = fit sublevel and combined models
#
# This enables structural separation between:
#   • Endogenous state variables (e.g., disease incidence)
#   • Exogenous drivers (e.g., temperature, humidity, rainfall)
#
# If targets and predictors are NOT specified, the function defaults to a
# symmetric setting in which all non-time variables are treated as both
# targets and predictors.
#
# This structural control prevents scientifically implausible
# bidirectional inference in systems where domain knowledge
# dictates asymmetric temporal ordering (e.g., weather → dengue).
#
# ---------------------------------------------------------------

infer_network <- function(df,
                          time_col,
                          targets = NULL,
                          predictors = NULL,
                          representation_mode = c("sublevel", "all_predictors", "both"),
                          lag_units = 1,
                          run_pairwise = FALSE,
                          pairwise_representation_mode = representation_mode,
                          points_per_interval = 1,
                          spar = NULL,
                          outlier_method = "MAD",
                          outlier_thresh = 3.5,
                          outlier_action = "winsorize",
                          response_eps = 1e-6,
                          adaptive_jitter = TRUE,
                          jitter_predictors = FALSE,
                          jitter_min_class_count = 5,
                          jitter_scale_grid = c(
                            1e-8, 3e-8,
                            1e-7, 3e-7,
                            1e-6, 3e-6,
                            1e-5, 3e-5,
                            1e-4, 3e-4,
                            1e-3, 3e-3,
                            1e-2
                          ),
                          jitter_scale_basis = c(
                            "state_sd",
                            "state_range",
                            "derivative_sd",
                            "derivative_max_abs",
                            "absolute"
                          ),
                          model_iter = 3000,
                          model_warmup = 1000,
                          model_chains = 4,
                          model_cores = model_chains,
                          prior_mean = 0,
                          prior_sd = 2,
                          adapt_delta = 0.95,
                          compute_loo = FALSE,
                          loo_moment_match = FALSE,
                          loo_reloo = NULL,
                          loo_k_threshold = 0.7,
                          seed = 123,
                          show_progress = TRUE,
                          progress_bar = TRUE,
                          progress_callback = NULL,
                          debug = FALSE,
                          assign_debug_to_global = FALSE) {
  
  set_cirn_thread_limits()
  gc()
  
  representation_mode <- match.arg(representation_mode)
  pairwise_representation_mode <- match.arg(
    pairwise_representation_mode,
    choices = c("sublevel", "all_predictors", "both")
  )
  jitter_scale_basis <- match.arg(jitter_scale_basis)
  if (is.null(loo_reloo)) {
    loo_reloo <- isTRUE(compute_loo) && isTRUE(loo_moment_match)
  }
  
  if (length(lag_units) != 1 || is.na(lag_units) || lag_units < 1 || lag_units != floor(lag_units)) {
    stop("lag_units must be a single positive integer.")
  }
  lag_units <- as.integer(lag_units)
  
  if (!is.logical(adaptive_jitter) || length(adaptive_jitter) != 1 || is.na(adaptive_jitter)) {
    stop("adaptive_jitter must be TRUE or FALSE.")
  }
  
  if (!is.logical(jitter_predictors) ||
      length(jitter_predictors) != 1 ||
      is.na(jitter_predictors)) {
    stop("jitter_predictors must be TRUE or FALSE.")
  }
  
  if (length(jitter_min_class_count) != 1 ||
      is.na(jitter_min_class_count) ||
      jitter_min_class_count < 1 ||
      jitter_min_class_count != floor(jitter_min_class_count)) {
    stop("jitter_min_class_count must be a single positive integer.")
  }
  jitter_min_class_count <- as.integer(jitter_min_class_count)
  
  if (length(jitter_scale_grid) == 0 ||
      any(!is.finite(jitter_scale_grid)) ||
      any(jitter_scale_grid <= 0)) {
    stop("jitter_scale_grid must contain positive finite values.")
  }
  jitter_scale_grid <- sort(unique(as.numeric(jitter_scale_grid)))
  
  if (!is.logical(run_pairwise) || length(run_pairwise) != 1 || is.na(run_pairwise)) {
    stop("run_pairwise must be TRUE or FALSE.")
  }
  
  if (isTRUE(run_pairwise) && model_cores > 2) {
    warning(
      "run_pairwise = TRUE with model_cores = ", model_cores,
      " can exhaust macOS/OpenMP thread resources during long CIRN runs. ",
      "If OMP pthread errors occur, restart R and set model_cores = 1 or 2.",
      call. = FALSE
    )
  }
  
  if (!time_col %in% names(df)) {
    stop("time_col is not present in df.")
  }
  
  # ------------------------------------------------------------
  # 5.1 Identify targets and allowed predictors before preprocessing
  # ------------------------------------------------------------
  
  # CIRN should be agnostic to unrelated columns in the input file.
  # Therefore, derivative construction and complete-case filtering are
  # restricted to the declared targets and predictors only.
  
  candidate_vars <- setdiff(names(df), time_col)
  numeric_candidate_vars <- candidate_vars[
    vapply(df[candidate_vars], is.numeric, logical(1))
  ]
  
  if (is.null(targets)) {
    targets <- numeric_candidate_vars
  }
  
  if (is.null(predictors)) {
    predictors <- numeric_candidate_vars
  }
  
  if (length(targets) == 0 || length(predictors) == 0) {
    stop("At least one numeric target and one numeric predictor are required.")
  }
  
  if (!all(targets %in% names(df))) {
    stop("Some specified targets are not present in the dataset.")
  }
  
  if (!all(predictors %in% names(df))) {
    stop("Some specified predictors are not present in the dataset.")
  }
  
  analysis_vars <- unique(c(targets, predictors))
  non_numeric_vars <- analysis_vars[
    !vapply(df[analysis_vars], is.numeric, logical(1))
  ]
  
  if (length(non_numeric_vars) > 0) {
    stop(
      "All targets and predictors must be numeric. Non-numeric columns: ",
      paste(non_numeric_vars, collapse = ", ")
    )
  }
  
  df <- df %>%
    dplyr::select(dplyr::all_of(c(time_col, analysis_vars))) %>%
    dplyr::filter(stats::complete.cases(.))
  
  if (nrow(df) < 3) {
    stop("At least 3 complete observations are required for CIRN inference.")
  }
  
  # ------------------------------------------------------------
  # 5.2 Derivative computation
  # ------------------------------------------------------------
  
  df_derivs <- compute_derivatives(
    df,
    time_col,
    points_per_interval = points_per_interval,
    spar = spar,
    outlier_method = outlier_method,
    outlier_thresh = outlier_thresh,
    outlier_action = outlier_action
  )
  
  # ------------------------------------------------------------
  # 5.3 Predictor standardization
  # ------------------------------------------------------------
  
  # Keep derivative columns raw at this stage. Class labels are built
  # target-wise from the raw target derivative inside infer_for_target().
  # Model predictors are standardized only after Class construction and
  # temporal lag alignment.
  
  # ------------------------------------------------------------
  # 5.4 Progress accounting
  # ------------------------------------------------------------
  
  # A "fit" means one Bayesian logistic regression model. CIRN can
  # fit several models per target depending on representation_mode.
  # This tracker reports both the current fit and the total number
  # expected for the selected multivariable and pairwise settings.
  
  representation_fit_count <- function(mode) {
    switch(
      as.character(mode),
      sublevel = 3L,
      all_predictors = 1L,
      both = 4L
    )
  }
  
  total_model_fits <- length(targets) * representation_fit_count(representation_mode)
  if (run_pairwise) {
    total_model_fits <- total_model_fits +
      length(targets) * length(predictors) * representation_fit_count(pairwise_representation_mode)
  }
  
  progress_env <- new.env(parent = emptyenv())
  progress_env$total <- total_model_fits
  progress_env$completed <- 0L
  progress_env$started_at <- Sys.time()
  progress_env$bar <- NULL
  progress_env$use_bar <- isTRUE(show_progress) &&
    isTRUE(progress_bar) &&
    total_model_fits > 0
  
  if (progress_env$use_bar) {
    message("CIRN model-fitting progress: ", total_model_fits, " model fit(s) scheduled.")
    progress_env$bar <- utils::txtProgressBar(
      min = 0,
      max = total_model_fits,
      initial = 0,
      style = 3
    )
    
    on.exit({
      if (!is.null(progress_env$bar)) {
        utils::setTxtProgressBar(progress_env$bar, progress_env$completed)
        close(progress_env$bar)
        cat("\n")
      }
    }, add = TRUE)
  }
  
  make_progress_callback <- function(analysis_mode,
                                     pairwise_predictor = NA_character_) {
    
    if (!isTRUE(show_progress) && !is.function(progress_callback)) {
      return(NULL)
    }
    
    force(analysis_mode)
    force(pairwise_predictor)
    
    function(target,
             predictor_set,
             stage,
             status = NA_character_,
             n_predictors = NA_integer_,
             n_edges = NA_integer_,
             ...) {
      
      if (identical(stage, "finish")) {
        progress_env$completed <- min(progress_env$completed + 1L, progress_env$total)
        if (isTRUE(progress_env$use_bar) && !is.null(progress_env$bar)) {
          utils::setTxtProgressBar(progress_env$bar, progress_env$completed)
        }
      }
      
      percent_done <- if (progress_env$total > 0) {
        round(100 * progress_env$completed / progress_env$total, 1)
      } else {
        100
      }
      percent_left <- round(max(0, 100 - percent_done), 1)
      
      fit_index <- if (identical(stage, "start")) {
        min(progress_env$completed + 1L, progress_env$total)
      } else {
        progress_env$completed
      }
      
      elapsed_min <- as.numeric(difftime(Sys.time(), progress_env$started_at, units = "mins"))
      eta_min <- if (identical(stage, "finish") && progress_env$completed > 0 && progress_env$total > 0) {
        elapsed_min * (progress_env$total - progress_env$completed) / progress_env$completed
      } else {
        NA_real_
      }
      
      if (isTRUE(show_progress)) {
        if (isTRUE(progress_env$use_bar)) {
          cat("\n")
        }
        pairwise_text <- if (!is.na(pairwise_predictor)) {
          paste0(" | pairwise_predictor=", pairwise_predictor)
        } else {
          ""
        }
        eta_text <- if (is.finite(eta_min)) {
          paste0(" | est_remaining_min=", round(eta_min, 1))
        } else {
          ""
        }
        edge_text <- if (!is.na(n_edges)) {
          paste0(" | retained_edges=", n_edges)
        } else {
          ""
        }
        
        message(
          "[", stage, "] CIRN fit ", fit_index, "/", progress_env$total,
          " | done=", percent_done, "% | left=", percent_left, "%",
          " | mode=", analysis_mode,
          pairwise_text,
          " | target=d", target,
          " | representation=", predictor_set,
          " | status=", status,
          " | predictors=", n_predictors,
          edge_text,
          eta_text
        )
      }
      
      call_cirn_progress(
        progress_callback,
        target = target,
        predictor_set = predictor_set,
        stage = stage,
        status = status,
        n_predictors = n_predictors,
        n_edges = n_edges,
        analysis_mode = analysis_mode,
        pairwise_predictor = pairwise_predictor,
        completed = progress_env$completed,
        total = progress_env$total,
        percent_done = percent_done,
        percent_left = percent_left,
        elapsed_min = elapsed_min,
        estimated_remaining_min = eta_min
      )
    }
  }
  
  multivariable_progress <- make_progress_callback(
    analysis_mode = "multivariable"
  )
  
  # ------------------------------------------------------------
  # 5.5 Target-wise CIRN inference
  # ------------------------------------------------------------
  
  # For each target variable x_j:
  #   • Only predictors specified in 'predictors' are allowed
  #   • infer_for_target() keeps declared self-state and self-acceleration terms
  #     while excluding the target first derivative internally
  #   • representation_mode controls whether CIRN fits sublevel
  #     representations, one all-predictors model, or both
  #
  # This supports:
  #   • Epidemiological systems (weather → dengue)
  #   • Fully endogenous ecological systems
  #   • Hybrid constrained systems
  
  res <- purrr::map(
    targets,
    ~ tryCatch(
      infer_for_target(
        df_derivs = df_derivs,
        target = .x,
        time_col = time_col,
        predictors = predictors,
        representation_mode = representation_mode,
        lag_units = lag_units,
        response_eps = response_eps,
        adaptive_jitter = adaptive_jitter,
        jitter_predictors = jitter_predictors,
        jitter_min_class_count = jitter_min_class_count,
        jitter_scale_grid = jitter_scale_grid,
        jitter_scale_basis = jitter_scale_basis,
        model_iter = model_iter,
        model_warmup = model_warmup,
        model_chains = model_chains,
        model_cores = model_cores,
        prior_mean = prior_mean,
        prior_sd = prior_sd,
        adapt_delta = adapt_delta,
        compute_loo = compute_loo,
        loo_moment_match = loo_moment_match,
        loo_reloo = loo_reloo,
        loo_k_threshold = loo_k_threshold,
        seed = seed,
        progress_callback = multivariable_progress,
        debug = debug,
        assign_debug_to_global = assign_debug_to_global
      ),
      error = function(e) {
        warning(
          "CIRN target fit failed for target = ", .x, ": ",
          conditionMessage(e)
        )
        make_empty_target_result(
          target = .x,
          representation_mode = representation_mode,
          status = paste0("fit_failed: ", conditionMessage(e)),
          n_predictors = length(predictors)
        )
      }
    )
  )
  
  
  # ------------------------------------------------------------
  # 5.6 Collect fitted Bayesian models
  # ------------------------------------------------------------
  
  # Models are indexed by target variable name.
  # Enables posterior diagnostics and inspection.
  
  models <- setNames(
    purrr::map(res, "models"),
    targets
  )
  
  debug_tables <- setNames(
    purrr::map(res, "debug"),
    targets
  )
  
  
  # ------------------------------------------------------------
  # 5.7 Aggregate inferred regulatory edges
  # ------------------------------------------------------------
  
  # Each target contributes its own inferred edges.
  # These are combined into a single network table.
  
  place_method_last <- function(tbl) {
    if (!"method" %in% names(tbl)) {
      return(tbl)
    }
    
    tbl[, c(setdiff(names(tbl), "method"), "method"), drop = FALSE]
  }
  
  safe_fit_table <- function(fit, element, fallback = tibble::tibble()) {
    if (is.null(fit) || inherits(fit, "error") || is.null(fit[[element]])) {
      return(fallback)
    }
    fit[[element]]
  }
  
  edges <- purrr::map_df(res, "edges")
  if (nrow(edges) > 0) {
    edges <- edges %>%
      dplyr::mutate(
        analysis_mode = "multivariable",
        pairwise_predictor = NA_character_
      ) %>%
      place_method_last()
  }
  
  all_coefficients <- purrr::map_df(res, "all_coefficients")
  if (nrow(all_coefficients) > 0) {
    all_coefficients <- all_coefficients %>%
      dplyr::mutate(
        analysis_mode = "multivariable",
        pairwise_predictor = NA_character_
      ) %>%
      place_method_last()
  }
  
  vif_group <- purrr::map_df(res, "vif_group")
  vif_pairs <- purrr::map_df(res, "vif_pairs")
  
  
  # ------------------------------------------------------------
  # 5.8 Optional pairwise CIRN inference
  # ------------------------------------------------------------
  
  # Pairwise mode fits CIRN one target-regulator pair at a time.
  # This is useful for checking whether an inferred edge is robust
  # to conditioning on other predictors. It is intentionally stored
  # independently from the primary multivariable CIRN output.
  
  pairwise <- list(
    edges = tibble::tibble(),
    all_coefficients = tibble::tibble(),
    diagnostics = tibble::tibble(),
    vif_group = tibble::tibble(),
    vif_pairs = tibble::tibble(),
    models = list()
  )
  
  if (run_pairwise) {
    pairwise_grid <- tidyr::crossing(
      target = targets,
      predictor = predictors
    )
    
    pairwise_fits <- purrr::pmap(
      pairwise_grid,
      function(target, predictor) {
        tryCatch(
          infer_for_target(
            df_derivs = df_derivs,
            target = target,
            time_col = time_col,
            predictors = predictor,
            representation_mode = pairwise_representation_mode,
            lag_units = lag_units,
            response_eps = response_eps,
            adaptive_jitter = adaptive_jitter,
            jitter_predictors = jitter_predictors,
            jitter_min_class_count = jitter_min_class_count,
            jitter_scale_grid = jitter_scale_grid,
            jitter_scale_basis = jitter_scale_basis,
            model_iter = model_iter,
            model_warmup = model_warmup,
            model_chains = model_chains,
            model_cores = model_cores,
            prior_mean = prior_mean,
            prior_sd = prior_sd,
            adapt_delta = adapt_delta,
            compute_loo = compute_loo,
            loo_moment_match = loo_moment_match,
            loo_reloo = loo_reloo,
            loo_k_threshold = loo_k_threshold,
            seed = seed,
            progress_callback = make_progress_callback(
              analysis_mode = "pairwise",
              pairwise_predictor = predictor
            ),
            debug = debug,
            assign_debug_to_global = FALSE
          ),
          error = function(e) {
            warning(
              "Pairwise CIRN fit failed for target = ", target,
              ", predictor = ", predictor, ": ", conditionMessage(e)
            )
            make_empty_target_result(
              target = target,
              representation_mode = pairwise_representation_mode,
              status = paste0("pairwise_fit_failed: ", conditionMessage(e)),
              n_predictors = 1L
            )
          }
        )
      }
    )
    
    pairwise_names <- paste(pairwise_grid$target, pairwise_grid$predictor, sep = "__")
    
    pairwise$edges <- purrr::map2_df(
      pairwise_fits,
      seq_along(pairwise_fits),
      function(fit, i) {
        fit_edges <- safe_fit_table(fit, "edges", empty_bayes_result_tibble)
        if (nrow(fit_edges) == 0) {
          return(tibble::tibble())
        }
        fit_edges %>%
          dplyr::mutate(
            analysis_mode = "pairwise",
            pairwise_target = pairwise_grid$target[[i]],
            pairwise_predictor = pairwise_grid$predictor[[i]]
          )
      }
    ) %>%
      place_method_last()
    
    pairwise$all_coefficients <- purrr::map2_df(
      pairwise_fits,
      seq_along(pairwise_fits),
      function(fit, i) {
        fit_coefficients <- safe_fit_table(
          fit,
          "all_coefficients",
          empty_bayes_all_coefficients_tibble
        )
        if (nrow(fit_coefficients) == 0) {
          return(tibble::tibble())
        }
        fit_coefficients %>%
          dplyr::mutate(
            analysis_mode = "pairwise",
            pairwise_target = pairwise_grid$target[[i]],
            pairwise_predictor = pairwise_grid$predictor[[i]]
          )
      }
    ) %>%
      place_method_last()
    
    pairwise$diagnostics <- purrr::map2_df(
      pairwise_fits,
      seq_along(pairwise_fits),
      function(fit, i) {
        diag_sets <- safe_fit_table(fit, "diagnostics", list())
        if (length(diag_sets) == 0) {
          return(
            tibble::tibble(
              target = pairwise_grid$target[[i]],
              pairwise_predictor = pairwise_grid$predictor[[i]],
              predictor_set = NA_character_,
              status = "missing_pairwise_diagnostics"
            )
          )
        }
        purrr::imap_dfr(
          diag_sets,
          function(diag_set, set_name) {
            diag <- diag_set$diagnostics
            
            tibble::tibble(
              target = pairwise_grid$target[[i]],
              pairwise_predictor = pairwise_grid$predictor[[i]],
              predictor_set = set_name,
              response_mode = if (!is.null(diag$response_mode)) {
                diag$response_mode
              } else {
                NA_character_
              },
              response_label = if (!is.null(diag$response_label)) {
                diag$response_label
              } else {
                NA_character_
              },
              response_source = if (!is.null(diag$response_source)) {
                diag$response_source
              } else {
                NA_character_
              },
              dominant_direction = if (!is.null(diag$dominant_direction)) {
                diag$dominant_direction
              } else {
                NA_real_
              },
              jitter_used = if (!is.null(diag$jitter_used)) {
                diag$jitter_used
              } else {
                FALSE
              },
              jitter_scale = if (!is.null(diag$jitter_scale)) {
                diag$jitter_scale
              } else {
                NA_real_
              },
              jitter_scale_factor = if (!is.null(diag$jitter_scale_factor)) {
                diag$jitter_scale_factor
              } else {
                NA_real_
              },
              jitter_basis = if (!is.null(diag$jitter_basis)) {
                diag$jitter_basis
              } else {
                NA_character_
              },
              jitter_basis_value = if (!is.null(diag$jitter_basis_value)) {
                diag$jitter_basis_value
              } else {
                NA_real_
              },
              jitter_attempts = if (!is.null(diag$jitter_attempts)) {
                diag$jitter_attempts
              } else {
                NA_integer_
              },
              predictor_jitter_used = if (!is.null(diag$predictor_jitter_used)) {
                diag$predictor_jitter_used
              } else {
                FALSE
              },
              predictor_jitter_scale_factor = if (!is.null(diag$predictor_jitter_scale_factor)) {
                diag$predictor_jitter_scale_factor
              } else {
                NA_real_
              },
              predictor_jitter_variables = if (!is.null(diag$predictor_jitter_variables)) {
                diag$predictor_jitter_variables
              } else {
                NA_character_
              },
              class_0_count = if (!is.null(diag$class_0_count)) {
                diag$class_0_count
              } else {
                NA_integer_
              },
              class_1_count = if (!is.null(diag$class_1_count)) {
                diag$class_1_count
              } else {
                NA_integer_
              },
              min_class_count = if (!is.null(diag$min_class_count)) {
                diag$min_class_count
              } else {
                NA_integer_
              },
              n_predictors = if (!is.null(diag$n_predictors)) {
                diag$n_predictors
              } else {
                NA_integer_
              },
              response_trigger = if (!is.null(diag$response_trigger)) {
                diag$response_trigger
              } else {
                NA_character_
              },
              status = if (!is.null(diag$status)) {
                diag$status
              } else {
                NA_character_
              },
              max_rhat = if (length(diag$rhat) > 0 && any(!is.na(diag$rhat))) {
                max(diag$rhat, na.rm = TRUE)
              } else {
                NA_real_
              },
              min_neff = if (length(diag$neff_ratio) > 0 && any(!is.na(diag$neff_ratio))) {
                min(diag$neff_ratio, na.rm = TRUE)
              } else {
                NA_real_
              },
              loo_elpd = if (!is.null(diag$loo)) {
                diag$loo$estimates["elpd_loo", "Estimate"]
              } else {
                NA_real_
              }
            )
          }
        )
      }
    )
    
    pairwise$vif_group <- purrr::map2_df(
      pairwise_fits,
      seq_along(pairwise_fits),
      function(fit, i) {
        vif_tbl <- safe_fit_table(fit, "vif_group", empty_vif_group_tibble)
        if (nrow(vif_tbl) == 0) {
          return(tibble::tibble())
        }
        vif_tbl %>%
          dplyr::mutate(pairwise_predictor = pairwise_grid$predictor[[i]])
      }
    )
    
    pairwise$vif_pairs <- purrr::map2_df(
      pairwise_fits,
      seq_along(pairwise_fits),
      function(fit, i) {
        vif_tbl <- safe_fit_table(fit, "vif_pairs", empty_vif_pair_tibble)
        if (nrow(vif_tbl) == 0) {
          return(tibble::tibble())
        }
        vif_tbl %>%
          dplyr::mutate(pairwise_predictor = pairwise_grid$predictor[[i]])
      }
    )
    
    pairwise$models <- stats::setNames(
      purrr::map(pairwise_fits, ~ safe_fit_table(.x, "models", list())),
      pairwise_names
    )
  }
  
  edges_combined <- dplyr::bind_rows(edges, pairwise$edges) %>%
    place_method_last()
  
  all_coefficients_combined <- dplyr::bind_rows(
    all_coefficients,
    pairwise$all_coefficients
  ) %>%
    place_method_last()
  
  
  # ------------------------------------------------------------
  # 5.9 Aggregate diagnostics across targets
  # ------------------------------------------------------------
  
  # Diagnostics are reported for transparency only.
  # They DO NOT influence edge inclusion.
  
  diagnostics <- purrr::map_df(seq_along(targets), function(i) {
    
    target_name <- targets[i]
    diag_sets   <- res[[i]]$diagnostics
    
    bind_rows(
      lapply(names(diag_sets), function(set_name) {
        
        diag <- diag_sets[[set_name]]$diagnostics
        
        max_rhat <- if (length(diag$rhat) > 0 && any(!is.na(diag$rhat))) {
          max(diag$rhat, na.rm = TRUE)
        } else {
          NA_real_
        }
        
        min_neff <- if (length(diag$neff_ratio) > 0 && any(!is.na(diag$neff_ratio))) {
          min(diag$neff_ratio, na.rm = TRUE)
        } else {
          NA_real_
        }
        
        tibble(
          target = target_name,
          predictor_set = set_name,
          response_mode = if (!is.null(diag$response_mode)) {
            diag$response_mode
          } else {
            NA_character_
          },
          response_label = if (!is.null(diag$response_label)) {
            diag$response_label
          } else {
            NA_character_
          },
          response_source = if (!is.null(diag$response_source)) {
            diag$response_source
          } else {
            NA_character_
          },
          dominant_direction = if (!is.null(diag$dominant_direction)) {
            diag$dominant_direction
          } else {
            NA_real_
          },
          jitter_used = if (!is.null(diag$jitter_used)) {
            diag$jitter_used
          } else {
            FALSE
          },
          jitter_scale = if (!is.null(diag$jitter_scale)) {
            diag$jitter_scale
          } else {
            NA_real_
          },
          jitter_scale_factor = if (!is.null(diag$jitter_scale_factor)) {
            diag$jitter_scale_factor
          } else {
            NA_real_
          },
          jitter_basis = if (!is.null(diag$jitter_basis)) {
            diag$jitter_basis
          } else {
            NA_character_
          },
          jitter_basis_value = if (!is.null(diag$jitter_basis_value)) {
            diag$jitter_basis_value
          } else {
            NA_real_
          },
          jitter_attempts = if (!is.null(diag$jitter_attempts)) {
            diag$jitter_attempts
          } else {
            NA_integer_
          },
          predictor_jitter_used = if (!is.null(diag$predictor_jitter_used)) {
            diag$predictor_jitter_used
          } else {
            FALSE
          },
          predictor_jitter_scale_factor = if (!is.null(diag$predictor_jitter_scale_factor)) {
            diag$predictor_jitter_scale_factor
          } else {
            NA_real_
          },
          predictor_jitter_variables = if (!is.null(diag$predictor_jitter_variables)) {
            diag$predictor_jitter_variables
          } else {
            NA_character_
          },
          class_0_count = if (!is.null(diag$class_0_count)) {
            diag$class_0_count
          } else {
            NA_integer_
          },
          class_1_count = if (!is.null(diag$class_1_count)) {
            diag$class_1_count
          } else {
            NA_integer_
          },
          min_class_count = if (!is.null(diag$min_class_count)) {
            diag$min_class_count
          } else {
            NA_integer_
          },
          n_predictors = if (!is.null(diag$n_predictors)) {
            diag$n_predictors
          } else {
            NA_integer_
          },
          response_trigger = if (!is.null(diag$response_trigger)) {
            diag$response_trigger
          } else {
            NA_character_
          },
          status = if (!is.null(diag$status)) {
            diag$status
          } else {
            NA_character_
          },
          max_rhat = max_rhat,
          min_neff = min_neff,
          loo_elpd = if (!is.null(diag$loo)) {
            diag$loo$estimates["elpd_loo", "Estimate"]
          } else {
            NA_real_
          }
        )
      })
    )
  })
  
  
  # ------------------------------------------------------------
  # 5.10 Informative completion message
  # ------------------------------------------------------------
  
  if (nrow(edges) == 0) {
    message("No regulatory edges inferred under current HDI criterion.")
  } else {
    message("CIRN inference completed: ",
            nrow(edges),
            " regulatory edges inferred.")
  }
  
  
  # ------------------------------------------------------------
  # 5.11 Return CIRN results
  # ------------------------------------------------------------
  
  return(list(
    edges = edges,
    all_coefficients = all_coefficients,
    all_coefficients_combined = all_coefficients_combined,
    pairwise = pairwise,
    edges_combined = edges_combined,
    diagnostics = diagnostics,
    vif_group = vif_group,
    vif_pairs = vif_pairs,
    models = models,
    debug = debug_tables,
    settings = list(
      time_col = time_col,
      targets = targets,
      predictors = predictors,
      representation_mode = representation_mode,
      lag_units = lag_units,
      run_pairwise = run_pairwise,
      pairwise_representation_mode = pairwise_representation_mode,
      points_per_interval = points_per_interval,
      spar = spar,
      outlier_method = outlier_method,
      outlier_thresh = outlier_thresh,
      outlier_action = outlier_action,
      response_eps = response_eps,
      adaptive_jitter = adaptive_jitter,
      jitter_predictors = jitter_predictors,
      jitter_min_class_count = jitter_min_class_count,
      jitter_scale_grid = jitter_scale_grid,
      jitter_scale_basis = jitter_scale_basis,
      model_iter = model_iter,
      model_warmup = model_warmup,
      model_chains = model_chains,
      model_cores = model_cores,
      prior_mean = prior_mean,
      prior_sd = prior_sd,
      adapt_delta = adapt_delta,
      compute_loo = compute_loo,
      loo_moment_match = loo_moment_match,
      loo_reloo = loo_reloo,
      loo_k_threshold = loo_k_threshold,
      seed = seed,
      show_progress = show_progress,
      progress_bar = progress_bar,
      total_model_fits = total_model_fits,
      completed_model_fits = progress_env$completed
    )
  ))
}


# IMPORTANT:
# The inferred CIRN network represents a probabilistic structural
# summary of regulatory influence. It does NOT establish
# interventional causality and should not be interpreted as
# a forecasting model.

################################################################################
################################################################################


# ================================
# 6. Evaluation Metrics
# ================================

# --------------------------------------------------------------
# 6.1 Core evaluation for a single inferred network
#     Representation-agnostic CIRN evaluation
# --------------------------------------------------------------

# Evaluates signed edge recovery given a ground-truth adjacency matrix

evaluate_representation_agnostic <- function(true_adj, inferred_edges) {
  
  vars <- rownames(true_adj)
  if (is.null(vars) || is.null(colnames(true_adj)) || !all(vars == colnames(true_adj))) {
    stop("true_adj must be a square signed adjacency matrix with matching row and column names.")
  }
  
  # Initialize predicted adjacency (state-level)
  
  pred_adj <- matrix(0, nrow = length(vars), ncol = length(vars),
                     dimnames = list(vars, vars))
  
  # Build predicted adjacency from all inferred edges first. This is
  # necessary for correct false-positive accounting.
  
  if (!is.null(inferred_edges) && nrow(inferred_edges) > 0) {
    
    target_lookup <- stats::setNames(vars, paste0("d", vars))
    
    pred_edges <- inferred_edges %>%
      dplyr::mutate(
        predictor_base = base_cirn_term(.data$term),
        target_base = dplyr::case_when(
          .data$target %in% names(target_lookup) ~ target_lookup[.data$target],
          .data$target %in% vars ~ as.character(.data$target),
          TRUE ~ NA_character_
        ),
        inferred_sign = dplyr::case_when(
          .data$omega > 0 ~ 1,
          .data$omega < 0 ~ -1,
          .data$regulation_type == "activation" ~ 1,
          .data$regulation_type == "inhibition" ~ -1,
          TRUE ~ 0
        )
      ) %>%
      dplyr::filter(
        .data$predictor_base %in% vars,
        .data$target_base %in% vars,
        .data$inferred_sign != 0
      ) %>%
      dplyr::group_by(.data$predictor_base, .data$target_base) %>%
      dplyr::arrange(dplyr::desc(abs(.data$omega)), .by_group = TRUE) %>%
      dplyr::summarise(
        inferred_sign = dplyr::first(.data$inferred_sign),
        .groups = "drop"
      )
    
    for (row_id in seq_len(nrow(pred_edges))) {
      pred_adj[
        pred_edges$predictor_base[[row_id]],
        pred_edges$target_base[[row_id]]
      ] <- pred_edges$inferred_sign[[row_id]]
    }
  }
  
  # ----------------------------------------------------------
  # 6.1.1 Confusion matrix components (signed edge recovery)
  # ----------------------------------------------------------
  
  # NOTE:
  # Evaluation is performed on a signed adjacency matrix:
  #   •  0  : no regulatory interaction
  #   • +1  : activating regulation
  #   • −1  : inhibiting regulation
  #
  # A True Positive (TP) requires BOTH:
  #   (i) correct identification of an interaction, and
  #   (ii) correct regulatory sign.
  
  sign_errors <- sum(pred_adj != 0 & true_adj != 0 & sign(pred_adj) != sign(true_adj))
  
  # Sign Errors:
  #   A regulatory edge is present in both inferred and true networks,
  #   but the inferred activation/inhibition sign is wrong.
  
  TP <- sum(pred_adj != 0 & true_adj != 0 & sign(pred_adj) == sign(true_adj))
  
  # True Positives (TP):
  #   Correctly inferred regulatory edges with correct sign
  
  FP <- sum(pred_adj != 0 & true_adj == 0) + sign_errors
  
  # False Positives (FP):
  #   Spurious inferred edges where no true regulation exists, plus
  #   wrong-sign recoveries of true edges
  
  FN <- sum(true_adj != 0 & pred_adj == 0) + sign_errors
  
  # False Negatives (FN):
  #   True regulatory edges that CIRN failed to recover, plus true
  #   edges recovered with the wrong regulatory sign
  
  TN <- sum(pred_adj == 0 & true_adj == 0)
  
  # True Negatives (TN):
  #   Correct identification of non-interacting variable pairs
  #   (important in sparse regulatory networks)
  
  # ----------------------------------------------------------
  # Classical performance metrics
  # ----------------------------------------------------------
  
  precision <- TP / (TP + FP + 1e-9)
  
  # Precision:
  #   Fraction of inferred regulatory edges that are truly present
  #   High precision indicates low false discovery rate
  
  recall <- TP / (TP + FN + 1e-9)
  
  # Recall (Sensitivity):
  #   Fraction of true regulatory edges successfully recovered
  #   High recall indicates effective detection of true interactions
  
  specificity <- TN / (TN + FP + 1e-9)
  
  # Specificity:
  #   Fraction of true non-interactions correctly identified
  #   Particularly relevant when networks are sparse
  
  f1 <- 2 * precision * recall / (precision + recall + 1e-9)
  
  # F1-score:
  #   Harmonic mean of precision and recall
  #   Provides a single summary metric balancing false positives
  #   and false negatives
  
  balanced_accuracy <- (recall + specificity) / 2
  
  # Balanced accuracy:
  #   Mean of sensitivity and specificity
  #   Robust to class imbalance between interacting and
  #   non-interacting variable pairs
  
  # ----------------------------------------------------------
  # 6.1.2 Matthews Correlation Coefficient (robust to imbalance)
  # ----------------------------------------------------------
  
  # MCC summarizes the confusion matrix into a single correlation
  # coefficient between true and inferred edge states.
  # It accounts for all four outcomes (TP, FP, FN, TN) and remains
  # informative even under severe class imbalance.
  
  mcc_num <- (TP * TN - FP * FN)
  mcc_den <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + 1e-9
  mcc <- mcc_num / mcc_den
  
  # MCC interpretation:
  #   • mcc ∈ [−1, 1]
  #       +1 : perfect agreement between inferred and true networks
  #        0 : performance equivalent to random guessing
  #       −1 : complete disagreement
  #
  # MCC is particularly suitable for evaluating sparse regulatory
  # networks, where non-interactions vastly outnumber true edges.
  
  # ----------------------------------------------------------
  # 6.1.3 Structural Hamming Distance (edge presence only)
  # ----------------------------------------------------------
  
  # SHD quantifies structural disagreement between inferred and
  # true networks based on edge presence/absence only.
  
  shd <- sum((pred_adj != 0) != (true_adj != 0))
  signed_shd <- sum(pred_adj != true_adj)
  
  # SHD interpretation:
  #   • Counts the minimum number of edge insertions or deletions
  #     required to transform the inferred network into the
  #     true network
  #   • Ignores regulatory sign (activation vs inhibition)
  #   • Lower SHD values indicate better recovery of network structure
  
  # ----------------------------------------------------------
  # 6.1.4 Optional AUC (binary edge existence, sign ignored)
  # ----------------------------------------------------------
  
  truth_vec <- as.numeric(true_adj != 0)
  pred_vec  <- as.numeric(pred_adj != 0)
  
  if (exists("use_auc") && use_auc && length(unique(truth_vec)) == 2) {
    roc_obj <- pROC::roc(truth_vec, pred_vec, quiet = TRUE)
    auc_val <- as.numeric(pROC::auc(roc_obj))
  } else {
    auc_val <- NA_real_
  }
  
  # AUC (Area Under the ROC Curve):
  #   Measures the ability to discriminate between
  #   edge presence and absence
  #   • Regulatory sign is ignored
  #   • Included for comparability with classical
  #     network inference studies
  #   • NOT used for CIRN edge inclusion decisions
  
  list(
    TP = TP, FP = FP, FN = FN, TN = TN,
    sign_errors = sign_errors,
    precision = precision,
    recall = recall,
    specificity = specificity,
    balanced_accuracy = balanced_accuracy,
    f1 = f1,
    mcc = mcc,
    shd = shd,
    signed_shd = signed_shd
  )
}

# ----------------------------------------------------------
# 6.2 Presentation helper: long-form evaluation table
# ----------------------------------------------------------

metrics_to_long_table <- function(metrics_list) {
  
  tibble::enframe(
    metrics_list,
    name  = "Metric",
    value = "Value"
  ) %>%
    mutate(
      Value = round(as.numeric(unlist(Value)), 2),  # ← round to 2 decimals
      Metric = dplyr::case_when(
        Metric == "TP" ~ "True Positives (TP)",
        Metric == "FP" ~ "False Positives (FP)",
        Metric == "FN" ~ "False Negatives (FN)",
        Metric == "TN" ~ "True Negatives (TN)",
        Metric == "sign_errors" ~ "Wrong-Sign Recoveries",
        Metric == "precision" ~ "Precision",
        Metric == "recall" ~ "Recall",
        Metric == "specificity" ~ "Specificity",
        Metric == "balanced_accuracy" ~ "Balanced Accuracy",
        Metric == "f1" ~ "F1 score",
        Metric == "mcc" ~ "Matthews Correlation Coefficient (MCC)",
        Metric == "shd" ~ "Structural Hamming Distance (edge presence)",
        Metric == "signed_shd" ~ "Signed Structural Hamming Distance",
        Metric == "auc" ~ "AUC (edge existence)",
        TRUE ~ Metric
      )
    )
}


# ----------------------------------------------------------
# 6.3 Objective: sensitivity-analysis helpers
# ----------------------------------------------------------

# These helpers keep the primary CIRN inference path unchanged.
# They repeatedly call infer_network() under controlled perturbations
# so robustness can be reported as an empirical property of the method.

cirn_non_inference_config_keys <- function() {
  c(
    "run_ground_truth_validation",
    "true_adj",
    "run_sensitivity_analysis",
    "sensitivity_plan",
    "sensitivity_replicates",
    "sensitivity_noise_sd_fractions",
    "sensitivity_lag_units",
    "sensitivity_downsample_intervals",
    "sensitivity_target_sample_sizes",
    "sensitivity_missing_fractions",
    "sensitivity_inference_scope",
    "sensitivity_save_outputs",
    "sensitivity_show_progress",
    "sensitivity_progress_bar"
  )
}

resolve_cirn_sensitivity_scope <- function(scope = "use_config",
                                           representation_mode = "both",
                                           run_pairwise = FALSE,
                                           pairwise_representation_mode = NULL) {
  
  scope <- if (is.null(scope) || length(scope) == 0) {
    "use_config"
  } else {
    as.character(scope[[1]])
  }
  if (length(scope) == 0 || is.na(scope) || !nzchar(scope)) {
    scope <- "use_config"
  }
  
  scope_alias <- c(
    use_run_settings = "use_config",
    sublevel_all_predictors = "both"
  )
  if (scope %in% names(scope_alias)) {
    scope <- unname(scope_alias[[scope]])
  }
  
  valid_scopes <- c(
    "use_config",
    "sublevel",
    "all_predictors",
    "both",
    "pairwise_only",
    "everything"
  )
  if (!scope %in% valid_scopes) {
    stop(
      "sensitivity_inference_scope must be one of: ",
      paste(valid_scopes, collapse = ", "),
      "."
    )
  }
  
  representation_mode <- if (is.null(representation_mode) ||
                             !representation_mode %in% c("sublevel", "all_predictors", "both")) {
    "both"
  } else {
    representation_mode
  }
  
  pairwise_representation_mode <- if (is.null(pairwise_representation_mode) ||
                                      !pairwise_representation_mode %in% c("sublevel", "all_predictors", "both")) {
    representation_mode
  } else {
    pairwise_representation_mode
  }
  
  switch(
    scope,
    use_config = list(
      key = "use_config",
      title = "Use supplied CIRN configuration",
      representation_mode = representation_mode,
      run_pairwise = isTRUE(run_pairwise),
      pairwise_representation_mode = pairwise_representation_mode,
      edge_focus = "all",
      body = "Sensitivity uses the representation and pairwise settings supplied in base_config.",
      detail = "Best default when robustness checks should match the primary CIRN analysis configuration."
    ),
    sublevel = list(
      key = "sublevel",
      title = "Sublevel only",
      representation_mode = "sublevel",
      run_pairwise = FALSE,
      pairwise_representation_mode = "sublevel",
      edge_focus = "all",
      body = "Sensitivity fits only sublevel multivariable CIRN models.",
      detail = "Good fast robustness check with interpretable state, first-derivative, and second-derivative levels."
    ),
    all_predictors = list(
      key = "all_predictors",
      title = "All predictors only",
      representation_mode = "all_predictors",
      run_pairwise = FALSE,
      pairwise_representation_mode = "all_predictors",
      edge_focus = "all",
      body = "Sensitivity fits only the all-predictors multivariable CIRN model.",
      detail = "Useful for checking the full combined predictor set; inspect VIF and posterior diagnostics carefully."
    ),
    both = list(
      key = "both",
      title = "Sublevel + all predictors",
      representation_mode = "both",
      run_pairwise = FALSE,
      pairwise_representation_mode = "sublevel",
      edge_focus = "all",
      body = "Sensitivity fits both sublevel and all-predictors multivariable CIRN models, without pairwise CIRN.",
      detail = "Recommended for most serious non-pairwise robustness checks."
    ),
    pairwise_only = list(
      key = "pairwise_only",
      title = "Pairwise only",
      representation_mode = "sublevel",
      run_pairwise = TRUE,
      pairwise_representation_mode = pairwise_representation_mode,
      edge_focus = "pairwise",
      body = "Sensitivity focuses stability summaries on pairwise CIRN edges.",
      detail = "infer_network() still performs a minimal main sublevel fit internally before the pairwise add-on."
    ),
    everything = list(
      key = "everything",
      title = "Everything",
      representation_mode = "both",
      run_pairwise = TRUE,
      pairwise_representation_mode = "both",
      edge_focus = "all",
      body = "Sensitivity fits sublevel, all-predictors, pairwise sublevel, and pairwise all-predictors CIRN.",
      detail = "Best for final robustness appendices; expect longer runtime and more retained-edge rows."
    )
  )
}

make_cirn_sensitivity_plan <- function(noise_sd_fractions = c(0.01, 0.05, 0.10),
                                       lag_units = c(1, 2, 3),
                                       downsample_intervals = c(2, 5),
                                       target_sample_sizes = c(25, 50, 75, 100),
                                       missing_fractions = c(0.10, 0.25),
                                       replicates = 3) {
  
  if (length(replicates) != 1) {
    stop("sensitivity_replicates must be a single positive integer.")
  }
  replicates <- as.integer(replicates)
  if (is.na(replicates) || replicates < 1) {
    stop("sensitivity_replicates must be a positive integer.")
  }
  
  rows <- list(
    tibble::tibble(
      condition = "baseline",
      scenario = "baseline",
      value = NA_real_,
      replicate = 1L
    )
  )
  
  if (length(noise_sd_fractions) > 0) {
    noise_rows <- tibble::tibble(
      condition = paste0("noise_sd_fraction_", noise_sd_fractions),
      scenario = "noise_sd_fraction",
      value = as.numeric(noise_sd_fractions)
    )
    rows[[length(rows) + 1L]] <- noise_rows[
      rep(seq_len(nrow(noise_rows)), each = replicates),
      ,
      drop = FALSE
    ] %>%
      dplyr::mutate(replicate = rep(seq_len(replicates), times = nrow(noise_rows)))
  }
  
  if (length(lag_units) > 0) {
    rows[[length(rows) + 1L]] <- tibble::tibble(
      condition = paste0("lag_", as.integer(lag_units)),
      scenario = "lag_units",
      value = as.numeric(lag_units),
      replicate = 1L
    )
  }
  
  if (length(downsample_intervals) > 0) {
    rows[[length(rows) + 1L]] <- tibble::tibble(
      condition = paste0("downsample_every_", as.integer(downsample_intervals)),
      scenario = "downsample_interval",
      value = as.numeric(downsample_intervals),
      replicate = 1L
    )
  }
  
  if (length(target_sample_sizes) > 0) {
    rows[[length(rows) + 1L]] <- tibble::tibble(
      condition = paste0("target_n_", as.integer(target_sample_sizes)),
      scenario = "target_sample_size",
      value = as.numeric(target_sample_sizes),
      replicate = 1L
    )
  }
  
  if (length(missing_fractions) > 0) {
    missing_rows <- tibble::tibble(
      condition = paste0("row_missing_fraction_", missing_fractions),
      scenario = "row_missing_fraction",
      value = as.numeric(missing_fractions)
    )
    rows[[length(rows) + 1L]] <- missing_rows[
      rep(seq_len(nrow(missing_rows)), each = replicates),
      ,
      drop = FALSE
    ] %>%
      dplyr::mutate(replicate = rep(seq_len(replicates), times = nrow(missing_rows)))
  }
  
  dplyr::bind_rows(rows) %>%
    dplyr::mutate(
      sensitivity_id = dplyr::row_number(),
      scenario_value = dplyr::if_else(
        is.na(.data$value),
        "NA",
        as.character(.data$value)
      )
    ) %>%
    dplyr::select(
      sensitivity_id,
      condition,
      scenario,
      scenario_value,
      value,
      replicate
    )
}

prepare_cirn_sensitivity_dataset <- function(df,
                                             time_col,
                                             scenario,
                                             value = NA_real_,
                                             replicate = 1L,
                                             seed = 123) {
  
  if (!time_col %in% names(df)) {
    stop("time_col is not present in df.")
  }
  
  df <- df %>%
    dplyr::arrange(.data[[time_col]])
  
  numeric_vars <- setdiff(
    names(df)[vapply(df, is.numeric, logical(1))],
    time_col
  )
  
  scenario <- as.character(scenario)
  replicate <- as.integer(replicate)
  scenario_seed <- as.integer(
    (as.numeric(seed) + 1009L * replicate + sum(utf8ToInt(scenario))) %%
      .Machine$integer.max
  )
  
  if (scenario == "baseline") {
    return(df)
  }
  
  if (scenario == "noise_sd_fraction") {
    if (!is.finite(value) || value < 0) {
      stop("noise_sd_fraction must be non-negative.")
    }
    
    seed_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    old_seed <- if (seed_exists) {
      get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    } else {
      NULL
    }
    on.exit(restore_random_seed(seed_exists, old_seed), add = TRUE)
    set.seed(scenario_seed)
    
    for (var in numeric_vars) {
      var_sd <- stats::sd(df[[var]], na.rm = TRUE)
      if (is.finite(var_sd) && var_sd > 0) {
        df[[var]] <- df[[var]] + stats::rnorm(nrow(df), 0, value * var_sd)
      }
    }
    return(df)
  }
  
  if (scenario == "lag_units") {
    return(df)
  }
  
  if (scenario == "downsample_interval") {
    interval <- as.integer(value)
    if (is.na(interval) || interval < 1) {
      stop("downsample_interval must be a positive integer.")
    }
    return(df[seq(1, nrow(df), by = interval), , drop = FALSE])
  }
  
  if (scenario == "target_sample_size") {
    target_n <- as.integer(value)
    if (is.na(target_n) || target_n < 3) {
      stop("target_sample_size must be at least 3.")
    }
    target_n <- min(target_n, nrow(df))
    index <- unique(as.integer(round(seq(1, nrow(df), length.out = target_n))))
    return(df[index, , drop = FALSE])
  }
  
  if (scenario == "row_missing_fraction") {
    if (!is.finite(value) || value < 0 || value >= 1) {
      stop("row_missing_fraction must be in [0, 1).")
    }
    
    seed_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    old_seed <- if (seed_exists) {
      get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    } else {
      NULL
    }
    on.exit(restore_random_seed(seed_exists, old_seed), add = TRUE)
    set.seed(scenario_seed)
    
    remove_n <- floor(nrow(df) * value)
    if (remove_n <= 0) {
      return(df)
    }
    keep_index <- setdiff(seq_len(nrow(df)), sample(seq_len(nrow(df)), remove_n))
    keep_index <- sort(keep_index)
    if (length(keep_index) < 3) {
      stop("row_missing_fraction leaves fewer than 3 observations.")
    }
    return(df[keep_index, , drop = FALSE])
  }
  
  stop("Unsupported sensitivity scenario: ", scenario)
}

summarize_cirn_effective_sample_size <- function(res) {
  
  multivariable_diag <- if (!is.null(res$diagnostics) && nrow(res$diagnostics) > 0) {
    res$diagnostics %>%
      dplyr::mutate(
        analysis_mode = "multivariable",
        pairwise_predictor = NA_character_
      )
  } else {
    tibble::tibble()
  }
  
  pairwise_diag <- if (!is.null(res$pairwise$diagnostics) &&
                       nrow(res$pairwise$diagnostics) > 0) {
    res$pairwise$diagnostics %>%
      dplyr::mutate(analysis_mode = "pairwise")
  } else {
    tibble::tibble()
  }
  
  diagnostics <- dplyr::bind_rows(multivariable_diag, pairwise_diag)
  if (nrow(diagnostics) == 0) {
    return(tibble::tibble())
  }
  
  if (!"n_predictors" %in% names(diagnostics)) {
    diagnostics$n_predictors <- NA_integer_
  }
  
  diagnostics %>%
    dplyr::mutate(
      usable_n = .data$class_0_count + .data$class_1_count,
      minority_class_count = pmin(.data$class_0_count, .data$class_1_count),
      class_balance = dplyr::if_else(
        is.finite(.data$usable_n) & .data$usable_n > 0,
        .data$minority_class_count / .data$usable_n,
        NA_real_
      ),
      minority_per_predictor = dplyr::if_else(
        is.finite(.data$n_predictors) & .data$n_predictors > 0,
        .data$minority_class_count / .data$n_predictors,
        NA_real_
      ),
      sample_size_flag = dplyr::case_when(
        is.na(.data$status) ~ "unknown",
        .data$status != "completed" ~ paste0("not_fitted_", .data$status),
        !is.finite(.data$usable_n) ~ "unknown",
        .data$minority_class_count < 5 ~ "below_minimum",
        .data$minority_class_count < 10 ~ "small_minority_class",
        is.finite(.data$minority_per_predictor) &
          .data$minority_per_predictor < 5 ~ "low_minority_per_predictor",
        TRUE ~ "adequate_basic"
      )
    ) %>%
    dplyr::select(
      analysis_mode,
      target,
      predictor_set,
      pairwise_predictor,
      status,
      class_0_count,
      class_1_count,
      usable_n,
      minority_class_count,
      n_predictors,
      minority_per_predictor,
      class_balance,
      jitter_used,
      predictor_jitter_used,
      sample_size_flag
    )
}

state_level_edge_signature <- function(edges) {
  
  if (is.null(edges) || nrow(edges) == 0) {
    return(
      tibble::tibble(
        predictor_base = character(),
        target_base = character(),
        inferred_sign = integer(),
        sign_label = character()
      )
    )
  }
  
  edges %>%
    dplyr::mutate(
      predictor_base = base_cirn_term(.data$term),
      target_base = sub("^d", "", as.character(.data$target)),
      inferred_sign = dplyr::case_when(
        .data$omega > 0 ~ 1L,
        .data$omega < 0 ~ -1L,
        TRUE ~ 0L
      ),
      sign_label = dplyr::case_when(
        .data$inferred_sign > 0 ~ "activation",
        .data$inferred_sign < 0 ~ "inhibition",
        TRUE ~ "zero"
      )
    ) %>%
    dplyr::filter(.data$inferred_sign != 0) %>%
    dplyr::distinct(
      .data$predictor_base,
      .data$target_base,
      .data$inferred_sign,
      .data$sign_label
    )
}

summarize_sensitivity_edge_stability <- function(sensitivity_edges,
                                                 sensitivity_runs) {
  
  if (is.null(sensitivity_edges) || nrow(sensitivity_edges) == 0 ||
      is.null(sensitivity_runs) || nrow(sensitivity_runs) == 0) {
    return(tibble::tibble())
  }
  
  completed_runs <- sensitivity_runs %>%
    dplyr::filter(.data$run_status == "completed") %>%
    dplyr::count(.data$scenario, .data$scenario_value, name = "completed_runs")
  
  baseline_keys <- sensitivity_edges %>%
    dplyr::filter(.data$scenario == "baseline") %>%
    state_level_edge_signature() %>%
    dplyr::mutate(in_baseline = TRUE)
  
  detected_edges <- sensitivity_edges %>%
    dplyr::mutate(
      predictor_base = base_cirn_term(.data$term),
      target_base = sub("^d", "", as.character(.data$target)),
      inferred_sign = dplyr::case_when(
        .data$omega > 0 ~ 1L,
        .data$omega < 0 ~ -1L,
        TRUE ~ 0L
      ),
      sign_label = dplyr::case_when(
        .data$inferred_sign > 0 ~ "activation",
        .data$inferred_sign < 0 ~ "inhibition",
        TRUE ~ "zero"
      )
    ) %>%
    dplyr::filter(.data$inferred_sign != 0) %>%
    dplyr::distinct(
      .data$sensitivity_id,
      .data$scenario,
      .data$scenario_value,
      .data$condition,
      .data$replicate,
      .data$predictor_base,
      .data$target_base,
      .data$inferred_sign,
      .data$sign_label
    )
  
  edge_keys <- detected_edges %>%
    dplyr::distinct(
      .data$predictor_base,
      .data$target_base,
      .data$inferred_sign,
      .data$sign_label
    )
  
  if (nrow(edge_keys) == 0 || nrow(completed_runs) == 0) {
    return(tibble::tibble())
  }
  
  scenario_edge_grid <- merge(
    completed_runs,
    edge_keys,
    by = NULL,
    all = TRUE
  )
  
  detected_counts <- detected_edges %>%
    dplyr::group_by(
      .data$scenario,
      .data$scenario_value,
      .data$predictor_base,
      .data$target_base,
      .data$inferred_sign,
      .data$sign_label
    ) %>%
    dplyr::summarise(
      detected_runs = dplyr::n_distinct(.data$sensitivity_id),
      conditions = paste(unique(.data$condition), collapse = "; "),
      .groups = "drop"
    )
  
  scenario_edge_grid %>%
    dplyr::left_join(
      detected_counts,
      by = c(
        "scenario",
        "scenario_value",
        "predictor_base",
        "target_base",
        "inferred_sign",
        "sign_label"
      )
    ) %>%
    dplyr::mutate(
      detected_runs = dplyr::coalesce(.data$detected_runs, 0L),
      conditions = dplyr::coalesce(.data$conditions, ""),
      detection_rate = dplyr::if_else(
        is.finite(.data$completed_runs) & .data$completed_runs > 0,
        .data$detected_runs / .data$completed_runs,
        NA_real_
      )
    ) %>%
    dplyr::left_join(
      baseline_keys,
      by = c("predictor_base", "target_base", "inferred_sign", "sign_label")
    ) %>%
    dplyr::mutate(in_baseline = dplyr::coalesce(.data$in_baseline, FALSE)) %>%
    dplyr::arrange(
      .data$scenario,
      .data$scenario_value,
      dplyr::desc(.data$detection_rate),
      .data$predictor_base,
      .data$target_base
    )
}

summarize_cirn_sensitivity_feature_edge_stability <- function(sensitivity_edges,
                                                              sensitivity_runs,
                                                              edge_focus = c("all", "pairwise")) {
  
  edge_focus <- match.arg(edge_focus)
  
  if (is.null(sensitivity_edges) || nrow(sensitivity_edges) == 0 ||
      is.null(sensitivity_runs) || nrow(sensitivity_runs) == 0) {
    return(tibble::tibble())
  }
  
  completed_runs <- sensitivity_runs %>%
    dplyr::filter(.data$run_status == "completed") %>%
    dplyr::count(.data$scenario, .data$scenario_value, name = "completed_runs")
  
  if (nrow(completed_runs) == 0) {
    return(tibble::tibble())
  }
  
  strongest_omega <- function(x) {
    x <- x[is.finite(x)]
    if (length(x) == 0) {
      return(NA_real_)
    }
    x[which.max(abs(x))]
  }
  
  edge_source <- sensitivity_edges
  for (col in c("analysis_mode", "predictor_set")) {
    if (!col %in% names(edge_source)) {
      edge_source[[col]] <- NA_character_
    }
  }
  
  if (identical(edge_focus, "pairwise")) {
    edge_source <- edge_source %>%
      dplyr::filter(dplyr::coalesce(as.character(.data$analysis_mode), "") == "pairwise")
  }
  
  if (nrow(edge_source) == 0) {
    return(tibble::tibble())
  }
  
  detected_edges <- edge_source %>%
    dplyr::filter(
      !is.na(.data$term),
      !is.na(.data$target),
      is.finite(.data$omega)
    ) %>%
    dplyr::mutate(
      term = as.character(.data$term),
      target = as.character(.data$target),
      inferred_sign = dplyr::case_when(
        .data$omega > 0 ~ 1L,
        .data$omega < 0 ~ -1L,
        TRUE ~ 0L
      ),
      regulation_type = dplyr::case_when(
        .data$inferred_sign > 0 ~ "activation",
        .data$inferred_sign < 0 ~ "inhibition",
        TRUE ~ "zero"
      ),
      mode_group = dplyr::coalesce(as.character(.data$analysis_mode), "not_recorded"),
      predictor_set = dplyr::coalesce(as.character(.data$predictor_set), "not_recorded")
    ) %>%
    dplyr::filter(.data$inferred_sign != 0) %>%
    dplyr::group_by(
      .data$sensitivity_id,
      .data$scenario,
      .data$scenario_value,
      .data$condition,
      .data$replicate,
      .data$term,
      .data$target,
      .data$inferred_sign,
      .data$regulation_type
    ) %>%
    dplyr::summarise(
      omega = strongest_omega(.data$omega),
      n_coefficients = dplyr::n(),
      modes = paste(sort(unique(.data$mode_group)), collapse = ", "),
      predictor_sets = paste(sort(unique(.data$predictor_set)), collapse = ", "),
      .groups = "drop"
    )
  
  if (nrow(detected_edges) == 0) {
    return(tibble::tibble())
  }
  
  edge_keys <- detected_edges %>%
    dplyr::distinct(.data$term, .data$target)
  
  scenario_edge_grid <- merge(completed_runs, edge_keys, by = NULL, all = TRUE)
  
  detected_counts <- detected_edges %>%
    dplyr::group_by(.data$scenario, .data$scenario_value, .data$term, .data$target) %>%
    dplyr::summarise(
      activation_detected_runs = dplyr::n_distinct(.data$sensitivity_id[.data$inferred_sign > 0]),
      inhibition_detected_runs = dplyr::n_distinct(.data$sensitivity_id[.data$inferred_sign < 0]),
      activation_omega = strongest_omega(.data$omega[.data$inferred_sign > 0]),
      inhibition_omega = strongest_omega(.data$omega[.data$inferred_sign < 0]),
      total_detected_runs = dplyr::n_distinct(.data$sensitivity_id),
      conditions = paste(sort(unique(.data$condition)), collapse = "; "),
      modes = paste(sort(unique(.data$modes)), collapse = "; "),
      predictor_sets = paste(sort(unique(.data$predictor_sets)), collapse = "; "),
      .groups = "drop"
    )
  
  baseline_keys <- detected_edges %>%
    dplyr::filter(.data$scenario == "baseline") %>%
    dplyr::distinct(.data$term, .data$target) %>%
    dplyr::mutate(in_baseline = TRUE)
  
  scenario_edge_grid %>%
    dplyr::left_join(detected_counts, by = c("scenario", "scenario_value", "term", "target")) %>%
    dplyr::mutate(
      activation_detected_runs = dplyr::coalesce(.data$activation_detected_runs, 0L),
      inhibition_detected_runs = dplyr::coalesce(.data$inhibition_detected_runs, 0L),
      total_detected_runs = dplyr::coalesce(.data$total_detected_runs, 0L),
      conditions = dplyr::coalesce(.data$conditions, ""),
      modes = dplyr::coalesce(.data$modes, ""),
      predictor_sets = dplyr::coalesce(.data$predictor_sets, ""),
      activation_rate = dplyr::if_else(.data$completed_runs > 0, .data$activation_detected_runs / .data$completed_runs, NA_real_),
      inhibition_rate = dplyr::if_else(.data$completed_runs > 0, .data$inhibition_detected_runs / .data$completed_runs, NA_real_),
      detection_rate = pmax(.data$activation_rate, .data$inhibition_rate, na.rm = TRUE),
      cell_status = dplyr::case_when(
        .data$activation_rate > 0 & .data$inhibition_rate > 0 ~ "mixed sign",
        .data$activation_rate > 0 ~ "activation",
        .data$inhibition_rate > 0 ~ "inhibition",
        TRUE ~ "not detected"
      ),
      fill_value = dplyr::case_when(
        .data$cell_status == "activation" ~ .data$activation_rate,
        .data$cell_status == "inhibition" ~ -.data$inhibition_rate,
        .data$cell_status == "mixed sign" ~ NA_real_,
        TRUE ~ 0
      ),
      edge = paste0(
        format_cirn_node_label(.data$term),
        " -> ",
        format_cirn_node_label(.data$target)
      )
    ) %>%
    dplyr::left_join(baseline_keys, by = c("term", "target")) %>%
    dplyr::mutate(in_baseline = dplyr::coalesce(.data$in_baseline, FALSE)) %>%
    dplyr::arrange(
      dplyr::desc(.data$in_baseline),
      .data$edge,
      .data$scenario,
      .data$scenario_value
    )
}

plot_cirn_sensitivity_feature_heatmap <- function(feature_stability_tbl) {
  
  if (is.null(feature_stability_tbl) || nrow(feature_stability_tbl) == 0) {
    return(
      ggplot2::ggplot() +
        ggplot2::annotate(
          "text",
          x = 0,
          y = 0,
          label = "No retained feature-level edges are available for the sensitivity heatmap.",
          size = 4.5,
          color = "#52657a",
          lineheight = 1.08
        ) +
        ggplot2::xlim(-1, 1) +
        ggplot2::ylim(-1, 1) +
        ggplot2::theme_void()
    )
  }
  
  plot_tbl <- feature_stability_tbl %>%
    dplyr::mutate(
      scenario_label = paste(.data$scenario, .data$scenario_value),
      edge = as.character(.data$edge)
    ) %>%
    dplyr::group_by(.data$edge) %>%
    dplyr::mutate(max_detection = max(.data$detection_rate, na.rm = TRUE)) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(edge = stats::reorder(.data$edge, .data$max_detection))
  
  ggplot2::ggplot(
    plot_tbl,
    ggplot2::aes(x = .data$scenario_label, y = .data$edge, fill = .data$fill_value)
  ) +
    ggplot2::geom_tile(color = "white", linewidth = 0.7) +
    ggplot2::scale_fill_gradient2(
      low = "#d62728",
      mid = "#eeeeee",
      high = "#2ca02c",
      midpoint = 0,
      limits = c(-1, 1),
      breaks = c(-1, 0, 1),
      labels = c("Inhibition", "Not detected", "Activation"),
      name = "Signed detection rate",
      na.value = "#9467bd",
      oob = scales::squish
    ) +
    ggplot2::labs(
      x = "Sensitivity scenario",
      y = "Inferred edge",
      caption = "Color intensity is the detection rate. Green = activation, red = inhibition, grey = not detected, purple = mixed sign across retained runs or modes."
    ) +
    theme_cirn_publication(base_size = 12) +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
      legend.position = "bottom",
      plot.caption = ggplot2::element_text(color = "#5e7084", hjust = 0)
    )
}

run_cirn_sensitivity_analysis <- function(df,
                                          time_col,
                                          base_config,
                                          sensitivity_plan = NULL,
                                          sensitivity_inference_scope = NULL,
                                          true_adj = NULL,
                                          output_dir = NULL,
                                          save_outputs = TRUE,
                                          show_progress = FALSE,
                                          progress_bar = FALSE,
                                          progress_callback = NULL) {
  
  if (is.null(sensitivity_inference_scope) &&
      "sensitivity_inference_scope" %in% names(base_config)) {
    sensitivity_inference_scope <- base_config$sensitivity_inference_scope
  }
  if (is.null(sensitivity_inference_scope)) {
    sensitivity_inference_scope <- "use_config"
  }
  
  if (is.null(sensitivity_plan)) {
    sensitivity_plan <- make_cirn_sensitivity_plan(
      noise_sd_fractions = base_config$sensitivity_noise_sd_fractions,
      lag_units = base_config$sensitivity_lag_units,
      downsample_intervals = base_config$sensitivity_downsample_intervals,
      target_sample_sizes = base_config$sensitivity_target_sample_sizes,
      missing_fractions = base_config$sensitivity_missing_fractions,
      replicates = base_config$sensitivity_replicates
    )
  }
  
  required_plan_cols <- c(
    "sensitivity_id",
    "condition",
    "scenario",
    "scenario_value",
    "value",
    "replicate"
  )
  missing_plan_cols <- setdiff(required_plan_cols, names(sensitivity_plan))
  if (length(missing_plan_cols) > 0) {
    stop(
      "sensitivity_plan is missing required columns: ",
      paste(missing_plan_cols, collapse = ", ")
    )
  }
  
  inference_config <- base_config[
    setdiff(names(base_config), cirn_non_inference_config_keys())
  ]
  
  sensitivity_scope <- resolve_cirn_sensitivity_scope(
    scope = sensitivity_inference_scope,
    representation_mode = inference_config$representation_mode,
    run_pairwise = isTRUE(inference_config$run_pairwise),
    pairwise_representation_mode = inference_config$pairwise_representation_mode
  )
  
  inference_config$representation_mode <- sensitivity_scope$representation_mode
  inference_config$run_pairwise <- isTRUE(sensitivity_scope$run_pairwise)
  inference_config$pairwise_representation_mode <- sensitivity_scope$pairwise_representation_mode
  inference_config$show_progress <- show_progress
  inference_config$progress_bar <- progress_bar
  
  run_rows <- list()
  edge_rows <- list()
  diagnostic_rows <- list()
  sample_rows <- list()
  metric_rows <- list()
  total_runs <- nrow(sensitivity_plan)
  
  for (row_index in seq_len(nrow(sensitivity_plan))) {
    
    plan_row <- sensitivity_plan[row_index, , drop = FALSE]
    call_sensitivity_progress <- function(status,
                                          message = NA_character_,
                                          completed = row_index - 1L,
                                          percent_done = NULL,
                                          scenario_n = NA_integer_,
                                          runtime_seconds = NA_real_) {
      if (!is.function(progress_callback)) {
        return(invisible(NULL))
      }
      if (is.null(percent_done)) {
        percent_done <- 100 * completed / max(total_runs, 1L)
      }
      call_cirn_progress(
        progress_callback,
        row_index = row_index,
        total = total_runs,
        completed = completed,
        percent_done = percent_done,
        sensitivity_id = plan_row$sensitivity_id,
        condition = plan_row$condition,
        scenario = plan_row$scenario,
        scenario_value = plan_row$scenario_value,
        replicate = plan_row$replicate,
        status = status,
        message = message,
        scenario_n = scenario_n,
        runtime_seconds = runtime_seconds
      )
      invisible(NULL)
    }
    
    message(
      "Running CIRN sensitivity scenario: ",
      plan_row$condition,
      " (replicate ", plan_row$replicate, ")"
    )
    call_sensitivity_progress("running")
    
    scenario_df <- tryCatch(
      prepare_cirn_sensitivity_dataset(
        df = df,
        time_col = time_col,
        scenario = plan_row$scenario,
        value = plan_row$value,
        replicate = plan_row$replicate,
        seed = inference_config$seed
      ),
      error = function(e) e
    )
    
    if (inherits(scenario_df, "error")) {
      run_rows[[row_index]] <- tibble::tibble(
        sensitivity_id = plan_row$sensitivity_id,
        condition = plan_row$condition,
        scenario = plan_row$scenario,
        scenario_value = plan_row$scenario_value,
        replicate = plan_row$replicate,
        raw_n = nrow(df),
        scenario_n = NA_integer_,
        run_status = "failed_data_preparation",
        message = conditionMessage(scenario_df),
        runtime_seconds = NA_real_
      )
      call_sensitivity_progress(
        "failed_data_preparation",
        message = conditionMessage(scenario_df),
        completed = row_index
      )
      next
    }
    
    scenario_config <- inference_config
    if (plan_row$scenario == "lag_units") {
      scenario_config$lag_units <- as.integer(plan_row$value)
    }
    scenario_config$progress_callback <- function(stage = NA_character_,
                                                  target = NA_character_,
                                                  predictor_set = NA_character_,
                                                  analysis_mode = NA_character_,
                                                  pairwise_predictor = NA_character_,
                                                  completed = NA_integer_,
                                                  total = NA_integer_,
                                                  percent_done = NA_real_,
                                                  ...) {
      inner_percent <- suppressWarnings(as.numeric(percent_done))
      if (!is.finite(inner_percent)) {
        inner_percent <- 0
      }
      inner_percent <- max(0, min(100, inner_percent))
      
      value_or_unknown <- function(value) {
        if (length(value) == 0 || is.null(value) || is.na(value)) {
          return("?")
        }
        as.character(value)
      }
      
      pairwise_text <- if (!is.na(pairwise_predictor)) {
        paste0("; pairwise=", pairwise_predictor)
      } else {
        ""
      }
      
      call_sensitivity_progress(
        paste0("running_", value_or_unknown(stage)),
        message = paste0(
          "CIRN fit ",
          value_or_unknown(completed),
          "/",
          value_or_unknown(total),
          "; mode=",
          value_or_unknown(analysis_mode),
          pairwise_text,
          "; target=d",
          value_or_unknown(target),
          "; representation=",
          value_or_unknown(predictor_set)
        ),
        completed = row_index - 1L,
        percent_done = 100 * ((row_index - 1L) + inner_percent / 100) / max(total_runs, 1L),
        scenario_n = nrow(scenario_df)
      )
    }
    
    timing <- system.time(
      fit <- tryCatch(
        do.call(
          infer_network,
          c(
            list(
              df = scenario_df,
              time_col = time_col
            ),
            scenario_config
          )
        ),
        error = function(e) e
      )
    )
    
    if (inherits(fit, "error")) {
      run_rows[[row_index]] <- tibble::tibble(
        sensitivity_id = plan_row$sensitivity_id,
        condition = plan_row$condition,
        scenario = plan_row$scenario,
        scenario_value = plan_row$scenario_value,
        replicate = plan_row$replicate,
        raw_n = nrow(df),
        scenario_n = nrow(scenario_df),
        run_status = "failed_inference",
        message = conditionMessage(fit),
        runtime_seconds = as.numeric(timing["elapsed"])
      )
      call_sensitivity_progress(
        "failed_inference",
        message = conditionMessage(fit),
        completed = row_index,
        scenario_n = nrow(scenario_df),
        runtime_seconds = as.numeric(timing["elapsed"])
      )
      next
    }
    
    run_rows[[row_index]] <- tibble::tibble(
      sensitivity_id = plan_row$sensitivity_id,
      condition = plan_row$condition,
      scenario = plan_row$scenario,
      scenario_value = plan_row$scenario_value,
      replicate = plan_row$replicate,
      raw_n = nrow(df),
      scenario_n = nrow(scenario_df),
      run_status = "completed",
      message = NA_character_,
      runtime_seconds = as.numeric(timing["elapsed"])
    )
    call_sensitivity_progress(
      "completed",
      completed = row_index,
      scenario_n = nrow(scenario_df),
      runtime_seconds = as.numeric(timing["elapsed"])
    )
    
    edge_rows[[row_index]] <- fit$edges_combined %>%
      dplyr::mutate(
        sensitivity_id = plan_row$sensitivity_id,
        condition = plan_row$condition,
        scenario = plan_row$scenario,
        scenario_value = plan_row$scenario_value,
        replicate = plan_row$replicate,
        .before = 1
      )
    
    diagnostic_rows[[row_index]] <- fit$diagnostics %>%
      dplyr::mutate(
        sensitivity_id = plan_row$sensitivity_id,
        condition = plan_row$condition,
        scenario = plan_row$scenario,
        scenario_value = plan_row$scenario_value,
        replicate = plan_row$replicate,
        .before = 1
      )
    
    sample_rows[[row_index]] <- summarize_cirn_effective_sample_size(fit) %>%
      dplyr::mutate(
        sensitivity_id = plan_row$sensitivity_id,
        condition = plan_row$condition,
        scenario = plan_row$scenario,
        scenario_value = plan_row$scenario_value,
        replicate = plan_row$replicate,
        .before = 1
      )
    
    if (!is.null(true_adj)) {
      metric <- tryCatch(
        evaluate_representation_agnostic(true_adj, fit$edges),
        error = function(e) e
      )
      
      if (!inherits(metric, "error")) {
        metric_rows[[row_index]] <- tibble::as_tibble(metric) %>%
          dplyr::mutate(
            sensitivity_id = plan_row$sensitivity_id,
            condition = plan_row$condition,
            scenario = plan_row$scenario,
            scenario_value = plan_row$scenario_value,
            replicate = plan_row$replicate,
            .before = 1
          )
      }
    }
  }
  
  sensitivity_runs <- dplyr::bind_rows(run_rows)
  sensitivity_edges <- dplyr::bind_rows(edge_rows)
  sensitivity_diagnostics <- dplyr::bind_rows(diagnostic_rows)
  sensitivity_sample_size <- dplyr::bind_rows(sample_rows)
  sensitivity_metrics <- dplyr::bind_rows(metric_rows)
  sensitivity_edge_stability <- summarize_sensitivity_edge_stability(
    sensitivity_edges,
    sensitivity_runs
  )
  sensitivity_feature_edge_stability <- summarize_cirn_sensitivity_feature_edge_stability(
    sensitivity_edges,
    sensitivity_runs,
    edge_focus = sensitivity_scope$edge_focus
  )
  
  if (isTRUE(save_outputs) && !is.null(output_dir)) {
    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    }
    
    utils::write.csv(
      sensitivity_runs,
      file.path(output_dir, "CIRN_sensitivity_runs.csv"),
      row.names = FALSE
    )
    utils::write.csv(
      sensitivity_edges,
      file.path(output_dir, "CIRN_sensitivity_edges.csv"),
      row.names = FALSE
    )
    utils::write.csv(
      sensitivity_sample_size,
      file.path(output_dir, "CIRN_sensitivity_effective_sample_size.csv"),
      row.names = FALSE
    )
    utils::write.csv(
      sensitivity_edge_stability,
      file.path(output_dir, "CIRN_sensitivity_edge_stability.csv"),
      row.names = FALSE
    )
    utils::write.csv(
      sensitivity_feature_edge_stability,
      file.path(output_dir, "CIRN_sensitivity_feature_edge_stability.csv"),
      row.names = FALSE
    )
    
    if (nrow(sensitivity_metrics) > 0) {
      utils::write.csv(
        sensitivity_metrics,
        file.path(output_dir, "CIRN_sensitivity_ground_truth_metrics.csv"),
        row.names = FALSE
      )
    }
  }
  
  list(
    plan = sensitivity_plan,
    runs = sensitivity_runs,
    edges = sensitivity_edges,
    diagnostics = sensitivity_diagnostics,
    effective_sample_size = sensitivity_sample_size,
    ground_truth_metrics = sensitivity_metrics,
    edge_stability = sensitivity_edge_stability,
    feature_edge_stability = sensitivity_feature_edge_stability,
    sensitivity_scope = sensitivity_scope
  )
}


################################################################################
################################################################################


# ================================
# 7. Visualization
# ================================

prepare_plot_vars <- function(df, vars, time_col) {
  vars <- unique(as.character(unlist(vars)))
  vars <- vars[nzchar(vars)]
  vars <- setdiff(vars, time_col)
  
  if (length(vars) == 0) {
    stop("No variables were provided for plotting.")
  }
  
  missing_vars <- setdiff(vars, names(df))
  if (length(missing_vars) > 0) {
    stop(
      "Variables not found in data: ",
      paste(missing_vars, collapse = ", ")
    )
  }
  
  vars
}

make_variable_palette <- function(vars) {
  vars <- unique(as.character(vars))
  setNames(publication_palette(length(vars)), vars)
}

publication_palette <- function(n) {
  base_colors <- c(
    "#0072B2", # blue
    "#D55E00", # vermillion
    "#009E73", # green
    "#CC79A7", # reddish purple
    "#E69F00", # orange
    "#56B4E9", # sky blue
    "#332288", # indigo
    "#882255"  # wine
  )
  
  if (n <= length(base_colors)) {
    return(base_colors[seq_len(n)])
  }
  
  grDevices::colorRampPalette(base_colors)(n)
}

theme_cirn_publication <- function(base_size = 12) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      legend.key       = element_rect(fill = "white", color = NA),
      strip.background = element_rect(fill = "grey92", color = "grey40")
    )
}

# --------------------------------------------------------------
# 7.1 Raw state variables (before smoothing)
# --------------------------------------------------------------

# Purpose:
#   • Visualize original observed data prior to smoothing
#   • Identify noise, outliers, and temporal irregularities
#   • Provide baseline reference for derivative-based analyses
#
# All variables are plotted:
#   • In a single panel
#   • On a shared time axis
#   • On a shared value axis
#
# This plot is diagnostic only (appendix-level).

plot_raw_data_all_vars <- function(df, vars, time_col) {
  
  vars <- prepare_plot_vars(df, vars, time_col)
  variable_palette <- make_variable_palette(vars)
  
  plot_df <- df %>%
    dplyr::select(all_of(c(time_col, vars))) %>%
    tidyr::pivot_longer(
      cols = all_of(vars),
      names_to  = "variable",
      values_to = "value"
    ) %>%
    dplyr::mutate(
      variable = factor(variable, levels = vars)
    )
  
  ggplot(
    plot_df,
    aes(x = .data[[time_col]], y = value, color = variable)
  ) +
    geom_line(alpha = 1.0, linewidth = 1.0, na.rm = TRUE) +
    scale_color_manual(
      values = variable_palette
    ) +
    labs(
      x = "Time",
      y = "Observed value",
      title = "Raw state variables (unsmoothed)",
      color = "Variable"
    ) +
    theme_cirn_publication(base_size = 12) +
    theme(
      legend.position = "bottom"
    ) +
    guides(
      colour = guide_legend(
        override.aes = list(linewidth = 1.5, alpha = 1)
      )
    )
}

# --------------------------------------------------------------
# 7.2 State and derivatives across all variables (shared panels)
# --------------------------------------------------------------

# Purpose:
#   • Compare dynamics across variables within each derivative order
#   • Assess relative magnitude and stability of derivatives
#   • Ensure consistent scaling across system components
#
# Layout:
#   • Panel 1: All state variables
#   • Panel 2: All first derivatives (smoothed)
#   • Panel 3: All second derivatives (smoothed)
#
# All panels share the same x- and y-axes.

plot_all_vars_by_derivative_order <- function(df,
                                              vars,
                                              time_col,
                                              points_per_interval = 1,
                                              spar = NULL,
                                              outlier_method = "MAD",
                                              outlier_thresh = 3.5,
                                              outlier_action = "winsorize") {
  
  vars <- prepare_plot_vars(df, vars, time_col)
  variable_palette <- make_variable_palette(vars)
  
  # Compute derivatives using the same preprocessing choices used
  # in the CIRN inference configuration.
  
  df_d <- compute_derivatives(
    df,
    time_col,
    points_per_interval = points_per_interval,
    spar = spar,
    outlier_method = outlier_method,
    outlier_thresh = outlier_thresh,
    outlier_action = outlier_action
  )
  
  plot_df <- lapply(vars, function(v) {
    
    tibble(
      time = df_d[[time_col]],
      variable = v,
      
      # RAW state variable interpolated to derivative time grid
      
      state = approx(
        x = df[[time_col]],
        y = df[[v]],
        xout = df_d[[time_col]]
      )$y,
      
      d1 = df_d[[paste0(v, "_d1")]],
      d2 = df_d[[paste0(v, "_d2")]]
    )
    
  }) %>%
    dplyr::bind_rows() %>%
    tidyr::pivot_longer(
      cols = c(state, d1, d2),
      names_to  = "quantity",
      values_to = "value"
    ) %>%
    dplyr::mutate(
      quantity = factor(
        quantity,
        levels = c("state","d1","d2"),
        labels = c(
          "Original State Variables",
          "First Derivatives",
          "Second Derivatives"
        )
      )
    ) %>%
    dplyr::mutate(
      variable = factor(variable, levels = vars)
    )
  
  ggplot(plot_df, aes(x = time, y = value, color = variable)) +
    geom_line(alpha = 1.0, linewidth = 1.0, na.rm = TRUE) +
    scale_color_manual(
      values = variable_palette
    ) +
    facet_wrap(~ quantity, ncol = 1, scales = "free_y") +
    labs(
      x = "Time",
      y = "Value",
      title = "Original State Variables and Temporal Derivatives",
      color = "Variable"
    ) +
    theme_cirn_publication(base_size = 12) +
    theme(legend.position = "bottom") +
    guides(
      colour = guide_legend(
        override.aes = list(linewidth = 1.5, alpha = 1)
      )
    )
}

# --------------------------------------------------------------
# 7.3 State variable and temporal derivatives (single variable)
# --------------------------------------------------------------

# Purpose:
#   • Visual sanity check of the raw system dynamics
#   • Assess smoothness and numerical stability of derivative estimates
#   • Examine how temporal change relates to inferred regulation
#
# This plot displays, on a shared time axis:
#   1) The original state variable x(t)
#   2) Its first temporal derivative dx/dt
#   3) Its second temporal derivative d²x/dt²
#
# Each quantity is shown in its own panel with its own y-scale,
# but all panels are aligned in time to facilitate interpretation.

plot_derivative_example <- function(df,
                                    var,
                                    time_col,
                                    points_per_interval = 1,
                                    spar = NULL,
                                    outlier_method = "MAD",
                                    outlier_thresh = 3.5,
                                    outlier_action = "winsorize") {
  
  var <- prepare_plot_vars(df, var, time_col)[1]
  
  # Compute derivatives using the same preprocessing choices used
  # in the CIRN inference configuration.
  
  df_d <- compute_derivatives(
    df,
    time_col,
    points_per_interval = points_per_interval,
    spar = spar,
    outlier_method = outlier_method,
    outlier_thresh = outlier_thresh,
    outlier_action = outlier_action
  )
  
  state_label <- paste0(var, "(t)")
  d1_label    <- paste0("d", var, "(t)/dt")
  d2_label    <- paste0("d²", var, "(t)/dt²")
  
  plot_df <- tibble(
    time = df_d[[time_col]],
    
    # ORIGINAL state variable interpolated to derivative grid
    
    state = approx(
      x = df[[time_col]],
      y = df[[var]],
      xout = df_d[[time_col]]
    )$y,
    
    d1 = df_d[[paste0(var, "_d1")]],
    d2 = df_d[[paste0(var, "_d2")]]
    
  ) %>%
    tidyr::pivot_longer(
      cols = c(state, d1, d2),
      names_to = "quantity",
      values_to = "value"
    ) %>%
    dplyr::mutate(
      quantity = factor(
        quantity,
        levels = c("state", "d1", "d2"),
        labels = c(state_label, d1_label, d2_label)
      )
    )
  
  quantity_palette <- setNames(
    c("#111111", "#0072B2", "#D55E00"),
    c(state_label, d1_label, d2_label)
  )
  
  ggplot(plot_df, aes(x = time, y = value, color = quantity)) +
    geom_line(linewidth = 1.0, na.rm = TRUE) +
    facet_wrap(~ quantity, ncol = 1, scales = "free_y") +
    scale_color_manual(
      values = quantity_palette,
      guide = "none"
    ) +
    labs(
      x = "Time",
      y = NULL,
      title = paste("Original state variable and temporal derivatives of", var)
    ) +
    theme_cirn_publication(base_size = 12)
}


# --------------------------------------------------------------
# 7.4 Adaptive minimal-jitter diagnostic
# --------------------------------------------------------------

# Purpose:
#   • Visualize the target-specific perturbation used only for
#     response construction when adaptive minimal jitter is triggered.
#   • Confirm that the jitter is small relative to the target trajectory.
#   • Show how the minimally jittered derivative creates a valid
#     two-class directional response.
#
# IMPORTANT:
#   • This plot is a response-construction diagnostic.
#   • It is not a new observed or simulated system trajectory.
#   • Predictors are not jittered by CIRN.

plot_adaptive_jitter_diagnostic <- function(df,
                                            target,
                                            time_col,
                                            points_per_interval = 1,
                                            spar = NULL,
                                            outlier_method = "MAD",
                                            outlier_thresh = 3.5,
                                            outlier_action = "winsorize",
                                            response_eps = 1e-6,
                                            jitter_min_class_count = 5,
                                            jitter_scale_grid = c(
                                              1e-8, 3e-8,
                                              1e-7, 3e-7,
                                              1e-6, 3e-6,
                                              1e-5, 3e-5,
                                              1e-4, 3e-4,
                                              1e-3, 3e-3,
                                              1e-2
                                            ),
                                            jitter_scale_basis = "state_sd",
                                            seed = 123) {
  
  target <- prepare_plot_vars(df, target, time_col)[1]
  target_d1 <- paste0(target, "_d1")
  jittered_state_col <- paste0(target, "_jittered")
  jittered_d1_col <- paste0(target, "_d1_jittered")
  
  df_d <- compute_derivatives(
    df,
    time_col,
    points_per_interval = points_per_interval,
    spar = spar,
    outlier_method = outlier_method,
    outlier_thresh = outlier_thresh,
    outlier_action = outlier_action
  )
  
  raw_response <- classify_directional_response(
    df = df_d,
    derivative_col = target_d1,
    response_eps = response_eps
  )
  raw_counts <- response_class_counts(raw_response$Class)
  
  jitter_result <- adaptive_jitter_directional_response(
    df_source = df_d,
    target = target,
    time_col = time_col,
    target_d1 = target_d1,
    response_eps = response_eps,
    min_class_count = jitter_min_class_count,
    jitter_scale_grid = jitter_scale_grid,
    jitter_scale_basis = jitter_scale_basis,
    seed = seed
  )
  
  if (!isTRUE(jitter_result$success) ||
      is.null(jitter_result$data_full) ||
      !all(c(jittered_state_col, jittered_d1_col) %in% names(jitter_result$data_full))) {
    return(
      ggplot() +
        annotate(
          "text",
          x = 0.5,
          y = 0.5,
          label = paste("Adaptive jitter was not available for target", target),
          size = 5
        ) +
        theme_void()
    )
  }
  
  jitter_full <- jitter_result$data_full
  jitter_response <- jitter_result$data
  
  quantity_levels <- c(
    "Target trajectory",
    "Added jitter",
    "First derivative"
  )
  
  plot_df <- dplyr::bind_rows(
    tibble::tibble(
      time = jitter_full[[time_col]],
      quantity = "Target trajectory",
      series = "Smoothed target",
      value = jitter_full[[target]]
    ),
    tibble::tibble(
      time = jitter_full[[time_col]],
      quantity = "Target trajectory",
      series = "Minimally jittered target",
      value = jitter_full[[jittered_state_col]]
    ),
    tibble::tibble(
      time = jitter_full[[time_col]],
      quantity = "Added jitter",
      series = "Added jitter",
      value = jitter_full[[jittered_state_col]] - jitter_full[[target]]
    ),
    tibble::tibble(
      time = jitter_full[[time_col]],
      quantity = "First derivative",
      series = "Raw derivative",
      value = jitter_full[[target_d1]]
    ),
    tibble::tibble(
      time = jitter_full[[time_col]],
      quantity = "First derivative",
      series = "Jittered derivative",
      value = jitter_full[[jittered_d1_col]]
    )
  ) %>%
    dplyr::mutate(
      quantity = factor(.data$quantity, levels = quantity_levels),
      series = factor(
        .data$series,
        levels = c(
          "Smoothed target",
          "Minimally jittered target",
          "Added jitter",
          "Raw derivative",
          "Jittered derivative"
        )
      )
    )
  
  class_df <- jitter_response %>%
    dplyr::transmute(
      time = .data[[time_col]],
      value = .data[[jittered_d1_col]],
      Class = factor(
        .data$Class,
        levels = c(0, 1),
        labels = c("Decreasing class", "Increasing class")
      ),
      quantity = factor("First derivative", levels = quantity_levels)
    )
  
  hline_df <- tibble::tibble(
    quantity = factor("First derivative", levels = quantity_levels),
    yintercept = c(-response_eps, response_eps)
  )
  
  subtitle_text <- paste0(
    "Jitter scale factor = ", signif(jitter_result$jitter_scale_factor, 4),
    "; jitter SD = ", signif(jitter_result$jitter_scale, 4),
    "; basis = ", jitter_result$jitter_basis,
    "; raw classes: 0 = ", raw_counts$class_0_count,
    ", 1 = ", raw_counts$class_1_count,
    "; jittered classes: 0 = ", jitter_result$class_0_count,
    ", 1 = ", jitter_result$class_1_count
  )
  
  ggplot(plot_df, aes(x = time, y = value, color = series, linetype = series)) +
    geom_line(linewidth = 0.9, na.rm = TRUE) +
    geom_hline(
      data = hline_df,
      aes(yintercept = yintercept),
      inherit.aes = FALSE,
      linetype = "dashed",
      color = "#D55E00",
      linewidth = 0.4
    ) +
    geom_point(
      data = class_df,
      aes(x = time, y = value, fill = Class),
      inherit.aes = FALSE,
      shape = 21,
      size = 1.7,
      color = "white",
      stroke = 0.2,
      alpha = 0.9,
      na.rm = TRUE
    ) +
    facet_wrap(~ quantity, ncol = 1, scales = "free_y") +
    scale_color_manual(
      values = c(
        "Smoothed target" = "#111111",
        "Minimally jittered target" = "#D62728",
        "Added jitter" = "#D55E00",
        "Raw derivative" = "#555555",
        "Jittered derivative" = "#009E73"
      ),
      drop = FALSE
    ) +
    scale_linetype_manual(
      values = c(
        "Smoothed target" = "solid",
        "Minimally jittered target" = "dashed",
        "Added jitter" = "solid",
        "Raw derivative" = "solid",
        "Jittered derivative" = "dashed"
      ),
      drop = FALSE
    ) +
    scale_fill_manual(
      values = c(
        "Decreasing class" = "#D55E00",
        "Increasing class" = "#009E73"
      ),
      drop = FALSE
    ) +
    labs(
      x = "Time",
      y = NULL,
      title = paste("Adaptive minimal-jitter diagnostic for target", target),
      subtitle = subtitle_text,
      color = NULL,
      linetype = NULL,
      fill = "Jittered derivative class"
    ) +
    theme_cirn_publication(base_size = 12) +
    theme(
      legend.position = "bottom",
      plot.subtitle = element_text(size = 9)
    )
}


# --------------------------------------------------------------
# 7.5 State and derivative trajectories for all variables
# --------------------------------------------------------------

# Purpose:
#   • Exploratory visualization across all system components
#   • Identify variables with unstable derivatives or noisy dynamics
#   • Intended for diagnostic and quality-control purposes only
#     (not for main-text figures)

plot_all_derivatives <- function(df,
                                 vars,
                                 time_col,
                                 points_per_interval = 1,
                                 spar = NULL,
                                 outlier_method = "MAD",
                                 outlier_thresh = 3.5,
                                 outlier_action = "winsorize") {
  vars <- prepare_plot_vars(df, vars, time_col)
  for (v in vars) {
    print(
      plot_derivative_example(
        df = df,
        var = v,
        time_col = time_col,
        points_per_interval = points_per_interval,
        spar = spar,
        outlier_method = outlier_method,
        outlier_thresh = outlier_thresh,
        outlier_action = outlier_action
      )
    )
  }
}

format_cirn_node_label <- function(node_id) {
  label <- as.character(node_id)
  label <- ifelse(
    grepl("_d1$", label),
    paste0("d", sub("_d1$", "", label)),
    label
  )
  label <- ifelse(
    grepl("_d2$", label),
    paste0("d²", sub("_d2$", "", label)),
    label
  )
  label
}

rescale_cirn_edge_width <- function(omega, to = c(2, 7)) {
  omega_abs <- abs(omega)
  
  if (length(omega_abs) == 0) {
    return(numeric(0))
  }
  
  if (all(is.na(omega_abs))) {
    return(rep(mean(to), length(omega_abs)))
  }
  
  omega_range <- range(omega_abs, na.rm = TRUE)
  
  if (!is.finite(diff(omega_range)) || diff(omega_range) == 0) {
    return(rep(mean(to), length(omega_abs)))
  }
  
  scales::rescale(omega_abs, to = to)
}

cirn_panel_titles <- function() {
  c(
    original = "Original State Predictors",
    first_derivative = "First-Derivative Predictors",
    second_derivative = "Second-Derivative Predictors",
    all_predictors = "All Predictors (Joint Model)"
  )
}

cirn_predictor_sets_for_mode <- function(edges = NULL,
                                         representation_mode = c("sublevel",
                                                                 "all_predictors",
                                                                 "both"),
                                         predictor_sets = NULL) {
  
  panel_titles <- cirn_panel_titles()
  
  if (!is.null(predictor_sets)) {
    return(intersect(predictor_sets, names(panel_titles)))
  }
  
  representation_mode <- match.arg(representation_mode)
  
  requested_sets <- switch(
    representation_mode,
    sublevel = c("original", "first_derivative", "second_derivative"),
    all_predictors = "all_predictors",
    both = c("original", "first_derivative", "second_derivative", "all_predictors")
  )
  
  present_sets <- character(0)
  if (!is.null(edges) && "predictor_set" %in% names(edges)) {
    present_sets <- unique(as.character(edges$predictor_set))
  }
  
  # Always show the mode-requested panels, even when a representation
  # has no retained posterior-supported edges. Extra fitted sets are
  # appended so older result objects still display completely.
  
  unique(c(requested_sets, setdiff(present_sets, requested_sets)))
}

prepare_cirn_edges_for_plot <- function(inferred_edges,
                                        aggregate_duplicate_edges = TRUE) {
  
  if (is.null(inferred_edges) || nrow(inferred_edges) == 0) {
    return(tibble::tibble())
  }
  
  plot_edges <- inferred_edges %>%
    dplyr::mutate(
      predictor_set = as.character(.data$predictor_set),
      omega_abs = abs(.data$omega)
    )
  
  if (!"response_mode" %in% names(plot_edges)) {
    plot_edges$response_mode <- NA_character_
  }
  if (!"response_label" %in% names(plot_edges)) {
    plot_edges$response_label <- NA_character_
  }
  if (!"response_source" %in% names(plot_edges)) {
    plot_edges$response_source <- NA_character_
  }
  if (!"dominant_direction" %in% names(plot_edges)) {
    plot_edges$dominant_direction <- NA_real_
  }
  if (!"jitter_used" %in% names(plot_edges)) {
    plot_edges$jitter_used <- FALSE
  }
  if (!"jitter_scale" %in% names(plot_edges)) {
    plot_edges$jitter_scale <- NA_real_
  }
  if (!"jitter_scale_factor" %in% names(plot_edges)) {
    plot_edges$jitter_scale_factor <- NA_real_
  }
  if (!"response_trigger" %in% names(plot_edges)) {
    plot_edges$response_trigger <- NA_character_
  }
  if (!"effect_interpretation" %in% names(plot_edges)) {
    plot_edges$effect_interpretation <- NA_character_
  }
  
  if (!aggregate_duplicate_edges) {
    return(plot_edges)
  }
  
  plot_edges %>%
    dplyr::group_by(.data$term, .data$target, .data$regulation_type) %>%
    dplyr::arrange(dplyr::desc(.data$omega_abs), .by_group = TRUE) %>%
    dplyr::summarise(
      omega = dplyr::first(.data$omega),
      odds_ratio = dplyr::first(.data$odds_ratio),
      sign = dplyr::first(.data$sign),
      rel_strength = paste(unique(.data$rel_strength), collapse = ", "),
      hdi_lower95 = dplyr::first(.data$hdi_lower95),
      hdi_upper95 = dplyr::first(.data$hdi_upper95),
      eti_lower95 = dplyr::first(.data$eti_lower95),
      eti_upper95 = dplyr::first(.data$eti_upper95),
      method = paste(unique(.data$method), collapse = ", "),
      predictor_set = paste(unique(.data$predictor_set), collapse = ", "),
      response_mode = paste(unique(.data$response_mode), collapse = ", "),
      response_label = paste(unique(.data$response_label), collapse = ", "),
      response_source = paste(unique(.data$response_source), collapse = ", "),
      dominant_direction = dplyr::first(.data$dominant_direction),
      jitter_used = any(.data$jitter_used, na.rm = TRUE),
      jitter_scale = dplyr::first(.data$jitter_scale),
      jitter_scale_factor = dplyr::first(.data$jitter_scale_factor),
      response_trigger = paste(unique(.data$response_trigger), collapse = ", "),
      effect_interpretation = paste(unique(.data$effect_interpretation), collapse = ", "),
      support_n = dplyr::n(),
      .groups = "drop"
    )
}

make_cirn_node_table <- function(node_ids, target_ids = character()) {
  
  node_ids <- unique(as.character(node_ids))
  
  node_type <- dplyr::case_when(
    node_ids %in% target_ids ~ "Target derivative response",
    grepl("_d1$", node_ids) ~ "First-derivative predictor",
    grepl("_d2$", node_ids) ~ "Second-derivative predictor",
    TRUE ~ "Original-state predictor"
  )
  
  node_fill <- dplyr::case_when(
    node_type == "Original-state predictor" ~ "#fff4e6",
    node_type == "First-derivative predictor" ~ "#e8f3ff",
    node_type == "Second-derivative predictor" ~ "#ffe8f6",
    TRUE ~ "#ffffff"
  )
  
  node_border <- dplyr::case_when(
    node_type == "Original-state predictor" ~ "#f28e2b",
    node_type == "First-derivative predictor" ~ "#1f77b4",
    node_type == "Second-derivative predictor" ~ "#cc2e9b",
    TRUE ~ "#111111"
  )
  
  data.frame(
    id = node_ids,
    label = format_cirn_node_label(node_ids),
    title = paste0(format_cirn_node_label(node_ids), "<br>", node_type),
    shape = "circle",
    size = ifelse(node_ids %in% target_ids, 42, 38),
    color.background = node_fill,
    color.border = node_border,
    borderWidth = ifelse(node_ids %in% target_ids, 3, 2),
    stringsAsFactors = FALSE
  )
}

make_cirn_edge_table <- function(plot_edges, width_range = c(2, 7)) {
  
  if (is.null(plot_edges) || nrow(plot_edges) == 0) {
    return(data.frame())
  }
  
  data.frame(
    from = plot_edges$term,
    to = plot_edges$target,
    color = ifelse(
      plot_edges$regulation_type == "activation",
      "#2ca02c",
      "#d62728"
    ),
    width = rescale_cirn_edge_width(plot_edges$omega, to = width_range),
    arrows = "to",
    smooth = TRUE,
    title = paste0(
      "<b>", format_cirn_node_label(plot_edges$term), " -> ",
      format_cirn_node_label(plot_edges$target), "</b><br>",
      "Representation: ", plot_edges$predictor_set, "<br>",
      "Response mode: ", ifelse(
        "response_mode" %in% names(plot_edges),
        plot_edges$response_mode,
        "not recorded"
      ), "<br>",
      "Response source: ", ifelse(
        "response_source" %in% names(plot_edges),
        plot_edges$response_source,
        "not recorded"
      ), "<br>",
      "Adaptive jitter used: ", ifelse(
        "jitter_used" %in% names(plot_edges),
        plot_edges$jitter_used,
        "not recorded"
      ), "<br>",
      "Jitter scale factor: ", ifelse(
        "jitter_scale_factor" %in% names(plot_edges),
        signif(plot_edges$jitter_scale_factor, 4),
        "not recorded"
      ), "<br>",
      "Effect interpretation: ", ifelse(
        "effect_interpretation" %in% names(plot_edges),
        plot_edges$effect_interpretation,
        "not recorded"
      ), "<br>",
      "Regulation: ", plot_edges$regulation_type, "<br>",
      "omega = ", round(plot_edges$omega, 3), "<br>",
      "95% HDI = [", round(plot_edges$hdi_lower95, 3), ", ",
      round(plot_edges$hdi_upper95, 3), "]"
    ),
    stringsAsFactors = FALSE
  )
}

cirn_network_legend <- function(position = c("side", "bottom")) {
  
  position <- match.arg(position)
  
  legend_items <- list(
    div(style = "color:#2ca02c; font-weight:700;", "Activation (+)"),
    div(style = "color:#d62728; font-weight:700;", "Inhibition (-)"),
    div("Edge width proportional to |omega|"),
    div(style = "color:#f28e2b; font-weight:700;", "State predictor"),
    div(style = "color:#1f77b4; font-weight:700;", "First derivative"),
    div(style = "color:#cc2e9b; font-weight:700;", "Second derivative"),
    div(style = "font-weight:700;", "Black border = target response")
  )
  
  if (position == "side") {
    return(
      div(
        style = paste(
          "width:250px; flex:0 0 250px; align-self:stretch;",
          "border-left:1px solid #dddddd; padding:16px 18px;",
          "font-size:15px; line-height:1.45; background:#ffffff;"
        ),
        h3(style = "margin:0 0 12px 0; text-align:left;", "Legend"),
        div(style = "display:flex; flex-direction:column; gap:12px;", legend_items)
      )
    )
  }
  
  div(
    style = paste(
      "display:flex; flex-wrap:wrap; gap:22px; align-items:center;",
      "justify-content:center; margin-top:12px; font-size:15px;"
    ),
    legend_items
  )
}

# --------------------------------------------------
# 7.5 PUBLICATION-GRADE CIRN NETWORK OUTPUT 1
# --------------------------------------------------

# Visual encoding:
#   • Output 1 is the Full CIRN network:
#       posterior-supported edges from the CIRN sublevels only:
#       state, first-derivative, and second-derivative predictors.
#   • The all-predictors and pairwise fits are retained for consistency
#     checks, sensitivity interpretation, and edge-support summaries, but
#     they are not merged into the primary Full CIRN network graph.
#   • Nodes retain their predictor representation:
#       state, first derivative, second derivative, or target derivative.
#   • Edge color encodes activation/inhibition.
#   • Edge width encodes |omega_ij|, the posterior mean magnitude.
#
# This figure is intended for MAIN TEXT presentation
# and summarizes the final CIRN-inferred regulatory structure.
# --------------------------------------------------

plot_network_main <- function(inferred_edges,
                              representation_mode = c("sublevel",
                                                      "all_predictors",
                                                      "both"),
                              title = "Output 1: Full CIRN",
                              subtitle = "Posterior-supported network across CIRN state, first-derivative, and second-derivative sublevels",
                              aggregate_duplicate_edges = TRUE,
                              height = "680px",
                              legend_position = c("side", "bottom")) {
  
  representation_mode <- match.arg(representation_mode)
  legend_position <- match.arg(legend_position)
  sublevel_predictor_sets <- c("original", "first_derivative", "second_derivative")
  
  if (!is.null(inferred_edges) && nrow(inferred_edges) > 0) {
    inferred_edges <- inferred_edges %>%
      dplyr::filter(.data$predictor_set %in% sublevel_predictor_sets)
  }
  
  if (is.null(inferred_edges) || nrow(inferred_edges) == 0) {
    return(
      htmltools::browsable(
        div(
          style = "font-family:Arial, sans-serif; padding:20px;",
          h2(title),
          p(subtitle),
          div(
            style = paste(
              "border:1px solid #cccccc; padding:28px;",
              "text-align:center; color:#666666;"
            ),
            paste(
              "No posterior-supported sublevel CIRN edges were retained by",
              "the HDI rule. Pairwise and all-predictors fits, if enabled,",
              "are reported only in consistency and diagnostic outputs."
            )
          )
        )
      )
    )
  }
  
  plot_edges <- prepare_cirn_edges_for_plot(
    inferred_edges,
    aggregate_duplicate_edges = aggregate_duplicate_edges
  )
  
  node_ids <- unique(c(plot_edges$term, plot_edges$target))
  target_ids <- unique(plot_edges$target)
  
  nodes <- make_cirn_node_table(node_ids, target_ids = target_ids)
  edges <- make_cirn_edge_table(plot_edges)
  
  edge_summary <- paste0(
    "Mode: ", representation_mode,
    " | retained sublevel edges: ", nrow(inferred_edges),
    " | plotted unique edges: ", nrow(plot_edges)
  )
  
  network <- visNetwork(nodes, edges, width = "100%", height = height) %>%
    
    visEdges(arrows = "to") %>%
    
    visNodes(
      borderWidth = 2,
      font = list(
        color = "navy",
        size = 20,
        align = "center"
      )
    ) %>%
    
    visPhysics(
      solver = "forceAtlas2Based",
      stabilization = TRUE,
      forceAtlas2Based = list(
        gravitationalConstant = -80,
        springLength = 180
      )
    ) %>%
    
    visInteraction(
      dragNodes = TRUE,
      dragView = TRUE,
      zoomView = TRUE
    ) %>%
    visOptions(
      highlightNearest = list(enabled = TRUE, degree = 1, hover = TRUE),
      nodesIdSelection = TRUE
    ) %>%
    visLayout(randomSeed = 123)
  
  htmltools::browsable(
    div(
      style = paste(
        "font-family:Arial, sans-serif;",
        "max-width:1600px; margin:0 auto; padding:10px 16px;"
      ),
      h2(style = "text-align:center; margin-bottom:4px;", title),
      p(style = "text-align:center; margin-top:0; color:#555555;", subtitle),
      p(style = "text-align:center; margin-top:0; color:#777777;", edge_summary),
      div(
        style = paste(
          "display:flex; gap:16px; align-items:stretch;",
          "justify-content:center; width:100%;"
        ),
        div(style = "flex:1 1 auto; min-width:0;", network),
        if (legend_position == "side") cirn_network_legend("side")
      ),
      if (legend_position == "bottom") cirn_network_legend("bottom")
    )
  )
}

# --------------------------------------------------
# 7.6 PUBLICATION-GRADE CIRN NETWORK OUTPUT 2
# --------------------------------------------------

# Purpose:
#   • Output 2 is the Individual CIRN view:
#       one panel per fitted representation.
#   • In "sublevel" mode, the state, first-derivative, and second-derivative
#     level panels are shown.
#   • In "all_predictors" mode, only the joint model panel is shown.
#   • In "both" mode, all four panels are shown.
#
# Visual encoding is consistent across panels:
#   • Edge color : regulatory sign
#   • Edge width : |omega_ij| (posterior mean magnitude)
#
# Intended for APPENDIX or supplementary presentation.
# --------------------------------------------------

plot_cirn_panels <- function(edges,
                             predictor_sets = NULL,
                             representation_mode = c("sublevel",
                                                     "all_predictors",
                                                     "both"),
                             height = "430px",
                             legend_position = c("bottom", "side")) {
  
  representation_mode <- match.arg(representation_mode)
  legend_position <- match.arg(legend_position)
  panel_titles <- cirn_panel_titles()
  
  predictor_sets <- cirn_predictor_sets_for_mode(
    edges = edges,
    representation_mode = representation_mode,
    predictor_sets = predictor_sets
  )
  
  if (length(predictor_sets) == 0) {
    predictor_sets <- c("original", "first_derivative", "second_derivative")
  }
  
  panel_basis <- dplyr::case_when(
    length(predictor_sets) <= 1 ~ "100%",
    length(predictor_sets) == 2 ~ "48%",
    length(predictor_sets) == 3 ~ "31%",
    TRUE ~ "24%"
  )
  
  panel_min_width <- ifelse(length(predictor_sets) >= 4, "250px", "320px")
  
  # ------------------------------------------------
  # Helper function: single panel
  # ------------------------------------------------
  
  plot_single_panel <- function(edges_sub, panel_title, panel_key) {
    
    if (nrow(edges_sub) == 0) {
      return(
        div(
          style = paste0(
            "flex:1 1 ", panel_basis, "; min-width:", panel_min_width,
            "; padding:10px;"
          ),
          div(
            style = paste(
              "height:", height, "; border:1px solid #d9d9d9;",
              "display:flex; flex-direction:column; align-items:center;",
              "justify-content:center; text-align:center; background:#fafafa;"
            ),
            h3(style = "margin-bottom:8px;", panel_title),
            p(
              style = "color:#777777; margin:0 24px;",
              "No posterior-supported edges were retained for this representation."
            )
          )
        )
      )
    }
    
    plot_edges <- prepare_cirn_edges_for_plot(
      edges_sub,
      aggregate_duplicate_edges = TRUE
    )
    
    node_ids <- unique(c(plot_edges$term, plot_edges$target))
    target_ids <- unique(plot_edges$target)
    
    nodes <- make_cirn_node_table(node_ids, target_ids = target_ids)
    edges_vis <- make_cirn_edge_table(plot_edges, width_range = c(1.5, 6))
    
    div(
      style = paste0(
        "flex:1 1 ", panel_basis, "; min-width:", panel_min_width,
        "; padding:10px;"
      ),
      
      h3(panel_title, style = "text-align:center; margin-bottom:4px;"),
      p(
        paste0("Retained edges: ", nrow(edges_sub)),
        style = "text-align:center; margin-top:0; color:#777777;"
      ),
      
      visNetwork(nodes, edges_vis, width = "100%", height = height) %>%
        
        visEdges(arrows = "to") %>%
        
        visNodes(
          borderWidth = 2,
          font = list(
            color = "navy",
            size = 18,
            align = "center"
          )
        ) %>%
        
        visPhysics(
          solver = "forceAtlas2Based",
          stabilization = TRUE,
          forceAtlas2Based = list(
            gravitationalConstant = -80,
            springLength = 180
          )
        ) %>%
        
        visInteraction(
          dragNodes = TRUE,
          dragView = TRUE,
          zoomView = TRUE
        ) %>%
        visOptions(
          highlightNearest = list(enabled = TRUE, degree = 1, hover = TRUE),
          nodesIdSelection = TRUE
        ) %>%
        visLayout(randomSeed = match(panel_key, names(panel_titles)) + 100)
    )
  }
  
  panel_list <- purrr::map(
    predictor_sets,
    ~ plot_single_panel(
      edges %>% dplyr::filter(.data$predictor_set == .x),
      panel_titles[[.x]],
      .x
    )
  )
  
  mode_note <- switch(
    representation_mode,
    sublevel = paste(
      "Mode: sublevel. CIRN displays the state, first-derivative,",
      "and second-derivative level networks."
    ),
    all_predictors = paste(
      "Mode: all_predictors. CIRN displays the joint model using",
      "state, first-derivative, and second-derivative predictors together."
    ),
    both = paste(
      "Mode: both. CIRN displays the three sublevel",
      "networks plus the joint all-predictors network."
    )
  )
  
  page <- tagList(
    div(
      style = paste(
        "font-family:Arial, sans-serif;",
        "max-width:1700px; margin:0 auto; padding:10px 16px;"
      ),
      h2(
        style = "text-align:center; margin-bottom:4px;",
        "Output 2: Individual CIRN"
      ),
      p(
        style = "text-align:center; margin-top:0; color:#555555;",
        "Sublevel and joint posterior-supported networks"
      ),
      p(style = "text-align:center; margin-top:0; color:#777777;", mode_note),
      div(
        style = paste(
          "display:flex; gap:16px; align-items:stretch;",
          "justify-content:center; width:100%;"
        ),
        do.call(
          div,
          c(
            list(
              style = paste(
                "flex:1 1 auto; min-width:0;",
                "display:flex; flex-wrap:nowrap; justify-content:center;"
              )
            ),
            panel_list
          )
        ),
        if (legend_position == "side") cirn_network_legend("side")
      ),
      if (legend_position == "bottom") cirn_network_legend("bottom")
    ),
  )
  
  htmltools::browsable(page)
}

plot_cirn_outputs <- function(edges,
                              representation_mode = c("sublevel",
                                                      "all_predictors",
                                                      "both"),
                              predictor_sets = NULL) {
  
  representation_mode <- match.arg(representation_mode)
  
  htmltools::browsable(
    tagList(
      plot_network_main(
        edges,
        representation_mode = representation_mode,
        title = "Output 1: Full CIRN",
        subtitle = paste(
          "Posterior-supported network across CIRN state,",
          "first-derivative, and second-derivative sublevels"
        )
      ),
      hr(style = "margin:32px 0;"),
      plot_cirn_panels(
        edges,
        predictor_sets = predictor_sets,
        representation_mode = representation_mode
      )
    )
  )
}

plot_pairwise_network <- function(pairwise_edges,
                                  representation_mode = c("sublevel",
                                                          "all_predictors",
                                                          "both"),
                                  title = "Pairwise CIRN",
                                  subtitle = "Edges inferred from one target-regulator pair at a time") {
  
  representation_mode <- match.arg(representation_mode)
  
  if (is.null(pairwise_edges) || nrow(pairwise_edges) == 0) {
    return(
      htmltools::browsable(
        div(
          style = "font-family:Arial, sans-serif; padding:20px;",
          h2(title),
          div(
            style = paste(
              "border:1px solid #cccccc; padding:28px;",
              "text-align:center; color:#666666;"
            ),
            "No posterior-supported pairwise edges were retained."
          )
        )
      )
    )
  }
  
  plot_network_main(
    pairwise_edges,
    representation_mode = representation_mode,
    title = title,
    subtitle = subtitle,
    aggregate_duplicate_edges = TRUE
  )
}

base_cirn_term <- function(term) {
  term <- as.character(term)
  term <- sub("_d1$", "", term)
  term <- sub("_d2$", "", term)
  term
}

summarize_edge_consistency <- function(edges,
                                       pairwise_edges = NULL,
                                       collapse_derivative_terms = FALSE) {
  
  strongest_omega <- function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) return(NA_real_)
    x[which.max(abs(x))]
  }
  
  collapse_regulation <- function(x) {
    x <- sort(unique(as.character(x[!is.na(x)])))
    if (length(x) == 0) return(NA_character_)
    if (length(x) == 1) return(x)
    "mixed"
  }
  
  all_edges <- dplyr::bind_rows(edges, pairwise_edges)
  
  if (is.null(all_edges) || nrow(all_edges) == 0) {
    return(
      tibble::tibble(
        term = character(),
        target = character(),
        regulation_type = character(),
        in_pairwise = logical(),
        in_sublevel = logical(),
        in_all_predictors = logical(),
        omega_pairwise = numeric(),
        omega_sublevel = numeric(),
        omega_all_predictors = numeric(),
        abs_omega_pairwise = numeric(),
        abs_omega_sublevel = numeric(),
        abs_omega_all_predictors = numeric(),
        regulation_type_pairwise = character(),
        regulation_type_sublevel = character(),
        regulation_type_all_predictors = character(),
        n_sources = integer(),
        source_modes = character(),
        appears_in_all_three = logical(),
        has_sign_conflict = logical()
      )
    )
  }
  
  if (!"analysis_mode" %in% names(all_edges)) {
    all_edges$analysis_mode <- "multivariable"
  }
  if (!"omega" %in% names(all_edges)) {
    all_edges$omega <- NA_real_
  }
  if (!"regulation_type" %in% names(all_edges)) {
    all_edges$regulation_type <- dplyr::case_when(
      all_edges$omega > 0 ~ "activation",
      all_edges$omega < 0 ~ "inhibition",
      TRUE ~ NA_character_
    )
  }
  
  all_edges <- all_edges %>%
    dplyr::mutate(
      analysis_mode = as.character(.data$analysis_mode),
      source_mode = dplyr::case_when(
        .data$analysis_mode == "pairwise" ~ "pairwise",
        .data$predictor_set == "all_predictors" ~ "all_predictors",
        .data$predictor_set %in% c("original", "first_derivative", "second_derivative") ~ "sublevel",
        TRUE ~ as.character(.data$predictor_set)
      ),
      consistency_term = if (collapse_derivative_terms) {
        base_cirn_term(.data$term)
      } else {
        as.character(.data$term)
      }
    )
  
  cell_summary <- all_edges %>%
    dplyr::group_by(
      term = .data$consistency_term,
      .data$target,
      .data$source_mode
    ) %>%
    dplyr::summarise(
      omega = strongest_omega(.data$omega),
      abs_omega = abs(strongest_omega(.data$omega)),
      regulation_type = collapse_regulation(.data$regulation_type),
      coefficient_count = dplyr::n(),
      .groups = "drop"
    )
  
  source_summary <- cell_summary %>%
    dplyr::group_by(.data$term, .data$target) %>%
    dplyr::summarise(
      source_mode_list = list(sort(unique(.data$source_mode))),
      regulation_type = collapse_regulation(.data$regulation_type),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      in_pairwise = purrr::map_lgl(.data$source_mode_list, ~ "pairwise" %in% .x),
      in_sublevel = purrr::map_lgl(.data$source_mode_list, ~ "sublevel" %in% .x),
      in_all_predictors = purrr::map_lgl(.data$source_mode_list, ~ "all_predictors" %in% .x),
      n_sources = .data$in_pairwise + .data$in_sublevel + .data$in_all_predictors,
      source_modes = purrr::map_chr(.data$source_mode_list, ~ paste(.x, collapse = ", ")),
      appears_in_all_three = .data$in_pairwise & .data$in_sublevel & .data$in_all_predictors,
      has_sign_conflict = .data$regulation_type == "mixed"
    )
  
  source_values <- cell_summary %>%
    dplyr::filter(.data$source_mode %in% c("pairwise", "sublevel", "all_predictors")) %>%
    dplyr::select(
      term,
      target,
      source_mode,
      omega,
      abs_omega,
      regulation_type
    ) %>%
    tidyr::pivot_wider(
      names_from = source_mode,
      values_from = c(omega, abs_omega, regulation_type),
      names_glue = "{.value}_{source_mode}"
    )
  
  edge_consistency <- source_summary %>%
    dplyr::left_join(source_values, by = c("term", "target"))
  
  expected_cols <- c(
    "omega_pairwise", "omega_sublevel", "omega_all_predictors",
    "abs_omega_pairwise", "abs_omega_sublevel", "abs_omega_all_predictors",
    "regulation_type_pairwise", "regulation_type_sublevel", "regulation_type_all_predictors"
  )
  for (col in expected_cols) {
    if (!col %in% names(edge_consistency)) {
      edge_consistency[[col]] <- if (grepl("^regulation_type", col)) NA_character_ else NA_real_
    }
  }
  
  edge_consistency %>%
    dplyr::select(
      term,
      target,
      regulation_type,
      in_pairwise,
      in_sublevel,
      in_all_predictors,
      omega_pairwise,
      omega_sublevel,
      omega_all_predictors,
      abs_omega_pairwise,
      abs_omega_sublevel,
      abs_omega_all_predictors,
      regulation_type_pairwise,
      regulation_type_sublevel,
      regulation_type_all_predictors,
      n_sources,
      source_modes,
      appears_in_all_three,
      has_sign_conflict
    ) %>%
    dplyr::arrange(
      dplyr::desc(.data$appears_in_all_three),
      dplyr::desc(.data$n_sources),
      dplyr::desc(.data$has_sign_conflict),
      .data$target,
      .data$term
    )
}

plot_edge_consistency <- function(edge_consistency,
                                  show_regulation_label = FALSE,
                                  encode_strength = TRUE,
                                  strength_scale = c("within_mode", "global"),
                                  strength_alpha_range = c(0.25, 1)) {
  
  strength_scale <- match.arg(strength_scale)
  
  if (is.null(edge_consistency) || nrow(edge_consistency) == 0) {
    return(
      ggplot() +
        annotate(
          "text",
          x = 0,
          y = 0,
          label = "No posterior-supported edges to compare."
        ) +
        theme_void()
    )
  }
  
  strongest_omega <- function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) return(NA_real_)
    x[which.max(abs(x))]
  }
  
  collapse_regulation <- function(x) {
    x <- sort(unique(as.character(x[!is.na(x)])))
    if (length(x) == 0) return(NA_character_)
    if (length(x) == 1) return(x)
    "mixed"
  }
  
  scale_alpha_values <- function(abs_omega, present) {
    alpha_values <- rep(1, length(abs_omega))
    detected_idx <- present & is.finite(abs_omega)
    
    if (!any(detected_idx)) {
      return(alpha_values)
    }
    
    alpha_min <- min(strength_alpha_range, na.rm = TRUE)
    alpha_max <- max(strength_alpha_range, na.rm = TRUE)
    omega_range <- range(abs_omega[detected_idx], na.rm = TRUE)
    
    if (diff(omega_range) == 0) {
      alpha_values[detected_idx] <- alpha_max
    } else {
      alpha_values[detected_idx] <- scales::rescale(
        abs_omega[detected_idx],
        to = c(alpha_min, alpha_max),
        from = omega_range
      )
    }
    
    alpha_values
  }
  
  plot_input <- edge_consistency
  for (col in c("in_pairwise", "in_sublevel", "in_all_predictors")) {
    if (!col %in% names(plot_input)) plot_input[[col]] <- FALSE
  }
  for (col in c("omega_pairwise", "omega_sublevel", "omega_all_predictors")) {
    if (!col %in% names(plot_input)) plot_input[[col]] <- NA_real_
  }
  if (!"regulation_type_pairwise" %in% names(plot_input)) {
    plot_input$regulation_type_pairwise <- ifelse(plot_input$in_pairwise, plot_input$regulation_type, NA_character_)
  }
  if (!"regulation_type_sublevel" %in% names(plot_input)) {
    plot_input$regulation_type_sublevel <- ifelse(plot_input$in_sublevel, plot_input$regulation_type, NA_character_)
  }
  if (!"regulation_type_all_predictors" %in% names(plot_input)) {
    plot_input$regulation_type_all_predictors <- ifelse(plot_input$in_all_predictors, plot_input$regulation_type, NA_character_)
  }
  
  edge_df <- plot_input %>%
    dplyr::mutate(
      edge_base = paste0(
        format_cirn_node_label(.data$term),
        " -> ",
        format_cirn_node_label(.data$target)
      ),
      edge = if (show_regulation_label) {
        paste0(.data$edge_base, " [", .data$regulation_type, "]")
      } else {
        .data$edge_base
      }
    )
  
  plot_df <- dplyr::bind_rows(
    edge_df %>%
      dplyr::transmute(
        edge = .data$edge,
        source = "Pairwise",
        present = .data$in_pairwise,
        omega = .data$omega_pairwise,
        regulation_for_cell = .data$regulation_type_pairwise
      ),
    edge_df %>%
      dplyr::transmute(
        edge = .data$edge,
        source = "Sublevel",
        present = .data$in_sublevel,
        omega = .data$omega_sublevel,
        regulation_for_cell = .data$regulation_type_sublevel
      ),
    edge_df %>%
      dplyr::transmute(
        edge = .data$edge,
        source = "All predictors",
        present = .data$in_all_predictors,
        omega = .data$omega_all_predictors,
        regulation_for_cell = .data$regulation_type_all_predictors
      )
  ) %>%
    dplyr::group_by(.data$edge, .data$source) %>%
    dplyr::summarise(
      present = any(.data$present),
      omega = strongest_omega(.data$omega),
      regulation_for_cell = collapse_regulation(.data$regulation_for_cell[.data$present]),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      source = factor(.data$source, levels = c("Pairwise", "Sublevel", "All predictors")),
      abs_omega = abs(.data$omega),
      tile_status = dplyr::case_when(
        !.data$present ~ "Not detected",
        .data$regulation_for_cell == "activation" ~ "Activation",
        .data$regulation_for_cell == "inhibition" ~ "Inhibition",
        .data$regulation_for_cell == "mixed" ~ "Mixed sign",
        TRUE ~ "Detected"
      )
    ) %>%
    dplyr::group_by(.data$edge) %>%
    dplyr::mutate(n_sources = sum(.data$present)) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(edge = stats::reorder(.data$edge, .data$n_sources))
  
  plot_df$strength_value <- 1
  if (encode_strength) {
    if (strength_scale == "global") {
      plot_df$strength_value <- scale_alpha_values(plot_df$abs_omega, plot_df$present)
    } else {
      plot_df <- plot_df %>%
        dplyr::group_by(.data$source) %>%
        dplyr::mutate(strength_value = scale_alpha_values(.data$abs_omega, .data$present)) %>%
        dplyr::ungroup()
    }
  }
  plot_df <- plot_df %>%
    dplyr::mutate(
      fill_value = dplyr::case_when(
        !.data$present ~ 0,
        .data$regulation_for_cell == "activation" ~ .data$strength_value,
        .data$regulation_for_cell == "inhibition" ~ -.data$strength_value,
        .data$regulation_for_cell == "mixed" ~ NA_real_,
        TRUE ~ 0
      ),
      cell_label = dplyr::case_when(
        .data$present & is.finite(.data$omega) ~ sprintf("\u03c9 = %.2f", .data$omega),
        TRUE ~ ""
      )
    )
  
  strength_caption <- NULL
  if (encode_strength && strength_scale == "within_mode") {
    strength_caption <- paste(
      "Color intensity is scaled within each inference mode by relative |\u03c9|.",
      "Sublevel summarizes the state, first-derivative, and second-derivative levels."
    )
  }
  if (encode_strength && strength_scale == "global") {
    strength_caption <- paste(
      "Color intensity is scaled globally by relative |\u03c9|.",
      "Sublevel summarizes the state, first-derivative, and second-derivative levels."
    )
  }
  
  ggplot(plot_df, aes(x = .data$source, y = .data$edge, fill = .data$fill_value)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_label(
      aes(label = .data$cell_label),
      color = "#111111",
      fill = "#ffffff",
      alpha = 0.88,
      fontface = "bold",
      size = 3.8,
      linewidth = 0,
      label.padding = grid::unit(0.14, "lines"),
      na.rm = TRUE,
      show.legend = FALSE
    ) +
    scale_fill_gradient2(
      low = "#d62728",
      mid = "#eeeeee",
      high = "#2ca02c",
      midpoint = 0,
      limits = c(-1, 1),
      breaks = c(-1, 0, 1),
      labels = c("Inhibition", "Not detected", "Activation"),
      name = NULL,
      na.value = "#9467bd",
      guide = ggplot2::guide_colorbar(
        barwidth = grid::unit(5.5, "cm"),
        barheight = grid::unit(0.35, "cm"),
        ticks = FALSE,
        label.position = "bottom",
        title.position = "top"
      ),
      oob = scales::squish
    ) +
    labs(
      title = "CIRN Edge Consistency Across Inference Modes",
      x = NULL,
      y = "Inferred edge",
      caption = strength_caption
    ) +
    theme_cirn_publication(base_size = 12) +
    theme(
      legend.position = "bottom",
      axis.text.x = element_text(face = "bold"),
      axis.text.y = element_text(face = "bold", size = 13),
      axis.title.y = element_text(face = "bold", size = 14)
    )
}

format_publication_number <- function(x, digits = 2) {
  ifelse(
    is.na(x) | !is.finite(x),
    "",
    formatC(x, format = "f", digits = digits)
  )
}

format_publication_or <- function(x, digits = 2) {
  ifelse(
    is.na(x) | !is.finite(x),
    "",
    ifelse(
      abs(x) < 0.01,
      formatC(x, format = "e", digits = digits),
      formatC(x, format = "f", digits = digits)
    )
  )
}

format_publication_hdi <- function(lower, upper, digits = 2) {
  ifelse(
    is.na(lower) | is.na(upper),
    "",
    paste0(
      "[",
      format_publication_number(lower, digits = digits),
      ", ",
      format_publication_number(upper, digits = digits),
      "]"
    )
  )
}

format_publication_support <- function(in_pairwise,
                                       in_sublevel,
                                       in_all_predictors) {
  purrr::pmap_chr(
    list(in_pairwise, in_sublevel, in_all_predictors),
    function(pairwise, sublevel, all_predictors) {
      support_codes <- c(
        if (isTRUE(pairwise)) "P" else character(),
        if (isTRUE(sublevel)) "S" else character(),
        if (isTRUE(all_predictors)) "A" else character()
      )
      
      if (length(support_codes) == 0) {
        return("")
      }
      
      paste(support_codes, collapse = ",")
    }
  )
}

format_publication_jitter <- function(jitter_used,
                                      predictor_jitter_used = FALSE) {
  jitter_used <- dplyr::coalesce(as.logical(jitter_used), FALSE)
  predictor_jitter_used <- dplyr::coalesce(as.logical(predictor_jitter_used), FALSE)
  
  dplyr::case_when(
    jitter_used & predictor_jitter_used ~ "Yes (response + predictor)",
    jitter_used ~ "Yes (response)",
    predictor_jitter_used ~ "Yes (predictor)",
    TRUE ~ "No (raw)"
  )
}

cirn_term_order <- function(term, predictor_order = NULL) {
  term <- as.character(term)
  base_term <- base_cirn_term(term)
  representation_order <- dplyr::case_when(
    grepl("_d1$", term) ~ 2L,
    grepl("_d2$", term) ~ 3L,
    TRUE ~ 1L
  )
  
  if (!is.null(predictor_order) && length(predictor_order) > 0) {
    base_order <- match(base_term, predictor_order)
    base_order[is.na(base_order)] <- length(predictor_order) + match(
      base_term[is.na(base_order)],
      sort(unique(base_term[is.na(base_order)]))
    )
  } else {
    base_order <- match(base_term, sort(unique(base_term)))
  }
  
  representation_order * 100000L + base_order
}

make_publication_edge_table <- function(edges_combined,
                                        edge_consistency,
                                        predictor_order = NULL) {
  if (is.null(edges_combined) || nrow(edges_combined) == 0) {
    return(tibble::tibble())
  }
  
  if (is.null(edge_consistency) || nrow(edge_consistency) == 0) {
    edge_consistency <- summarize_edge_consistency(
      edges = edges_combined,
      pairwise_edges = NULL,
      collapse_derivative_terms = FALSE
    )
  }
  
  support_lookup <- edge_consistency %>%
    dplyr::select(
      term,
      target,
      in_pairwise,
      in_sublevel,
      in_all_predictors
    )
  
  publication_table <- edges_combined %>%
    dplyr::left_join(support_lookup, by = c("term", "target")) %>%
    dplyr::mutate(
      in_pairwise = dplyr::coalesce(.data$in_pairwise, FALSE),
      in_sublevel = dplyr::coalesce(.data$in_sublevel, FALSE),
      in_all_predictors = dplyr::coalesce(.data$in_all_predictors, FALSE),
      support = format_publication_support(
        .data$in_pairwise,
        .data$in_sublevel,
        .data$in_all_predictors
      ),
      cirn_edge = paste0(
        format_cirn_node_label(.data$term),
        ifelse(.data$regulation_type == "activation", " -> ", " -| "),
        format_cirn_node_label(.data$target)
      ),
      class_0_1 = dplyr::case_when(
        !is.na(.data$class_0_count) & !is.na(.data$class_1_count) ~ paste0(
          .data$class_0_count,
          "/",
          .data$class_1_count
        ),
        TRUE ~ ""
      ),
      jitter = format_publication_jitter(
        .data$jitter_used,
        dplyr::coalesce(.data$predictor_jitter_used, FALSE)
      ),
      term_order = cirn_term_order(.data$term, predictor_order = predictor_order),
      target_order = match(.data$target, sort(unique(.data$target))),
      analysis_mode_order = dplyr::case_when(
        .data$analysis_mode == "multivariable" ~ 1L,
        .data$analysis_mode == "pairwise" ~ 2L,
        TRUE ~ 3L
      ),
      predictor_set_order = dplyr::case_when(
        .data$predictor_set == "original" ~ 1L,
        .data$predictor_set == "first_derivative" ~ 2L,
        .data$predictor_set == "second_derivative" ~ 3L,
        .data$predictor_set == "all_predictors" ~ 4L,
        TRUE ~ 5L
      )
    ) %>%
    dplyr::arrange(
      .data$term_order,
      .data$target_order,
      .data$analysis_mode_order,
      .data$predictor_set_order
    ) %>%
    dplyr::transmute(
      Mode = .data$analysis_mode,
      `Predictor set` = .data$predictor_set,
      `CIRN edge` = .data$cirn_edge,
      omega = format_publication_number(.data$omega),
      OR = format_publication_or(.data$odds_ratio),
      `95% HDI` = format_publication_hdi(.data$hdi_lower95, .data$hdi_upper95),
      Strength = .data$rel_strength,
      Support = .data$support,
      `Class (0/1)` = .data$class_0_1,
      Jitter = .data$jitter
    )
  
  publication_table
}

make_all_coefficients_report_table <- function(all_coefficients_combined,
                                               predictor_order = NULL) {
  if (is.null(all_coefficients_combined) || nrow(all_coefficients_combined) == 0) {
    return(tibble::tibble())
  }
  
  coefficient_table <- all_coefficients_combined
  
  if (!"retained" %in% names(coefficient_table)) {
    coefficient_table$retained <- coefficient_table$hdi_lower95 > 0 |
      coefficient_table$hdi_upper95 < 0
  }
  if (!"credibility" %in% names(coefficient_table)) {
    coefficient_table$credibility <- ifelse(
      coefficient_table$retained,
      "credible",
      "uncertain"
    )
  }
  if (!"predictor_jitter_used" %in% names(coefficient_table)) {
    coefficient_table$predictor_jitter_used <- FALSE
  }
  
  coefficient_table %>%
    dplyr::mutate(
      cirn_coefficient = paste0(
        format_cirn_node_label(.data$term),
        " -> ",
        format_cirn_node_label(.data$target)
      ),
      hdi_decision = ifelse(.data$retained, "Retained", "Not retained"),
      class_0_1 = dplyr::case_when(
        !is.na(.data$class_0_count) & !is.na(.data$class_1_count) ~ paste0(
          .data$class_0_count,
          "/",
          .data$class_1_count
        ),
        TRUE ~ ""
      ),
      jitter = format_publication_jitter(
        .data$jitter_used,
        dplyr::coalesce(.data$predictor_jitter_used, FALSE)
      ),
      term_order = cirn_term_order(.data$term, predictor_order = predictor_order),
      target_order = match(.data$target, sort(unique(.data$target))),
      decision_order = ifelse(.data$retained, 1L, 2L),
      analysis_mode_order = dplyr::case_when(
        .data$analysis_mode == "multivariable" ~ 1L,
        .data$analysis_mode == "pairwise" ~ 2L,
        TRUE ~ 3L
      ),
      predictor_set_order = dplyr::case_when(
        .data$predictor_set == "original" ~ 1L,
        .data$predictor_set == "first_derivative" ~ 2L,
        .data$predictor_set == "second_derivative" ~ 3L,
        .data$predictor_set == "all_predictors" ~ 4L,
        TRUE ~ 5L
      )
    ) %>%
    dplyr::arrange(
      .data$term_order,
      .data$target_order,
      .data$analysis_mode_order,
      .data$predictor_set_order,
      .data$decision_order
    ) %>%
    dplyr::transmute(
      Mode = .data$analysis_mode,
      `Predictor set` = .data$predictor_set,
      `CIRN coefficient` = .data$cirn_coefficient,
      omega = format_publication_number(.data$omega),
      OR = format_publication_or(.data$odds_ratio),
      `95% HDI` = format_publication_hdi(.data$hdi_lower95, .data$hdi_upper95),
      `Regulation type` = .data$regulation_type,
      `HDI decision` = .data$hdi_decision,
      Credibility = .data$credibility,
      `Class (0/1)` = .data$class_0_1,
      Jitter = .data$jitter
    )
}

summarize_publication_vif_status <- function(status) {
  status <- unique(as.character(status[!is.na(status)]))
  if (length(status) == 0) return("not_available")
  
  priority <- c(
    "exact_collinearity",
    "aliased_coefficient",
    "rank_deficient_model",
    "severe_collinearity",
    "high_collinearity",
    "single_predictor",
    "ok",
    "not_available"
  )
  
  matched <- priority[priority %in% status]
  if (length(matched) > 0) return(matched[[1]])
  paste(sort(status), collapse = "; ")
}

format_publication_diagnostic_number <- function(x, digits = 2) {
  ifelse(
    is.na(x),
    "",
    ifelse(
      is.infinite(x),
      "Inf",
      formatC(x, format = "f", digits = digits)
    )
  )
}

make_publication_diagnostics_table <- function(diagnostics_table,
                                               pairwise_diagnostics_table = NULL,
                                               effective_sample_size_table = NULL,
                                               vif_group_table = NULL,
                                               pairwise_vif_group_table = NULL) {
  
  multivariable_diag <- if (!is.null(diagnostics_table) && nrow(diagnostics_table) > 0) {
    diagnostics_table %>%
      dplyr::mutate(
        analysis_mode = "multivariable",
        pairwise_predictor = NA_character_
      )
  } else {
    tibble::tibble()
  }
  
  pairwise_diag <- if (!is.null(pairwise_diagnostics_table) &&
                       nrow(pairwise_diagnostics_table) > 0) {
    pairwise_diagnostics_table %>%
      dplyr::mutate(analysis_mode = "pairwise")
  } else {
    tibble::tibble()
  }
  
  diagnostics_all <- dplyr::bind_rows(multivariable_diag, pairwise_diag)
  if (nrow(diagnostics_all) == 0) {
    return(tibble::tibble())
  }
  
  for (col in c(
    "target", "predictor_set", "pairwise_predictor", "status",
    "class_0_count", "class_1_count", "n_predictors",
    "jitter_used", "predictor_jitter_used", "max_rhat", "min_neff",
    "loo_elpd", "response_mode", "response_source", "response_trigger"
  )) {
    if (!col %in% names(diagnostics_all)) {
      diagnostics_all[[col]] <- NA
    }
  }
  
  multivariable_vif <- if (!is.null(vif_group_table) && nrow(vif_group_table) > 0) {
    vif_group_table %>%
      dplyr::mutate(
        analysis_mode = "multivariable",
        pairwise_predictor = NA_character_
      )
  } else {
    tibble::tibble()
  }
  
  pairwise_vif <- if (!is.null(pairwise_vif_group_table) &&
                      nrow(pairwise_vif_group_table) > 0) {
    pairwise_vif_group_table %>%
      dplyr::mutate(analysis_mode = "pairwise")
  } else {
    tibble::tibble()
  }
  
  vif_all <- dplyr::bind_rows(multivariable_vif, pairwise_vif)
  
  if (nrow(vif_all) > 0) {
    for (col in c(
      "target", "predictor_set", "pairwise_predictor", "vif",
      "status", "aliased", "model_rank", "model_columns",
      "rank_deficient"
    )) {
      if (!col %in% names(vif_all)) {
        vif_all[[col]] <- NA
      }
    }
    
    vif_summary <- vif_all %>%
      dplyr::mutate(
        target_response = as.character(.data$target),
        pairwise_predictor = as.character(.data$pairwise_predictor)
      ) %>%
      dplyr::group_by(
        .data$analysis_mode,
        .data$target_response,
        .data$predictor_set,
        .data$pairwise_predictor
      ) %>%
      dplyr::summarise(
        max_vif = if (all(is.na(.data$vif))) NA_real_ else max(.data$vif, na.rm = TRUE),
        vif_status = summarize_publication_vif_status(.data$status),
        rank_deficient = any(.data$rank_deficient, na.rm = TRUE),
        aliased = any(.data$aliased, na.rm = TRUE),
        model_rank = {
          value <- .data$model_rank[!is.na(.data$model_rank)]
          if (length(value) == 0) NA_integer_ else value[[1]]
        },
        model_columns = {
          value <- .data$model_columns[!is.na(.data$model_columns)]
          if (length(value) == 0) NA_integer_ else value[[1]]
        },
        .groups = "drop"
      )
  } else {
    vif_summary <- tibble::tibble(
      analysis_mode = character(),
      target_response = character(),
      predictor_set = character(),
      pairwise_predictor = character(),
      max_vif = numeric(),
      vif_status = character(),
      rank_deficient = logical(),
      aliased = logical(),
      model_rank = integer(),
      model_columns = integer()
    )
  }
  
  diagnostics_all %>%
    dplyr::mutate(
      target_response = ifelse(
        grepl("^d", as.character(.data$target)),
        as.character(.data$target),
        paste0("d", as.character(.data$target))
      ),
      pairwise_predictor = as.character(.data$pairwise_predictor),
      usable_n = .data$class_0_count + .data$class_1_count,
      minority_class_count = pmin(.data$class_0_count, .data$class_1_count),
      minority_per_predictor = dplyr::if_else(
        is.finite(.data$n_predictors) & .data$n_predictors > 0,
        .data$minority_class_count / .data$n_predictors,
        NA_real_
      ),
      class_balance = dplyr::if_else(
        is.finite(.data$usable_n) & .data$usable_n > 0,
        .data$minority_class_count / .data$usable_n,
        NA_real_
      ),
      sample_size_flag = dplyr::case_when(
        is.na(.data$status) ~ "unknown",
        .data$status != "completed" ~ paste0("not_fitted_", .data$status),
        !is.finite(.data$usable_n) ~ "unknown",
        .data$minority_class_count < 5 ~ "below_minimum",
        .data$minority_class_count < 10 ~ "small_minority_class",
        is.finite(.data$minority_per_predictor) &
          .data$minority_per_predictor < 5 ~ "low_minority_per_predictor",
        TRUE ~ "adequate_basic"
      )
    ) %>%
    dplyr::left_join(
      vif_summary,
      by = c(
        "analysis_mode",
        "target_response",
        "predictor_set",
        "pairwise_predictor"
      )
    ) %>%
    dplyr::mutate(
      rank_deficient = dplyr::coalesce(.data$rank_deficient, FALSE),
      aliased = dplyr::coalesce(.data$aliased, FALSE),
      rank_alias = dplyr::case_when(
        .data$rank_deficient & .data$aliased ~ "Rank deficient; aliased",
        .data$rank_deficient ~ "Rank deficient",
        .data$aliased ~ "Aliased",
        !is.na(.data$model_rank) & !is.na(.data$model_columns) ~ "Full rank/no alias",
        TRUE ~ "Not available"
      ),
      class_0_1 = dplyr::case_when(
        !is.na(.data$class_0_count) & !is.na(.data$class_1_count) ~ paste0(
          .data$class_0_count,
          "/",
          .data$class_1_count
        ),
        TRUE ~ ""
      ),
      jitter = format_publication_jitter(
        .data$jitter_used,
        dplyr::coalesce(.data$predictor_jitter_used, FALSE)
      ),
      analysis_mode_order = dplyr::case_when(
        .data$analysis_mode == "multivariable" ~ 1L,
        .data$analysis_mode == "pairwise" ~ 2L,
        TRUE ~ 3L
      ),
      predictor_set_order = dplyr::case_when(
        .data$predictor_set == "original" ~ 1L,
        .data$predictor_set == "first_derivative" ~ 2L,
        .data$predictor_set == "second_derivative" ~ 3L,
        .data$predictor_set == "all_predictors" ~ 4L,
        TRUE ~ 5L
      )
    ) %>%
    dplyr::arrange(
      .data$analysis_mode_order,
      .data$target_response,
      .data$pairwise_predictor,
      .data$predictor_set_order
    ) %>%
    dplyr::transmute(
      Mode = .data$analysis_mode,
      `Target response` = .data$target_response,
      `Predictor set` = .data$predictor_set,
      `Pairwise predictor` = dplyr::coalesce(.data$pairwise_predictor, ""),
      Status = .data$status,
      Predictors = .data$n_predictors,
      `Class (0/1)` = .data$class_0_1,
      `Usable n` = .data$usable_n,
      `Minority/predictor` = format_publication_diagnostic_number(.data$minority_per_predictor),
      `Class balance` = format_publication_diagnostic_number(.data$class_balance),
      `Max Rhat` = format_publication_diagnostic_number(.data$max_rhat),
      `Min ESS ratio` = format_publication_diagnostic_number(.data$min_neff),
      `Max VIF` = format_publication_diagnostic_number(.data$max_vif),
      `VIF status` = dplyr::coalesce(.data$vif_status, "not_available"),
      `Rank/Alias` = .data$rank_alias,
      Jitter = .data$jitter,
      `Sample-size flag` = .data$sample_size_flag,
      `LOO ELPD` = format_publication_diagnostic_number(.data$loo_elpd)
    )
}

latex_escape_text <- function(x) {
  x <- as.character(x)
  x <- gsub("\\\\", "\\\\textbackslash{}", x)
  x <- gsub("([_&#%$])", "\\\\\\1", x, perl = TRUE)
  x
}

latex_cirn_symbol <- function(x) {
  x <- as.character(x)
  escaped <- latex_escape_text(x)
  ifelse(
    grepl("^[A-Za-z][A-Za-z0-9]*$", x),
    escaped,
    paste0("\\mathrm{", escaped, "}")
  )
}

format_cirn_node_label_latex <- function(node_id) {
  node_id <- as.character(node_id)
  ifelse(
    grepl("_d1$", node_id),
    paste0("d", latex_cirn_symbol(sub("_d1$", "", node_id))),
    ifelse(
      grepl("_d2$", node_id),
      paste0("d^{2}", latex_cirn_symbol(sub("_d2$", "", node_id))),
      ifelse(
        grepl("^d[A-Za-z0-9_]+$", node_id),
        paste0("d", latex_cirn_symbol(sub("^d", "", node_id))),
        latex_cirn_symbol(node_id)
      )
    )
  )
}

format_cirn_edge_latex <- function(edge_label) {
  vapply(
    as.character(edge_label),
    function(label) {
      if (grepl(" -\\| ", label)) {
        parts <- strsplit(label, " -\\| ")[[1]]
        return(paste0(
          "$",
          format_cirn_node_label_latex(parts[1]),
          " \\dashv ",
          format_cirn_node_label_latex(parts[2]),
          "$"
        ))
      }
      
      parts <- strsplit(label, " -> ")[[1]]
      paste0(
        "$",
        format_cirn_node_label_latex(parts[1]),
        " \\longrightarrow ",
        format_cirn_node_label_latex(parts[2]),
        "$"
      )
    }
    ,
    character(1)
  )
}

write_publication_edge_table_latex <- function(publication_table,
                                               output_dir,
                                               filename = "CIRN_Publication_Edge_Table.tex",
                                               label = "tab:cirn_publication_edge_table") {
  if (is.null(publication_table) || nrow(publication_table) == 0) {
    return(NA_character_)
  }
  
  table_for_latex <- publication_table %>%
    dplyr::mutate(
      Mode = latex_escape_text(.data$Mode),
      `Predictor set` = latex_escape_text(.data$`Predictor set`),
      `CIRN edge` = format_cirn_edge_latex(.data$`CIRN edge`),
      Strength = latex_escape_text(.data$Strength),
      Support = latex_escape_text(.data$Support),
      Jitter = latex_escape_text(.data$Jitter)
    )
  
  header <- c(
    "\\begin{table*}[t!]",
    "\\centering",
    "\\caption{\\textbf{Publication-ready CIRN edge-list output.} The table reports posterior-supported rows from \\texttt{edges\\_combined}, with cross-mode support summarized from \\texttt{edge\\_consistency}.}",
    paste0("\\label{", label, "}"),
    "\\scriptsize",
    "\\setlength{\\tabcolsep}{2.2pt}",
    "\\renewcommand{\\arraystretch}{1.05}",
    "\\begin{tabular}{@{}lp{0.16\\textwidth}p{0.15\\textwidth}rrp{0.18\\textwidth}llll@{}}",
    "\\toprule",
    "Mode & Predictor set & CIRN edge & $\\omega$ & OR & 95\\% HDI & Strength & Support & Class (0/1) & Jitter \\\\",
    "\\midrule"
  )
  
  body <- purrr::pmap_chr(
    table_for_latex,
    function(Mode,
             `Predictor set`,
             `CIRN edge`,
             omega,
             OR,
             `95% HDI`,
             Strength,
             Support,
             `Class (0/1)`,
             Jitter) {
      paste(
        Mode,
        `Predictor set`,
        `CIRN edge`,
        omega,
        OR,
        paste0("$", `95% HDI`, "$"),
        Strength,
        Support,
        `Class (0/1)`,
        Jitter,
        sep = " & "
      ) %>%
        paste0(" \\\\")
    }
  )
  
  footer <- c(
    "\\bottomrule",
    "\\end{tabular}",
    "",
    "\\vspace{1mm}",
    "\\footnotesize",
    "\\textit{Notes:} Mode is the raw \\texttt{analysis\\_mode} value from \\texttt{edges\\_combined}; Predictor set is the raw \\texttt{predictor\\_set} value. Rows are ordered by predictor term, with state terms followed by first- and second-derivative terms. Support gives cross-mode evidence for the same term--target pair (P=pairwise, S=sublevel/representation-specific multivariable, A=all predictors). $\\omega$ is the posterior mean coefficient, OR is $\\exp(\\omega)$, and the 95\\% HDI is the highest-density interval used for edge retention. Class (0/1) reports the number of decreasing/increasing target-direction labels used in the fitted edge model. Jitter reports whether the fitted edge used the raw directional response, adaptive response jitter, optional predictor jitter, or both.",
    "\\end{table*}"
  )
  
  output_file <- file.path(output_dir, filename)
  writeLines(c(header, body, footer), con = output_file, useBytes = TRUE)
  message("Saved publication-ready LaTeX edge table: ", output_file)
  output_file
}

write_publication_tables_workbook <- function(table1_edges,
                                              table2_diagnostics,
                                              output_dir,
                                              filename = "CIRN_Publication_Tables.xlsx") {
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  
  if (!requireNamespace("openxlsx", quietly = TRUE)) {
    warning(
      "Publication-ready XLSX workbook was not created because package 'openxlsx' ",
      "is not installed.",
      call. = FALSE
    )
    return(NA_character_)
  }
  
  prepare_sheet <- function(tbl, empty_message) {
    if (is.null(tbl) || !inherits(tbl, "data.frame") || nrow(tbl) == 0) {
      return(tibble::tibble(Note = empty_message))
    }
    tbl
  }
  
  table1_edges <- prepare_sheet(
    table1_edges,
    "No retained CIRN edges were available for Table 1."
  )
  table2_diagnostics <- prepare_sheet(
    table2_diagnostics,
    "No fitted-model diagnostics were available for Table 2."
  )
  
  workbook <- openxlsx::createWorkbook()
  
  header_style <- openxlsx::createStyle(
    textDecoration = "bold",
    fgFill = "#D9EAF7",
    border = "Bottom",
    halign = "center",
    valign = "center",
    wrapText = TRUE
  )
  
  body_style <- openxlsx::createStyle(
    valign = "top",
    wrapText = TRUE
  )
  
  add_publication_sheet <- function(sheet_name, tbl) {
    openxlsx::addWorksheet(workbook, sheet_name)
    openxlsx::writeData(workbook, sheet_name, tbl)
    openxlsx::freezePane(workbook, sheet_name, firstRow = TRUE)
    openxlsx::addStyle(
      workbook,
      sheet = sheet_name,
      style = header_style,
      rows = 1,
      cols = seq_len(ncol(tbl)),
      gridExpand = TRUE
    )
    if (nrow(tbl) > 0) {
      openxlsx::addStyle(
        workbook,
        sheet = sheet_name,
        style = body_style,
        rows = seq_len(nrow(tbl)) + 1,
        cols = seq_len(ncol(tbl)),
        gridExpand = TRUE,
        stack = TRUE
      )
    }
    openxlsx::setColWidths(
      workbook,
      sheet = sheet_name,
      cols = seq_len(ncol(tbl)),
      widths = "auto"
    )
  }
  
  add_publication_sheet("Table1_Edges", table1_edges)
  add_publication_sheet("Table2_Diagnostics", table2_diagnostics)
  
  output_file <- file.path(output_dir, filename)
  openxlsx::saveWorkbook(workbook, output_file, overwrite = TRUE)
  message("Saved publication-ready two-sheet workbook: ", output_file)
  output_file
}


# --------------------------------------------------
# 7.7 Posterior Effect Visualization (CIRN)
# --------------------------------------------------

# These plots visualize posterior uncertainty of CIRN
# regression coefficients ω_ij (posterior means of β_ij).
#
# Interpretation:
#   • ω_ij > 0  → activating regulation
#   • ω_ij < 0  → inhibiting regulation
#   • ω_ij ≈ 0  → weak or no evidence of regulation
#
# Zero serves as a "structural reference", not a point estimate.
# Exclusion of zero from the 95% HDI constitutes the CIRN
# edge inclusion criterion.
# --------------------------------------------------

# --------------------------------------------------
# 7.7.1 Posterior Effect Plot (single coefficient)
# --------------------------------------------------

# Purpose:
#   • Inspect posterior uncertainty for a single ω_ij
#   • Assess sign stability and exclusion of zero
#
# Visual encoding:
#   • Density curve      : posterior distribution
#   • Red dashed line    : zero (no regulation reference)
#   • Blue dashed lines  : 95% Highest Density Interval (HDI)
#
# Intended use:
#   • Exploratory diagnostics
#   • Appendix or supplementary figures (NOT main text)
# --------------------------------------------------

plot_posterior_effect <- function(model, term, label_expr = NULL) {
  
  # Parameter name as stored by brms
  
  param <- paste0("b_", term)
  
  # Safety check
  
  draws_df <- as_draws_df(model)
  if (!param %in% names(draws_df)) {
    stop("Parameter ", param, " not found in model.")
  }
  
  # Extract posterior draws
  
  draws <- draws_df[[param]]
  
  # Compute 95% Highest Density Interval
  
  hdi <- bayestestR::hdi(draws)
  
  # Density plot
  
  plot(
    density(draws),
    col = "#1F4E79",
    lwd = 2.2,
    main = if (!is.null(label_expr)) {
      label_expr
    } else {
      paste("Posterior of", format_cirn_node_label(term))
    },
    xlab = expression(beta)
  )
  
  # Reference lines
  
  abline(v = 0, lty = 2, col = "#d62728", lwd = 1.2)          # Null regulation
  abline(v = hdi$CI_low,  lty = 3, col = "#1F4E79", lwd = 1.2)
  abline(v = hdi$CI_high, lty = 3, col = "#1F4E79", lwd = 1.2)
}

# --------------------------------------------------
# 7.8 Posterior distributions for ALL coefficients
# --------------------------------------------------

# Purpose:
#   • Compare posterior uncertainty across predictors
#   • Assess relative effect magnitude and sign consistency
#
# Each facet corresponds to one ω_ij.
# The vertical zero line indicates absence of regulation.
#
# Intended use:
#   • Appendix or supplementary material
#   • Reviewer-facing diagnostic assessment
# --------------------------------------------------

plot_all_posterior_effects <- function(model, terms) {
  
  # Extract posterior draws
  
  draws <- as_draws_df(model)
  
  # Construct brms parameter names
  
  param_names <- paste0("b_", terms)
  
  # Safety check
  
  missing <- setdiff(param_names, names(draws))
  if (length(missing) > 0) {
    stop("Parameters not found in model: ", paste(missing, collapse = ", "))
  }
  
  # Long-format posterior draws
  
  plot_df <- draws %>%
    select(all_of(param_names)) %>%
    pivot_longer(
      cols = everything(),
      names_to = "parameter",
      values_to = "value"
    ) %>%
    mutate(
      parameter = format_cirn_node_label(gsub("^b_", "", parameter))
    )
  
  # Density plot
  
  ggplot(plot_df, aes(x = value)) +
    geom_density(fill = "#7BAAD0", color = "#1F4E79", linewidth = 1.0, alpha = 1.0) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#d62728", linewidth = 0.9) +
    facet_wrap(~ parameter, scales = "free", nrow = 1) +
    labs(
      title = "Posterior distributions of CIRN coefficients",
      x = expression(beta),
      y = "Posterior density"
    ) +
    theme_cirn_publication(base_size = 12)
}

# --------------------------------------------------
# 7.9 Posterior distributions across CIRN representations
# --------------------------------------------------

# Purpose:
#   • Compare posterior estimates ω_ij across
#       - original state variables
#       - first temporal derivatives
#       - second temporal derivatives
#       - optional all-predictors model
#
# This visualization highlights how inferred regulatory
# effects depend on the chosen dynamical representation.
#
# Intended use:
#   • Appendix or supplementary material
# --------------------------------------------------

plot_all_cirn_posteriors <- function(models_named) {
  
  plot_df <- purrr::imap_dfr(models_named, function(mod, model_name) {
    
    if (is.null(mod)) return(NULL)
    
    draws <- as_draws_df(mod)
    coef_terms <- setdiff(rownames(brms::fixef(mod)), "Intercept")
    
    purrr::map_dfr(coef_terms, function(term) {
      tibble(
        value = draws[[paste0("b_", term)]],
        term  = paste0(format_cirn_node_label(term), " (", model_name, ")")
      )
    })
  })
  
  if (nrow(plot_df) == 0) {
    return(
      ggplot() +
        annotate(
          "text",
          x = 0,
          y = 0,
          label = "No non-intercept posterior coefficients available."
        ) +
        theme_void()
    )
  }
  
  ggplot(plot_df, aes(x = value)) +
    geom_density(fill = "#7BAAD0", color = "#1F4E79", linewidth = 1.0, alpha = 1.0) +
    facet_wrap(~ term, scales = "free") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#d62728", linewidth = 0.9) +
    labs(
      x = expression(beta),
      y = "Posterior density",
      title = "Posterior distributions across CIRN representations"
    ) +
    theme_cirn_publication(base_size = 12)
}

# --------------------------------------------------
# 7.10 Diagnostic Summary Plots (CIRN models)
# --------------------------------------------------

# Purpose:
#   • Provide a concise overview of MCMC convergence
#     and sampling quality
#   • Diagnose sampling efficiency and overall model adequacy
#   • NOT used for CIRN edge inclusion decisions
#
# These diagnostics support model credibility and transparency
# but do not influence CIRN inference rules (HDI-based).
# --------------------------------------------------

plot_diagnostics_summary <- function(diagnostics_table) {
  
  # Two-panel layout:
  #   Left  : Sampling efficiency (ESS ratio)
  #   Right : Predictive adequacy (LOO ELPD)
  
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)
  
  par(mfrow = c(1, 2), mar = c(6, 4, 4, 2))
  
  has_diagnostics <- !is.null(diagnostics_table) &&
    nrow(diagnostics_table) > 0
  
  if (!has_diagnostics) {
    plot.new()
    title("Minimum ESS ratio")
    text(0.5, 0.5, "No fitted-model diagnostics available.")
    
    plot.new()
    title("LOO ELPD")
    text(0.5, 0.5, "LOO was not computed.")
    
    return(invisible(NULL))
  }
  
  diagnostics_table <- diagnostics_table %>%
    dplyr::mutate(
      min_neff = suppressWarnings(as.numeric(.data$min_neff)),
      loo_elpd = suppressWarnings(as.numeric(.data$loo_elpd)),
      model_label = paste(.data$target, .data$predictor_set, sep = "\n")
    )
  
  # --------------------------------------------------
  # Panel 1: Minimum Effective Sample Size (ESS) ratio
  # --------------------------------------------------
  
  # Interpretation:
  #   • ESS ratio ≈ 1   → excellent mixing
  #   • ESS ratio < 0.1 → poor mixing / autocorrelation
  #
  # We plot the MINIMUM ESS across parameters
  # to highlight worst-case sampling behavior.
  
  ess_table <- diagnostics_table %>%
    dplyr::filter(is.finite(.data$min_neff))
  
  if (nrow(ess_table) > 0) {
    barplot(
      ess_table$min_neff,
      names.arg = ess_table$model_label,
      las  = 2,
      main = "Minimum ESS ratio",
      ylab = "ESS / N",
      col  = "grey90",
      border = "grey40"
    )
    
    abline(h = 0.1, lty = 2, col = "#D55E00")  # heuristic warning threshold
  } else {
    plot.new()
    title("Minimum ESS ratio")
    text(0.5, 0.5, "ESS diagnostics unavailable.")
  }
  
  # --------------------------------------------------
  # Panel 2: Leave-One-Out Expected Log Predictive Density
  # --------------------------------------------------
  
  # Interpretation:
  #   • Higher ELPD indicates better out-of-sample predictive fit
  #   • Used ONLY to compare alternative CIRN representations
  #
  # IMPORTANT:
  #   • LOO plays NO role in edge inclusion or sign determination
  #   • Regulatory inference is based solely on HDI exclusion
  
  loo_table <- diagnostics_table %>%
    dplyr::filter(is.finite(.data$loo_elpd))
  
  if (nrow(loo_table) > 0) {
    dotchart(
      loo_table$loo_elpd,
      labels = paste(loo_table$target, loo_table$predictor_set),
      main = "LOO ELPD (higher is better)",
      xlab = "Expected log predictive density"
    )
  } else {
    plot.new()
    title("LOO ELPD")
    text(
      0.5,
      0.5,
      "LOO was not computed.\nSet compute_loo = TRUE to generate this panel."
    )
  }
  
  invisible(NULL)
}

# --------------------------------------------------
# 7.11 MCMC Diagnostics (Sampling Quality)
# --------------------------------------------------

# Purpose:
#   • Visually assess convergence and chain mixing
#   • Distinguish warm-up (burn-in) from the sampling phase
#   • Compare MCMC behavior across CIRN representations
#
# IMPORTANT:
#   • These plots are diagnostic ONLY
#   • They are NOT used for edge inclusion or sign determination
#   • Regulatory inference relies exclusively on HDI exclusion
# --------------------------------------------------

# --------------------------------------------------
# 7.11.1 Quick MCMC sanity check (single model)
# --------------------------------------------------

# Uses default brms diagnostic plots:
#   • Trace plots
#   • Posterior density plots
#   • Autocorrelation summaries
#
# Intended for:
#   • Rapid development and debugging checks
#   • Preliminary assessment only 

plot_mcmc_hist_trace <- function(model, pars = NULL) {
  
  # --------------------------------------------------
  # Extract posterior draws (sampling phase only)
  # --------------------------------------------------
  
  draws <- posterior::as_draws_df(model) %>%
    tibble::as_tibble()
  
  # Number of warm-up (burn-in) iterations per chain
  
  burnin <- model$fit@sim$warmup
  
  # --------------------------------------------------
  # Dynamically generate chain colors
  # --------------------------------------------------
  
  chain_ids <- sort(unique(draws$.chain))
  n_chains  <- length(chain_ids)
  
  chain_colors <- setNames(
    publication_palette(n_chains),
    as.character(chain_ids)
  )
  
  # --------------------------------------------------
  # Select parameters
  # --------------------------------------------------
  
  if (is.null(pars)) {
    pars <- colnames(draws)[grepl("^b_", colnames(draws))]
  }
  
  if (length(pars) == 0) {
    return(
      ggplot() +
        annotate(
          "text",
          x = 0,
          y = 0,
          label = "No non-intercept MCMC parameters available."
        ) +
        theme_void()
    )
  }
  
  parameter_levels <- format_cirn_node_label(gsub("^b_", "", pars))
  
  df_long <- draws %>%
    dplyr::select(.iteration, .chain, all_of(pars)) %>%
    tidyr::pivot_longer(
      cols = all_of(pars),
      names_to  = "parameter",
      values_to = "value"
    ) %>%
    dplyr::mutate(
      .chain    = factor(.chain),
      parameter = gsub("^b_", "", parameter),
      parameter = factor(
        format_cirn_node_label(parameter),
        levels = parameter_levels
      )
    )
  
  # --------------------------------------------------
  # Histogram (left): marginal posterior
  # --------------------------------------------------
  
  p_hist <- ggplot(df_long, aes(x = value)) +
    geom_histogram(
      bins  = 30,
      fill  = "#7BAAD0",
      color = "#111111",
      alpha = 1.0
    ) +
    facet_wrap(~ parameter, scales = "free", ncol = 1) +
    labs(x = NULL, y = "Count") +
    theme_cirn_publication(base_size = 11)
  
  # --------------------------------------------------
  # Trace plot (right): MCMC trajectory + burn-in
  # --------------------------------------------------
  
  p_trace <- ggplot(
    df_long,
    aes(x = .iteration, y = value, color = .chain)
  ) +
    geom_line(alpha = 1.0, linewidth = 0.3) +
    geom_vline(
      xintercept = 0,
      linetype   = "dashed",
      color      = "#D55E00",
      linewidth  = 0.6
    ) +
    facet_wrap(~ parameter, scales = "free", ncol = 1) +
    scale_color_manual(values = chain_colors) +
    labs(
      x     = "Iteration",
      y     = NULL,
      color = "Chain"
    ) +
    theme_cirn_publication(base_size = 11) +
    theme(
      legend.position = "right"
    ) +
    guides(
      colour = guide_legend(
        override.aes = list(linewidth = 1.4, alpha = 1)
      )
    )
  
  # --------------------------------------------------
  # Combine histogram + trace
  # --------------------------------------------------
  
  bayesplot::bayesplot_grid(
    p_hist,
    p_trace,
    grid_args = list(ncol = 2, rel_widths = c(1, 2))
  )
}

# --------------------------------------------------
# 7.11.2 Full trace diagnostics across CIRN representations
# --------------------------------------------------

# Produces a grid of trace plots with:
#   • Rows    → regression coefficients (ω_ij)
#   • Columns → CIRN representations
#               (original state, first derivative, second derivative,
#                optional all-predictors model)
#
# Each panel displays:
#   • All MCMC chains
#   • Full iteration range (warm-up + sampling)
#   • Explicit burn-in cutoff
#
# Intended for:
#   • Appendix or supplementary material
#   • Detailed convergence and mixing assessment

plot_all_traces_with_burnin <- function(models_list, target_var) {
  
  plot_data   <- list()
  burnin_vals <- list()
  all_chains  <- c()
  
  for (repr in names(models_list)) {
    
    model <- models_list[[repr]]
    if (is.null(model)) next
    
    # --------------------------------------------------
    # Extract posterior draws INCLUDING warm-up
    # --------------------------------------------------
    
    draws <- posterior::as_draws_df(model, include_warmup = TRUE) %>%
      tibble::as_tibble()
    
    # Warm-up (burn-in) iterations per chain
    
    burnin <- model$fit@sim$warmup
    
    # Track chain IDs globally (across representations)
    
    all_chains <- union(all_chains, unique(draws$.chain))
    
    # --------------------------------------------------
    # Non-intercept coefficients
    # --------------------------------------------------
    
    coef_names <- rownames(brms::fixef(model))
    coef_names <- coef_names[coef_names != "Intercept"]
    if (length(coef_names) == 0) next
    
    df_long <- draws %>%
      dplyr::select(
        .iteration,
        .chain,
        all_of(paste0("b_", coef_names))
      ) %>%
      tidyr::pivot_longer(
        cols = starts_with("b_"),
        names_to  = "parameter",
        values_to = "value"
      ) %>%
      dplyr::mutate(
        representation = repr,
        parameter      = format_cirn_node_label(gsub("^b_", "", parameter)),
        .chain         = factor(.chain)
      ) %>%
      dplyr::mutate(
        representation = factor(
          representation,
          levels = c(
            "original",
            "first_derivative",
            "second_derivative",
            "all_predictors"
          )
        )
      )
    
    plot_data[[repr]]   <- df_long
    burnin_vals[[repr]] <- burnin
  }
  
  plot_df <- dplyr::bind_rows(plot_data)
  
  if (nrow(plot_df) == 0) {
    return(
      ggplot() +
        annotate(
          "text",
          x = 0,
          y = 0,
          label = paste(
            "No non-intercept MCMC parameters available for target",
            format_cirn_node_label(target_var)
          )
        ) +
        theme_void()
    )
  }
  
  # --------------------------------------------------
  # Dynamically generate chain colors (global consistency)
  # --------------------------------------------------
  
  all_chains <- sort(unique(all_chains))
  chain_colors <- setNames(
    publication_palette(length(all_chains)),
    as.character(all_chains)
  )
  
  ggplot(plot_df, aes(x = .iteration, y = value, color = .chain)) +
    geom_line(alpha = 1.0, linewidth = 0.3) +
    facet_grid(parameter ~ representation, scales = "free_y") +
    geom_vline(
      xintercept = 0,
      linetype   = "dashed",
      color      = "#D55E00",
      linewidth  = 0.6
    ) +
    scale_color_manual(values = chain_colors) +
    labs(
      title    = paste(
        "MCMC trace diagnostics for target",
        format_cirn_node_label(target_var)
      ),
      subtitle = "Red dashed line marks end of warm-up (burn-in)",
      x        = "Iteration",
      y        = expression(beta),
      color    = "Chain"
    ) +
    theme_cirn_publication(base_size = 12) +
    theme(
      legend.position  = "bottom"
    ) +
    guides(
      colour = guide_legend(
        override.aes = list(linewidth = 1.2, alpha = 1)
      )
    )
}


################################################################################
################################################################################


##### User-configurable CIRN analysis run


# ==================================================
# 8. Data Loading, Configuration, and Pre-CIRN Checks
# ==================================================

# Purpose:
#   • Load a multivariate time-series dataset
#   • Set all CIRN analysis options in one central place
#   • Visualize raw trajectories and derivative construction before fitting
#   • Execute CIRN and summarize posterior-supported regulatory edges
#
# IMPORTANT SCOPE NOTE:
#   • CIRN inference is unsupervised and does not require ground truth.
#   • Ground-truth validation is optional and should be enabled only for
#     simulations or benchmark systems with a known signed adjacency matrix.
# ==================================================

# --------------------------------------------------
# 8.1 Load input data and identify the time column
# --------------------------------------------------

# Assumptions:
#   • First column corresponds to time.
#   • Remaining columns are observed state variables.
#   • Missing rows are removed before derivative construction.
#
# User setting:
#   • Option 1: set input_file explicitly below.
#   • Option 2: pass the file as the first command-line argument with Rscript.
#   • Option 3: leave input_file = NULL when exactly one CSV file is present
#     in the analysis directory; the script will use that CSV automatically.
#
# Design choice:
#   • Raw values are preserved here. Class labels are created later from
#     the raw target derivative, and model predictors are standardized only
#     after class construction and temporal lag alignment.

input_file <- NULL
# Examples:
#   input_file <- "my_dataset.csv"
#   input_file <- "/absolute/path/to/my_dataset.csv"
# Leave NULL to use the first command-line argument, the bundled example
# in data/Predator_Prey.csv, or the only CSV file in analysis_dir.
#
# Input priority:
#   1. Explicit path assigned to input_file above
#   2. First command-line argument
#   3. Bundled data/Predator_Prey.csv
#   4. The only CSV file in analysis_dir

cli_args <- commandArgs(trailingOnly = TRUE)
if (is.null(input_file) && length(cli_args) >= 1 && nzchar(cli_args[1])) {
  input_file <- cli_args[1]
}

if (is.null(input_file)) {
  bundled_example <- file.path(analysis_dir, "data", "Predator_Prey.csv")
  if (file.exists(bundled_example)) {
    input_file <- bundled_example
    message("No input file supplied; using bundled example: ", input_file)
  }
}

if (is.null(input_file)) {
  csv_candidates <- list.files(
    analysis_dir,
    pattern = "\\.csv$",
    full.names = FALSE,
    ignore.case = TRUE
  )
  
  if (length(csv_candidates) == 1) {
    input_file <- csv_candidates[[1]]
    message("No input_file supplied; using the only CSV in analysis_dir: ", input_file)
  } else if (length(csv_candidates) == 0) {
    stop(
      "No input file was supplied and no CSV file was found in analysis_dir. ",
      "Set input_file explicitly or run: Rscript CIRN_Algorithm.R path/to/data.csv"
    )
  } else {
    stop(
      "No input file was supplied and multiple CSV files were found in analysis_dir: ",
      paste(csv_candidates, collapse = ", "),
      ". Set input_file explicitly or run: Rscript CIRN_Algorithm.R path/to/data.csv"
    )
  }
}

input_path <- if (grepl("^(/|[A-Za-z]:)", input_file)) {
  input_file
} else {
  file.path(analysis_dir, input_file)
}

if (!file.exists(input_path)) {
  stop("Input file not found: ", input_path)
}

# Save all generated figures, diagnostics, and session information beside the
# input data file. This keeps outputs for each dataset inside that dataset's
# own analysis folder, even when the master CIRN script is stored elsewhere.
input_path <- normalizePath(input_path, mustWork = TRUE)
output_dir <- dirname(input_path)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
message("CIRN input file: ", input_path)
message("CIRN figure/output directory: ", output_dir)

my_data <- read.csv(input_path)

time_col <- NULL  # Set explicitly when the time column is not first.
if (is.null(time_col)) {
  time_col <- names(my_data)[1]
}

sim_df <- my_data


# --------------------------------------------------
# Optional known ground-truth network
# Use only for simulations or benchmark systems.
# --------------------------------------------------

# Leave NULL unless run_ground_truth_validation = TRUE.
true_adj_matrix <- NULL

# Example template:
#   truth_vars <- c("X", "Y")  # must match data column names exactly
#   true_adj_matrix <- matrix(
#     0,
#     nrow = length(truth_vars),
#     ncol = length(truth_vars),
#     dimnames = list(truth_vars, truth_vars)
#   )
#
# Rows = predictors/regulators
# Columns = targets/responses
# +1 = activation, -1 = inhibition, 0 = no edge
#
#   true_adj_matrix["X", "Y"] <-  1
#   true_adj_matrix["Y", "X"] <- -1


# --------------------------------------------------
# 8.2 Central CIRN run configuration
# --------------------------------------------------

# Change settings here only. The run block below automatically uses
# this configuration, so function defaults do not need manual editing.
#
# Target and predictor choices:
#   targets    = variables whose target response Y_i(t_k) is modeled.
#   predictors = variables allowed to act as candidate regulators.
#
# Inference modes:
#   "sublevel"       = state level, first-derivative level, and
#                      second-derivative level models are fitted.
#   "all_predictors" = one joint model uses all predictor representations.
#   "both"           = runs both approaches.
#
# Pairwise mode:
#   run_pairwise = TRUE adds one target-regulator model at a time. This is
#   diagnostic and is stored independently from the main multivariable output.


####################################################################################
############################# CIRN BEGIN CONFIGURATION #############################
####################################################################################

cirn_config <- list(
  
  # --- Variables to model ---
  #   Set to NULL to use all non-time numeric variables.
  #   To analyze only selected variables, use character vectors with exact column names:
  #   targets = c("x1")             # only these variables are modeled as targets
  #   predictors = c("x1", "x3")    # only these variables are tested as candidate regulators
  #   Targets and predictors may differ; do not include the time column.
  
  targets = NULL,
  predictors = NULL,
  
  # --- CIRN inference modes ---
  #   representation_mode controls the primary multivariable CIRN models:
  #   "sublevel"       = fit state, first-derivative, and second-derivative level models
  #   "all_predictors" = fit one joint state-derivative model per target
  #   "both"           = run both "sublevel" and "all_predictors" modes
  
  representation_mode = "both",
  
  #   Used only when run_pairwise = TRUE.
  #   NULL copies representation_mode. Otherwise use "sublevel", "all_predictors", or "both".
  
  pairwise_representation_mode = NULL,
  
  # --- Temporal alignment and pairwise option ---
  #   lag_units = 1 means predictors at t_{k-1} are used to model Y_i(t_k).
  #   Increase only when there is a justified longer response delay.
  
  lag_units = 1,
  
  #   FALSE is recommended for primary inference.
  #   TRUE adds lower-dimensional one-predictor-at-a-time diagnostic models.
  
  run_pairwise = TRUE,
  
  # --- Preprocessing and derivative construction ---
  #   points_per_interval controls the regularized derivative grid resolution.
  #   1 keeps one grid interval per observed interval; larger values densify the grid.
  
  points_per_interval = 1,
  
  #   NULL lets smooth.spline choose the smoothing level by GCV.
  #   Use a value in [0, 1] only for a planned smoothing-sensitivity analysis.
  
  spar = NULL,
  
  #   outlier_method = "none" leaves trajectories unchanged before smoothing.
  #   Use "MAD" only when robust outlier handling is part of the analysis plan.
  
  outlier_method = "none",
  outlier_thresh = 3.5,
  
  #   With outlier_method = "MAD": "keep", "winsorize", or "remove".
  #   "keep" records detected outliers but does not alter the data.
  
  outlier_action = "keep",
  
  #   Tolerance for removing near-zero raw target derivatives before fitting.
  #   1e-6 is a numerical safeguard; consider sensitivity checks for noisy empirical data.
  
  response_eps = 1e-6,
  
  #   Adaptive minimal jitter is used only when the raw target
  #   derivative response is one-class or nearly one-class.
  #   CIRN tries very small jitter levels and stops at the smallest
  #   level that creates at least jitter_min_class_count observations
  #   in each derivative class. Jitter is added to the target trajectory
  #   used to construct Class, not directly to Class labels.
  #   Every jittered fit is explicitly flagged in edges and diagnostics.
  #   Recommended jitter_min_class_count guide:
  #     n <= 100 usable time points      : 5
  #     100 < n <= 300 usable time points: 5 to 10
  #     n > 300 usable time points       : about 5% of usable points,
  #                                        without forcing artificial balance.
  
  adaptive_jitter = TRUE,
  
  #   Optional sensitivity mode only.
  #   FALSE keeps predictors on the original smoothed trajectories.
  #   TRUE, only when adaptive response jitter is triggered, jitters
  #   declared predictor state trajectories and recomputes their
  #   first/second derivatives before fitting. Keep FALSE for primary
  #   CIRN results unless explicitly reporting a sensitivity analysis.
  
  jitter_predictors = FALSE,
  jitter_min_class_count = 5,
  jitter_scale_grid = c(
    1e-8, 3e-8,
    1e-7, 3e-7,
    1e-6, 3e-6,
    1e-5, 3e-5,
    1e-4, 3e-4,
    1e-3, 3e-3,
    1e-2
  ),
  jitter_scale_basis = "state_sd",    
  
  #   Scale used to convert jitter_scale_grid values into an actual jitter SD.
  #   Options:
  #     "state_sd"           = fraction of the target state standard deviation; conservative default.
  #     "state_range"        = fraction of the target state range; useful for strongly monotonic trajectories.
  #     "derivative_sd"      = fraction of the target derivative standard deviation; usually very small.
  #     "derivative_max_abs" = fraction of the largest absolute target derivative; usually very small.
  #     "absolute"           = use jitter_scale_grid values directly as absolute jitter SD values.
  
  # --- Bayesian MCMC settings ---
  #   Total iterations per chain and warmup samples.
  #   Post-warmup draws per chain = model_iter - model_warmup.
  
  model_iter = 3000,
  model_warmup = 1000,
  model_chains = 4,
  
  #   Keep cores modest for long pairwise runs to avoid overloading the machine - 1 or 2 is enough.
  
  model_cores = 2,
  
  #   Normal(prior_mean, prior_sd) prior for regression coefficients.
  #   prior_sd is the standard deviation, not the variance.
  
  prior_mean = 0,
  prior_sd = 2,
  
  #   Increase adapt_delta, for example to 0.99, only if divergent transitions occur.
  
  adapt_delta = 0.95,
  
  # --- Optional LOO model comparison ---
  #   FALSE is recommended for routine network inference.
  #   TRUE adds leave-one-out model comparison and can substantially increase runtime.
  
  compute_loo = FALSE,
  
  #   Advanced LOO controls; keep defaults unless diagnosing problematic Pareto-k values.
  
  loo_moment_match = FALSE,
  loo_reloo = NULL,
  loo_k_threshold = 0.7,
  
  # --- Optional ground-truth validation ---
  #   Enable only for simulations or benchmarks where the signed true network is known.
  
  run_ground_truth_validation = FALSE,
  
  #   true_adj must be supplied explicitly when validation is enabled after "sim_df <- my_data".
  #   Rows = predictors, columns = targets; use +1 activation, -1 inhibition, 0 no edge.
  #   For empirical datasets without known ground truth, keep validation disabled, i.e. run_ground_truth_validation = FALSE
  #   true_adj = NULL if run_ground_truth_validation = FALSE, true_adj = true_adj_matrix if run_ground_truth_validation = TRUE
  
  true_adj = NULL,    
  
  # --- Optional Objective: robustness and sensitivity analysis ---
  #   Keep FALSE for ordinary/final CIRN runs.
  #   Set TRUE only when intentionally evaluating robustness under
  #   controlled perturbations. This reruns CIRN many times and can be slow,
  #   especially when Bayesian MCMC settings are large or run_pairwise = TRUE.
  #   Recommended use:
  #     FALSE = main analysis / primary reported network
  #     TRUE  = sensitivity analysis for dissertation/manuscript robustness tables
  
  run_sensitivity_analysis = FALSE,
  
  #   Leave NULL to automatically generate the sensitivity plan from the
  #   settings below. Advanced users may supply a custom tibble/data.frame
  #   with these required columns:
  #     sensitivity_id, condition, scenario, scenario_value, value, replicate
  #   Supported scenario values in the default helper:
  #     "baseline", "noise_sd_fraction", "lag_units",
  #     "downsample_interval", "target_sample_size", "row_missing_fraction"
  
  sensitivity_plan = NULL,
  
  #   Controls which inference modes are rerun inside sensitivity analysis.
  #   "use_config"      = use representation_mode, run_pairwise, and
  #                       pairwise_representation_mode exactly as supplied above.
  #   "sublevel"        = rerun only sublevel multivariable CIRN.
  #   "all_predictors"  = rerun only all-predictors multivariable CIRN.
  #   "both"            = rerun sublevel plus all-predictors, without pairwise.
  #   "pairwise_only"   = focus on pairwise CIRN retained edges.
  #   "everything"      = rerun sublevel, all-predictors, and both pairwise modes.
  
  sensitivity_inference_scope = "use_config",
  
  #   Number of repeated runs for stochastic sensitivity scenarios.
  #   Used for noise addition and random row-removal scenarios.
  #   Larger values give more stable robustness summaries but increase runtime.
  
  sensitivity_replicates = 3,
  
  #   Noise sensitivity:
  #   Adds Gaussian noise to each state variable before CIRN preprocessing.
  #   Values are fractions of each variable's empirical standard deviation.
  #   Example: 0.05 means noise SD = 5% of that variable's SD.
  
  sensitivity_noise_sd_fractions = c(0.01, 0.05, 0.10),
  
  #   Lag sensitivity:
  #   Reruns CIRN using alternative temporal lags.
  #   Example: lag_units = 2 means predictors at t_{k-2}
  #   are used to model the target response at t_k.
  
  sensitivity_lag_units = c(1, 2, 3),
  
  #   Sampling-frequency sensitivity:
  #   Downsamples the time series by keeping every nth observation.
  #   Example: 2 keeps every second time point; 5 keeps every fifth time point.
  
  sensitivity_downsample_intervals = c(2, 5),
  
  #   Effective sample-size sensitivity:
  #   Reruns CIRN after reducing the trajectory to approximately these
  #   numbers of time points. Useful for answering "how small is small?"
  #   in terms of usable sample size and edge recovery stability.
  
  sensitivity_target_sample_sizes = c(25, 50, 75, 100),
  
  #   Data sparsity / missingness sensitivity:
  #   Randomly removes the specified fraction of rows before CIRN preprocessing.
  #   Example: 0.10 removes 10% of observations; 0.25 removes 25%.
  
  sensitivity_missing_fractions = c(0.10, 0.25),
  
  #   Save sensitivity outputs as CSV files in the same folder as the input data.
  #   Outputs include run summaries, retained edges, effective sample-size
  #   diagnostics, edge stability, and ground-truth metrics when available.
  
  sensitivity_save_outputs = TRUE,
  
  #   Progress display for repeated sensitivity runs.
  #   Keep FALSE for cleaner console output; set TRUE when running long
  #   sensitivity analyses interactively.
  
  sensitivity_show_progress = FALSE,
  sensitivity_progress_bar = FALSE,
  
  # --- Reproducibility ---
  
  seed = 123,
  
  # --- Progress and debugging ---
  #   show_progress/progress_bar control user-facing runtime messages.
  
  show_progress = TRUE,
  progress_bar = TRUE,
  
  #   Optional function called by the pipeline for custom progress reporting.
  
  progress_callback = NULL,
  
  #   debug = TRUE stores additional intermediate objects and messages.
  #   assign_debug_to_global = TRUE exports selected debug objects to the global environment.
  #   Keep both FALSE for final scripted runs unless troubleshooting.
  
  debug = FALSE,
  assign_debug_to_global = FALSE
)

####################################################################################
############################## CIRN END CONFIGURATION ##############################
####################################################################################


if (is.null(cirn_config$pairwise_representation_mode)) {
  cirn_config$pairwise_representation_mode <- cirn_config$representation_mode
}

if (is.null(cirn_config$model_cores)) {
  cirn_config$model_cores <- cirn_config$model_chains
  message("model_cores = NULL; using model_cores = model_chains = ", cirn_config$model_cores, ".")
}

if (isTRUE(cirn_config$run_pairwise) && cirn_config$model_cores > 2) {
  warning(
    "This run uses run_pairwise = TRUE and model_cores = ", cirn_config$model_cores,
    ". If OMP pthread errors occur, restart R and set model_cores = 1 or 2.",
    call. = FALSE
  )
}

default_analysis_vars <- setdiff(names(my_data), time_col)
default_analysis_vars <- default_analysis_vars[
  vapply(my_data[default_analysis_vars], is.numeric, logical(1))
]

if (length(default_analysis_vars) == 0) {
  stop("No numeric non-time variables were found in the input data.")
}

# Keep aliases so plotting, reporting, and validation sections follow the
# same central configuration. NULL targets/predictors mean all numeric
# non-time variables.

targets <- if (!is.null(cirn_config$targets)) {
  cirn_config$targets
} else {
  default_analysis_vars
}

predictors <- if (!is.null(cirn_config$predictors)) {
  cirn_config$predictors
} else {
  default_analysis_vars
}

representation_mode <- cirn_config$representation_mode
lag_units <- cirn_config$lag_units
run_pairwise <- cirn_config$run_pairwise
debug <- cirn_config$debug
assign_debug_to_global <- cirn_config$assign_debug_to_global
vars <- unique(c(targets, predictors))
if (length(vars) == 0) {
  stop("No variables available for CIRN plotting or inference.")
}
representative_var <- vars[min(2, length(vars))]


# --------------------------------------------------
# 8.3 Pre-CIRN visual diagnostics
# --------------------------------------------------

# These figures are generated before model fitting so the analyst can
# inspect the data and derivative construction before interpreting CIRN
# edges. Derivative plots use the same smoothing, grid, and outlier options
# specified in cirn_config.

raw_timeseries_plot <- plot_raw_data_all_vars(
  df = my_data,
  vars = vars,
  time_col = time_col
)
raw_timeseries_plot

derivative_order_plot <- plot_all_vars_by_derivative_order(
  df = my_data,
  vars = vars,
  time_col = time_col,
  points_per_interval = cirn_config$points_per_interval,
  spar = cirn_config$spar,
  outlier_method = cirn_config$outlier_method,
  outlier_thresh = cirn_config$outlier_thresh,
  outlier_action = cirn_config$outlier_action
)
derivative_order_plot

representative_derivative_plot <- plot_derivative_example(
  df = sim_df,
  var = representative_var,
  time_col = time_col,
  points_per_interval = cirn_config$points_per_interval,
  spar = cirn_config$spar,
  outlier_method = cirn_config$outlier_method,
  outlier_thresh = cirn_config$outlier_thresh,
  outlier_action = cirn_config$outlier_action
)
representative_derivative_plot

# Optional: inspect each variable individually.
# plot_all_derivatives(
#   df = sim_df,
#   vars = vars,
#   time_col = time_col,
#   points_per_interval = cirn_config$points_per_interval,
#   spar = cirn_config$spar,
#   outlier_method = cirn_config$outlier_method,
#   outlier_thresh = cirn_config$outlier_thresh,
#   outlier_action = cirn_config$outlier_action
# )


# --------------------------------------------------
# 8.4 Run full CIRN network inference
# --------------------------------------------------

# This step executes the complete CIRN pipeline:
#   • Derivative computation
#   • Directional classification from raw target derivatives
#   • Adaptive minimal-jitter response encoding when enabled and needed
#   • Predictor lagging and standardization
#   • Bayesian logistic regression
#   • Posterior-based edge filtering
#
# Runtime reported here covers CIRN inference only. It excludes plotting,
# optional validation, and downstream summary table construction.

inference_config <- cirn_config[
  setdiff(names(cirn_config), cirn_non_inference_config_keys())
]

CIRN <- system.time(
  res <- do.call(
    infer_network,
    c(
      list(
        df = sim_df,
        time_col = time_col
      ),
      inference_config
    )
  )
)

message("CIRN inference runtime: ", round(CIRN["elapsed"], 2), " seconds")


# --------------------------------------------------
# 8.5 Extract CIRN results and sanity-check tables
# --------------------------------------------------

# Primary output:
#   • edges = multivariable CIRN edge table
#       - sublevel edges form the primary Full CIRN network
#       - all-predictors edges are retained for consistency checks
#   • pairwise_edges = optional pairwise edge table
#   • diagnostics_table = convergence and sampling summaries
#   • effective_sample_size_table = target-specific usable sample-size summary

edges <- res$edges
pairwise_edges <- res$pairwise$edges
edges_combined <- res$edges_combined
all_coefficients_combined <- res$all_coefficients_combined
diagnostics_table <- res$diagnostics
effective_sample_size_table <- summarize_cirn_effective_sample_size(res)
vif_group_table <- res$vif_group
vif_pair_table <- res$vif_pairs
debug_tables <- res$debug

edge_consistency_table <- summarize_edge_consistency(
  edges = edges,
  pairwise_edges = pairwise_edges,
  collapse_derivative_terms = FALSE
)

print(edges)
print(pairwise_edges)
print(all_coefficients_combined)
print(edge_consistency_table)
print(diagnostics_table)
print(effective_sample_size_table)
print(vif_group_table)
print(vif_pair_table)

# If debug = TRUE, inspect debug_tables[["X"]], debug_tables[["Y"]], etc.
# If assign_debug_to_global = TRUE, the same tables also appear as
# debug_full_X, debug_full_Y, etc. in the Global Environment.


# --------------------------------------------------
# 8.6 Extract representative models for diagnostics
# --------------------------------------------------

# These models are used only for posterior and MCMC diagnostic plots.
# Edge inclusion has already been decided by the 95% HDI rule.

available_targets <- names(res$models)[
  vapply(
    res$models,
    function(model_list) {
      any(vapply(model_list, function(model) inherits(model, "brmsfit"), logical(1)))
    },
    logical(1)
  )
]

models_by_target <- purrr::map(
  available_targets,
  function(this_target) {
    target_models <- list(
      original = res$models[[this_target]][["original"]],
      first_derivative = res$models[[this_target]][["first_derivative"]],
      second_derivative = res$models[[this_target]][["second_derivative"]],
      all_predictors = res$models[[this_target]][["all_predictors"]]
    )
    
    target_models[
      vapply(target_models, function(model) inherits(model, "brmsfit"), logical(1))
    ]
  }
)
names(models_by_target) <- available_targets
models_by_target <- models_by_target[
  vapply(models_by_target, length, integer(1)) > 0
]

target_name <- if (length(available_targets) > 0) {
  available_targets[1]
} else {
  NA_character_
}

selected_models <- if (!is.na(target_name)) {
  list(
    original = res$models[[target_name]][["original"]],
    first_derivative = res$models[[target_name]][["first_derivative"]],
    second_derivative = res$models[[target_name]][["second_derivative"]],
    all_predictors = res$models[[target_name]][["all_predictors"]]
  )
} else {
  list()
}

selected_models <- selected_models[
  !vapply(selected_models, is.null, logical(1))
]

if (length(selected_models) > 0) {
  purrr::walk(
    selected_models,
    ~ stopifnot(inherits(.x, "brmsfit"))
  )
} else {
  warning(
    "No fitted brms models are available for posterior or MCMC diagnostic plots.",
    call. = FALSE
  )
}

if (length(models_by_target) > 0) {
  message(
    "MCMC and posterior diagnostic plots will be generated for target(s): ",
    paste(names(models_by_target), collapse = ", ")
  )
}

model_original <- selected_models[["original"]]
model_first_derivative <- selected_models[["first_derivative"]]
model_second_derivative <- selected_models[["second_derivative"]]
model_all_predictors <- selected_models[["all_predictors"]]


# --------------------------------------------------
# 8.7 Structural sanity check: symmetric edges
# --------------------------------------------------

# Symmetric edges are not errors. They may indicate reciprocal feedback
# or coupled dynamics and should be interpreted in context.

symmetric_edges <- edges %>%
  filter(term != target) %>%
  dplyr::select(term, target) %>%
  inner_join(
    edges %>%
      filter(term != target) %>%
      select(term = target, target = term),
    by = c("term", "target")
  ) %>%
  distinct()

if (nrow(symmetric_edges) > 0) {
  message("Symmetric (bidirectional) interactions detected:")
  print(symmetric_edges)
} else {
  message("No symmetric (bidirectional) interactions detected.")
}


# --------------------------------------------------
# 8.8 Optional ground-truth validation
# --------------------------------------------------

# Use only for synthetic simulations or benchmark systems with a known
# signed adjacency matrix. Leave run_ground_truth_validation = FALSE for
# observational case studies without known ground truth.

evaluation_metrics <- NULL
evaluation_table_long <- tibble::tibble()

cirn_summary_table <- tibble(
  Metric = "CIRN inference runtime (seconds)",
  Value = round(CIRN["elapsed"], 2)
)

if (isTRUE(cirn_config$run_ground_truth_validation)) {
  
  if (is.null(cirn_config$true_adj)) {
    stop(
      "run_ground_truth_validation = TRUE requires cirn_config$true_adj. ",
      "Provide a signed adjacency matrix with +1 activation, -1 inhibition, and 0 no edge."
    )
  } else {
    true_adj <- cirn_config$true_adj
  }
  
  if (!is.matrix(true_adj) || is.null(rownames(true_adj)) || is.null(colnames(true_adj))) {
    stop("true_adj must be a signed adjacency matrix with row and column names.")
  }
  
  evaluation_metrics <- evaluate_representation_agnostic(
    true_adj,
    edges
  )
  
  evaluation_table_long <- metrics_to_long_table(
    evaluation_metrics
  )
  
  print(evaluation_table_long)
  
  cirn_summary_table <- tibble(
    Metric = c(
      "True Positives (TP)",
      "False Positives (FP)",
      "False Negatives (FN)",
      "True Negatives (TN)",
      "Wrong-Sign Recoveries",
      "Precision",
      "Recall",
      "Specificity",
      "Balanced Accuracy",
      "F1 score",
      "Matthews Correlation Coefficient (MCC)",
      "Structural Hamming Distance (edge presence)",
      "Signed Structural Hamming Distance",
      "CIRN inference runtime (seconds)"
    ),
    Value = c(
      evaluation_metrics$TP,
      evaluation_metrics$FP,
      evaluation_metrics$FN,
      evaluation_metrics$TN,
      evaluation_metrics$sign_errors,
      round(evaluation_metrics$precision, 2),
      round(evaluation_metrics$recall, 2),
      round(evaluation_metrics$specificity, 2),
      round(evaluation_metrics$balanced_accuracy, 2),
      round(evaluation_metrics$f1, 2),
      round(evaluation_metrics$mcc, 2),
      evaluation_metrics$shd,
      evaluation_metrics$signed_shd,
      round(CIRN["elapsed"], 2)
    )
  )
  
} else {
  message("Ground-truth validation skipped. Set run_ground_truth_validation = TRUE only for benchmark data.")
}

print(cirn_summary_table)


# --------------------------------------------------
# 8.9 Optional Objective: sensitivity analysis
# --------------------------------------------------

# This block operationalizes robustness analysis for research objectives.
# It is disabled by default because it reruns CIRN many times.
# When enabled, it generates:
#   • CIRN_sensitivity_runs.csv
#   • CIRN_sensitivity_edges.csv
#   • CIRN_sensitivity_effective_sample_size.csv
#   • CIRN_sensitivity_edge_stability.csv
#   • CIRN_sensitivity_feature_edge_stability.csv
#   • CIRN_sensitivity_ground_truth_metrics.csv, when true_adj is supplied

sensitivity_results <- NULL
sensitivity_plan <- tibble::tibble()
sensitivity_runs <- tibble::tibble()
sensitivity_edges <- tibble::tibble()
sensitivity_diagnostics <- tibble::tibble()
sensitivity_effective_sample_size <- tibble::tibble()
sensitivity_edge_stability <- tibble::tibble()
sensitivity_feature_edge_stability <- tibble::tibble()
sensitivity_ground_truth_metrics <- tibble::tibble()

if (isTRUE(cirn_config$run_sensitivity_analysis)) {
  
  sensitivity_truth <- if (isTRUE(cirn_config$run_ground_truth_validation)) {
    cirn_config$true_adj
  } else {
    NULL
  }
  
  sensitivity_results <- run_cirn_sensitivity_analysis(
    df = sim_df,
    time_col = time_col,
    base_config = cirn_config,
    sensitivity_plan = cirn_config$sensitivity_plan,
    sensitivity_inference_scope = cirn_config$sensitivity_inference_scope,
    true_adj = sensitivity_truth,
    output_dir = output_dir,
    save_outputs = cirn_config$sensitivity_save_outputs,
    show_progress = cirn_config$sensitivity_show_progress,
    progress_bar = cirn_config$sensitivity_progress_bar
  )
  
  sensitivity_plan <- sensitivity_results$plan
  sensitivity_runs <- sensitivity_results$runs
  sensitivity_edges <- sensitivity_results$edges
  sensitivity_diagnostics <- sensitivity_results$diagnostics
  sensitivity_effective_sample_size <- sensitivity_results$effective_sample_size
  sensitivity_edge_stability <- sensitivity_results$edge_stability
  sensitivity_feature_edge_stability <- sensitivity_results$feature_edge_stability
  sensitivity_ground_truth_metrics <- sensitivity_results$ground_truth_metrics
  
  print(sensitivity_runs)
  print(sensitivity_effective_sample_size)
  print(sensitivity_edge_stability)
  print(sensitivity_feature_edge_stability)
  
} else {
  message(
    "Objective: sensitivity analysis skipped. ",
    "Set run_sensitivity_analysis = TRUE when evaluating robustness, ",
    "effective sample size, and edge stability."
  )
}


################################################################################
################################################################################


# ======================================
# 9. CIRN Visualizations After Inference
# ======================================

# Purpose:
#   • Display the posterior-supported CIRN regulatory network
#   • Inspect sublevel networks and pairwise diagnostics
#   • Examine posterior uncertainty and MCMC convergence after fitting
#
# Raw and derivative diagnostic plots are intentionally generated before
# inference in Section 8.3.

# ------------------------------------------------------------------
# 9.1 Adaptive minimal-jitter diagnostics
# ------------------------------------------------------------------

# Purpose:
#   • Generated only for targets whose response construction used
#     adaptive minimal jitter.
#   • Confirms that the perturbation is small relative to the target
#     trajectory and documents the class balance created by jitter.
#   • Saved as a target-specific diagnostic beside the input data file.

jittered_targets <- diagnostics_table %>%
  dplyr::filter(!is.na(.data$jitter_used) & .data$jitter_used) %>%
  dplyr::distinct(.data$target) %>%
  dplyr::pull("target")

if (length(jittered_targets) > 0) {
  purrr::walk(
    jittered_targets,
    function(this_target) {
      jitter_plot <- plot_adaptive_jitter_diagnostic(
        df = sim_df,
        target = this_target,
        time_col = time_col,
        points_per_interval = cirn_config$points_per_interval,
        spar = cirn_config$spar,
        outlier_method = cirn_config$outlier_method,
        outlier_thresh = cirn_config$outlier_thresh,
        outlier_action = cirn_config$outlier_action,
        response_eps = cirn_config$response_eps,
        jitter_min_class_count = cirn_config$jitter_min_class_count,
        jitter_scale_grid = cirn_config$jitter_scale_grid,
        jitter_scale_basis = cirn_config$jitter_scale_basis,
        seed = cirn_config$seed
      )
      
      print(jitter_plot)
      
      output_file <- file.path(
        output_dir,
        paste0(
          "Target_",
          gsub("[^A-Za-z0-9]+", "_", this_target),
          "_Adaptive_Jitter_Diagnostic.png"
        )
      )
      
      tryCatch(
        {
          ggplot2::ggsave(
            filename = output_file,
            plot = jitter_plot,
            width = 10,
            height = 8,
            dpi = 300
          )
          message("Saved adaptive jitter diagnostic: ", output_file)
        },
        error = function(e) {
          warning(
            "Could not save adaptive jitter diagnostic for target ",
            this_target,
            ": ",
            conditionMessage(e)
          )
        }
      )
    }
  )
} else {
  message("No adaptive jitter diagnostic plots generated; no fitted target used adaptive jitter.")
}

# ------------------------------------------------------------------
# 9.2 Inferred regulatory network
# ------------------------------------------------------------------

# Purpose:
#   • Visualize the signed and directed regulatory structure
#     inferred by CIRN
#   • Encode regulatory direction, sign, and effect magnitude
#   • Provide a compact graphical summary of CIRN results
#
# This section produces two complementary network visualizations:
#   (i)   A single, publication-grade network for the main text
#   (ii)  A multi-panel sublevel/joint view for diagnostics

# Output 1: Full CIRN
#   • Aggregates posterior-supported edges across the three CIRN
#     sublevels only: state, first derivative, and second derivative.
#   • Does not merge all-predictors or pairwise edges.
#   • Intended as the primary graphical result of the algorithm.

cirn_output_full <- plot_network_main(
  edges,
  representation_mode = representation_mode
)

cirn_output_full

# Output 2: Individual CIRN
#   • Displays one network panel per fitted CIRN representation.
#   • In "both" mode, this includes original-state, first-derivative,
#     second-derivative, and all-predictors panels.
#   • Empty panels are retained and labeled so users can see that a
#     representation was fitted but produced no HDI-supported edges.


cirn_output_individual <- plot_cirn_panels(
  edges,
  representation_mode = representation_mode
)

cirn_output_individual

# Optional pairwise network and consistency map.
# These appear only when run_pairwise = TRUE produces retained edges.
if (nrow(pairwise_edges) > 0) {
  cirn_output_pairwise <- plot_pairwise_network(
    pairwise_edges,
    representation_mode = representation_mode
  )
  
  cirn_output_pairwise
}

if (nrow(edge_consistency_table) > 0) {
  cirn_output_edge_consistency <- plot_edge_consistency(
    edge_consistency_table,
    show_regulation_label = FALSE,
    encode_strength = TRUE,
    strength_scale = "within_mode"
  )
  
  cirn_output_edge_consistency
  
  edge_consistency_height <- min(
    14,
    max(4.8, 0.45 * nrow(edge_consistency_table) + 2.6)
  )
  
  edge_consistency_png <- file.path(
    output_dir,
    "CIRN_Edge_Consistency_Across_Modes.png"
  )
  edge_consistency_pdf <- file.path(
    output_dir,
    "CIRN_Edge_Consistency_Across_Modes.pdf"
  )
  
  tryCatch(
    {
      ggplot2::ggsave(
        filename = edge_consistency_png,
        plot = cirn_output_edge_consistency,
        width = 10,
        height = edge_consistency_height,
        dpi = 300
      )
      ggplot2::ggsave(
        filename = edge_consistency_pdf,
        plot = cirn_output_edge_consistency,
        width = 10,
        height = edge_consistency_height
      )
      message("Saved CIRN edge-consistency figure: ", edge_consistency_png)
      message("Saved CIRN edge-consistency figure: ", edge_consistency_pdf)
    },
    error = function(e) {
      warning(
        "Could not save CIRN edge-consistency figure: ",
        conditionMessage(e),
        call. = FALSE
      )
    }
  )
} else {
  cirn_output_edge_consistency <- NULL
  message("No edge-consistency figure generated; no retained edges were available to compare.")
}

# Optional combined viewer containing both official CIRN network outputs:
# plot_cirn_outputs(edges, representation_mode = representation_mode)

# ------------------------------------------------------------------
# 9.3 Posterior distributions of CIRN coefficients (ω)
# ------------------------------------------------------------------

# Purpose:
#   • Examine posterior uncertainty of inferred regulatory effects
#   • Confirm stability of regulatory sign
#   • Compare relative magnitudes across predictors
#
# These plots emphasize that CIRN inference is uncertainty-aware
# and posterior-based, not driven by point estimates or p-values.
#
# NOTE:
#   • These figures are exploratory and diagnostic
#   • Typically placed in the appendix or supplementary material


# Posterior distributions for all coefficients:
#   • Shown for each fitted CIRN representation
#   • Intercepts are excluded to focus on regulatory effects

plot_model_posteriors_if_available <- function(model) {
  coef_names <- rownames(brms::fixef(model))
  coef_names <- coef_names[coef_names != "Intercept"]
  
  if (length(coef_names) == 0) {
    return(invisible(NULL))
  }
  
  suppressWarnings(
    plot_all_posterior_effects(
      model = model,
      terms = coef_names
    )
  )
}

posterior_effect_suffix <- c(
  original = "Original",
  first_derivative = "1stDeriv",
  second_derivative = "2ndDeriv",
  all_predictors = "AllPredictors"
)

if (length(models_by_target) > 0) {
  purrr::iwalk(
    models_by_target,
    function(target_models, this_target) {
      purrr::iwalk(
        target_models,
        function(model, representation_name) {
          posterior_plot <- plot_model_posteriors_if_available(model)
          
          if (is.null(posterior_plot)) {
            return(invisible(NULL))
          }
          
          print(posterior_plot)
          
          output_suffix <- posterior_effect_suffix[[representation_name]]
          if (is.null(output_suffix)) {
            output_suffix <- gsub("[^A-Za-z0-9]+", "_", representation_name)
          }
          
          output_file <- file.path(
            output_dir,
            paste0(
              "Target_",
              gsub("[^A-Za-z0-9]+", "_", this_target),
              "_Posterior_Effects_",
              output_suffix,
              ".png"
            )
          )
          
          tryCatch(
            {
              ggplot2::ggsave(
                filename = output_file,
                plot = posterior_plot,
                width = 10,
                height = 5,
                dpi = 300
              )
              message("Saved posterior effect diagnostic: ", output_file)
            },
            error = function(e) {
              warning(
                "Could not save posterior effect diagnostic for target ",
                this_target,
                ", representation ",
                representation_name,
                ": ",
                conditionMessage(e)
              )
            }
          )
        }
      )
    }
  )
}

# Cross-representation posterior comparison for the selected target
#   • Highlights how inferred regulation depends on
#     state-based vs derivative-based predictors

if (length(models_by_target) > 0) {
  purrr::iwalk(
    models_by_target,
    function(target_models, this_target) {
      representation_posterior_plot <- plot_all_cirn_posteriors(target_models)
      print(representation_posterior_plot)
      
      output_file <- file.path(
        output_dir,
        paste0(
          "Target_",
          gsub("[^A-Za-z0-9]+", "_", this_target),
          "_All_Representation_Posteriors.png"
        )
      )
      
      tryCatch(
        {
          ggplot2::ggsave(
            filename = output_file,
            plot = representation_posterior_plot,
            width = 10,
            height = 8,
            dpi = 300
          )
          message("Saved cross-representation posterior diagnostic: ", output_file)
        },
        error = function(e) {
          warning(
            "Could not save cross-representation posterior diagnostic for target ",
            this_target,
            ": ",
            conditionMessage(e)
          )
        }
      )
    }
  )
}

# ------------------------------------------------------------------
# 9.4 Diagnostics summary (model-level)
# ------------------------------------------------------------------

# Purpose:
#   • Provide a compact overview of convergence and sampling quality
#   • Summarize R-hat, ESS, and predictive diagnostics
#
# These summaries support reporting transparency and may appear
# in the main text or supplementary material depending on venue.

plot_diagnostics_summary(diagnostics_table)

diagnostics_summary_file <- file.path(output_dir, "CIRN_Diagnostics_Summary.png")
diagnostics_device_open <- FALSE
tryCatch(
  {
    grDevices::png(
      filename = diagnostics_summary_file,
      width = 10,
      height = 5,
      units = "in",
      res = 300
    )
    diagnostics_device_open <- TRUE
    plot_diagnostics_summary(diagnostics_table)
    grDevices::dev.off()
    diagnostics_device_open <- FALSE
    message("Saved diagnostics summary: ", diagnostics_summary_file)
  },
  error = function(e) {
    if (isTRUE(diagnostics_device_open) && !is.null(grDevices::dev.list())) {
      grDevices::dev.off()
    }
    warning(
      "Could not save diagnostics summary: ",
      conditionMessage(e)
    )
  }
)

# ------------------------------------------------------------------
# 9.5 MCMC diagnostics (quick checks)
# ------------------------------------------------------------------
# Purpose:
#   • Perform rapid visual sanity checks using brms defaults
#   • Detect obvious sampling pathologies (divergence, poor mixing)
#
# IMPORTANT:
#   • These plots are NOT publication-grade
#   • Intended solely for development-time validation

mcmc_hist_trace_suffix <- c(
  original = "Original",
  first_derivative = "1stDeriv",
  second_derivative = "2ndDeriv",
  all_predictors = "AllPredictors"
)

if (length(models_by_target) > 0) {
  purrr::iwalk(
    models_by_target,
    function(target_models, this_target) {
      purrr::iwalk(
        target_models,
        function(model, representation_name) {
          mcmc_plot <- suppressWarnings(plot_mcmc_hist_trace(model))
          
          message(
            "Displaying MCMC histogram/trace diagnostic for target = ",
            this_target,
            ", representation = ",
            representation_name
          )
          print(mcmc_plot)
          
          output_suffix <- mcmc_hist_trace_suffix[[representation_name]]
          if (is.null(output_suffix)) {
            output_suffix <- gsub("[^A-Za-z0-9]+", "_", representation_name)
          }
          
          output_file <- file.path(
            output_dir,
            paste0(
              "Target_",
              gsub("[^A-Za-z0-9]+", "_", this_target),
              "_Hist_Trace_",
              output_suffix,
              ".png"
            )
          )
          
          tryCatch(
            {
              ggplot2::ggsave(
                filename = output_file,
                plot = mcmc_plot,
                width = 10,
                height = 8,
                dpi = 300
              )
              message("Saved MCMC histogram/trace diagnostic: ", output_file)
            },
            error = function(e) {
              warning(
                "Could not save MCMC histogram/trace diagnostic for target ",
                this_target,
                ", representation ",
                representation_name,
                ": ",
                conditionMessage(e)
              )
            }
          )
        }
      )
    }
  )
}


# ------------------------------------------------------------------
# 9.6 MCMC trace plots with burn-in (thesis-quality diagnostics)
# ------------------------------------------------------------------

# Purpose:
#   • Explicitly distinguish warm-up (burn-in) from sampling phase
#   • Visually assess convergence and mixing across chains
#   • Compare MCMC behavior across CIRN representations
#
# These figures provide detailed reviewer-facing evidence
# of sampling quality and are typically placed in the appendix.

# All parameters for a given target variable:
#   • Generates one trace plot per coefficient
#   • Includes all chains and explicit burn-in cutoff
#   • Used for deep convergence assessment

if (length(models_by_target) > 0) {
  purrr::iwalk(
    models_by_target,
    function(target_models, this_target) {
      burnin_trace_plot <- suppressWarnings(
        plot_all_traces_with_burnin(
          models_list = target_models,
          target_var = this_target
        )
      )
      
      print(burnin_trace_plot)
      
      output_file <- file.path(
        output_dir,
        paste0(
          "Target_",
          gsub("[^A-Za-z0-9]+", "_", this_target),
          "_MCMC_Traces_With_Burnin.png"
        )
      )
      
      tryCatch(
        {
          ggplot2::ggsave(
            filename = output_file,
            plot = burnin_trace_plot,
            width = 12,
            height = 8,
            dpi = 300
          )
          message("Saved MCMC trace diagnostic with burn-in: ", output_file)
        },
        error = function(e) {
          warning(
            "Could not save MCMC trace diagnostic with burn-in for target ",
            this_target,
            ": ",
            conditionMessage(e)
          )
        }
      )
    }
  )
}


################################################################################
################################################################################


# ==================================================================
# (Optional) Save figures for thesis / manuscript submission
# ==================================================================

# This section provides reproducible export commands for selected
# figures generated in Sections 8.3 and 9.
#
# IMPORTANT:
#   • Do NOT uncomment everything by default.
#   • Selectively export only figures required for:
#       – Main text
#       – Appendix / Supplementary material
#   • File names are stable and journal-ready.
#
# NOTE:
#   Figure numbering (Fig. 1, Fig. 2, Fig. S1–S6) is illustrative
#   and may be adjusted to match journal or thesis formatting.
#
# Recommendation:
#   Keep this section commented until final manuscript preparation.

# ------------------------------------------------------------------
# Main-text figures (FINAL)
# ------------------------------------------------------------------
# Only two figures are intended for the main text:
#   Fig. 1 – Core scientific result (inferred CIRN network)
#   Fig. 2 – Methodological illustration (state + derivatives)
#
# All other visualizations are supplementary / appendix material.

# ---------------------------------------------------------------
# Fig. 1: Inferred CIRN regulatory network (MAIN RESULT)
# ---------------------------------------------------------------
# Purpose:
#   • Primary output of the CIRN algorithm
#   • Displays signed, directed regulatory interactions
#   • Edge width ∝ |ω_ij| (posterior mean effect size)
#
# pdf("Fig1_CIRN_Network_Main.pdf", width = 7, height = 7)
# plot_network_main(edges, representation_mode = representation_mode)
# dev.off()

# ---------------------------------------------------------------
# Fig. 2: State variable and temporal derivatives (METHOD FIGURE)
# ---------------------------------------------------------------
# Purpose:
#   • Illustrates how CIRN transforms raw dynamics into predictors
#   • Shows x(t), dx/dt, and d²x/dt² for one representative variable
#   • Included for methodological clarity (not an inferred result)
#
# pdf("Fig2_State_and_Derivatives_Method.pdf", width = 7, height = 9)
# plot_derivative_example(
#   df = sim_df,
#   var = vars[1],
#   time_col = time_col,
#   points_per_interval = cirn_config$points_per_interval,
#   spar = cirn_config$spar,
#   outlier_method = cirn_config$outlier_method,
#   outlier_thresh = cirn_config$outlier_thresh,
#   outlier_action = cirn_config$outlier_action
# )
# dev.off()



# ------------------------------------------------------------------
# Appendix / supplementary figures
# ------------------------------------------------------------------

# ---------------------------------------------------------------
# Fig. S1: Time series and temporal derivatives (example variable)
# ---------------------------------------------------------------

# pdf("FigS1_TimeSeries_Derivatives_Example.pdf", width = 7, height = 9)
# plot_derivative_example(
#   df = sim_df,
#   var = vars[1],
#   time_col = time_col,
#   points_per_interval = cirn_config$points_per_interval,
#   spar = cirn_config$spar,
#   outlier_method = cirn_config$outlier_method,
#   outlier_thresh = cirn_config$outlier_thresh,
#   outlier_action = cirn_config$outlier_action
# )
# dev.off()

# ---------------------------------------------------------------
# Fig. S2: CIRN networks by predictor representation
#          (original, first derivative, second derivative,
#           optional all-predictors model)
# ---------------------------------------------------------------

# pdf("FigS2_CIRN_Representation_Panels.pdf", width = 12, height = 4)
# plot_cirn_panels(
#   edges,
#   predictor_sets = names(selected_models),
#   representation_mode = representation_mode
# )
# dev.off()

# ---------------------------------------------------------------
# Fig. S3: Posterior distributions of CIRN coefficients (ω)
#          Single representation (original)
# ---------------------------------------------------------------

# pdf("FigS3_Posterior_Coefficients_Original.pdf", width = 9, height = 4)
# plot_all_posterior_effects(
#   model = model_original,
#   terms = coef_names_original
# )
# dev.off()

# ---------------------------------------------------------------
# Fig. S4: Posterior comparison across CIRN representations
# ---------------------------------------------------------------

# pdf(paste0("FigS4_CIRN_Representation_Posteriors_", target_name, ".pdf"), width = 10, height = 4)
# plot_all_cirn_posteriors(res$models[[target_name]])
# dev.off()

# ---------------------------------------------------------------
# Fig. S5: Model diagnostics summary (ESS + LOO)
# ---------------------------------------------------------------

# pdf("FigS5_Diagnostics_Summary.pdf", width = 10, height = 4)
# plot_diagnostics_summary(diagnostics_table)
# dev.off()

# ---------------------------------------------------------------
# Fig. S6: Full MCMC trace plots with burn-in
#          Reviewer-facing convergence diagnostics
# ---------------------------------------------------------------

# pdf(paste0("FigS6_MCMC_Traces_With_Burnin_", target_name, ".pdf"), width = 12, height = 8)
# suppressWarnings(
#   plot_all_traces_with_burnin(
#     models_list = res$models[[target_name]],
#     target_var = target_name
#   )
# )
# dev.off()


################################################################################
################################################################################


# ==================================================================
# Save CIRN result-table compendium
# ==================================================================

# CSV files do not support multiple worksheet tabs. Therefore, this section
# saves each non-empty CIRN result table as an individual CSV file and, when
# the optional openxlsx package is installed, also saves one Excel workbook
# with one worksheet per table. This provides both machine-readable CSV files
# and a tabbed analysis compendium for review and reporting.

sanitize_output_name <- function(x, max_chars = 60) {
  x <- gsub("[^A-Za-z0-9_]+", "_", as.character(x))
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  ifelse(nchar(x) > max_chars, substr(x, 1, max_chars), x)
}

sanitize_excel_sheet_name <- function(x) {
  x <- gsub("[\\\\/\\?\\*\\[\\]:]", "_", as.character(x))
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  ifelse(nchar(x) > 31, substr(x, 1, 31), x)
}

is_non_empty_table <- function(x) {
  inherits(x, "data.frame") && nrow(x) > 0 && ncol(x) > 0
}

settings_to_table <- function(settings_list) {
  if (is.null(settings_list) || length(settings_list) == 0) {
    return(tibble::tibble())
  }
  
  tibble::tibble(
    setting = names(settings_list),
    value = vapply(
      settings_list,
      function(value) {
        if (length(value) == 0 || is.null(value)) {
          return(NA_character_)
        }
        paste(as.character(value), collapse = "; ")
      },
      character(1)
    )
  )
}

write_cirn_result_compendium <- function(result_tables,
                                         output_dir,
                                         basename = "CIRN_Results_Compendium") {
  
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  
  non_empty_tables <- result_tables[
    vapply(result_tables, is_non_empty_table, logical(1))
  ]
  
  if (length(non_empty_tables) == 0) {
    message("No non-empty CIRN result tables were available to save.")
    return(invisible(list(
      csv_dir = NA_character_,
      csv_files = character(),
      xlsx_file = NA_character_,
      saved_tables = character()
    )))
  }
  
  csv_dir <- file.path(output_dir, paste0(basename, "_CSV"))
  dir.create(csv_dir, recursive = TRUE, showWarnings = FALSE)
  
  csv_files <- purrr::imap_chr(
    non_empty_tables,
    function(tbl, table_name) {
      csv_file <- file.path(
        csv_dir,
        paste0(sanitize_output_name(table_name), ".csv")
      )
      utils::write.csv(tbl, csv_file, row.names = FALSE)
      csv_file
    }
  )
  
  message(
    "Saved ",
    length(csv_files),
    " non-empty CIRN result table(s) as CSV files in: ",
    csv_dir
  )
  
  xlsx_file <- NA_character_
  if (requireNamespace("openxlsx", quietly = TRUE)) {
    workbook <- openxlsx::createWorkbook()
    
    used_sheet_names <- character()
    purrr::iwalk(
      non_empty_tables,
      function(tbl, table_name) {
        sheet_name <- sanitize_excel_sheet_name(table_name)
        if (sheet_name == "") {
          sheet_name <- "table"
        }
        
        original_sheet_name <- sheet_name
        suffix <- 1L
        while (sheet_name %in% used_sheet_names) {
          suffix_text <- paste0("_", suffix)
          sheet_name <- paste0(
            substr(original_sheet_name, 1, 31 - nchar(suffix_text)),
            suffix_text
          )
          suffix <- suffix + 1L
        }
        used_sheet_names <<- c(used_sheet_names, sheet_name)
        
        openxlsx::addWorksheet(workbook, sheet_name)
        openxlsx::writeData(workbook, sheet_name, tbl)
        openxlsx::freezePane(workbook, sheet_name, firstRow = TRUE)
      }
    )
    
    xlsx_file <- file.path(output_dir, paste0(basename, ".xlsx"))
    openxlsx::saveWorkbook(workbook, xlsx_file, overwrite = TRUE)
    message("Saved tabbed CIRN result workbook: ", xlsx_file)
  } else {
    message(
      "Optional Excel workbook was not created because package 'openxlsx' ",
      "is not installed. CSV files were still saved."
    )
  }
  
  invisible(list(
    csv_dir = csv_dir,
    csv_files = csv_files,
    xlsx_file = xlsx_file,
    saved_tables = names(non_empty_tables)
  ))
}

publication_edge_table <- make_publication_edge_table(
  edges_combined = edges_combined,
  edge_consistency = edge_consistency_table,
  predictor_order = predictors
)

if (nrow(publication_edge_table) > 0) {
  print(publication_edge_table)
}

publication_edge_table_latex <- write_publication_edge_table_latex(
  publication_table = publication_edge_table,
  output_dir = output_dir,
  filename = "CIRN_Publication_Edge_Table.tex",
  label = "tab:cirn_publication_edge_table"
)

publication_diagnostics_table <- make_publication_diagnostics_table(
  diagnostics_table = diagnostics_table,
  pairwise_diagnostics_table = res$pairwise$diagnostics,
  effective_sample_size_table = effective_sample_size_table,
  vif_group_table = vif_group_table,
  pairwise_vif_group_table = res$pairwise$vif_group
)

if (nrow(publication_diagnostics_table) > 0) {
  print(publication_diagnostics_table)
}

publication_tables_xlsx <- write_publication_tables_workbook(
  table1_edges = publication_edge_table,
  table2_diagnostics = publication_diagnostics_table,
  output_dir = output_dir,
  filename = "CIRN_Publication_Tables.xlsx"
)

all_coefficients_table <- make_all_coefficients_report_table(
  all_coefficients_combined = all_coefficients_combined,
  predictor_order = predictors
)

if (nrow(all_coefficients_table) > 0) {
  print(all_coefficients_table)
  
  all_coefficients_csv <- file.path(
    output_dir,
    "CIRN_All_Coefficients_Retained_and_Not_Retained.csv"
  )
  utils::write.csv(
    all_coefficients_table,
    all_coefficients_csv,
    row.names = FALSE
  )
  message("Saved all-coefficients retained/not-retained CSV: ", all_coefficients_csv)
} else {
  all_coefficients_csv <- NA_character_
  message("No all-coefficients table was saved because no fitted coefficients were available.")
}

cirn_result_tables <- list(
  cirn_summary = cirn_summary_table,
  settings = settings_to_table(res$settings),
  edges = edges,
  pairwise_edges = pairwise_edges,
  edges_combined = edges_combined,
  all_coefficients_combined = all_coefficients_combined,
  all_coefficients_table = all_coefficients_table,
  publication_edge_table = publication_edge_table,
  publication_diagnostics_table = publication_diagnostics_table,
  diagnostics = diagnostics_table,
  pairwise_diagnostics = res$pairwise$diagnostics,
  effective_sample_size = effective_sample_size_table,
  edge_consistency = edge_consistency_table,
  vif_group = vif_group_table,
  vif_pairs = vif_pair_table,
  pairwise_vif_group = res$pairwise$vif_group,
  pairwise_vif_pairs = res$pairwise$vif_pairs,
  evaluation_metrics = evaluation_table_long,
  sensitivity_plan = sensitivity_plan,
  sensitivity_runs = sensitivity_runs,
  sensitivity_edges = sensitivity_edges,
  sensitivity_diagnostics = sensitivity_diagnostics,
  sensitivity_effective_sample_size = sensitivity_effective_sample_size,
  sensitivity_edge_stability = sensitivity_edge_stability,
  sensitivity_feature_edge_stability = sensitivity_feature_edge_stability,
  sensitivity_ground_truth_metrics = sensitivity_ground_truth_metrics
)

cirn_compendium_files <- write_cirn_result_compendium(
  result_tables = cirn_result_tables,
  output_dir = output_dir,
  basename = "CIRN_Results_Compendium"
)


################################################################################
#############                       ~ END ~                        #############
################################################################################


# ================================
# Reproducibility record
# ================================

cat("\n================ SESSION INFO ================\n")
sessionInfo()

writeLines(
  capture.output(sessionInfo()),
  con = file.path(output_dir, "sessionInfo_CIRN.txt")
)


################################################################################
############################### END OF CIRN SCRIPT #############################
################################################################################
