################################################################################
############# CLASSIFICATION-BASED INFERENCE OF REGULATORY NETWORKS ############
#############                      ~ CIRN ~                         ############
################################################################################
#
# Author:
#   Giovannie M. Entero (Gio)
#
# Affiliation:
#   PhD in Applied Mathematics
#   University of the Philippines Los Baños (UPLB)
#
# Supervisor:
#   Prof. Jomar F. Rabajante, D.Sc.
#   Full Professor, Dean of UPLB Graduate School
#   University of the Philippines Los Baños (UPLB)
#
# Research Area:
#   Data-driven inference of regulatory networks
#   with applications to ecological and environmental systems
#
# Dissertation:
#   Classification-Based Inference of Regulatory Networks (CIRN):
#   Uncovering Interaction Link Types among Environmental Variables
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
# By combining temporal derivatives, enforced causality through
# lagging, and Bayesian uncertainty quantification, CIRN provides
# a flexible framework that is applicable across domains,
# including gene regulatory networks, ecological systems, and
# environmental dynamics—primary ambition.
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
#     temporal index — first column with "t" as its name.
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
#   • Light smoothing is applied to reduce noise amplification during
#     numerical differentiation.
#
#   Implemented in:
#     infer_for_target()  →  na.omit()
#     compute_derivatives(df, time_col, smooth = TRUE)
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
# (4) Binary encoding of directional change
# --------------------------------------------------------------
#   • For each target variable x_j, the sign of its first temporal
#     derivative dx_j/dt is encoded as a binary response variable:
#         Class = 1  (increasing, dx_j/dt > 0)
#         Class = 0  (decreasing, dx_j/dt < 0)
#   • Near-zero derivatives are excluded to avoid numerical ambiguity.
#
#   Implemented in:
#     infer_for_target()
#       → construction of Class from target_d1
#
#
# --------------------------------------------------------------
# (5) Causally informed feature construction
# --------------------------------------------------------------
#   • Predictor variables (state variables and their derivatives)
#     are temporally lagged by one time step.
#   • This enforces causal asymmetry by ensuring predictors precede
#     the observed directional change of the target.
#   • The target first derivative is excluded from its own predictor set
#     because dX is represented by X_d1; lagged target state and
#     second derivative predictors may still be used when declared.
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
#     the probability of an increasing state given lagged predictors.
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
#   • Separate Bayesian models may be fitted using three predictor
#     representations:
#         (i)   original state variables,
#         (ii)  first-order derivatives,
#         (iii) second-order derivatives.
#   • Optionally, one combined all-predictors model fits original,
#     first-derivative, and second-derivative predictors together
#     for d[target], excluding d[target] itself.
#   • No representation is privileged a priori; consistency across
#     separate and combined fits can be used to assess robustness.
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
#############                       ~ START ~                        ############
################################################################################


# ==============================================================
# Working directory
# ==============================================================
# Sets the root directory for all data inputs and outputs used
# in this dissertation. Update this path if the project is
# moved to a different machine or folder structure.

setwd('/Users/gio/Desktop/My Dissertation/Dissertation Focus/Main Version/Different Systems')


################################################################################
################################################################################


# ==============================================================
# 0. Clean session & reproducibility
# ==============================================================
# This section ensures that the analysis starts from a clean and
# reproducible R session. This is important for transparency,
# debugging, and exact replication of results.

# Remove all objects from the global environment
rm(list = ls())

# Run garbage collection to free memory from any previously loaded objects
gc()

# Set a fixed random seed to ensure reproducibility of all stochastic 
# components (e.g., Bayesian sampling)
set.seed(143)

# Close any open graphics devices to avoid plot conflicts
if (!is.null(dev.list())) dev.off()

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
#   "dplyr", "tidyr", "purrr", "broom", "stats",
#   "ggplot2", "igraph", "car",
#   "brms", "cmdstanr", "bayestestR", "bayesplot", "pROC"
# ))
#
# IMPORTANT (Stan users on macOS):
#   - Xcode Command Line Tools must be installed
#   - CmdStan must be installed via:
#       cmdstanr::install_cmdstan()
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
# Fail early if the SDK is not visible to R to avoid obscure
# compilation errors later during model fitting.

if (Sys.getenv("SDKROOT") == "") {
  stop(
    "SDKROOT not set. ",
    "Restart RStudio or ensure ~/.Renviron contains:\n",
    "SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
  )
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
  # "MAD" or "none"; # threshold for modified Z-score; # "remove" or "winsorize" or "keep"
  # change spar = NULL to control smoothing, e.g. spar = 0.3 to have a moderate smoothing
  
  # --------------------------------------------------------------
  # 2.1 Ensure numeric, sorted time
  # --------------------------------------------------------------
  
  time_raw <- df[[time_col]]
  
  if (inherits(time_raw, "Date") || inherits(time_raw, "POSIXt")) {
    time_raw <- as.numeric(time_raw)
  } else if (!is.numeric(time_raw)) {
    stop("time_col must be numeric, Date, or POSIXt.")
  }
  
  ord <- order(time_raw)
  df <- df[ord, ]
  time_raw <- time_raw[ord]
  
  var_names <- setdiff(names(df), time_col)
  
  # --------------------------------------------------------------
  # 2.2 Construct globally regularized time grid
  # --------------------------------------------------------------
  
  t_min <- min(time_raw)
  t_max <- max(time_raw)
  
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
              "⚠ Potential outliers detected in variable '%s': %d observation(s). Action = %s",
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
# 3. Standardization of State Variables
# ==============================================================
# This function standardizes all numeric predictors prior to
# Bayesian logistic regression, which is essential for:
#
#   • numerical stability of MCMC sampling
#   • comparability of regression coefficients (ω)
#   • preventing scale dominance by large-magnitude variables
#
# IMPORTANT DESIGN CHOICES:
#   • The response variable "Class" is NEVER standardized
#     (it is binary by construction).
#   • The time variable is excluded to preserve temporal structure.
#   • Additional variables may be excluded if needed.
#
# Standardization is performed using:
#     Z = (X - mean(X)) / sd(X)
#
# This ensures that posterior coefficients β_ij reflect
# relative regulatory influence rather than raw units.
#
# Arguments:
#   df            : data frame containing variables
#   time_col      : name of the time column (default = "t")
#   extra_exclude : optional vector of column names to exclude
#
# Output:
#   A data frame where eligible numeric columns are standardized.
# ==============================================================

standardize_except_class_time <- function(df, time_col = "t", extra_exclude = NULL) {
  
  # Identify numeric columns
  num_vars <- sapply(df, is.numeric)
  
  # --------------------------------------------------------------
  # 3.1 Exclusions
  # --------------------------------------------------------------
  
  # Do NOT standardize the binary response variable
  if ("Class" %in% names(df)) {
    num_vars["Class"] <- FALSE
  }
  
  # Do NOT standardize time (preserves temporal spacing)
  if (time_col %in% names(df)) {
    num_vars[time_col] <- FALSE
  }
  
  # Optionally exclude additional variables
  if (!is.null(extra_exclude)) {
    for (col in extra_exclude) {
      if (col %in% names(df)) {
        num_vars[col] <- FALSE
      }
    }
  }
  
  # --------------------------------------------------------------
  # 3.2 Apply standardization
  # --------------------------------------------------------------
  # scale() centers variables to mean 0 and variance 1
  
  df[num_vars] <- scale(df[num_vars])
  
  return(df)
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
#   P( sign(dx_j / dt) | x_i(t), dx_i/dt, d²x_i/dt² )
#
# via Bayesian logistic regression, where:
#   • The response is the DIRECTION of change of x_j
#   • Predictors are lagged state variables and derivatives
#   • Temporal causality is explicitly enforced
#
# The output consists of:
#   • Inferred directed, signed edges
#   • Posterior diagnostics
#   • Fitted Bayesian models (for visualization and diagnostics)
#
# DESIGN CHOICE:
# CIRN employs a single-step temporal lag to enforce causality
# while maintaining interpretability and tractability.
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
  regulation_type = character(),
  sign = numeric(),
  rel_strength = character(),
  hdi_lower95 = numeric(),
  hdi_upper95 = numeric(),
  eti_lower95 = numeric(),
  eti_upper95 = numeric(),
  method = character()
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
                             representation_mode = c("separate", "all_predictors", "both"),
                             lag_units = 1,
                             response_eps = 1e-6,
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
  if (is.null(loo_reloo)) {
    loo_reloo <- isTRUE(compute_loo) && isTRUE(loo_moment_match)
  }
  
  if (length(lag_units) != 1 || is.na(lag_units) || lag_units < 1 || lag_units != floor(lag_units)) {
    stop("lag_units must be a single positive integer.")
  }
  lag_units <- as.integer(lag_units)
  
  # --------------------------------------------------------------
  # 4.1 Data preparation
  # --------------------------------------------------------------
  # Remove incomplete rows BEFORE derivative computation
  # (avoids propagation of NA values)
  #df <- na.omit(df)
  
  # Compute smoothed variables and numerical derivatives
  #df_derivs <- compute_derivatives(df, time_col)
  
  # Name of the first derivative of the target variable
  target_d1 <- paste0(target, "_d1")
  
  # --------------------------------------------------------------
  # 4.2 Binary response construction (CIRN core idea)
  # --------------------------------------------------------------
  # The response variable encodes the LOCAL DIRECTION of change
  # of the target variable x_j.
  #
  # Class = 1  → x_j is increasing
  # Class = 0  → x_j is decreasing
  #
  # Near-zero derivatives are discarded to avoid numerical noise.
  
  eps <- response_eps  # tolerance for numerical stability
  
  df_derivs <- df_derivs %>%
    mutate(
      Class = case_when(
        .data[[target_d1]] >  eps ~ 1,
        .data[[target_d1]] < -eps ~ 0,
        TRUE ~ NA_real_
      )
    ) %>%
    na.omit()
  
  # NOTE (1):
  # CIRN intentionally models only the DIRECTION of temporal change
  # (sign of dx_j/dt), not its magnitude. This abstraction improves
  # robustness to noise, scale differences, and sampling variability,
  # while preserving causal directionality and interpretability.
  # Quantitative effect sizes and trajectory prediction are outside
  # the scope of the CIRN framework (see Chapter III).
  
  # NOTE (2):
  # Class imbalance (unequal frequencies of increasing vs decreasing
  # states) is handled implicitly through Bayesian posterior uncertainty
  # rather than explicit resampling or threshold adjustment.
  
  # --------------------------------------------------------------
  # 4.3 Enforcing temporal causality (CRITICAL STEP)
  # --------------------------------------------------------------
  # This step enforces the causal direction assumed by CIRN.
  #
  # In CIRN, we do NOT model the state x_j(t) itself.
  # Instead, we model the "direction of change" of the target variable:
  #
  #     Class_j(t_k) = sign( d x_j / d t evaluated at t_k )
  #
  # To ensure causal interpretability, predictors must strictly
  # precede the response in time.
  #
  # Therefore, ALL predictor variables are lagged by lag_units time step(s):
  #
  #     x_i(t_{k-lag_units})  →  sign( d x_j / d t ) at t_k
  #
  # The default is lag_units = 1, which is the standard CIRN
  # assumption. Larger values allow the analyst to test delayed
  # regulatory effects when the sampling design or domain knowledge
  # suggests that responses occur after more than one observed interval.
  #
  # This construction guarantees:
  #
  #   • Temporal causality:
  #       Predictor values occur BEFORE the observed change
  #
  #   • No future information leakage:
  #       x_i(t_{k+1}) is never used to explain x_j(t_k)
  #
  #   • Directional interpretability of inferred edges:
  #       An edge x_i → x_j means:
  #       "The state of x_i at time t_k influences whether x_j
  #        increases or decreases immediately afterward."
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
  # statistically symmetric and causally meaningless.
  # --------------------------------------------------------------
  
  # Epistemic clarification:
  # Temporal lagging enforces directional asymmetry consistent with
  # causal reasoning under observational data. However, CIRN does not
  # claim interventional causality; inferred edges represent
  # probabilistically supported causal hypotheses conditional on
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
  }
  
  # Structural hygiene:
  # Remove time from predictor space, but keep the target state when declared.
  # Exclude only the target first derivative because d[target] is target_d1.
  predictors <- setdiff(predictors, time_col)
  
  # Construct full list of predictor columns across representations
  # (state variables + first derivatives + second derivatives)
  predictor_cols <- c(
    predictors,
    paste0(predictors, "_d1"),
    paste0(predictors, "_d2")
  )
  
  # Drop the target first derivative, then keep only columns that exist
  # after derivative computation. For any target, this allows the target
  # state and target_d2 while excluding target_d1.
  predictor_cols <- setdiff(predictor_cols, target_d1)
  predictor_cols <- intersect(predictor_cols, names(df_derivs))
  
  # Enforce temporal causality (lag predictors only)
  df_derivs <- df_derivs %>%
    mutate(across(all_of(predictor_cols), ~ dplyr::lag(.x, n = lag_units))) %>%
    na.omit()
  
  
  # --------------------------------------------------------------
  # 4.4 Debug snapshot (OPTIONAL -- development & validation only)
  # --------------------------------------------------------------
  # By default, CIRN does not write debug tables to the Global
  # Environment. Set debug = TRUE to retain the fully processed
  # model-ready data for each target inside the returned result.
  # Set assign_debug_to_global = TRUE only if you also want objects
  # such as debug_full_<target> to appear in RStudio.
  
  debug_full <- NULL
  
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
  # 4.5 Predictor representations (CIRN feature sets)
  # --------------------------------------------------------------
  # CIRN explicitly evaluates multiple representations of regulatory
  # influence in order to capture different mechanistic hypotheses
  # about how system components interact over time.
  #
  # Rather than assuming a single functional form, CIRN decomposes
  # potential regulation into three complementary feature spaces:
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
  #   • The target first derivative is excluded because it defines the response
  #   • Each representation is evaluated independently
  #   • Optionally, all representations can be fitted together in one
  #     equation-style model for d[target]
  #   • Together these layers form the multi-representation
  #     structure of the CIRN framework
  # --------------------------------------------------------------
  
  # Representation-specific predictor sets
  # (built from the already cleaned predictor base)
  
  predictors_original <- intersect(predictors, names(df_derivs))
  predictors_d1       <- setdiff(
    intersect(paste0(predictors, "_d1"), names(df_derivs)),
    target_d1
  )
  predictors_d2       <- intersect(paste0(predictors, "_d2"), names(df_derivs))
  
  
  # --------------------------------------------------------------
  # 4.6 Bayesian logistic regression (CIRN core engine)
  # --------------------------------------------------------------
  # For a fixed target variable x_j, CIRN models the *direction*
  # of its local temporal change using Bayesian logistic regression:
  #
  #   Class_j(t_k) = 1  if  dx_j/dt > 0
  #                = 0  if  dx_j/dt < 0
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
  #   • predictors x_i are temporally lagged to enforce
  #     causal asymmetry
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
      return(list(
        coefficients = empty_bayes_result_tibble,
        diagnostics = list(
          rhat = NA_real_,
          neff_ratio = NA_real_,
          loo = NULL
        ),
        model = NULL,
        vif_group = empty_vif_group_tibble,
        vif_pairs = empty_vif_pair_tibble
      ))
    }
    
    # ----------------------------------------------------------
    # 4.6.2 Model specification
    # ----------------------------------------------------------
    # Logistic regression on the direction of change:
    #
    #   logit( P(Class_j(t_k) = 1) )
    #     = β_0 + Σ_i β_ij x_i(t_{k-1})
    #
    # where:
    #   • Class_j(t_k) = sign(dx_j/dt)
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
        "⚠️ High multicollinearity detected (VIF > 5) for target = ",
        paste0("d", target),
        ", predictor_set = ",
        predictor_set,
        ": ",
        paste(high_vif_terms, collapse = ", ")
      )
    }
    
    if (length(exact_pair_terms) > 0) {
      warning(
        "⚠️ Exact or near-exact pairwise collinearity detected for target = ",
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
    
    model <- brm(                           # Bayesian logistic regression via brms (Stan backend)
      formula = form,                       # Logistic regression formula for CIRN classification
      data    = df_derivs,                  # Preprocessed data containing state variables and derivatives
      family  = bernoulli(link = "logit"),  # Binary outcome with logit link (direction of change)
      
      # --------------------------------------------------
      # Prior specification
      # --------------------------------------------------
      
      prior = coef_prior,  # Regression coefficients are assigned weakly
      # informative Normal priors centered at zero.
      # This reflects the assumption of no regulation
      # a priori while allowing moderate-to-strong
      # effects when supported by the data.
      
      # --------------------------------------------------
      # MCMC configuration
      # --------------------------------------------------
      
      iter = model_iter,    # Total number of MCMC iterations per chain (includes warm-up).
      # This setting was sufficient to achieve
      # stable convergence and adequate effective
      # sample sizes across all CIRN models.
      # If convergence diagnostics are unsatisfactory
      # (e.g., R-hat > 1.01, low ESS, or poor mixing
      # in trace plots), this value should be increased,
      # together with warmup (e.g., iter = 4000,
      # warmup = 1500).
      
      warmup = model_warmup,  # Number of warm-up (burn-in) iterations per chain.
      # Warm-up iterations are used for step-size
      # adaptation and are excluded from posterior
      # inference.
      
      chains = model_chains,  # Number of independent MCMC chains.
      # Four chains are the default and recommended
      # setting in modern Bayesian workflows (Stan/brms),
      # providing reliable convergence diagnostics
      # while balancing computational cost.
      # Increasing the number of chains is generally
      # unnecessary unless convergence issues persist.
      
      cores = model_cores,    # Number of CPU cores used for parallel sampling.
      # Typically set equal to the number of chains.
      
      control = list(
        adapt_delta = adapt_delta  # Increase HMC target acceptance rate to improve
        # numerical stability of sampling.
        # This helps reduce divergent transitions that
        # can arise from:
        #   • highly correlated predictors,
        #   • derivative-based features,
        #   • near-separation in logistic regression.
        # A higher adapt_delta enforces smaller step sizes,
        # leading to more accurate exploration of the
        # posterior geometry at the cost of slightly
        # increased computation time.
        #
        # NOTE:
        #   If divergent transitions persist, increase this
        #   value (e.g., to 0.99) rather than ignoring
        #   divergences.
      ),
      
      seed = seed,          # Random seed for reproducibility.
      
      refresh = 0,          # Suppress sampling progress output.
      
      backend = "cmdstanr"  # Use CmdStan via cmdstanr backend for
      # improved performance and diagnostics.
    )
    
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
      }
    )
    
    # Warn (but do not stop) if convergence is questionable
    if (any(diagnostics$rhat > 1.01, na.rm = TRUE)) {
      warning(
        "⚠️ Potential convergence issues detected for target = ",
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
    
    coef_tbl <- ps %>%
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
        target = paste0("d", target)
      ) %>%
      rowwise() %>%
      mutate(
        hdi_bounds  = list(get_hdi(posterior_draws[[paste0("b_", term)]])),
        hdi_lower95 = hdi_bounds[[1]],
        hdi_upper95 = hdi_bounds[[2]],
        credibility = ifelse(
          hdi_lower95 > 0 | hdi_upper95 < 0,
          "credible",
          "uncertain"
        )
      ) %>%
      ungroup() %>%
      filter(credibility == "credible") %>%
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
        method
      )
    
    
    # If no coefficients survive the HDI criterion,
    # return a structured empty result instead of NULL
    if (nrow(coef_tbl) == 0) {
      coef_tbl <- empty_bayes_result_tibble %>%
        mutate(target = paste0("d", target))
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
      coefficients = coef_tbl,
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
  # For a fixed target variable x_j, each representation defines
  # a separate Bayesian logistic regression model of the form:
  #
  #   logit( P(Class_j(t_k) = 1) )
  #     = β_0 + Σ_i β_ij^(r) x_i^(r)(t_{k-1})
  #
  # where:
  #   • r ∈ {original, first_derivative, second_derivative}
  #     denotes a separate CIRN representation
  #   • x_i^(r)(t_{k-1}) are temporally lagged predictors constructed
  #     from representation r
  #   • all_predictors uses the union of these predictor columns
  #
  # The representation_mode argument controls which models are fit:
  #   • "separate"       : original CIRN behavior; fit the three
  #                        representations independently
  #   • "all_predictors" : fit one combined model using all state,
  #                        first-derivative, and second-derivative
  #                        predictors together, excluding d[target]
  #   • "both"           : fit the three separate representations and
  #                        the combined all-predictors model
  #
  # These models are:
  #   • Fit independently
  #   • Diagnosed independently
  #   • Optionally compared via LOO for predictive adequacy
  #
  # No representation is privileged a priori. The combined model is
  # intended as an optional equation-approximation layer, while the
  # separate models remain useful for representation sensitivity.
  
  results <- list()
  
  if (representation_mode %in% c("separate", "both")) {
    results <- c(
      results,
      list(
        # State-based regulation:
        # x_i(t_k) → sign(dx_j/dt)
        original = run_bayes_logit(predictors_original, "original"),
        
        # Rate-based regulation:
        # dx_i/dt(t_k) → sign(dx_j/dt)
        first_derivative = run_bayes_logit(predictors_d1, "first_derivative"),
        
        # Acceleration-based regulation:
        # d²x_i/dt²(t_k) → sign(dx_j/dt)
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
  #   • Each representation contributes candidate edges independently
  #   • Only coefficients satisfying the CIRN inference rule
  #     (95% HDI excludes zero) are retained
  #   • No weighting, voting, or averaging is applied across
  #     representations
  #
  # Consequently:
  #   • A regulator–target pair may appear in multiple representations
  #   • Each occurrence corresponds to a distinct dynamical hypothesis
  #     (state-based, rate-based, acceleration-based regulation, or
  #      the optional all-predictors equation-style model)
  #
  # The aggregated result constitutes the final CIRN edge list,
  # preserving representation-specific mechanistic interpretation.
  
  edges_tbl <- bind_rows(
    purrr::map(results, "coefficients"),
    .id = "predictor_set"
  ) %>%
    
    # Safety filter: remove rows with missing posterior means
    # (can occur if a model returns no significant coefficients)
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
    diagnostics = results,
    models = lapply(results, function(x) x$model),
    vif_group = purrr::map_df(results, "vif_group"),
    vif_pairs = purrr::map_df(results, "vif_pairs"),
    debug = debug_full
  )
}


################################################################################
################################################################################


# ================================
# 5. Full Network Inference (STRUCTURALLY CONSTRAINED VERSION)
# ================================
#
# This version of infer_network() extends CIRN by allowing the user
# to explicitly specify:
#
#   • targets    : endogenous variables to be modeled dynamically
#   • predictors : variables allowed to act as regulators
#   • representation_mode :
#       - "separate"       = original three-representation CIRN
#       - "all_predictors" = one combined model for d[target]
#       - "both"           = fit separate and combined models
#
# This enables structural separation between:
#   • Endogenous state variables (e.g., disease incidence)
#   • Exogenous drivers (e.g., temperature, humidity, rainfall)
#
# If targets and predictors are NOT specified,
# the function defaults to the original symmetric CIRN behavior,
# treating all non-time variables as both targets and predictors.
#
# This structural control prevents scientifically implausible
# bidirectional inference in systems where domain knowledge
# dictates asymmetric causality (e.g., weather → dengue).
#
# ---------------------------------------------------------------

infer_network <- function(df,
                          time_col,
                          targets = NULL,
                          predictors = NULL,
                          representation_mode = c("separate", "all_predictors", "both"),
                          lag_units = 1,
                          run_pairwise = FALSE,
                          pairwise_representation_mode = representation_mode,
                          points_per_interval = 1,
                          spar = NULL,
                          outlier_method = "MAD",
                          outlier_thresh = 3.5,
                          outlier_action = "winsorize",
                          response_eps = 1e-6,
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
    choices = c("separate", "all_predictors", "both")
  )
  if (is.null(loo_reloo)) {
    loo_reloo <- isTRUE(compute_loo) && isTRUE(loo_moment_match)
  }
  
  if (length(lag_units) != 1 || is.na(lag_units) || lag_units < 1 || lag_units != floor(lag_units)) {
    stop("lag_units must be a single positive integer.")
  }
  lag_units <- as.integer(lag_units)
  
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
  
  df <- na.omit(df)
  
  # ------------------------------------------------------------
  # 5.0.1 Derivative computation
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
  # 5.0.2 Standardization (applied AFTER derivative computation)
  # ------------------------------------------------------------
  df_derivs <- standardize_except_class_time(
    df_derivs,
    time_col = time_col
  )
  
  # ------------------------------------------------------------
  # 5.1 Identify endogenous targets
  # ------------------------------------------------------------
  # If no targets specified → symmetric CIRN behavior
  
  if (is.null(targets)) {
    targets <- setdiff(names(df), time_col)
  }
  
  # Safety check
  if (!all(targets %in% names(df))) {
    stop("Some specified targets are not present in the dataset.")
  }
  
  
  # ------------------------------------------------------------
  # 5.2 Identify allowed predictors
  # ------------------------------------------------------------
  # If no predictors specified → symmetric CIRN behavior
  
  if (is.null(predictors)) {
    predictors <- setdiff(names(df), time_col)
  }
  
  # Safety check
  if (!all(predictors %in% names(df))) {
    stop("Some specified predictors are not present in the dataset.")
  }
  
  # ------------------------------------------------------------
  # 5.2.1 Progress accounting
  # ------------------------------------------------------------
  # A "fit" means one Bayesian logistic regression model. CIRN can
  # fit several models per target depending on representation_mode.
  # This tracker reports both the current fit and the total number
  # expected for the selected multivariable and pairwise settings.
  
  representation_fit_count <- function(mode) {
    switch(
      as.character(mode),
      separate = 3L,
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
  # 5.3 Target-wise CIRN inference
  # ------------------------------------------------------------
  # For each target variable x_j:
  #   • Only predictors specified in 'predictors' are allowed
  #   • infer_for_target() keeps declared self-state and self-acceleration terms
  #     while excluding the target first derivative internally
  #   • representation_mode controls whether CIRN fits separate
  #     representations, one all-predictors model, or both
  #
  # This supports:
  #   • Epidemiological systems (weather → dengue)
  #   • Fully endogenous ecological systems
  #   • Hybrid constrained systems
  
  res <- purrr::map(
    targets,
    ~ infer_for_target(
      df_derivs = df_derivs,
      target = .x,
      time_col = time_col,
      predictors = predictors,
      representation_mode = representation_mode,
      lag_units = lag_units,
      response_eps = response_eps,
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
    )
  )
  
  
  # ------------------------------------------------------------
  # 5.4 Collect fitted Bayesian models
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
  # 5.5 Aggregate inferred regulatory edges
  # ------------------------------------------------------------
  # Each target contributes its own inferred edges.
  # These are combined into a single network table.
  
  edges <- purrr::map_df(res, "edges")
  if (nrow(edges) > 0) {
    edges <- edges %>%
      dplyr::mutate(
        analysis_mode = "multivariable",
        pairwise_predictor = NA_character_
      )
  }
  vif_group <- purrr::map_df(res, "vif_group")
  vif_pairs <- purrr::map_df(res, "vif_pairs")
  
  
  # ------------------------------------------------------------
  # 5.5.1 Optional pairwise CIRN inference
  # ------------------------------------------------------------
  # Pairwise mode fits CIRN one target-regulator pair at a time.
  # This is useful for checking whether an inferred edge is robust
  # to conditioning on other predictors. It is intentionally stored
  # separately from the primary multivariable CIRN output.
  
  pairwise <- list(
    edges = tibble::tibble(),
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
        infer_for_target(
          df_derivs = df_derivs,
          target = target,
          time_col = time_col,
          predictors = predictor,
          representation_mode = pairwise_representation_mode,
          lag_units = lag_units,
          response_eps = response_eps,
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
        )
      }
    )
    
    pairwise_names <- paste(pairwise_grid$target, pairwise_grid$predictor, sep = "__")
    
    pairwise$edges <- purrr::map2_df(
      pairwise_fits,
      seq_along(pairwise_fits),
      function(fit, i) {
        fit$edges %>%
          dplyr::mutate(
            analysis_mode = "pairwise",
            pairwise_target = pairwise_grid$target[[i]],
            pairwise_predictor = pairwise_grid$predictor[[i]]
          )
      }
    )
    
    pairwise$diagnostics <- purrr::map2_df(
      pairwise_fits,
      seq_along(pairwise_fits),
      function(fit, i) {
        purrr::imap_dfr(
          fit$diagnostics,
          function(diag_set, set_name) {
            diag <- diag_set$diagnostics
            
            tibble::tibble(
              target = pairwise_grid$target[[i]],
              pairwise_predictor = pairwise_grid$predictor[[i]],
              predictor_set = set_name,
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
        fit$vif_group %>%
          dplyr::mutate(pairwise_predictor = pairwise_grid$predictor[[i]])
      }
    )
    
    pairwise$vif_pairs <- purrr::map2_df(
      pairwise_fits,
      seq_along(pairwise_fits),
      function(fit, i) {
        fit$vif_pairs %>%
          dplyr::mutate(pairwise_predictor = pairwise_grid$predictor[[i]])
      }
    )
    
    pairwise$models <- stats::setNames(
      purrr::map(pairwise_fits, "models"),
      pairwise_names
    )
  }
  
  edges_combined <- dplyr::bind_rows(edges, pairwise$edges)
  
  
  # ------------------------------------------------------------
  # 5.6 Aggregate diagnostics across targets
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
  # 5.7 Informative completion message
  # ------------------------------------------------------------
  
  if (nrow(edges) == 0) {
    message("⚠️ No regulatory edges inferred under current HDI criterion.")
  } else {
    message("✅ CIRN inference completed: ",
            nrow(edges),
            " regulatory edges inferred.")
  }
  
  
  # ------------------------------------------------------------
  # 5.8 Return CIRN results
  # ------------------------------------------------------------
  
  return(list(
    edges = edges,
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
  
  # Initialize predicted adjacency (state-level)
  pred_adj <- matrix(0, nrow = length(vars), ncol = length(vars),
                     dimnames = list(vars, vars))
  
  # For each true pair (i, j), check if ANY representation recovers it
  for (i in vars) {
    for (j in vars) {
      
      if (true_adj[i, j] == 0) next
      
      # All inferred edges that could explain j
      candidate_edges <- inferred_edges %>%
        filter(target == paste0("d", j)) %>%
        filter(term %in% c(
          i,
          paste0(i, "_d1"),
          paste0(i, "_d2")
        ))
      
      if (nrow(candidate_edges) == 0) next
      
      # Check if ANY recovered edge has correct sign
      recovered <- any(
        (candidate_edges$regulation_type == "activation" & true_adj[i, j] > 0) |
          (candidate_edges$regulation_type == "inhibition" & true_adj[i, j] < 0)
      )
      
      if (recovered) {
        pred_adj[i, j] <- sign(true_adj[i, j])
      }
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
  
  TP <- sum(pred_adj != 0 & true_adj != 0 & sign(pred_adj) == sign(true_adj))
  # True Positives (TP):
  #   Correctly inferred regulatory edges with correct sign
  
  FP <- sum(pred_adj != 0 & true_adj == 0)
  # False Positives (FP):
  #   Spurious inferred edges where no true regulation exists
  #   (i.e., false discoveries in the inferred network)
  
  FN <- sum(true_adj != 0 & pred_adj == 0)
  # False Negatives (FN):
  #   True regulatory edges that CIRN failed to recover
  #   (i.e., missed interactions)
  
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
    precision = precision,
    recall = recall,
    specificity = specificity,
    balanced_accuracy = balanced_accuracy,
    f1 = f1,
    mcc = mcc,
    shd = shd
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
        Metric == "precision" ~ "Precision",
        Metric == "recall" ~ "Recall",
        Metric == "specificity" ~ "Specificity",
        Metric == "balanced_accuracy" ~ "Balanced Accuracy",
        Metric == "f1" ~ "F1 score",
        Metric == "mcc" ~ "Matthews Correlation Coefficient (MCC)",
        Metric == "shd" ~ "Structural Hamming Distance (SHD)",
        Metric == "auc" ~ "AUC (edge existence)",
        TRUE ~ Metric
      )
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

plot_all_vars_by_derivative_order <- function(df, vars, time_col) {
  
  vars <- prepare_plot_vars(df, vars, time_col)
  variable_palette <- make_variable_palette(vars)
  
  # Compute derivatives
  df_d <- compute_derivatives(df, time_col)
  
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
# Each quantity is shown in a separate panel with its own y-scale,
# but all panels are aligned in time to facilitate interpretation.

plot_derivative_example <- function(df, var, time_col) {
  
  var <- prepare_plot_vars(df, var, time_col)[1]
  
  # Compute derivatives (regularized grid)
  df_d <- compute_derivatives(df, time_col)
  
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
# 7.4 State and derivative trajectories for all variables
# --------------------------------------------------------------
# Purpose:
#   • Exploratory visualization across all system components
#   • Identify variables with unstable derivatives or noisy dynamics
#   • Intended for diagnostic and quality-control purposes only
#     (not for main-text figures)

plot_all_derivatives <- function(df, vars, time_col) {
  vars <- prepare_plot_vars(df, vars, time_col)
  for (v in vars) {
    print(plot_derivative_example(df, v, time_col))
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
                                         representation_mode = c("separate",
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
    separate = c("original", "first_derivative", "second_derivative"),
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
#       all posterior-supported edges across fitted representations.
#   • Nodes retain their predictor representation:
#       state, first derivative, second derivative, or target derivative.
#   • Edge color encodes activation/inhibition.
#   • Edge width encodes |omega_ij|, the posterior mean magnitude.
#
# This figure is intended for MAIN TEXT presentation
# and summarizes the final CIRN-inferred regulatory structure.
# --------------------------------------------------

plot_network_main <- function(inferred_edges,
                              representation_mode = c("separate",
                                                      "all_predictors",
                                                      "both"),
                              title = "Output 1: Full CIRN",
                              subtitle = "All posterior-supported edges across fitted predictor representations",
                              aggregate_duplicate_edges = TRUE,
                              height = "680px",
                              legend_position = c("side", "bottom")) {
  
  representation_mode <- match.arg(representation_mode)
  legend_position <- match.arg(legend_position)
  
  if (nrow(inferred_edges) == 0) {
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
            "No posterior-supported edges were retained by the CIRN HDI rule."
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
    " | retained edges: ", nrow(inferred_edges),
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
#   • In "separate" mode, the three original CIRN panels are shown.
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
                             representation_mode = c("separate",
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
    separate = paste(
      "Mode: separate. CIRN displays the three original",
      "representation-specific networks."
    ),
    all_predictors = paste(
      "Mode: all_predictors. CIRN displays the joint model using",
      "state, first-derivative, and second-derivative predictors together."
    ),
    both = paste(
      "Mode: both. CIRN displays the three representation-specific",
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
        "Representation-specific posterior-supported networks"
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
                              representation_mode = c("separate",
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
          "Aggregated posterior-supported network across fitted",
          "CIRN predictor representations"
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
                                  representation_mode = c("separate",
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
        in_separate = logical(),
        in_all_predictors = logical(),
        omega_pairwise = numeric(),
        omega_separate = numeric(),
        omega_all_predictors = numeric(),
        abs_omega_pairwise = numeric(),
        abs_omega_separate = numeric(),
        abs_omega_all_predictors = numeric(),
        regulation_type_pairwise = character(),
        regulation_type_separate = character(),
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
        .data$predictor_set %in% c("original", "first_derivative", "second_derivative") ~ "separate",
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
      in_separate = purrr::map_lgl(.data$source_mode_list, ~ "separate" %in% .x),
      in_all_predictors = purrr::map_lgl(.data$source_mode_list, ~ "all_predictors" %in% .x),
      n_sources = .data$in_pairwise + .data$in_separate + .data$in_all_predictors,
      source_modes = purrr::map_chr(.data$source_mode_list, ~ paste(.x, collapse = ", ")),
      appears_in_all_three = .data$in_pairwise & .data$in_separate & .data$in_all_predictors,
      has_sign_conflict = .data$regulation_type == "mixed"
    )
  
  source_values <- cell_summary %>%
    dplyr::filter(.data$source_mode %in% c("pairwise", "separate", "all_predictors")) %>%
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
    "omega_pairwise", "omega_separate", "omega_all_predictors",
    "abs_omega_pairwise", "abs_omega_separate", "abs_omega_all_predictors",
    "regulation_type_pairwise", "regulation_type_separate", "regulation_type_all_predictors"
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
      in_separate,
      in_all_predictors,
      omega_pairwise,
      omega_separate,
      omega_all_predictors,
      abs_omega_pairwise,
      abs_omega_separate,
      abs_omega_all_predictors,
      regulation_type_pairwise,
      regulation_type_separate,
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
  for (col in c("in_pairwise", "in_separate", "in_all_predictors")) {
    if (!col %in% names(plot_input)) plot_input[[col]] <- FALSE
  }
  for (col in c("omega_pairwise", "omega_separate", "omega_all_predictors")) {
    if (!col %in% names(plot_input)) plot_input[[col]] <- NA_real_
  }
  if (!"regulation_type_pairwise" %in% names(plot_input)) {
    plot_input$regulation_type_pairwise <- ifelse(plot_input$in_pairwise, plot_input$regulation_type, NA_character_)
  }
  if (!"regulation_type_separate" %in% names(plot_input)) {
    plot_input$regulation_type_separate <- ifelse(plot_input$in_separate, plot_input$regulation_type, NA_character_)
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
        source = "Separate",
        present = .data$in_separate,
        omega = .data$omega_separate,
        regulation_for_cell = .data$regulation_type_separate
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
      source = factor(.data$source, levels = c("Pairwise", "Separate", "All predictors")),
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
      )
    )
  
  strength_caption <- NULL
  if (encode_strength && strength_scale == "within_mode") {
    strength_caption <- "Color intensity is scaled within each inference mode by relative |omega|."
  }
  if (encode_strength && strength_scale == "global") {
    strength_caption <- "Color intensity is scaled globally by relative |omega|."
  }
  
  ggplot(plot_df, aes(x = .data$source, y = .data$edge, fill = .data$fill_value)) +
    geom_tile(color = "white", linewidth = 0.8) +
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
      axis.text.x = element_text(face = "bold")
    )
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
  par(mfrow = c(1, 2), mar = c(6, 4, 4, 2))
  
  # --------------------------------------------------
  # Panel 1: Minimum Effective Sample Size (ESS) ratio
  # --------------------------------------------------
  # Interpretation:
  #   • ESS ratio ≈ 1   → excellent mixing
  #   • ESS ratio < 0.1 → poor mixing / autocorrelation
  #
  # We plot the MINIMUM ESS across parameters
  # to highlight worst-case sampling behavior.
  
  barplot(
    diagnostics_table$min_neff,
    names.arg = paste(
      diagnostics_table$target,
      diagnostics_table$predictor_set,
      sep = "\n"
    ),
    las  = 2,
    main = "Minimum ESS ratio",
    ylab = "ESS / N",
    col  = "grey90",
    border = "grey40"
  )
  
  abline(h = 0.1, lty = 2, col = "#D55E00")  # heuristic warning threshold
  
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
  
  dotchart(
    diagnostics_table$loo_elpd,
    labels = paste(
      diagnostics_table$target,
      diagnostics_table$predictor_set
    ),
    main = "LOO ELPD (higher is better)",
    xlab = "Expected log predictive density"
  )
  
  # Reset graphics parameters
  par(mfrow = c(1, 1))
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


##### Dataset: Cardiovascular Disease


# ==================================================
# 8. Data Loading, CIRN Inference, and Validation
# ==================================================
# Purpose:
#   • Load and preprocess multivariate time-series data
#   • Execute the full CIRN inference pipeline end-to-end
#   • (Optionally) validate inferred structure against
#     known ground truth in simulation settings
#
# IMPORTANT SCOPE NOTE:
#   • CIRN inference itself is fully unsupervised and
#     does NOT require ground truth
#   • Validation steps are included ONLY for simulation
#     studies and benchmarking purposes
# ==================================================

# --------------------------------------------------
# 8.1 Load input data, identify time column, and standardize variables
# --------------------------------------------------
# Assumptions about input dataset:
#   • First column corresponds to time (t)
#   • Remaining columns are observed state variables
#   • Rows are already temporally ordered
#
# NOTE:
#   • Missing values are removed prior to inference
#   • This avoids numerical instability in derivative
#     estimation and Bayesian fitting
#
# CIRN preprocessing requirements:
#   • The time column must be preserved untransformed
#   • State variables are standardized to:
#       - improve numerical stability of Bayesian sampling
#       - allow comparable coefficient magnitudes across predictors
#       - support weakly informative priors (Normal(0,2))
#
# NOTE:
#   • Standardization does NOT affect:
#       - the sign of inferred regulation
#       - the direction of inferred interactions
#
#   • Standardization may be skipped ONLY if:
#       (i) all variables are already on comparable numerical scales, and
#       (ii) extreme magnitude differences across predictors are absent.
#
#   • When variables differ substantially in scale (e.g., temperature vs.
#     population counts), standardization is strongly recommended to
#     ensure stable and interpretable Bayesian estimation.

my_data <- read.csv("Cardiovascular_Disease.csv") %>%   # replace this with your dataset
  na.omit(); time_col <- names(my_data)[1]; sim_df <- my_data


# --------------------------------------------------
# 8.2 Run full CIRN network inference
# --------------------------------------------------
# This step executes the complete CIRN pipeline:
#   • Derivative computation
#   • Directional classification
#   • Bayesian logistic regression
#   • Posterior-based edge filtering
#
# CIRN can evaluate:
#   1) Separate representations:
#        original state variables, first derivatives, second derivatives
#   2) One all-predictors model:
#        original + first derivative + second derivative predictors together,
#        excluding the target first derivative d[target]
#   3) Both approaches for direct comparison
#
# Output:
#   • Signed, directed regulatory edges
#   • Full diagnostic information
#   • Fitted Bayesian models (for transparency only)

# Purpose:
#   • Execute the CIRN network inference algorithm
#   • Measure the computational runtime of the inference step
#
# IMPORTANT:
#   • Specifying targets and predictors can reduce false positive predictions,
#     making it a structurally constrained CIRN inference.
#   • The runtime recorded here corresponds ONLY to the
#     execution of the CIRN inference procedure.
#   • It explicitly excludes:
#       - ground-truth validation
#       - performance metric computation
#       - visualization and diagnostic plots
#
#   This ensures that the reported runtime reflects the
#   computational cost of the CIRN algorithm itself,
#   allowing fair comparison with other network inference
#   methods.

# Target and predictor choices:
#   targets    = variables whose directional change Y_i(t_k) will be modeled.
#   predictors = variables allowed to act as candidate regulators.
#
# Important:
#   These settings do not make CIRN pairwise by default. If multiple predictors
#   are listed, CIRN fits them jointly within each selected representation.
#
# Example multivariable CIRN run:
#   targets <- c("X", "Y")
#   predictors <- c("X", "Y")
#   This fits target-wise multivariable models using all allowed regulators.
#
# Example pairwise-style CIRN run:
#   targets <- c("Y")
#   predictors <- c("X")
#   This tests whether X predicts the directional change of Y.
#
# To run all pairwise target-regulator fits automatically, set
# run_pairwise = TRUE in cirn_config below.

# --------------------------------------------------
# 8.2.1 Central CIRN run configuration
# --------------------------------------------------
# Change settings here only. The infer_network() call below
# automatically uses this configuration, so you do not need to edit
# function defaults or update the same option in multiple places.
#
# Key options:
#   representation_mode:
#     "separate"       = original CIRN behavior; separate models for
#                        original-state, first-derivative, and
#                        second-derivative predictors.
#     "all_predictors" = one joint model using all predictor representations.
#     "both"           = runs both approaches.
#
#   lag_units:
#     1 = standard CIRN lag, using predictors at t_{k-1}.
#     2 = uses predictors at t_{k-2}, etc.
#
#   run_pairwise:
#     FALSE = regular multivariable CIRN only.
#     TRUE  = additionally fit one target-regulator pair at a time.

cirn_config <- list(
  # --- Variables to model ---
  targets = c("Cardio"),                                # Target variables whose directional change will be modeled
  predictors = c("NO2", "SO2", "O3", "Rspar"), #c("λcl", "LacI", "TetR"), # Candidate regulators allowed to predict each target
  
  # --- CIRN inference modes ---
  representation_mode = "both",                        # Options: "separate", "all_predictors", "both"
  pairwise_representation_mode = NULL,                 # NULL = use the same setting as representation_mode
  
  # --- Temporal alignment and pairwise option ---
  lag_units = 2,                                      # Number of time steps predictors are lagged; 1 = standard CIRN
  run_pairwise = TRUE,                                 # TRUE = also fit one predictor-target pair at a time
  
  # --- Preprocessing and derivative construction ---
  points_per_interval = 1,                             # Time-grid resolution for derivative estimation
  spar = NULL,                                         # Spline smoothing; NULL = automatic GCV choice
  outlier_method = "none",                             # Options: "none", "MAD"
  outlier_thresh = 3.5,                                # Modified Z-score threshold if outlier_method = "MAD"
  outlier_action = "keep",                             # Options: "keep", "winsorize", "remove"
  response_eps = 1e-6,                                 # Derivative threshold for removing near-zero changes
  
  # --- Bayesian MCMC settings ---
  model_iter = 3000,                                   # Total MCMC iterations per chain
  model_warmup = 1000,                                 # Warm-up/burn-in iterations per chain
  model_chains = 4,                                    # Number of MCMC chains
  model_cores = 1,                                  # NULL = follow model_chains; use 1 or 2 for long pairwise runs
  prior_mean = 0,                                      # Mean of Normal prior for regression coefficients
  prior_sd = 2,                                        # SD of Normal prior for regression coefficients
  
  adapt_delta = 0.95,                                  # Higher values may reduce divergent transitions
  
  # --- Optional LOO model comparison ---
  compute_loo = TRUE,                                  # TRUE = compute optional LOO diagnostics after fitting; set to FALSE for faster runs
  loo_moment_match = TRUE,                             # TRUE = use moment matching for LOO; slower and may recompile; ; set to FALSE for faster runs
  loo_reloo = NULL,                                    # NULL = TRUE when compute_loo and loo_moment_match are TRUE
  loo_k_threshold = 0.7,                               # Pareto-k cutoff used when reloo is active
  
  # --- Reproducibility ---
  seed = 123,                                          # Reproducibility seed
  
  # --- Progress and debugging ---
  show_progress = TRUE,                                # TRUE = print percent-complete messages during fitting
  progress_bar = TRUE,                                 # TRUE = show a console progress bar between model fits
  progress_callback = NULL,                            # Optional custom progress function; usually leave NULL
  debug = TRUE,                                        # TRUE = store preprocessing/lagging debug tables
  assign_debug_to_global = TRUE                        # TRUE = export debug tables to the global environment
)


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

# Keep these aliases so the plotting and reporting sections below
# automatically follow the same central configuration.

targets <- cirn_config$targets
predictors <- cirn_config$predictors
representation_mode <- cirn_config$representation_mode
lag_units <- cirn_config$lag_units
run_pairwise <- cirn_config$run_pairwise
debug <- cirn_config$debug
assign_debug_to_global <- cirn_config$assign_debug_to_global

CIRN <- system.time(
  res <- do.call(
    infer_network,
    c(
      list(
        df = sim_df,
        time_col = time_col
      ),
      cirn_config
    )
  )
)

message("⏱ CIRN inference runtime: ", round(CIRN["elapsed"], 2), " seconds")

# Optional
#cirn_runtime_table <- tibble(
#  Metric = "CIRN inference runtime",
#  Value  = round(CIRN["elapsed"], 2),
#  Units  = "seconds"
# )
#print(cirn_runtime_table)

# --------------------------------------------------
# 8.4 Extract results and sanity checks
# --------------------------------------------------
# Extract:
#   • edges       : final CIRN-inferred regulatory network
#   • diagnostics : convergence and sampling summaries
#
# These outputs constitute the PRIMARY inference results.

edges <- res$edges
pairwise_edges <- res$pairwise$edges
edges_combined <- res$edges_combined
diagnostics_table <- res$diagnostics
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
print(edge_consistency_table)
print(diagnostics_table)
print(vif_group_table)
print(vif_pair_table)

# If debug = TRUE, inspect debug_tables[["X"]], debug_tables[["Y"]], etc.
# If assign_debug_to_global = TRUE, the same tables also appear as
# debug_full_X, debug_full_Y, etc. in the Global Environment.

# --------------------------------------------------
# 8.5 Extract representative models (diagnostics only)
# --------------------------------------------------
# Purpose:
#   • Enable detailed posterior inspection
#   • Generate diagnostic and appendix figures
#
# IMPORTANT:
#   • These models are NOT required for inference
#   • Edge inclusion has already been decided
#     via HDI-based rules

target_name <- targets[1]    # Change the index to inspect another target.

model_original <- res$models[[target_name]][["original"]]
model_first_derivative <- res$models[[target_name]][["first_derivative"]]
model_second_derivative <- res$models[[target_name]][["second_derivative"]]
model_all_predictors <- res$models[[target_name]][["all_predictors"]]

selected_models <- list(
  original = model_original,
  first_derivative = model_first_derivative,
  second_derivative = model_second_derivative,
  all_predictors = model_all_predictors
)

selected_models <- selected_models[
  !vapply(selected_models, is.null, logical(1))
]

stopifnot(length(selected_models) > 0)
purrr::walk(
  selected_models,
  ~ stopifnot(inherits(.x, "brmsfit"))
)

# --------------------------------------------------
# 8.6 Structural sanity check: symmetric edges
# --------------------------------------------------
# Purpose:
#   • Identify bidirectional (reciprocal) regulation
#   • Support biological or mechanistic interpretation
#
# NOTE:
#   • Symmetric edges are NOT treated as errors
#   • They may reflect feedback mechanisms or
#     coupled system dynamics

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
  message("ℹ️ Symmetric (bidirectional) interactions detected:")
  print(symmetric_edges)
} else {
  message("✅ No symmetric (bidirectional) interactions detected.")
}

# --------------------------------------------------
# 8.7 OPTIONAL: Ground-truth validation (simulation only)
# --------------------------------------------------
# IMPORTANT:
#   • This block is OPTIONAL and simulation-specific
#   • MUST be removed for real ecological datasets
#   • Does NOT influence CIRN inference or runtime
#
# Purpose:
#   • Quantitatively evaluate CIRN performance
#   • Enable benchmarking against known dynamics

vars <- setdiff(names(sim_df), time_col)

true_adj <- matrix(
  0,
  nrow = length(vars),
  ncol = length(vars),
  dimnames = list(vars, vars)
)

storage.mode(true_adj) <- "numeric"

# Air pollutants positively affect cardiovascular outcome
#true_adj["NO2",  "Cardio"] <-  1
#true_adj["SO2",  "Cardio"] <-  1
#true_adj["O3",   "Cardio"] <-  1

# Solar radiation influences atmospheric chemistry
#true_adj["Rspar","O3"]     <-  1
#true_adj["Rspar","NO2"]    <- -1

# --------------------------------------------------
# 8.8 Quantitative performance evaluation (OPTIONAL)
# --------------------------------------------------
# Representation-agnostic evaluation:
#   • Aggregates inferred edges across CIRN representations
#   • Compares inferred network against ground truth
#
# Metrics include:
#   • Precision, recall, specificity
#   • Balanced accuracy, F1-score
#   • MCC, SHD

evaluation_metrics <- evaluate_representation_agnostic(
  true_adj,
  edges
)

evaluation_table_long <- metrics_to_long_table(
  evaluation_metrics
)

print(evaluation_table_long)

# --------------------------------------------------
# 8.9 CIRN Summary (Accuracy + Runtime) [optional]
# --------------------------------------------------
# Purpose:
#   • Provide a compact summary of:
#       - inference accuracy (if ground truth exists)
#       - computational cost (always reported)
#
# NOTE:
#   • Accuracy metrics are CONDITIONAL on simulation data
#   • Runtime is ALWAYS available and applicable

cirn_summary_table <- tibble(
  Metric = c(
    "True Positives (TP)",
    "False Positives (FP)",
    "False Negatives (FN)",
    "True Negatives (TN)",
    "Precision",
    "Recall",
    "Specificity",
    "Balanced Accuracy",
    "F1 score",
    "Matthews Correlation Coefficient (MCC)",
    "Structural Hamming Distance (SHD)",
    "CIRN inference runtime (seconds)"
  ),
  Value = c(
    evaluation_metrics$TP,
    evaluation_metrics$FP,
    evaluation_metrics$FN,
    evaluation_metrics$TN,
    round(evaluation_metrics$precision, 2),
    round(evaluation_metrics$recall, 2),
    round(evaluation_metrics$specificity, 2),
    round(evaluation_metrics$balanced_accuracy, 2),
    round(evaluation_metrics$f1, 2),
    round(evaluation_metrics$mcc, 2),
    evaluation_metrics$shd,
    round(CIRN["elapsed"], 2)
  )
)

print(cirn_summary_table)


################################################################################
################################################################################


# ================================
# 9. Visualization
# ================================
# Purpose:
#   • Generate all visual outputs associated with CIRN inference
#   • Support exploratory analysis, methodological diagnostics,
#     and final presentation of results
#
# This section produces three classes of figures:
#   (i)   Exploratory sanity-check plots used during development
#   (ii)  Publication-grade figures intended for the main manuscript
#   (iii) Diagnostic and uncertainty plots intended for the appendix
#
# IMPORTANT:
#   • Not all plots generated here are intended for publication
#   • Several figures are retained to ensure transparency,
#     reproducibility, and reviewer-facing validation

# ------------------------------------------------------------------
# 9.1 Time series: state variables and temporal derivatives
# ------------------------------------------------------------------
# Purpose:
#   • Verify the raw temporal behavior of observed state variables
#   • Assess smoothness and fidelity after spline-based preprocessing
#   • Ensure numerical derivatives (first and second order) are
#     stable, interpretable, and consistent with the underlying dynamics
#
# These visualizations establish a direct conceptual link between:
#   • the original observed time-series data,
#   • the construction of derivative-based predictors, and
#   • the subsequent CIRN network inference procedure.
#
# Collectively, these plots serve as diagnostic and quality-control
# tools to confirm that preprocessing choices do not introduce
# artifacts or distort the temporal structure of the data.
#
# NOTE:
#   • These figures are exploratory and diagnostic in nature
#   • They are typically included only in the appendix or
#     supplementary material, not in the main manuscript
#
# The sequence of plots below follows a logical progression:
#   (i)   raw data inspection (prior to smoothing),
#   (ii)  detailed single-variable derivative inspection, and
#   (iii) system-wide comparison across variables and derivative orders.


# Raw state variables (unsmoothed):
#   • Displays the original observed data before any preprocessing
#   • Provides a baseline reference for subsequent smoothing
#   • Helps identify noise, outliers, and irregular temporal patterns

vars <- unique(c(targets, predictors))
representative_var <- vars[min(2, length(vars))]
plot_raw_data_all_vars(my_data, vars, time_col) # run this before standardizing


# System-wide dynamics by derivative order:
#   • Aggregates all variables into shared panels by derivative order
#   • Panel 1: all state variables
#   • Panel 2: all first derivatives
#   • Panel 3: all second derivatives
#
# Panels share a common time axis, while y-axes are scaled per
# derivative order to preserve visibility across different magnitudes.
# This visualization supports direct comparison of relative dynamics
# and stability across the system.

plot_all_vars_by_derivative_order(my_data, vars, time_col)

# Populate one
# Single representative variable:
#   • Displays the state variable and its first and second derivatives
#     on aligned time axes
#   • Used for rapid inspection of derivative behavior
#   • Confirms correct temporal alignment, scaling, and numerical stability

plot_derivative_example(
  df = sim_df,
  var = representative_var,
  time_col = time_col
)

# Populate all
# All variables (one at a time):
#   • Iterates the single-variable diagnostic across all system components
#   • Enables detection of variables with noisy or unstable derivatives
#   • Supports preprocessing quality control prior to CIRN inference

#plot_all_derivatives(
#  df = sim_df,
#  vars = vars,
#  time_col = time_col
#)

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
#   (ii)  A multi-panel representation-specific view for diagnostics

# Output 1: Full CIRN
#   • Aggregates all posterior-supported edges across fitted
#     CIRN predictor representations.
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
  plot_edge_consistency(edge_consistency_table)
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
#   • Shown separately for each fitted CIRN representation
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

purrr::walk(selected_models, plot_model_posteriors_if_available)

# Cross-representation posterior comparison for the selected target
#   • Highlights how inferred regulation depends on
#     state-based vs derivative-based predictors

plot_all_cirn_posteriors(
  res$models[[target_name]]
)

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

mcmc_hist_trace_plots <- purrr::imap(
  selected_models,
  ~ suppressWarnings(plot_mcmc_hist_trace(.x))
)

mcmc_hist_trace_suffix <- c(
  original = "Original",
  first_derivative = "1stDeriv",
  second_derivative = "2ndDeriv",
  all_predictors = "AllPredictors"
)

purrr::iwalk(
  mcmc_hist_trace_plots,
  ~ {
    message("Displaying MCMC histogram/trace diagnostic for: ", .y)
    print(.x)
    
    output_suffix <- mcmc_hist_trace_suffix[[.y]]
    if (is.null(output_suffix)) {
      output_suffix <- gsub("[^A-Za-z0-9]+", "_", .y)
    }
    
    output_file <- paste0(
      "Target_",
      gsub("[^A-Za-z0-9]+", "_", target_name),
      "_Hist_Trace_",
      output_suffix,
      ".png"
    )
    
    tryCatch(
      {
        ggplot2::ggsave(
          filename = output_file,
          plot = .x,
          width = 10,
          height = 8,
          dpi = 300
        )
        message("Saved MCMC histogram/trace diagnostic: ", output_file)
      },
      error = function(e) {
        warning(
          "Could not save MCMC histogram/trace diagnostic for ",
          .y,
          ": ",
          conditionMessage(e)
        )
      }
    )
  }
)


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

suppressWarnings(
  plot_all_traces_with_burnin(
    models_list = res$models[[target_name]],
    target_var = target_name
  )
)


################################################################################
################################################################################


# ==================================================================
# (Optional) Save figures for thesis / manuscript submission
# ==================================================================
# This section provides reproducible export commands for all figures
# generated in Section 9.
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
# plot_network_main(edges)
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
# plot_derivative_example(sim_df, vars[1], time_col)
# dev.off()



# ------------------------------------------------------------------
# Appendix / supplementary figures
# ------------------------------------------------------------------

# ---------------------------------------------------------------
# Fig. S1: Time series and temporal derivatives (example variable)
# ---------------------------------------------------------------
# pdf("FigS1_TimeSeries_Derivatives_Example.pdf", width = 7, height = 9)
# plot_derivative_example(sim_df, vars[1], time_col)
# dev.off()

# ---------------------------------------------------------------
# Fig. S2: CIRN networks by predictor representation
#          (original, first derivative, second derivative,
#           optional all-predictors model)
# ---------------------------------------------------------------
# pdf("FigS2_CIRN_Representation_Panels.pdf", width = 12, height = 4)
# plot_cirn_panels(edges, predictor_sets = names(selected_models))
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
#   )s
# )
# dev.off()


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
  con = "sessionInfo_CIRN.txt"
)


################################################################################
#######################🫰🫰🫰🫰🫰 THANK YOU!!! 🫰🫰🫰🫰🫰#######################
################################################################################
