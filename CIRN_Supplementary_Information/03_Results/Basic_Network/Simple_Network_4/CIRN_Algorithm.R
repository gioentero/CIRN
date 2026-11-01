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
#   • Separate Bayesian models are fitted using three predictor
#     representations:
#         (i)   original state variables,
#         (ii)  first-order derivatives,
#         (iii) second-order derivatives.
#   • No representation is privileged a priori; consistency across
#     representations is used to assess robustness.
#
#   Implemented in:
#     infer_for_target()
#       → results$original
#       → results$first_derivative
#       → results$second_derivative
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

setwd('/Users/gio/Desktop/My Dissertation/Dissertation Focus/Main Version/1_Basic_Net_Raw/Simple_Network_4')


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

compute_derivatives <- function(df, time_col, points_per_interval = 10, spar = NULL, outlier_method = "MAD", outlier_thresh = 3.5, outlier_action = "winsorize") {
  # "MAD" or "none"; # threshold for modified Z-score; # "remove" or "winsorize"
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
  rel_strength = numeric(),
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

infer_for_target <- function(df_derivs,
                             target,
                             time_col,
                             predictors = NULL,
                             debug = FALSE,
                             assign_debug_to_global = FALSE) { 
  
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
  
  eps <- 1e-6  # tolerance for numerical stability
  
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
  # Therefore, ALL predictor variables are lagged by one time step:
  #
  #     x_i(t_k)  →  sign( d x_j / d t ) over (t_k, t_{k+1})
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
    mutate(across(all_of(predictor_cols), dplyr::lag)) %>%
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
    
    # ----------------------------------------------------------
    # 4.6.1 Guard clause: no predictors available
    # ----------------------------------------------------------
    # This situation can arise if:
    #   • All candidate predictors were removed upstream
    #   • The representation yields no valid features
    #
    # Instead of failing, we return a well-defined empty result.
    if (length(pred_set) == 0) {
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
    
    model <- brm(                           # Bayesian logistic regression via brms (Stan backend)
      formula = form,                       # Logistic regression formula for CIRN classification
      data    = df_derivs,                  # Preprocessed data containing state variables and derivatives
      family  = bernoulli(link = "logit"),  # Binary outcome with logit link (direction of change)
      
      # --------------------------------------------------
      # Prior specification
      # --------------------------------------------------
      
      prior = prior(normal(0, 2), class = "b"),  # Regression coefficients are assigned weakly
      # informative Normal priors centered at zero.
      # This reflects the assumption of no regulation
      # a priori while allowing moderate-to-strong
      # effects when supported by the data.
      
      # --------------------------------------------------
      # MCMC configuration
      # --------------------------------------------------
      
      iter = 3000,          # Total number of MCMC iterations per chain (includes warm-up).
      # This setting was sufficient to achieve
      # stable convergence and adequate effective
      # sample sizes across all CIRN models.
      # If convergence diagnostics are unsatisfactory
      # (e.g., R-hat > 1.01, low ESS, or poor mixing
      # in trace plots), this value should be increased,
      # together with warmup (e.g., iter = 4000,
      # warmup = 1500).
      
      warmup = 1000,        # Number of warm-up (burn-in) iterations per chain.
      # Warm-up iterations are used for step-size
      # adaptation and are excluded from posterior
      # inference.
      
      chains = 4,           # Number of independent MCMC chains.
      # Four chains are the default and recommended
      # setting in modern Bayesian workflows (Stan/brms),
      # providing reliable convergence diagnostics
      # while balancing computational cost.
      # Increasing the number of chains is generally
      # unnecessary unless convergence issues persist.
      
      cores = 4,            # Number of CPU cores used for parallel sampling.
      # Typically set equal to the number of chains.
      
      control = list(
        adapt_delta = 0.95  # Increase HMC target acceptance rate to improve
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
      
      seed = 123,           # Random seed for reproducibility.
      
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
      loo = tryCatch(
        brms::loo(model, moment_match = TRUE),
        error = function(e) NULL
      )
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
  # 4.7 Fit all CIRN representations
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
  # For a fixed target variable x_j, each representation defines
  # a separate Bayesian logistic regression model of the form:
  #
  #   logit( P(Class_j(t_k) = 1) )
  #     = β_0 + Σ_i β_ij^(r) x_i^(r)(t_{k-1})
  #
  # where:
  #   • r ∈ {original, first_derivative, second_derivative}
  #     denotes the CIRN representation
  #   • x_i^(r)(t_{k-1}) are temporally lagged predictors constructed
  #     from representation r
  #
  # These models are:
  #   • Fit independently
  #   • Diagnosed independently
  #   • Optionally compared via LOO for predictive adequacy
  #
  # No representation is privileged a priori; consistency across
  # representations is used to assess robustness of inferred
  # regulatory relationships.
  
  results <- list(
    
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
  #     (state-based, rate-based, or acceleration-based regulation)
  #
  # The aggregated result constitutes the final CIRN edge list,
  # preserving representation-specific mechanistic interpretation.
  
  edges_tbl <- bind_rows(
    
    # State-based regulatory effects
    original = results$original$coefficients,
    
    # Rate-based regulatory effects
    first_derivative = results$first_derivative$coefficients,
    
    # Acceleration-based regulatory effects
    second_derivative = results$second_derivative$coefficients,
    
    # Record the originating representation
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
      rel_strength = case_when(
        n() == 1 ~ "credible",
        n() == 2 ~ ifelse(
          omega_abs == max(omega_abs),
          "strong",
          "weak"
        ),
        omega_abs <= quantile(omega_abs, 0.33, na.rm = TRUE) ~ "weak",
        omega_abs <= quantile(omega_abs, 0.66, na.rm = TRUE) ~ "moderate",
        TRUE ~ "strong"
      )
    ) %>%
    ungroup() %>%
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
                          debug = FALSE,
                          assign_debug_to_global = FALSE) {
  
  df <- na.omit(df)
  
  # ------------------------------------------------------------
  # 5.0.1 Derivative computation
  # ------------------------------------------------------------
  df_derivs <- compute_derivatives(df, time_col)
  
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
  # 5.3 Target-wise CIRN inference
  # ------------------------------------------------------------
  # For each target variable x_j:
  #   • Only predictors specified in 'predictors' are allowed
  #   • infer_for_target() keeps declared self-state and self-acceleration terms
  #     while excluding the target first derivative internally
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
  vif_group <- purrr::map_df(res, "vif_group")
  vif_pairs <- purrr::map_df(res, "vif_pairs")
  
  
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
    diagnostics = diagnostics,
    vif_group = vif_group,
    vif_pairs = vif_pairs,
    models = models,
    debug = debug_tables
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

# --------------------------------------------------
# 7.5 PUBLICATION-GRADE CIRN NETWORK (FINAL)
# Legend placed in a separate panel to avoid overlap
# --------------------------------------------------
# Visual encoding:
#   • Nodes       : system variables
#   • Arrows      : direction of regulation
#   • Edge color  : regulatory sign
#       - Green   = activation (ω_ij > 0)
#       - Red     = inhibition (ω_ij < 0)
#   • Edge width  : |ω_ij| (magnitude of posterior mean)
#
# This figure is intended for MAIN TEXT presentation
# and summarizes the final CIRN-inferred regulatory structure.
# --------------------------------------------------

plot_network_main <- function(inferred_edges) {
  
  # ------------------------
  # Safety check
  # ------------------------
  if (nrow(inferred_edges) == 0) {
    stop("No edges to plot: inferred_edges is empty.")
  }
  
  # ------------------------
  # Build node list
  # ------------------------
  # Only structural information is passed to the network object;
  # sign and magnitude are used later for styling.
  
  node_ids <- unique(c(inferred_edges$term, inferred_edges$target))
  
  nodes <- data.frame(
    id = node_ids,
    label = format_cirn_node_label(node_ids),
    shape = "circle"
  )
  
  # ------------------------
  # Node styling
  # ------------------------
  # Clean, journal-friendly appearance
  
  nodes$size <- 45
  nodes$color.background <- "white"
  nodes$color.border <- "black"
  nodes$borderWidth <- 2
  
  # ------------------------
  # Edge styling
  # ------------------------
  # Color encodes sign of regulation
  
  edges <- data.frame(
    from = inferred_edges$term,
    to   = inferred_edges$target
  )
  
  edges$color <- ifelse(
    inferred_edges$regulation_type == "activation",
    "#2ca02c",  # green = activation
    "#d62728"   # red   = inhibition
  )
  
  # Edge width encodes |ω_ij| (posterior mean magnitude)
  # Rescaling ensures visual comparability
  
  edges$width <- scales::rescale(
    abs(inferred_edges$omega),
    to = c(2, 6)
  )
  
  # Strong arrows to emphasize directionality
  
  edges$arrows <- "to"
  
  # Slight curvature improves visibility for reciprocal edges
  
  edges$smooth <- TRUE
  
  # ------------------------
  # Interactive layout
  # ------------------------
  # Nodes can be dragged manually to separate overlapping variables.
  # Physics-based layout automatically spreads nodes but allows
  # manual repositioning for publication-quality arrangement.
  
  visNetwork(nodes, edges, width = "100%", height = "720px") %>%
    
    visEdges(
      arrows = "to"
    ) %>%
    
    visNodes(
      borderWidth = 2,
      font = list(
        color = "navy",
        size = 20,
        align = "center"   # label centered inside node
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
      dragNodes = TRUE,   # allows manual node movement
      dragView = TRUE,
      zoomView = TRUE
    ) %>%
    
    # ------------------------
  # Custom legend (publication cleaner)
  # ------------------------
  visLegend(
    addNodes = data.frame(
      label = c(
        "Activation (+)\n────────────▶",
        "Inhibition (–)\n────────────▶",
        "Edge width ∝ |ω| (effect size)"
      ),
      shape = "text",
      font = list(
        size = 18,
        color = c("#2ca02c", "#d62728", "black")
      )
    ),
    useGroups = FALSE,
    position = "right",
    width = 0.3,
    ncol = 1
  )
}

# --------------------------------------------------
# 7.6 CLEAN multi-panel CIRN network 
# --------------------------------------------------
# Purpose:
#   • Compare CIRN-inferred regulatory structure across
#     predictor representations
#   • Panels correspond to:
#       1) Original state variables
#       2) First temporal derivatives
#       3) Second temporal derivatives
#
# Visual encoding is consistent across panels:
#   • Edge color : regulatory sign
#   • Edge width : |ω_ij| (posterior mean magnitude)
#
# Intended for APPENDIX or supplementary presentation.
# --------------------------------------------------

plot_cirn_panels <- function(edges) {
  
  # ------------------------------------------------
  # Helper function: single panel
  # ------------------------------------------------
  plot_single_panel <- function(edges_sub, panel_title) {
    
    # If no edges exist for this representation
    if (nrow(edges_sub) == 0) {
      return(
        div(style="flex:1; text-align:center;",
            h3(panel_title))
      )
    }
    
    # Build node list
    node_ids <- unique(c(edges_sub$term, edges_sub$target))
    
    nodes <- data.frame(
      id = node_ids,
      label = format_cirn_node_label(node_ids),
      shape = "circle",
      size = 40,
      color.background = "white",
      color.border = "black",
      borderWidth = 2
    )
    
    # -------------------------
    # Edge styling
    # -------------------------
    edges_vis <- data.frame(
      from = edges_sub$term,
      to   = edges_sub$target,
      color = ifelse(
        edges_sub$regulation_type == "activation",
        "#2ca02c",   # green
        "#d62728"    # red
      ),
      width = scales::rescale(
        abs(edges_sub$omega),
        to = c(2, 6)
      ),
      arrows = "to",
      smooth = TRUE
    )
    
    # Interactive network panel
    div(
      style="flex:1; padding:10px;",
      
      h3(panel_title, style="text-align:center;"),
      
      visNetwork(nodes, edges_vis, width="100%", height="500px") %>%
        
        visEdges(arrows="to") %>%
        
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
        
        visLayout(randomSeed = 123)
    )
  }
  
  # -------------------------
  # Panel 1: Original predictors
  # -------------------------
  p1 <- plot_single_panel(
    edges %>% dplyr::filter(predictor_set == "original"),
    "Original State Predictors"
  )
  
  # -------------------------
  # Panel 2: First derivatives
  # -------------------------
  p2 <- plot_single_panel(
    edges %>% dplyr::filter(predictor_set == "first_derivative"),
    "First-Derivative Predictors"
  )
  
  # -------------------------
  # Panel 3: Second derivatives
  # -------------------------
  p3 <- plot_single_panel(
    edges %>% dplyr::filter(predictor_set == "second_derivative"),
    "Second-Derivative Predictors"
  )
  
  # -------------------------
  # Unified legend (bottom)
  # -------------------------
  legend_html <- div(
    style="text-align:center; margin-top:15px;",
    HTML("
      <b>Legend</b><br>
      <span style='color:#2ca02c;'>Activation (+)</span>
      &nbsp;&nbsp;&nbsp;
      <span style='color:#d62728;'>Inhibition (−)</span>
      &nbsp;&nbsp;&nbsp;
      Edge width ∝ |ω| (effect size)
    ")
  )
  
  # Combine panels
  page <- tagList(
    div(style="display:flex; justify-content:space-between;", p1, p2, p3),
    legend_html
  )
  
  # Save and open interactive figure
  html_file <- tempfile(fileext = ".html")
  htmltools::browsable(page)
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
#               (original state, first derivative, second derivative)
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
          levels = c("original", "first_derivative", "second_derivative")
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


##### Dataset: Simple Network 2


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

my_data <- read.csv("Simple_Network_4.csv") %>%   # replace this with your dataset
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
# CIRN evaluates three representations:
#   1) Original state variables
#   2) First temporal derivatives
#   3) Second temporal derivatives
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

targets <- c("X", "Y")         # If there are endogenous variable; allowed targets
predictors <- c("X", "Y")      # Allowed regulators

# Debug tables are useful for checking preprocessing and lagging.
# Keep FALSE for routine dissertation runs to avoid cluttering RStudio.
debug <- TRUE
assign_debug_to_global <- TRUE

CIRN <- system.time(
  res <- infer_network(
    df = sim_df,
    time_col = time_col,
    targets = targets,
    predictors = predictors,
    debug = debug,
    assign_debug_to_global = assign_debug_to_global
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
diagnostics_table <- res$diagnostics
vif_group_table <- res$vif_group
vif_pair_table <- res$vif_pairs
debug_tables <- res$debug

print(edges)
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

target_name <- targets[2]    # Change the index to inspect another target.

model_original <- res$models[[target_name]][["original"]]
model_first_derivative <- res$models[[target_name]][["first_derivative"]]
model_second_derivative <- res$models[[target_name]][["second_derivative"]]

stopifnot(
  inherits(model_original, "brmsfit"),
  inherits(model_first_derivative, "brmsfit"),
  inherits(model_second_derivative, "brmsfit")
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

# Example predator–prey structure
true_adj["X", "Y"] <-  1
true_adj["Y", "X"] <- -1

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

plot_all_derivatives(
  df = sim_df,
  vars = vars,
  time_col = time_col
)

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

# Main-text figure:
#   • Represents the aggregated CIRN network
#   • Intended as the primary graphical result of the algorithm

plot_network_main(edges)

# Appendix figure:
#   • Displays networks inferred from each CIRN representation
#   • Supports assessment of robustness across dynamical assumptions

plot_cirn_panels(edges)

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
#   • Shown separately for each CIRN representation
#   • Intercepts are excluded to focus on regulatory effects

coef_names_original <- rownames(brms::fixef(model_original))
coef_names_original <- coef_names_original[coef_names_original != "Intercept"]

coef_names_1stDerivative <- rownames(brms::fixef(model_first_derivative))
coef_names_1stDerivative <- coef_names_1stDerivative[coef_names_1stDerivative != "Intercept"]

coef_names_2ndDerivative <- rownames(brms::fixef(model_second_derivative))
coef_names_2ndDerivative <- coef_names_2ndDerivative[coef_names_2ndDerivative != "Intercept"]

suppressWarnings(
  plot_all_posterior_effects(
    model = model_original,
    terms = coef_names_original
  )
)

suppressWarnings(
  plot_all_posterior_effects(
    model = model_first_derivative,
    terms = coef_names_1stDerivative
  )
)

suppressWarnings(
  plot_all_posterior_effects(
    model = model_second_derivative,
    terms = coef_names_2ndDerivative
  )
)

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

suppressWarnings(
  plot_mcmc_hist_trace(model_original)
)

suppressWarnings(
  plot_mcmc_hist_trace(model_first_derivative)
)

suppressWarnings(
  plot_mcmc_hist_trace(model_second_derivative)
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
#          (original, first derivative, second derivative)
# ---------------------------------------------------------------
# pdf("FigS2_CIRN_Representation_Panels.pdf", width = 12, height = 4)
# plot_cirn_panels(edges)
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

