################################################################################
# GRANGER CAUSALITY COMPARATOR FOR CIRN
################################################################################
#
# Purpose:
#   Baseline directed predictive network inference using pairwise Granger
#   causality. The current comparator reports directed predictive evidence.
#   It does not assign activation or inhibition signs by default.
#
# Main safeguards:
#   - No hardcoded working directory or dataset.
#   - Numeric non-time variables are selected automatically unless specified.
#   - Stationarity is checked with the Augmented Dickey-Fuller test when possible.
#   - First differencing can be applied consistently when nonstationarity is found.
#   - Near-perfect collinearity and failed tests fail closed.
#   - Output uses a CIRN-compatible edge-list schema where sign is NA by default.
################################################################################

required_packages <- c("dplyr", "tidyr", "lmtest", "tseries", "ggplot2")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    "Missing required package(s): ",
    paste(missing_packages, collapse = ", "),
    ". Install them before running this script.",
    call. = FALSE
  )
}

library(dplyr)
library(tidyr)
library(lmtest)
library(tseries)
library(ggplot2)

script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) getwd()
)

read_input_csv <- function(input_file) {
  input_path <- if (grepl("^(/|[A-Za-z]:)", input_file)) {
    input_file
  } else {
    file.path(script_dir, input_file)
  }
  if (!file.exists(input_path)) {
    stop("Input file not found: ", input_path, call. = FALSE)
  }
  read.csv(input_path)
}

validate_analysis_columns <- function(df, time_col, variables = NULL) {
  if (!time_col %in% names(df)) {
    stop("time_col not found in data: ", time_col, call. = FALSE)
  }

  candidate_vars <- setdiff(names(df), time_col)
  numeric_vars <- candidate_vars[
    vapply(df[candidate_vars], is.numeric, logical(1))
  ]

  if (is.null(variables)) {
    variables <- numeric_vars
  }

  missing_vars <- setdiff(variables, names(df))
  if (length(missing_vars) > 0) {
    stop("Requested variable(s) not found: ", paste(missing_vars, collapse = ", "), call. = FALSE)
  }

  nonnumeric_vars <- variables[
    !vapply(df[variables], is.numeric, logical(1))
  ]
  if (length(nonnumeric_vars) > 0) {
    stop("Granger variables must be numeric: ", paste(nonnumeric_vars, collapse = ", "), call. = FALSE)
  }

  if (length(variables) < 2) {
    stop("At least two numeric variables are required for Granger inference.", call. = FALSE)
  }

  variables
}

safe_adf_pvalue <- function(x) {
  if (length(unique(stats::na.omit(x))) < 3) {
    return(NA_real_)
  }
  tryCatch(
    suppressWarnings(tseries::adf.test(x)$p.value),
    error = function(e) NA_real_
  )
}

check_stationarity <- function(df,
                               time_col,
                               variables = NULL,
                               alpha = 0.05,
                               first_difference_if_any = TRUE) {
  variables <- validate_analysis_columns(df, time_col, variables)
  stationarity_table <- data.frame(
    variable = variables,
    adf_p_value = vapply(df[variables], safe_adf_pvalue, numeric(1)),
    stringsAsFactors = FALSE
  )

  stationarity_table$nonstationary <- ifelse(
    is.na(stationarity_table$adf_p_value),
    NA,
    stationarity_table$adf_p_value > alpha
  )

  should_difference <- first_difference_if_any &&
    any(stationarity_table$nonstationary %in% TRUE)

  df_out <- df
  if (should_difference) {
    for (v in variables) {
      df_out[[v]] <- c(NA_real_, diff(df[[v]]))
    }
    df_out <- stats::na.omit(df_out)
  }

  list(
    data = df_out,
    stationarity = stationarity_table,
    differenced = should_difference
  )
}

safe_granger_test <- function(target, cause, max_lag) {
  tryCatch(
    lmtest::grangertest(target ~ cause, order = max_lag),
    error = function(e) NULL
  )
}

infer_granger_network <- function(df,
                                  time_col,
                                  variables = NULL,
                                  max_lag = 10,
                                  p_threshold = 0.05,
                                  collinearity_threshold = 0.999,
                                  p_adjust_method = "none") {
  variables <- validate_analysis_columns(df, time_col, variables)
  p_adjust_method <- match.arg(
    p_adjust_method,
    choices = c("none", stats::p.adjust.methods)
  )

  if (length(max_lag) != 1 || is.na(max_lag) || max_lag < 1 || max_lag != floor(max_lag)) {
    stop("max_lag must be a positive integer.", call. = FALSE)
  }
  max_lag <- as.integer(max_lag)

  if (nrow(df) <= (max_lag + 2L)) {
    stop("Not enough rows for requested max_lag.", call. = FALSE)
  }

  data_mat <- df[, variables, drop = FALSE]
  start_time <- Sys.time()

  test_rows <- list()
  row_id <- 1L

  for (cause in variables) {
    for (target in variables) {
      if (cause == target) {
        next
      }

      corr_val <- stats::cor(data_mat[[cause]], data_mat[[target]], use = "pairwise.complete.obs")
      if (!is.na(corr_val) && abs(corr_val) > collinearity_threshold) {
        test_rows[[row_id]] <- data.frame(
          Cause = cause,
          Effect = target,
          p_value = NA_real_,
          Strength = NA_real_,
          status = "skipped_collinearity",
          stringsAsFactors = FALSE
        )
        row_id <- row_id + 1L
        next
      }

      test <- safe_granger_test(
        target = data_mat[[target]],
        cause = data_mat[[cause]],
        max_lag = max_lag
      )

      if (is.null(test)) {
        test_rows[[row_id]] <- data.frame(
          Cause = cause,
          Effect = target,
          p_value = NA_real_,
          Strength = NA_real_,
          status = "failed_test",
          stringsAsFactors = FALSE
        )
        row_id <- row_id + 1L
        next
      }

      p_value <- test$`Pr(>F)`[2]
      strength <- -log10(max(p_value, 1e-16))

      test_rows[[row_id]] <- data.frame(
        Cause = cause,
        Effect = target,
        p_value = p_value,
        Strength = strength,
        status = "tested",
        stringsAsFactors = FALSE
      )
      row_id <- row_id + 1L
    }
  }

  test_table <- dplyr::bind_rows(test_rows)
  tested_mask <- !is.na(test_table$p_value)
  test_table$p_adjusted <- NA_real_
  test_table$p_adjusted[tested_mask] <- if (p_adjust_method == "none") {
    test_table$p_value[tested_mask]
  } else {
    stats::p.adjust(test_table$p_value[tested_mask], method = p_adjust_method)
  }

  p_matrix <- matrix(
    NA_real_,
    nrow = length(variables),
    ncol = length(variables),
    dimnames = list(variables, variables)
  )
  strength_matrix <- p_matrix

  if (nrow(test_table) > 0) {
    for (i in seq_len(nrow(test_table))) {
      cause <- test_table$Cause[[i]]
      effect <- test_table$Effect[[i]]
      p_matrix[cause, effect] <- test_table$p_value[[i]]
      strength_matrix[cause, effect] <- test_table$Strength[[i]]
    }
  }

  retained <- test_table %>%
    dplyr::filter(
      .data$status == "tested",
      !is.na(.data$p_adjusted),
      .data$p_adjusted < p_threshold
    )

  edges <- if (nrow(retained) > 0) {
    retained %>%
      dplyr::transmute(
        predictor_set = "granger_causality",
        term = .data$Cause,
        target = .data$Effect,
        omega = .data$Strength,
        odds_ratio = NA_real_,
        sign = NA_real_,
        regulation_type = "directed_predictive",
        rel_strength = .data$Strength,
        hdi_lower95 = NA_real_,
        hdi_upper95 = NA_real_,
        eti_lower95 = NA_real_,
        eti_upper95 = NA_real_,
        method = "Granger causality",
        analysis_mode = "pairwise_directed_predictive",
        p_value = .data$p_value,
        p_adjusted = .data$p_adjusted,
        status = .data$status
      )
  } else {
    data.frame()
  }

  runtime_seconds <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  list(
    edges = edges,
    test_table = test_table,
    runtime = runtime_seconds,
    p_matrix = p_matrix,
    strength_matrix = strength_matrix,
    settings = list(
      max_lag = max_lag,
      p_threshold = p_threshold,
      p_adjust_method = p_adjust_method,
      collinearity_threshold = collinearity_threshold
    )
  )
}

plot_granger_heatmap <- function(p_matrix) {
  heatmap_df <- as.data.frame(p_matrix) %>%
    tibble::rownames_to_column("Cause") %>%
    tidyr::pivot_longer(-Cause, names_to = "Target", values_to = "p_value") %>%
    dplyr::filter(!is.na(.data$p_value))

  ggplot2::ggplot(heatmap_df, ggplot2::aes(.data$Cause, .data$Target, fill = -log10(.data$p_value))) +
    ggplot2::geom_tile(color = "white") +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.2e", .data$p_value)), size = 4) +
    ggplot2::scale_fill_gradient(low = "#e8f8f5", high = "#27ae60", name = "-log10(p)") +
    ggplot2::labs(title = "Granger Causality Significance", x = "Cause", y = "Target") +
    ggplot2::theme_minimal(base_size = 13)
}

################################################################################
# User-configurable run block
################################################################################

input_file <- NULL
cli_args <- commandArgs(trailingOnly = TRUE)
if (is.null(input_file) && length(cli_args) >= 1 && nzchar(cli_args[1])) {
  input_file <- cli_args[1]
}

if (!is.null(input_file)) {
  sim_df <- read_input_csv(input_file) %>% stats::na.omit()
  time_col <- names(sim_df)[1]

  granger_config <- list(
    variables = NULL,
    stationarity_alpha = 0.05,
    first_difference_if_any = TRUE,
    max_lag = 10,
    p_threshold = 0.05,
    p_adjust_method = "none",
    collinearity_threshold = 0.999
  )

  stationarity_res <- check_stationarity(
    df = sim_df,
    time_col = time_col,
    variables = granger_config$variables,
    alpha = granger_config$stationarity_alpha,
    first_difference_if_any = granger_config$first_difference_if_any
  )

  granger_res <- infer_granger_network(
    df = stationarity_res$data,
    time_col = time_col,
    variables = granger_config$variables,
    max_lag = granger_config$max_lag,
    p_threshold = granger_config$p_threshold,
    collinearity_threshold = granger_config$collinearity_threshold,
    p_adjust_method = granger_config$p_adjust_method
  )

  granger_edges <- granger_res$edges
  print(stationarity_res$stationarity)
  print(granger_edges)
}

################################################################################
# END
################################################################################
