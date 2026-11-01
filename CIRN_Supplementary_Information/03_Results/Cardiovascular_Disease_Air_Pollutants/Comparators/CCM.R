################################################################################
# CONVERGENT CROSS MAPPING COMPARATOR FOR CIRN
################################################################################
#
# Purpose:
#   Nonlinear state-space comparator based on Convergent Cross Mapping (CCM).
#   CCM provides directed cross-map evidence, but the current comparator does
#   not directly assign activation or inhibition signs.
#
# Direction convention:
#   To evaluate cause -> effect, the effect manifold is used to cross-map the
#   putative cause. In rEDM terms, this is columns = effect, target = cause.
#
# Main safeguards:
#   - No hardcoded working directory or dataset.
#   - Numeric non-time variables are selected automatically unless specified.
#   - Embedding dimension can be supplied or estimated from the data.
#   - All ordered pairs are scored; retained edges require explicit thresholding.
#   - Output uses a CIRN-compatible edge-list schema where sign is NA.
################################################################################

required_packages <- c("dplyr", "tidyr", "ggplot2", "rEDM")
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
library(ggplot2)
library(rEDM)

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
    stop("CCM variables must be numeric: ", paste(nonnumeric_vars, collapse = ", "), call. = FALSE)
  }

  if (length(variables) < 2) {
    stop("At least two numeric variables are required for CCM inference.", call. = FALSE)
  }

  variables
}

select_embedding_dimension <- function(df,
                                       variable,
                                       max_E = 10,
                                       lib = NULL,
                                       pred = NULL) {
  if (is.null(lib)) {
    lib <- c(1, nrow(df))
  }
  if (is.null(pred)) {
    pred <- c(1, nrow(df))
  }

  embed_out <- rEDM::EmbedDimension(
    dataFrame = df,
    lib = lib,
    pred = pred,
    columns = variable,
    target = variable,
    maxE = max_E
  )

  embed_out$E[which.max(embed_out$rho)]
}

safe_ccm <- function(df, E, columns, target, lib_sizes, num_samples) {
  tryCatch(
    rEDM::CCM(
      dataFrame = df,
      E = E,
      columns = columns,
      target = target,
      libSizes = lib_sizes,
      sample = num_samples
    ),
    error = function(e) NULL
  )
}

extract_final_rho <- function(ccm_out) {
  if (is.null(ccm_out)) {
    return(NA_real_)
  }

  numeric_cols <- names(ccm_out)[vapply(ccm_out, is.numeric, logical(1))]
  numeric_cols <- setdiff(numeric_cols, c("LibSize", "libSize", "librarySize"))
  if (length(numeric_cols) == 0) {
    return(NA_real_)
  }

  value <- tail(ccm_out[[numeric_cols[[1]]]], 1)
  as.numeric(value)
}

infer_network_ccm <- function(df,
                              time_col,
                              variables = NULL,
                              E = NULL,
                              max_E = 10,
                              lib_sizes = NULL,
                              num_samples = 100,
                              rho_threshold = 0.30,
                              min_rho_gain = 0.00) {
  variables <- validate_analysis_columns(df, time_col, variables)

  if (is.null(E)) {
    E_values <- vapply(
      variables,
      function(v) select_embedding_dimension(df, v, max_E = max_E),
      numeric(1)
    )
    E <- as.integer(round(stats::median(E_values, na.rm = TRUE)))
  } else {
    E_values <- rep(E, length(variables))
    names(E_values) <- variables
  }

  if (length(E) != 1 || is.na(E) || E < 1 || E != floor(E)) {
    stop("E must be a positive integer.", call. = FALSE)
  }
  E <- as.integer(E)

  N <- nrow(df)
  max_lib <- N - (E - 1L)
  if (max_lib < 5) {
    stop("Not enough rows for the requested embedding dimension.", call. = FALSE)
  }

  if (is.null(lib_sizes)) {
    start_lib <- max(5L, floor(max_lib * 0.20))
    lib_sizes <- paste(start_lib, max_lib, 5)
  }

  start_time <- Sys.time()
  score_rows <- list()
  row_id <- 1L

  for (cause in variables) {
    for (effect in variables) {
      if (cause == effect) {
        next
      }

      ccm_out <- safe_ccm(
        df = df,
        E = E,
        columns = effect,
        target = cause,
        lib_sizes = lib_sizes,
        num_samples = num_samples
      )

      final_rho <- extract_final_rho(ccm_out)

      initial_rho <- NA_real_
      if (!is.null(ccm_out)) {
        numeric_cols <- names(ccm_out)[vapply(ccm_out, is.numeric, logical(1))]
        numeric_cols <- setdiff(numeric_cols, c("LibSize", "libSize", "librarySize"))
        if (length(numeric_cols) > 0) {
          initial_rho <- as.numeric(head(ccm_out[[numeric_cols[[1]]]], 1))
        }
      }

      rho_gain <- final_rho - initial_rho
      retained <- !is.na(final_rho) &&
        final_rho >= rho_threshold &&
        !is.na(rho_gain) &&
        rho_gain >= min_rho_gain

      score_rows[[row_id]] <- data.frame(
        Cause = cause,
        Effect = effect,
        CCM_rho = final_rho,
        CCM_rho_initial = initial_rho,
        CCM_rho_gain = rho_gain,
        retained = retained,
        status = ifelse(is.null(ccm_out), "failed_ccm", "tested"),
        stringsAsFactors = FALSE
      )
      row_id <- row_id + 1L
    }
  }

  score_table <- dplyr::bind_rows(score_rows)

  rho_matrix <- matrix(
    NA_real_,
    nrow = length(variables),
    ncol = length(variables),
    dimnames = list(variables, variables)
  )

  if (nrow(score_table) > 0) {
    for (i in seq_len(nrow(score_table))) {
      rho_matrix[score_table$Cause[[i]], score_table$Effect[[i]]] <- score_table$CCM_rho[[i]]
    }
  }

  retained <- score_table %>% dplyr::filter(.data$retained)

  edges <- if (nrow(retained) > 0) {
    retained %>%
      dplyr::transmute(
        predictor_set = "ccm",
        term = .data$Cause,
        target = .data$Effect,
        omega = .data$CCM_rho,
        odds_ratio = NA_real_,
        sign = NA_real_,
        regulation_type = "directed_cross_map",
        rel_strength = .data$CCM_rho,
        hdi_lower95 = NA_real_,
        hdi_upper95 = NA_real_,
        eti_lower95 = NA_real_,
        eti_upper95 = NA_real_,
        method = "Convergent Cross Mapping",
        analysis_mode = "pairwise_directed_state_space",
        ccm_rho_initial = .data$CCM_rho_initial,
        ccm_rho_gain = .data$CCM_rho_gain,
        status = .data$status
      )
  } else {
    data.frame()
  }

  runtime_seconds <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  list(
    edges = edges,
    score_table = score_table,
    rho_matrix = rho_matrix,
    runtime = runtime_seconds,
    E = E,
    E_values = E_values,
    settings = list(
      E = E,
      max_E = max_E,
      lib_sizes = lib_sizes,
      num_samples = num_samples,
      rho_threshold = rho_threshold,
      min_rho_gain = min_rho_gain
    )
  )
}

plot_ccm_heatmap <- function(rho_matrix) {
  df <- as.data.frame(rho_matrix) %>%
    tibble::rownames_to_column("Cause") %>%
    tidyr::pivot_longer(-Cause, names_to = "Effect", values_to = "rho") %>%
    dplyr::filter(!is.na(.data$rho))

  ggplot2::ggplot(df, ggplot2::aes(.data$Cause, .data$Effect, fill = .data$rho)) +
    ggplot2::geom_tile(color = "white") +
    ggplot2::geom_text(ggplot2::aes(label = round(.data$rho, 2)), size = 5) +
    ggplot2::scale_fill_gradient(low = "#e8f8f5", high = "#27ae60", name = "rho") +
    ggplot2::theme_minimal(base_size = 13) +
    ggplot2::labs(title = "Convergent Cross Mapping", x = "Cause", y = "Effect")
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

  ccm_config <- list(
    variables = NULL,
    E = NULL,
    max_E = 10,
    lib_sizes = NULL,
    num_samples = 100,
    rho_threshold = 0.30,
    min_rho_gain = 0.00
  )

  ccm_res <- infer_network_ccm(
    df = sim_df,
    time_col = time_col,
    variables = ccm_config$variables,
    E = ccm_config$E,
    max_E = ccm_config$max_E,
    lib_sizes = ccm_config$lib_sizes,
    num_samples = ccm_config$num_samples,
    rho_threshold = ccm_config$rho_threshold,
    min_rho_gain = ccm_config$min_rho_gain
  )

  ccm_edges <- ccm_res$edges
  print(ccm_edges)
}

################################################################################
# END
################################################################################
