################################################################################
# PARTIAL CROSS MAPPING COMPARATOR FOR CIRN
################################################################################
#
# Purpose:
#   Nonlinear state-space comparator based on Partial Cross Mapping (PCM).
#   PCM provides directed conditional cross-map evidence, but the current
#   comparator does not directly assign activation or inhibition signs.
#
# Main safeguards:
#   - No hardcoded working directory or dataset.
#   - Numeric non-time variables are selected automatically unless specified.
#   - PCM is skipped cleanly when fewer than three variables are available.
#   - All ordered pairs are scored; retained edges require explicit thresholding.
#   - Output uses a CIRN-compatible edge-list schema where sign is NA.
################################################################################

required_packages <- c("dplyr", "tidyr", "ggplot2", "tEDM")
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
library(tEDM)

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
    stop("PCM variables must be numeric: ", paste(nonnumeric_vars, collapse = ", "), call. = FALSE)
  }

  variables
}

safe_pcm <- function(df, cause, effect, conds, libsizes, E, tau, threads) {
  tryCatch(
    tEDM::pcm(
      data = df,
      cause = cause,
      effect = effect,
      conds = conds,
      libsizes = libsizes,
      E = E,
      tau = tau,
      threads = threads
    ),
    error = function(e) NULL
  )
}

extract_pcm_strength <- function(pcm_res) {
  if (is.null(pcm_res) || is.null(pcm_res$pxmap)) {
    return(NA_real_)
  }

  pxmap <- pcm_res$pxmap
  numeric_cols <- names(pxmap)[vapply(pxmap, is.numeric, logical(1))]
  numeric_cols <- setdiff(numeric_cols, c("LibSize", "libSize", "librarySize"))
  if (length(numeric_cols) == 0) {
    return(NA_real_)
  }

  as.numeric(tail(pxmap[[numeric_cols[[1]]]], 1))
}

infer_network_pcm <- function(df,
                              time_col,
                              variables = NULL,
                              E = 3,
                              tau = 1,
                              libsizes = seq(10, 100, 10),
                              strength_threshold = 0.30,
                              threads = 1) {
  variables <- validate_analysis_columns(df, time_col, variables)

  if (length(variables) < 3) {
    empty_matrix <- matrix(
      NA_real_,
      nrow = length(variables),
      ncol = length(variables),
      dimnames = list(variables, variables)
    )
    return(list(
      edges = data.frame(),
      score_table = data.frame(),
      strength_matrix = empty_matrix,
      runtime = 0,
      status = "skipped_requires_at_least_three_variables"
    ))
  }

  if (length(E) != 1 || is.na(E) || E < 1 || E != floor(E)) {
    stop("E must be a positive integer.", call. = FALSE)
  }
  if (length(tau) != 1 || is.na(tau) || tau < 1 || tau != floor(tau)) {
    stop("tau must be a positive integer.", call. = FALSE)
  }

  start_time <- Sys.time()
  score_rows <- list()
  row_id <- 1L

  for (cause in variables) {
    for (effect in variables) {
      if (cause == effect) {
        next
      }

      conds <- setdiff(variables, c(cause, effect))
      pcm_res <- safe_pcm(
        df = df,
        cause = cause,
        effect = effect,
        conds = conds,
        libsizes = libsizes,
        E = as.integer(E),
        tau = as.integer(tau),
        threads = threads
      )

      strength <- extract_pcm_strength(pcm_res)
      retained <- !is.na(strength) && strength >= strength_threshold

      score_rows[[row_id]] <- data.frame(
        Cause = cause,
        Effect = effect,
        PCM_strength = strength,
        conditioning_variables = paste(conds, collapse = ";"),
        retained = retained,
        status = ifelse(is.null(pcm_res), "failed_pcm", "tested"),
        stringsAsFactors = FALSE
      )
      row_id <- row_id + 1L
    }
  }

  score_table <- dplyr::bind_rows(score_rows)

  strength_matrix <- matrix(
    NA_real_,
    nrow = length(variables),
    ncol = length(variables),
    dimnames = list(variables, variables)
  )

  if (nrow(score_table) > 0) {
    for (i in seq_len(nrow(score_table))) {
      strength_matrix[score_table$Cause[[i]], score_table$Effect[[i]]] <- score_table$PCM_strength[[i]]
    }
  }

  retained <- score_table %>% dplyr::filter(.data$retained)

  edges <- if (nrow(retained) > 0) {
    retained %>%
      dplyr::transmute(
        predictor_set = "pcm",
        term = .data$Cause,
        target = .data$Effect,
        omega = .data$PCM_strength,
        odds_ratio = NA_real_,
        sign = NA_real_,
        regulation_type = "directed_conditional_cross_map",
        rel_strength = .data$PCM_strength,
        hdi_lower95 = NA_real_,
        hdi_upper95 = NA_real_,
        eti_lower95 = NA_real_,
        eti_upper95 = NA_real_,
        method = "Partial Cross Mapping",
        analysis_mode = "pairwise_directed_state_space",
        conditioning_variables = .data$conditioning_variables,
        status = .data$status
      )
  } else {
    data.frame()
  }

  runtime_seconds <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  list(
    edges = edges,
    score_table = score_table,
    strength_matrix = strength_matrix,
    runtime = runtime_seconds,
    status = "completed",
    settings = list(
      E = E,
      tau = tau,
      libsizes = libsizes,
      strength_threshold = strength_threshold,
      threads = threads
    )
  )
}

plot_pcm_heatmap <- function(strength_matrix) {
  df <- as.data.frame(strength_matrix) %>%
    tibble::rownames_to_column("Cause") %>%
    tidyr::pivot_longer(-Cause, names_to = "Effect", values_to = "strength") %>%
    dplyr::filter(!is.na(.data$strength))

  ggplot2::ggplot(df, ggplot2::aes(.data$Cause, .data$Effect, fill = .data$strength)) +
    ggplot2::geom_tile(color = "white") +
    ggplot2::geom_text(ggplot2::aes(label = round(.data$strength, 2)), size = 5) +
    ggplot2::scale_fill_gradient(low = "white", high = "darkblue", name = "PCM") +
    ggplot2::theme_minimal(base_size = 13) +
    ggplot2::labs(title = "Partial Cross Mapping", x = "Cause", y = "Effect")
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

  pcm_config <- list(
    variables = NULL,
    E = 3,
    tau = 1,
    libsizes = seq(10, 100, 10),
    strength_threshold = 0.30,
    threads = 1
  )

  pcm_res <- infer_network_pcm(
    df = sim_df,
    time_col = time_col,
    variables = pcm_config$variables,
    E = pcm_config$E,
    tau = pcm_config$tau,
    libsizes = pcm_config$libsizes,
    strength_threshold = pcm_config$strength_threshold,
    threads = pcm_config$threads
  )

  pcm_edges <- pcm_res$edges
  print(pcm_edges)
}

################################################################################
# END
################################################################################
