options(shiny.maxRequestSize = 200 * 1024^2)

required_packages <- c(
  "shiny", "bslib", "DT", "ggplot2", "dplyr", "tidyr", "purrr",
  "tibble", "readr", "readxl", "openxlsx", "visNetwork", "igraph",
  "jsonlite", "deSolve", "randomForest", "brms", "cmdstanr",
  "bayestestR", "posterior", "bayesplot", "car", "broom",
  "htmltools", "htmlwidgets", "scales"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "The CIRN Studio app needs these packages: ",
    paste(missing_packages, collapse = ", "),
    ". Install them before launching the app."
  )
}

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(DT)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(tibble)
  library(readr)
  library(readxl)
  library(openxlsx)
  library(visNetwork)
  library(igraph)
  library(jsonlite)
  library(deSolve)
  library(randomForest)
  library(brms)
  library(cmdstanr)
  library(posterior)
  library(bayestestR)
  library(car)
  library(broom)
  library(scales)
  library(htmltools)
  library(htmlwidgets)
})

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

first_or <- function(x, default = "") {
  if (is.null(x) || length(x) == 0) {
    default
  } else {
    x[[1]]
  }
}

app_dir <- normalizePath(getwd(), mustWork = TRUE)
algorithm_path <- file.path(app_dir, "CIRN_Algorithm.R")
example_path <- file.path(app_dir, "data", "Predator_Prey.csv")

# Load reusable CIRN definitions without executing the script-style analysis
# section near the end of CIRN_Algorithm.R.
assignment_name <- function(expr) {
  if (!is.call(expr) || length(expr) < 3) {
    return(NULL)
  }
  op <- as.character(expr[[1]])
  if (length(op) == 0 || !op %in% c("<-", "=")) {
    return(NULL)
  }
  lhs <- expr[[2]]
  if (is.symbol(lhs)) {
    return(as.character(lhs))
  }
  NULL
}

assignment_rhs <- function(expr) {
  if (!is.call(expr) || length(expr) < 3) {
    return(NULL)
  }
  expr[[3]]
}

is_function_assignment <- function(expr) {
  rhs <- assignment_rhs(expr)
  is.call(rhs) && identical(rhs[[1]], as.name("function"))
}

load_cirn_definitions <- function(path, env = .GlobalEnv) {
  if (!file.exists(path)) {
    stop("CIRN_Algorithm.R was not found at: ", path)
  }

  exprs <- parse(path)
  loaded <- character()

  for (expr in exprs) {
    name <- assignment_name(expr)
    if (is.null(name)) {
      next
    }

    should_load <- is_function_assignment(expr) || grepl("^empty_", name)
    if (!should_load) {
      next
    }

    eval(expr, envir = env)
    loaded <- c(loaded, name)
  }

  required_objects <- c(
    "compute_derivatives",
    "infer_network",
    "plot_adaptive_jitter_diagnostic",
    "evaluate_representation_agnostic",
    "metrics_to_long_table",
    "make_cirn_sensitivity_plan",
    "resolve_cirn_sensitivity_scope",
    "run_cirn_sensitivity_analysis",
    "summarize_cirn_sensitivity_feature_edge_stability",
    "format_cirn_node_label"
  )

  missing_objects <- required_objects[!vapply(required_objects, exists, logical(1), envir = env)]
  if (length(missing_objects) > 0) {
    stop(
      "The following CIRN definitions were not loaded: ",
      paste(missing_objects, collapse = ", ")
    )
  }

  invisible(loaded)
}

load_cirn_definitions(algorithm_path, env = environment())

safe_dt <- function(data, page_length = 8, scroll_x = TRUE, scroll_y = NULL) {
  dt_options <- list(
    pageLength = page_length,
    scrollX = scroll_x,
    autoWidth = TRUE,
    dom = "tip"
  )

  if (!is.null(scroll_y) && length(scroll_y) == 1L && nzchar(scroll_y)) {
    dt_options$scrollY <- scroll_y
    dt_options$scrollCollapse <- TRUE
  }

  DT::datatable(
    data,
    rownames = FALSE,
    filter = "top",
    options = dt_options
  )
}

panel_box <- function(title, ..., class = "studio-panel") {
  tags$div(
    class = class,
    tags$h3(title),
    ...
  )
}

contextual_tab_header <- function(title, purpose, expected_input, resulting_output) {
  tags$section(
    class = "guide-hero contextual-tab-header",
    `aria-label` = paste(title, "tab overview"),
    tags$h2(title),
    tags$p(class = "contextual-tab-purpose", purpose),
    tags$div(
      class = "contextual-tab-cues",
      tags$p(
        class = "contextual-tab-cue",
        tags$strong("Expected input:"),
        tags$span(expected_input)
      ),
      tags$p(
        class = "contextual-tab-cue",
        tags$strong("Resulting output:"),
        tags$span(resulting_output)
      )
    )
  )
}

metric_box <- function(label, value_output) {
  tags$div(
    class = "metric-card",
    tags$div(class = "label", label),
    tags$div(class = "value", uiOutput(value_output))
  )
}

status_pill <- function(label, type = "info", title = NULL) {
  tags$span(
    class = paste("status-pill", paste0("status-", type)),
    title = title,
    tags$span(class = "status-pill-label", label)
  )
}

help_line <- function(text) {
  tags$p(class = "subtle", text)
}

button_label <- function(...) {
  tags$span(
    ...,
    class = "studio-action-button-label",
    style = "color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;opacity:1 !important;"
  )
}

read_any_table <- function(file_info) {
  req(file_info)
  ext <- tolower(tools::file_ext(file_info$name))

  if (ext %in% c("xlsx", "xls")) {
    as.data.frame(readxl::read_excel(file_info$datapath), check.names = FALSE)
  } else {
    as.data.frame(
      readr::read_csv(file_info$datapath, show_col_types = FALSE),
      check.names = FALSE
    )
  }
}

read_matrix_file <- function(file_info) {
  dat <- read_any_table(file_info)
  if (nrow(dat) == 0 || ncol(dat) == 0) {
    stop("The uploaded adjacency file is empty.")
  }

  first_col <- dat[[1]]
  first_col_is_labels <- !is.numeric(first_col) ||
    anyDuplicated(names(dat)) ||
    names(dat)[1] %in% c("", "X", "...1", "row", "node", "source")

  if (first_col_is_labels && ncol(dat) > 1) {
    row_ids <- as.character(first_col)
    dat <- dat[-1]
  } else {
    row_ids <- names(dat)
  }

  mat <- as.matrix(data.frame(lapply(dat, as.numeric), check.names = FALSE))
  colnames(mat) <- names(dat)
  rownames(mat) <- row_ids

  if (nrow(mat) != ncol(mat)) {
    stop("Adjacency matrix must be square.")
  }
  if (is.null(rownames(mat)) || is.null(colnames(mat)) ||
      !all(rownames(mat) %in% colnames(mat))) {
    stop("Adjacency matrix needs matching row and column names.")
  }

  mat <- mat[rownames(mat), rownames(mat), drop = FALSE]
  mat[!is.finite(mat)] <- 0
  mat
}

numeric_columns <- function(df) {
  names(df)[vapply(df, is.numeric, logical(1))]
}

state_columns <- function(df, time_col) {
  setdiff(numeric_columns(df), time_col)
}

normalize_data <- function(df, time_col, method) {
  vars <- state_columns(df, time_col)
  if (length(vars) == 0 || method == "none") {
    return(df)
  }

  for (v in vars) {
    x <- df[[v]]
    if (method == "zscore") {
      s <- stats::sd(x, na.rm = TRUE)
      m <- mean(x, na.rm = TRUE)
      if (is.finite(s) && s > 0) {
        df[[v]] <- (x - m) / s
      }
    }
    if (method == "minmax") {
      lo <- min(x, na.rm = TRUE)
      hi <- max(x, na.rm = TRUE)
      if (is.finite(lo) && is.finite(hi) && hi > lo) {
        df[[v]] <- (x - lo) / (hi - lo)
      }
    }
  }
  df
}

quality_summary <- function(df, time_col) {
  vars <- state_columns(df, time_col)
  time_values <- df[[time_col]]
  time_diff <- diff(sort(unique(time_values)))

  warnings <- character()
  if (length(vars) == 0) {
    warnings <- c(warnings, "No numeric state variables detected.")
  }
  if (anyDuplicated(time_values)) {
    warnings <- c(warnings, "Duplicate time values detected; CIRN will average duplicates during derivative computation.")
  }
  if (any(!stats::complete.cases(df[, c(time_col, vars), drop = FALSE]))) {
    warnings <- c(warnings, "Missing values detected; CIRN inference uses complete cases for selected variables.")
  }
  if (length(time_diff) > 0 && max(time_diff, na.rm = TRUE) / min(time_diff, na.rm = TRUE) > 3) {
    warnings <- c(warnings, "Irregular time spacing detected; inspect derivative behavior carefully.")
  }
  if (nrow(df) < 20) {
    warnings <- c(warnings, "Small time series; Bayesian estimates and derivative signs may be unstable.")
  }

  list(
    missing = tibble(
      variable = names(df),
      missing_n = colSums(is.na(df)),
      missing_pct = round(100 * colMeans(is.na(df)), 2)
    ),
    time = tibble(
      metric = c("Rows", "Unique time points", "Minimum spacing", "Median spacing", "Maximum spacing"),
      value = c(
        nrow(df),
        length(unique(time_values)),
        ifelse(length(time_diff) > 0, min(time_diff, na.rm = TRUE), NA_real_),
        ifelse(length(time_diff) > 0, stats::median(time_diff, na.rm = TRUE), NA_real_),
        ifelse(length(time_diff) > 0, max(time_diff, na.rm = TRUE), NA_real_)
      )
    ),
    variables = tibble(
      variable = vars,
      mean = vapply(df[vars], mean, numeric(1), na.rm = TRUE),
      sd = vapply(df[vars], stats::sd, numeric(1), na.rm = TRUE),
      min = vapply(df[vars], min, numeric(1), na.rm = TRUE),
      median = vapply(df[vars], stats::median, numeric(1), na.rm = TRUE),
      max = vapply(df[vars], max, numeric(1), na.rm = TRUE)
    ),
    warnings = warnings
  )
}

fallback_derivatives <- function(df, time_col) {
  vars <- state_columns(df, time_col)
  df <- df[order(df[[time_col]]), , drop = FALSE]
  t <- df[[time_col]]
  for (v in vars) {
    x <- df[[v]]
    d1 <- rep(NA_real_, length(x))
    d2 <- rep(NA_real_, length(x))
    if (length(x) >= 3) {
      d1[1] <- (x[2] - x[1]) / (t[2] - t[1])
      d1[length(x)] <- (x[length(x)] - x[length(x) - 1]) / (t[length(x)] - t[length(x) - 1])
      d1[2:(length(x) - 1)] <- (x[3:length(x)] - x[1:(length(x) - 2)]) /
        (t[3:length(x)] - t[1:(length(x) - 2)])
      d2[1] <- (d1[2] - d1[1]) / (t[2] - t[1])
      d2[length(x)] <- (d1[length(x)] - d1[length(x) - 1]) /
        (t[length(x)] - t[length(x) - 1])
      d2[2:(length(x) - 1)] <- (d1[3:length(x)] - d1[1:(length(x) - 2)]) /
        (t[3:length(x)] - t[1:(length(x) - 2)])
    }
    df[[paste0(v, "_d1")]] <- d1
    df[[paste0(v, "_d2")]] <- d2
  }
  df
}

response_preview_table <- function(df_derivs, targets, eps) {
  purrr::map_dfr(targets, function(target) {
    d1 <- paste0(target, "_d1")
    if (!d1 %in% names(df_derivs)) {
      return(tibble(target = target, class_0 = NA_integer_, class_1 = NA_integer_, blank = NA_integer_, usable = NA_integer_))
    }
    x <- df_derivs[[d1]]
    class_1 <- sum(x > eps, na.rm = TRUE)
    class_0 <- sum(x < -eps, na.rm = TRUE)
    blank <- sum(abs(x) <= eps | is.na(x), na.rm = TRUE)
    tibble(
      target = target,
      class_0 = class_0,
      class_1 = class_1,
      blank = blank,
      usable = class_0 + class_1,
      minority_class = pmin(class_0, class_1),
      blank_pct = round(100 * blank / length(x), 2),
      status = dplyr::case_when(
        class_0 == 0 | class_1 == 0 ~ "one_class",
        pmin(class_0, class_1) < 5 ~ "sparse_class",
        blank / length(x) > 0.5 ~ "many_blanks",
        TRUE ~ "ready"
      )
    )
  })
}

base_term <- function(term) {
  if (exists("base_cirn_term", mode = "function")) {
    return(base_cirn_term(term))
  }
  term <- as.character(term)
  term <- sub("_d1$", "", term)
  term <- sub("_d2$", "", term)
  term
}

format_node <- function(x) {
  if (exists("format_cirn_node_label", mode = "function")) {
    return(format_cirn_node_label(x))
  }
  x
}

CIRN_ACTIVATION_COLOR <- "#2ca02c"
CIRN_INHIBITION_COLOR <- "#d62728"
CIRN_UNCERTAIN_COLOR <- "#66788A"
CIRN_STATE_COLOR <- "#f28e2b"
CIRN_FIRST_DERIVATIVE_COLOR <- "#1f77b4"
CIRN_SECOND_DERIVATIVE_COLOR <- "#cc2e9b"
CIRN_TARGET_BORDER_COLOR <- "#111111"
CIRN_STATE_FILL <- "#fff4e6"
CIRN_FIRST_DERIVATIVE_FILL <- "#e8f3ff"
CIRN_SECOND_DERIVATIVE_FILL <- "#ffe8f6"
CIRN_TARGET_FILL <- "#fff0b3"
CIRN_PLOT_BASE_SIZE <- 13.5

node_type_style <- function(node_ids, target_ids = character()) {
  node_type <- dplyr::case_when(
    node_ids %in% target_ids ~ "Target derivative response",
    grepl("_d1$", node_ids) ~ "First-derivative predictor",
    grepl("_d2$", node_ids) ~ "Second-derivative predictor",
    TRUE ~ "Original-state predictor"
  )

  tibble(
    node_type = node_type,
    background = dplyr::case_when(
      node_type == "Original-state predictor" ~ CIRN_STATE_FILL,
      node_type == "First-derivative predictor" ~ CIRN_FIRST_DERIVATIVE_FILL,
      node_type == "Second-derivative predictor" ~ CIRN_SECOND_DERIVATIVE_FILL,
      TRUE ~ CIRN_TARGET_FILL
    ),
    border = dplyr::case_when(
      node_type == "Original-state predictor" ~ CIRN_STATE_COLOR,
      node_type == "First-derivative predictor" ~ CIRN_FIRST_DERIVATIVE_COLOR,
      node_type == "Second-derivative predictor" ~ CIRN_SECOND_DERIVATIVE_COLOR,
      TRUE ~ CIRN_TARGET_BORDER_COLOR
    )
  )
}

make_network_widget <- function(edges, height = "650px", show_legend = TRUE) {
  if (is.null(edges) || nrow(edges) == 0) {
    nodes <- data.frame(id = "No edges", label = "No retained edges", group = "empty", shape = "circle")
    return(visNetwork::visNetwork(nodes, data.frame(), height = height) %>%
             visNetwork::visOptions(highlightNearest = TRUE))
  }

  edges <- edges %>%
    dplyr::mutate(
      source = as.character(.data$term),
      target_node = as.character(.data$target),
      sign = dplyr::case_when(
        .data$omega > 0 ~ "activation",
        .data$omega < 0 ~ "inhibition",
        TRUE ~ "uncertain"
      ),
      title = paste0(
        format_node(.data$source), " -> ", format_node(.data$target_node),
        "<br>omega = ", signif(.data$omega, 4),
        "<br>Edge width proportional to |omega|",
        "<br>HDI = [", signif(.data$hdi_lower95, 4), ", ", signif(.data$hdi_upper95, 4), "]",
        "<br>Jitter: ", ifelse(.data$jitter_used, "yes", "no")
      )
    )

  node_ids <- unique(c(edges$source, edges$target_node))
  node_style <- node_type_style(node_ids, target_ids = unique(edges$target_node))
  nodes <- tibble(
    id = node_ids,
    label = format_node(node_ids),
    title = node_style$node_type,
    group = node_style$node_type,
    shape = "circle",
    size = ifelse(node_ids %in% edges$target_node, 44, 42),
    color.background = node_style$background,
    color.border = node_style$border,
    borderWidth = ifelse(node_ids %in% edges$target_node, 4, 3),
    font.size = 24,
    font.color = "#102033",
    font.face = "bold"
  ) %>%
    as.data.frame()

  abs_omega <- abs(edges$omega)
  edge_width <- if (length(unique(abs_omega[is.finite(abs_omega)])) <= 1) {
    rep(4, nrow(edges))
  } else {
    pmax(
      1.5,
      scales::rescale(
        abs_omega,
        to = c(2, 7),
        from = range(abs_omega, na.rm = TRUE)
      )
    )
  }

  edge_df <- edges %>%
    transmute(
      from = .data$source,
      to = .data$target_node,
      arrows = "to",
      label = ifelse(.data$omega > 0, "+", "-"),
      title = .data$title,
      color = ifelse(.data$omega > 0, CIRN_ACTIVATION_COLOR, CIRN_INHIBITION_COLOR),
      width = edge_width
    ) %>%
    as.data.frame()

  legend_nodes <- data.frame(
    label = c("State predictor", "First derivative", "Second derivative", "Target response"),
    shape = "circle",
    color.background = c(CIRN_STATE_FILL, CIRN_FIRST_DERIVATIVE_FILL, CIRN_SECOND_DERIVATIVE_FILL, CIRN_TARGET_FILL),
    color.border = c(CIRN_STATE_COLOR, CIRN_FIRST_DERIVATIVE_COLOR, CIRN_SECOND_DERIVATIVE_COLOR, CIRN_TARGET_BORDER_COLOR),
    borderWidth = c(2, 2, 2, 4),
    font.color = c(CIRN_STATE_COLOR, CIRN_FIRST_DERIVATIVE_COLOR, CIRN_SECOND_DERIVATIVE_COLOR, "#111111"),
    font.face = "bold",
    stringsAsFactors = FALSE
  )

  legend_edges <- data.frame(
    label = c("Activation (+)", "Inhibition (-)", "small |omega|", "large |omega|"),
    color = c(CIRN_ACTIVATION_COLOR, CIRN_INHIBITION_COLOR, CIRN_ACTIVATION_COLOR, CIRN_ACTIVATION_COLOR),
    arrows = "to",
    width = c(3, 3, 1.5, 7),
    font.align = "top",
    stringsAsFactors = FALSE
  )

  network <- visNetwork::visNetwork(nodes, edge_df, height = height) %>%
    visNetwork::visNodes(
      shape = "circle",
      font = list(size = 24, color = "#102033", face = "bold", align = "center")
    ) %>%
    visNetwork::visEdges(smooth = list(type = "dynamic")) %>%
    visNetwork::visOptions(highlightNearest = TRUE, nodesIdSelection = TRUE) %>%
    visNetwork::visPhysics(
      solver = "forceAtlas2Based",
      stabilization = TRUE,
      forceAtlas2Based = list(gravitationalConstant = -120, springLength = 220)
    ) %>%
    visNetwork::visLayout(randomSeed = 123, improvedLayout = TRUE)

  if (!isTRUE(show_legend)) {
    return(network)
  }

  network %>%
    visNetwork::visLegend(
      addNodes = legend_nodes,
      addEdges = legend_edges,
      useGroups = FALSE,
      position = "right",
      width = 0.28,
      ncol = 1,
      zoom = FALSE,
      main = list(
        text = "Legend",
        style = "font-family:Arial, sans-serif;font-weight:700;font-size:18px;text-align:left;color:#111111;"
      )
    )
}

plot_empty_message <- function(message) {
  ggplot() +
    annotate("text", x = 0, y = 0, label = message, size = 4.5, color = "#52657a", lineheight = 1.08) +
    xlim(-1, 1) +
    ylim(-1, 1) +
    theme_void()
}

cirn_plot_typography <- function(plot_obj) {
  if (!inherits(plot_obj, "ggplot")) {
    return(plot_obj)
  }

  panel_count <- tryCatch(
    {
      layout <- ggplot2::ggplot_build(plot_obj)$layout$layout
      length(unique(as.character(layout$PANEL)))
    },
    error = function(e) 1L
  )
  solo_plot <- isTRUE(panel_count <= 1L)

  plot_obj +
    ggplot2::theme(
      text = ggplot2::element_text(
        size = if (solo_plot) 15 else CIRN_PLOT_BASE_SIZE,
        color = "#102033"
      ),
      plot.title = ggplot2::element_text(
        size = if (solo_plot) 20 else 17,
        face = "bold",
        color = "#102033",
        margin = ggplot2::margin(b = 8)
      ),
      plot.subtitle = ggplot2::element_text(
        size = if (solo_plot) 14 else 12.5,
        color = "#52657A",
        lineheight = 1.08,
        margin = ggplot2::margin(b = 10)
      ),
      axis.title.x = ggplot2::element_text(
        size = if (solo_plot) 16 else 13.5,
        face = "bold",
        color = "#102033",
        margin = ggplot2::margin(t = 9)
      ),
      axis.title.y = ggplot2::element_text(
        size = if (solo_plot) 16 else 13.5,
        face = "bold",
        color = "#102033",
        margin = ggplot2::margin(r = 9)
      ),
      axis.text = ggplot2::element_text(
        size = if (solo_plot) 13 else 11.5,
        color = "#35485D"
      ),
      strip.text = ggplot2::element_text(
        size = if (solo_plot) 15 else 12.5,
        face = "bold",
        color = "#102033",
        margin = ggplot2::margin(7, 7, 7, 7)
      ),
      legend.title = ggplot2::element_text(
        size = if (solo_plot) 14 else 12,
        face = "bold",
        color = "#102033"
      ),
      legend.text = ggplot2::element_text(
        size = if (solo_plot) 13 else 11.5,
        color = "#35485D"
      ),
      plot.caption = ggplot2::element_text(
        size = if (solo_plot) 11.5 else 10.5,
        color = "#5E7084",
        lineheight = 1.08,
        margin = ggplot2::margin(t = 10)
      )
    )
}

as_result_tbl <- function(tbl) {
  if (is.null(tbl)) {
    return(tibble())
  }
  as_tibble(tbl)
}

ensure_column <- function(tbl, name, default) {
  if (!name %in% names(tbl)) {
    tbl[[name]] <- rep(default, nrow(tbl))
  }
  tbl
}

add_mode_group <- function(tbl) {
  tbl <- as_result_tbl(tbl)
  if (nrow(tbl) == 0) {
    return(tbl)
  }
  tbl <- ensure_column(tbl, "analysis_mode", "multivariable")
  tbl <- ensure_column(tbl, "predictor_set", NA_character_)
  tbl %>%
    mutate(
      analysis_mode = as.character(.data$analysis_mode),
      predictor_set = as.character(.data$predictor_set),
      mode_group = case_when(
        .data$analysis_mode == "pairwise" ~ "pairwise",
        .data$predictor_set == "all_predictors" ~ "all_predictors",
        .data$predictor_set %in% c("original", "first_derivative", "second_derivative") ~ "sublevel",
        TRUE ~ "other"
      )
    )
}

filter_result_mode <- function(tbl, mode = c("combined", "sublevel", "all_predictors", "pairwise")) {
  mode <- match.arg(mode)
  tbl <- add_mode_group(tbl)
  if (nrow(tbl) == 0 || mode == "combined") {
    return(tbl)
  }
  tbl %>% filter(.data$mode_group == mode)
}

empty_cirn_result <- function(message = NULL) {
  list(
    edges = tibble(),
    all_coefficients = tibble(),
    all_coefficients_combined = tibble(),
    pairwise = list(
      edges = tibble(),
      all_coefficients = tibble(),
      diagnostics = tibble(),
      vif_group = tibble(),
      vif_pairs = tibble(),
      models = list()
    ),
    edges_combined = tibble(),
    diagnostics = tibble(
      target = NA_character_,
      predictor_set = NA_character_,
      status = "app_run_failed",
      message = message %||% "CIRN run failed before producing a result object."
    ),
    vif_group = tibble(),
    vif_pairs = tibble(),
    models = list(),
    debug = list(),
    settings = list(error = message)
  )
}

summarize_mode_consistency <- function(edges) {
  tbl <- add_mode_group(edges)
  if (nrow(tbl) == 0 || !all(c("term", "target", "omega") %in% names(tbl))) {
    return(tibble())
  }

  tbl %>%
    filter(
      .data$mode_group %in% c("sublevel", "all_predictors", "pairwise"),
      is.finite(.data$omega)
    ) %>%
    mutate(
      source_base = base_term(.data$term),
      target_base = sub("^d", "", as.character(.data$target)),
      sign_label = case_when(
        .data$omega > 0 ~ "activation",
        .data$omega < 0 ~ "inhibition",
        TRUE ~ "uncertain"
      ),
      predictor_set = coalesce(as.character(.data$predictor_set), "not_recorded")
    ) %>%
    group_by(.data$source_base, .data$target_base, .data$sign_label, .data$mode_group) %>%
    summarise(
      terms = paste(sort(unique(format_node(.data$term))), collapse = ", "),
      predictor_sets = paste(sort(unique(.data$predictor_set)), collapse = ", "),
      n_coefficients = n(),
      mean_omega = mean(.data$omega, na.rm = TRUE),
      max_abs_omega = max(abs(.data$omega), na.rm = TRUE),
      .groups = "drop"
    ) %>%
    group_by(.data$source_base, .data$target_base, .data$sign_label) %>%
    summarise(
      edge = paste(dplyr::first(.data$source_base), "->", dplyr::first(.data$target_base)),
      modes = paste(sort(unique(.data$mode_group)), collapse = ", "),
      mode_count = n_distinct(.data$mode_group),
      has_sublevel = any(.data$mode_group == "sublevel"),
      has_all_predictors = any(.data$mode_group == "all_predictors"),
      has_pairwise = any(.data$mode_group == "pairwise"),
      mean_omega = mean(.data$mean_omega, na.rm = TRUE),
      max_abs_omega = max(.data$max_abs_omega, na.rm = TRUE),
      support_detail = paste(.data$mode_group, .data$predictor_sets, sep = ": ", collapse = " | "),
      .groups = "drop"
    ) %>%
    mutate(
      consistency_class = case_when(
        .data$mode_count >= 3 ~ "consistent across three modes",
        .data$mode_count == 2 ~ "consistent across two modes",
        TRUE ~ "single-mode only"
      )
    ) %>%
    arrange(desc(.data$mode_count), desc(.data$max_abs_omega), .data$edge)
}

consistency_edges <- function(consistency_tbl, min_modes = 2) {
  tbl <- as_result_tbl(consistency_tbl)
  if (nrow(tbl) == 0) {
    return(tibble())
  }

  tbl %>%
    filter(.data$mode_count >= min_modes) %>%
    mutate(
      consistency_omega = ifelse(.data$sign_label == "activation", .data$max_abs_omega, -.data$max_abs_omega)
    ) %>%
    transmute(
      term = .data$source_base,
      target = paste0("d", .data$target_base),
      omega = .data$consistency_omega,
      odds_ratio = exp(.data$consistency_omega),
      sign = sign(.data$consistency_omega),
      regulation_type = .data$sign_label,
      rel_strength = .data$consistency_class,
      hdi_lower95 = NA_real_,
      hdi_upper95 = NA_real_,
      eti_lower95 = NA_real_,
      eti_upper95 = NA_real_,
      predictor_set = .data$consistency_class,
      analysis_mode = "consistency",
      pairwise_predictor = NA_character_,
      jitter_used = FALSE,
      method = paste0("Mode consistency: ", .data$modes)
    )
}

spread_positions <- function(n) {
  if (n <= 0) {
    numeric(0)
  } else if (n == 1) {
    0
  } else {
    seq(1, -1, length.out = n)
  }
}

plot_static_network <- function(edges, title, subtitle = NULL) {
  tbl <- as_result_tbl(edges)
  if (nrow(tbl) == 0 || !all(c("term", "target", "omega") %in% names(tbl))) {
    return(plot_empty_message(paste0("No retained edges are available for ", title, ".")))
  }

  tbl <- add_mode_group(tbl) %>%
    filter(is.finite(.data$omega), !is.na(.data$term), !is.na(.data$target)) %>%
    mutate(
      source = as.character(.data$term),
      target_node = as.character(.data$target),
      sign_label = case_when(
        .data$omega > 0 ~ "activation",
        .data$omega < 0 ~ "inhibition",
        TRUE ~ "uncertain"
      ),
      predictor_set = coalesce(as.character(.data$predictor_set), "not_recorded")
    )

  if (nrow(tbl) == 0) {
    return(plot_empty_message(paste0("No finite retained edges are available for ", title, ".")))
  }

  edge_tbl <- tbl %>%
    group_by(.data$source, .data$target_node, .data$sign_label) %>%
    summarise(
      omega = mean(.data$omega, na.rm = TRUE),
      mean_abs_omega = mean(abs(.data$omega), na.rm = TRUE),
      n_edges = n(),
      predictor_sets = paste(sort(unique(.data$predictor_set)), collapse = ", "),
      modes = paste(sort(unique(.data$mode_group)), collapse = ", "),
      .groups = "drop"
    )

  source_nodes <- sort(unique(edge_tbl$source))
  target_nodes <- sort(unique(edge_tbl$target_node))
  node_tbl <- bind_rows(
    tibble(id = source_nodes, label = format_node(source_nodes), x = 0, y = spread_positions(length(source_nodes))),
    tibble(id = target_nodes, label = format_node(target_nodes), x = 1, y = spread_positions(length(target_nodes)))
  ) %>%
    distinct(id, .keep_all = TRUE) %>%
    bind_cols(node_type_style(.$id, target_ids = target_nodes)) %>%
    mutate(
      node_legend_label = case_when(
        .data$node_type == "Original-state predictor" ~ "State predictor",
        .data$node_type == "First-derivative predictor" ~ "First derivative",
        .data$node_type == "Second-derivative predictor" ~ "Second derivative",
        TRUE ~ "Target response"
      )
    )

  edge_plot <- edge_tbl %>%
    left_join(node_tbl %>% select(source = id, x_source = x, y_source = y), by = "source") %>%
    left_join(node_tbl %>% select(target_node = id, x_target = x, y_target = y), by = "target_node")

  ggplot() +
    geom_segment(
      data = edge_plot,
      aes(
        x = .data$x_source + 0.06,
        xend = .data$x_target - 0.06,
        y = .data$y_source,
        yend = .data$y_target,
        color = .data$sign_label,
        linewidth = .data$mean_abs_omega
      ),
      arrow = grid::arrow(type = "closed", length = grid::unit(0.16, "inches")),
      lineend = "round",
      alpha = 0.84,
      show.legend = FALSE
    ) +
    geom_point(
      data = filter(node_tbl, .data$node_type == "Original-state predictor"),
      aes(.data$x, .data$y),
      shape = 21,
      size = 16,
      stroke = 1.25,
      fill = CIRN_STATE_FILL,
      color = CIRN_STATE_COLOR
    ) +
    geom_point(
      data = filter(node_tbl, .data$node_type == "First-derivative predictor"),
      aes(.data$x, .data$y),
      shape = 21,
      size = 16,
      stroke = 1.25,
      fill = CIRN_FIRST_DERIVATIVE_FILL,
      color = CIRN_FIRST_DERIVATIVE_COLOR
    ) +
    geom_point(
      data = filter(node_tbl, .data$node_type == "Second-derivative predictor"),
      aes(.data$x, .data$y),
      shape = 21,
      size = 16,
      stroke = 1.25,
      fill = CIRN_SECOND_DERIVATIVE_FILL,
      color = CIRN_SECOND_DERIVATIVE_COLOR
    ) +
    geom_point(
      data = filter(node_tbl, .data$node_type == "Target derivative response"),
      aes(.data$x, .data$y),
      shape = 21,
      size = 16,
      stroke = 1.85,
      fill = CIRN_TARGET_FILL,
      color = CIRN_TARGET_BORDER_COLOR
    ) +
    geom_text(
      data = node_tbl,
      aes(.data$x, .data$y, label = .data$label),
      size = 3.8,
      fontface = "bold",
      color = "#102033",
      lineheight = 0.9
    ) +
    annotate("segment", x = 1.18, xend = 1.18, y = -1.08, yend = 1.08, color = "#d7dee6", linewidth = 0.4) +
    annotate("text", x = 1.25, y = 0.95, label = "Legend", hjust = 0, fontface = "bold", size = 5.1, color = "#111111") +
    annotate("text", x = 1.25, y = 0.72, label = "Activation (+)", hjust = 0, fontface = "bold", size = 4.1, color = CIRN_ACTIVATION_COLOR) +
    annotate("text", x = 1.25, y = 0.52, label = "Inhibition (-)", hjust = 0, fontface = "bold", size = 4.1, color = CIRN_INHIBITION_COLOR) +
    annotate("text", x = 1.25, y = 0.27, label = "Edge thickness: larger |omega| = thicker arrow", hjust = 0, size = 3.7, color = "#111111") +
    annotate("segment", x = 1.25, xend = 1.42, y = 0.11, yend = 0.11, color = CIRN_ACTIVATION_COLOR, linewidth = 0.65, lineend = "round", arrow = grid::arrow(type = "closed", length = grid::unit(0.09, "inches"))) +
    annotate("segment", x = 1.25, xend = 1.42, y = -0.03, yend = -0.03, color = CIRN_ACTIVATION_COLOR, linewidth = 2.35, lineend = "round", arrow = grid::arrow(type = "closed", length = grid::unit(0.09, "inches"))) +
    annotate("point", x = 1.28, y = -0.25, shape = 21, size = 6, stroke = 1.1, fill = CIRN_STATE_FILL, color = CIRN_STATE_COLOR) +
    annotate("text", x = 1.36, y = -0.25, label = "State predictor", hjust = 0, fontface = "bold", size = 3.7, color = CIRN_STATE_COLOR) +
    annotate("point", x = 1.28, y = -0.43, shape = 21, size = 6, stroke = 1.1, fill = CIRN_FIRST_DERIVATIVE_FILL, color = CIRN_FIRST_DERIVATIVE_COLOR) +
    annotate("text", x = 1.36, y = -0.43, label = "First derivative", hjust = 0, fontface = "bold", size = 3.7, color = CIRN_FIRST_DERIVATIVE_COLOR) +
    annotate("point", x = 1.28, y = -0.61, shape = 21, size = 6, stroke = 1.1, fill = CIRN_SECOND_DERIVATIVE_FILL, color = CIRN_SECOND_DERIVATIVE_COLOR) +
    annotate("text", x = 1.36, y = -0.61, label = "Second derivative", hjust = 0, fontface = "bold", size = 3.7, color = CIRN_SECOND_DERIVATIVE_COLOR) +
    annotate("point", x = 1.28, y = -0.82, shape = 21, size = 6, stroke = 1.5, fill = CIRN_TARGET_FILL, color = CIRN_TARGET_BORDER_COLOR) +
    annotate("text", x = 1.36, y = -0.82, label = "Black border = target response", hjust = 0, fontface = "bold", size = 3.7, color = "#111111") +
    scale_color_manual(values = c(activation = CIRN_ACTIVATION_COLOR, inhibition = CIRN_INHIBITION_COLOR, uncertain = CIRN_UNCERTAIN_COLOR), guide = "none") +
    scale_linewidth_continuous(range = c(0.6, 2.7), guide = "none") +
    coord_cartesian(xlim = c(-0.2, 1.95), ylim = c(-1.15, 1.15), clip = "off") +
    labs(title = title, subtitle = subtitle) +
    theme_void(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 16, color = "#102033"),
      plot.subtitle = element_text(size = 11, color = "#5e7084", margin = ggplot2::margin(b = 12)),
      legend.position = "none",
      plot.background = element_rect(fill = "white", color = NA),
      plot.margin = ggplot2::margin(18, 54, 18, 30)
    )
}

plot_consistency_heatmap <- function(consistency_tbl) {
  tbl <- as_result_tbl(consistency_tbl)
  if (nrow(tbl) == 0) {
    return(plot_empty_message("Run multiple CIRN modes to view edge consistency."))
  }

  if (!"edge" %in% names(tbl)) {
    if (all(c("term", "target") %in% names(tbl))) {
      tbl <- tbl %>%
        mutate(edge = paste(format_node(.data$term), "->", format_node(.data$target)))
    } else {
      tbl$edge <- paste("edge", seq_len(nrow(tbl)))
    }
  }
  if (!"sign_label" %in% names(tbl)) {
    tbl$sign_label <- if ("regulation_type" %in% names(tbl)) {
      as.character(tbl$regulation_type)
    } else {
      "retained"
    }
  }
  if (!"has_sublevel" %in% names(tbl)) {
    tbl$has_sublevel <- if ("in_sublevel" %in% names(tbl)) dplyr::coalesce(as.logical(tbl$in_sublevel), FALSE) else FALSE
  }
  if (!"has_all_predictors" %in% names(tbl)) {
    tbl$has_all_predictors <- if ("in_all_predictors" %in% names(tbl)) dplyr::coalesce(as.logical(tbl$in_all_predictors), FALSE) else FALSE
  }
  if (!"has_pairwise" %in% names(tbl)) {
    tbl$has_pairwise <- if ("in_pairwise" %in% names(tbl)) dplyr::coalesce(as.logical(tbl$in_pairwise), FALSE) else FALSE
  }
  if (!"mode_count" %in% names(tbl)) {
    tbl$mode_count <- if ("n_sources" %in% names(tbl)) {
      as.integer(tbl$n_sources)
    } else {
      rowSums(cbind(tbl$has_sublevel, tbl$has_all_predictors, tbl$has_pairwise), na.rm = TRUE)
    }
  }

  tbl %>%
    transmute(
      edge_label = paste(.data$edge, .data$sign_label),
      Sublevel = .data$has_sublevel,
      `All predictors` = .data$has_all_predictors,
      Pairwise = .data$has_pairwise,
      mode_count = .data$mode_count
    ) %>%
    pivot_longer(cols = c("Sublevel", "All predictors", "Pairwise"), names_to = "mode", values_to = "present") %>%
    ggplot(aes(.data$mode, reorder(.data$edge_label, .data$mode_count), fill = .data$present)) +
    geom_tile(color = "white", linewidth = 0.8) +
    scale_fill_manual(values = c("FALSE" = "#EEF3F7", "TRUE" = "#168A93"), labels = c("Absent", "Present")) +
    labs(x = NULL, y = NULL, fill = NULL) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(face = "bold"),
      legend.position = "bottom"
    )
}

edge_consistency_from_result <- function(res) {
  if (is.null(res)) {
    return(tibble())
  }

  main_edges <- res$edges %||% tibble()
  pairwise_edges <- if (!is.null(res$pairwise)) {
    res$pairwise$edges %||% tibble()
  } else {
    tibble()
  }

  if (nrow(as_result_tbl(main_edges)) == 0 && nrow(as_result_tbl(pairwise_edges)) == 0) {
    main_edges <- res$edges_combined %||% tibble()
  }

  summarize_edge_consistency(
    edges = main_edges,
    pairwise_edges = pairwise_edges,
    collapse_derivative_terms = FALSE
  )
}

prepare_edges_for_benchmark <- function(edges) {
  edges <- as_result_tbl(edges)
  if (nrow(edges) == 0) {
    return(edges)
  }

  if (!"regulation_type" %in% names(edges)) {
    edges <- edges %>%
      mutate(
        regulation_type = case_when(
          is.finite(.data$omega) & .data$omega > 0 ~ "activation",
          is.finite(.data$omega) & .data$omega < 0 ~ "inhibition",
          TRUE ~ "uncertain"
        )
      )
  }

  edges
}

effective_sample_size_from_result <- function(res) {
  if (is.null(res)) {
    return(tibble())
  }

  if (exists("summarize_cirn_effective_sample_size", mode = "function")) {
    out <- tryCatch(summarize_cirn_effective_sample_size(res), error = function(e) tibble())
    if (nrow(as_result_tbl(out)) > 0) {
      return(out)
    }
  }

  prepare_diag <- function(tbl, mode) {
    tbl <- as_result_tbl(tbl)
    if (nrow(tbl) == 0) {
      return(tibble())
    }
    required <- c("class_0_count", "class_1_count", "n_predictors", "status")
    for (nm in required) {
      if (!nm %in% names(tbl)) {
        tbl[[nm]] <- NA
      }
    }
    if (!"analysis_mode" %in% names(tbl)) {
      tbl$analysis_mode <- mode
    } else {
      tbl$analysis_mode <- coalesce(as.character(tbl$analysis_mode), mode)
    }
    tbl
  }

  multivariable_diag <- prepare_diag(res$diagnostics, "multivariable")
  pairwise_diag <- if (!is.null(res$pairwise)) {
    prepare_diag(res$pairwise$diagnostics, "pairwise")
  } else {
    tibble()
  }

  bind_rows(multivariable_diag, pairwise_diag) %>%
    mutate(
      class_0_count = suppressWarnings(as.numeric(.data$class_0_count)),
      class_1_count = suppressWarnings(as.numeric(.data$class_1_count)),
      n_predictors = suppressWarnings(as.numeric(.data$n_predictors)),
      usable_n = .data$class_0_count + .data$class_1_count,
      minority_class_count = pmin(.data$class_0_count, .data$class_1_count),
      class_balance = if_else(is.finite(.data$usable_n) & .data$usable_n > 0, .data$minority_class_count / .data$usable_n, NA_real_),
      minority_per_predictor = if_else(is.finite(.data$n_predictors) & .data$n_predictors > 0, .data$minority_class_count / .data$n_predictors, NA_real_),
      sample_size_flag = case_when(
        is.na(.data$status) ~ "unknown",
        .data$status != "completed" ~ paste0("not_fitted_", .data$status),
        !is.finite(.data$usable_n) ~ "unknown",
        .data$minority_class_count < 5 ~ "below_minimum",
        .data$minority_class_count < 10 ~ "small_minority_class",
        is.finite(.data$minority_per_predictor) & .data$minority_per_predictor < 5 ~ "low_minority_per_predictor",
        TRUE ~ "adequate_basic"
      )
    ) %>%
    select(
      any_of(c(
        "target", "analysis_mode", "predictor_set", "pairwise_predictor",
        "status", "class_0_count", "class_1_count", "usable_n",
        "minority_class_count", "n_predictors", "minority_per_predictor",
        "class_balance", "jitter_used", "predictor_jitter_used",
        "sample_size_flag"
      ))
    )
}

plot_mode_consistency_grid <- function(edge_consistency_tbl) {
  tbl <- as_result_tbl(edge_consistency_tbl)
  if (nrow(tbl) == 0) {
    return(plot_empty_message("No posterior-supported edges are available for the edge-consistency grid."))
  }

  plot_edge_consistency(
    tbl,
    show_regulation_label = FALSE,
    encode_strength = TRUE,
    strength_scale = "within_mode"
  )
}

summarize_sensitivity_feature_edge_stability <- function(sr, edge_focus = c("all", "pairwise")) {
  edge_focus <- match.arg(edge_focus)
  if (is.null(sr) || is.null(sr$edges) || is.null(sr$runs) ||
      nrow(sr$edges) == 0 || nrow(sr$runs) == 0) {
    return(tibble())
  }

  completed_runs <- sr$runs %>%
    filter(.data$run_status == "completed") %>%
    count(.data$scenario, .data$scenario_value, name = "completed_runs")

  if (nrow(completed_runs) == 0) {
    return(tibble())
  }

  strongest_omega <- function(x) {
    x <- x[is.finite(x)]
    if (length(x) == 0) {
      return(NA_real_)
    }
    x[which.max(abs(x))]
  }

  edge_source <- sr$edges
  for (col in c("analysis_mode", "predictor_set")) {
    if (!col %in% names(edge_source)) {
      edge_source[[col]] <- NA_character_
    }
  }
  if (identical(edge_focus, "pairwise")) {
    edge_source <- edge_source %>%
      filter(coalesce(as.character(.data$analysis_mode), "") == "pairwise")
  }

  if (nrow(edge_source) == 0) {
    return(tibble())
  }

  detected_edges <- edge_source %>%
    filter(
      !is.na(.data$term),
      !is.na(.data$target),
      is.finite(.data$omega)
    ) %>%
    mutate(
      term = as.character(.data$term),
      target = as.character(.data$target),
      inferred_sign = case_when(
        .data$omega > 0 ~ 1L,
        .data$omega < 0 ~ -1L,
        TRUE ~ 0L
      ),
      regulation_type = case_when(
        .data$inferred_sign > 0 ~ "activation",
        .data$inferred_sign < 0 ~ "inhibition",
        TRUE ~ "zero"
      ),
      mode_group = coalesce(as.character(.data$analysis_mode), "not_recorded"),
      predictor_set = coalesce(as.character(.data$predictor_set), "not_recorded")
    ) %>%
    filter(.data$inferred_sign != 0) %>%
    group_by(
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
    summarise(
      omega = strongest_omega(.data$omega),
      n_coefficients = n(),
      modes = paste(sort(unique(.data$mode_group)), collapse = ", "),
      predictor_sets = paste(sort(unique(.data$predictor_set)), collapse = ", "),
      .groups = "drop"
    )

  if (nrow(detected_edges) == 0) {
    return(tibble())
  }

  edge_keys <- detected_edges %>%
    distinct(.data$term, .data$target)

  scenario_edge_grid <- merge(completed_runs, edge_keys, by = NULL, all = TRUE)

  detected_counts <- detected_edges %>%
    group_by(.data$scenario, .data$scenario_value, .data$term, .data$target) %>%
    summarise(
      activation_detected_runs = n_distinct(.data$sensitivity_id[.data$inferred_sign > 0]),
      inhibition_detected_runs = n_distinct(.data$sensitivity_id[.data$inferred_sign < 0]),
      activation_omega = strongest_omega(.data$omega[.data$inferred_sign > 0]),
      inhibition_omega = strongest_omega(.data$omega[.data$inferred_sign < 0]),
      total_detected_runs = n_distinct(.data$sensitivity_id),
      conditions = paste(sort(unique(.data$condition)), collapse = "; "),
      modes = paste(sort(unique(.data$modes)), collapse = "; "),
      predictor_sets = paste(sort(unique(.data$predictor_sets)), collapse = "; "),
      .groups = "drop"
    )

  baseline_keys <- detected_edges %>%
    filter(.data$scenario == "baseline") %>%
    distinct(.data$term, .data$target) %>%
    mutate(in_baseline = TRUE)

  scenario_edge_grid %>%
    left_join(detected_counts, by = c("scenario", "scenario_value", "term", "target")) %>%
    mutate(
      activation_detected_runs = coalesce(.data$activation_detected_runs, 0L),
      inhibition_detected_runs = coalesce(.data$inhibition_detected_runs, 0L),
      total_detected_runs = coalesce(.data$total_detected_runs, 0L),
      conditions = coalesce(.data$conditions, ""),
      modes = coalesce(.data$modes, ""),
      predictor_sets = coalesce(.data$predictor_sets, ""),
      activation_rate = if_else(.data$completed_runs > 0, .data$activation_detected_runs / .data$completed_runs, NA_real_),
      inhibition_rate = if_else(.data$completed_runs > 0, .data$inhibition_detected_runs / .data$completed_runs, NA_real_),
      detection_rate = pmax(.data$activation_rate, .data$inhibition_rate, na.rm = TRUE),
      cell_status = case_when(
        .data$activation_rate > 0 & .data$inhibition_rate > 0 ~ "mixed sign",
        .data$activation_rate > 0 ~ "activation",
        .data$inhibition_rate > 0 ~ "inhibition",
        TRUE ~ "not detected"
      ),
      fill_value = case_when(
        .data$cell_status == "activation" ~ .data$activation_rate,
        .data$cell_status == "inhibition" ~ -.data$inhibition_rate,
        .data$cell_status == "mixed sign" ~ NA_real_,
        TRUE ~ 0
      ),
      edge = paste0(format_node(.data$term), " -> ", format_node(.data$target))
    ) %>%
    left_join(baseline_keys, by = c("term", "target")) %>%
    mutate(in_baseline = coalesce(.data$in_baseline, FALSE)) %>%
    arrange(desc(.data$in_baseline), .data$edge, .data$scenario, .data$scenario_value)
}

plot_sensitivity_feature_heatmap <- function(feature_stability_tbl) {
  tbl <- as_result_tbl(feature_stability_tbl)
  if (nrow(tbl) == 0) {
    return(plot_empty_message("No retained feature-level edges are available for the sensitivity heatmap."))
  }

  plot_tbl <- tbl %>%
    mutate(
      scenario_label = paste(.data$scenario, .data$scenario_value),
      edge = as.character(.data$edge)
    ) %>%
    group_by(.data$edge) %>%
    mutate(max_detection = max(.data$detection_rate, na.rm = TRUE)) %>%
    ungroup() %>%
    mutate(edge = stats::reorder(.data$edge, .data$max_detection))

  ggplot(plot_tbl, aes(x = .data$scenario_label, y = .data$edge, fill = .data$fill_value)) +
    geom_tile(color = "white", linewidth = 0.7) +
    scale_fill_gradient2(
      low = CIRN_INHIBITION_COLOR,
      mid = "#eeeeee",
      high = CIRN_ACTIVATION_COLOR,
      midpoint = 0,
      limits = c(-1, 1),
      breaks = c(-1, 0, 1),
      labels = c("Inhibition", "Not detected", "Activation"),
      name = NULL,
      na.value = "#9467bd",
      oob = scales::squish
    ) +
    labs(
      x = "Sensitivity scenario",
      y = "Inferred edge",
      caption = "Signed detection rate: green = activation; red = inhibition; grey = not detected; purple = mixed sign."
    ) +
    guides(
      fill = guide_colorbar(
        label.position = "bottom",
        barwidth = grid::unit(9.4, "cm"),
        barheight = grid::unit(0.48, "cm")
      )
    ) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "bottom",
      legend.text = element_text(size = 10, margin = ggplot2::margin(t = 7)),
      legend.box.margin = ggplot2::margin(t = 10, r = 0, b = 14, l = 0),
      legend.margin = ggplot2::margin(t = 4, r = 0, b = 10, l = 0),
      plot.caption = element_text(color = "#5e7084", hjust = 0, margin = ggplot2::margin(t = 12)),
      plot.margin = ggplot2::margin(t = 10, r = 18, b = 44, l = 10)
    )
}

resolve_sensitivity_scope <- function(scope,
                                      run_representation_mode,
                                      run_pairwise,
                                      run_pairwise_representation_mode) {
  if (exists("resolve_cirn_sensitivity_scope", mode = "function")) {
    cfg <- resolve_cirn_sensitivity_scope(
      scope = scope %||% "use_run_settings",
      representation_mode = run_representation_mode %||% "both",
      run_pairwise = isTRUE(run_pairwise),
      pairwise_representation_mode = run_pairwise_representation_mode %||% "sublevel"
    )
    
    if (identical(cfg$key, "use_config")) {
      cfg$key <- "use_run_settings"
      cfg$title <- "Use Run CIRN settings"
      cfg$body <- "Sensitivity uses the same representation and pairwise choices currently selected in the Run CIRN Algorithm tab."
      cfg$detail <- "Best default when you want robustness checks to match the main analysis exactly."
    }
    
    return(cfg)
  }
  
  scope <- scope %||% "use_run_settings"
  run_representation_mode <- run_representation_mode %||% "both"
  run_pairwise_representation_mode <- run_pairwise_representation_mode %||% "sublevel"

  switch(
    scope,
    use_run_settings = list(
      key = "use_run_settings",
      title = "Use Run CIRN settings",
      representation_mode = run_representation_mode,
      run_pairwise = isTRUE(run_pairwise),
      pairwise_representation_mode = run_pairwise_representation_mode,
      edge_focus = "all",
      body = "Sensitivity uses the same representation and pairwise choices currently selected in the Run CIRN Algorithm tab.",
      detail = "Best default when you want robustness checks to match the main analysis exactly."
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
      detail = "Useful for checking the full combined predictor set, but interpret with VIF and coefficient diagnostics because collinearity can be stronger."
    ),
    sublevel_all_predictors = list(
      key = "sublevel_all_predictors",
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
      pairwise_representation_mode = run_pairwise_representation_mode,
      edge_focus = "pairwise",
      body = "Sensitivity focuses the stability table and heatmap on pairwise CIRN edges.",
      detail = "The core algorithm still performs a minimal main sublevel fit internally, but the displayed feature-level stability summary is filtered to pairwise retained edges."
    ),
    everything = list(
      key = "everything",
      title = "Everything",
      representation_mode = "both",
      run_pairwise = TRUE,
      pairwise_representation_mode = "both",
      edge_focus = "all",
      body = "Sensitivity fits sublevel, all-predictors, pairwise sublevel, and pairwise all-predictors CIRN.",
      detail = "Best for final robustness appendix or dissertation-grade stress testing; expect longer runtime and more retained-edge rows."
    ),
    resolve_sensitivity_scope(
      "use_run_settings",
      run_representation_mode,
      run_pairwise,
      run_pairwise_representation_mode
    )
  )
}

is_fitted_bayes_model <- function(x) {
  inherits(x, "brmsfit")
}

bayes_target_choices <- function(res) {
  if (is.null(res) || is.null(res$models) || length(res$models) == 0) {
    return(character())
  }
  names(res$models)[
    vapply(
      res$models,
      function(model_list) {
        is.list(model_list) && any(vapply(model_list, is_fitted_bayes_model, logical(1)))
      },
      logical(1)
    )
  ]
}

bayes_representation_choices <- function(res, target) {
  if (is.null(res) || is.null(target) || !target %in% names(res$models)) {
    return(character())
  }
  model_list <- res$models[[target]]
  choices <- names(model_list)[vapply(model_list, is_fitted_bayes_model, logical(1))]
  display <- c(
    original = "State predictors",
    first_derivative = "First-derivative predictors",
    second_derivative = "Second-derivative predictors",
    all_predictors = "All predictors"
  )
  stats::setNames(choices, ifelse(choices %in% names(display), display[choices], choices))
}

selected_bayes_model <- function(res, target, representation) {
  if (is.null(res) || is.null(target) || is.null(representation) ||
      !target %in% names(res$models)) {
    return(NULL)
  }
  model <- res$models[[target]][[representation]]
  if (is_fitted_bayes_model(model)) model else NULL
}

bayes_model_terms <- function(model, max_terms = Inf) {
  if (!is_fitted_bayes_model(model)) {
    return(character())
  }
  terms <- tryCatch(rownames(brms::fixef(model)), error = function(e) character())
  terms <- setdiff(terms, "Intercept")
  if (is.finite(max_terms)) {
    terms <- head(terms, max_terms)
  }
  terms
}

safe_model_plot <- function(expr, empty_message) {
  tryCatch(
    suppressWarnings(expr),
    error = function(e) plot_empty_message(paste(empty_message, conditionMessage(e), sep = "\n"))
  )
}

draw_plot_object <- function(plot_obj) {
  if (inherits(plot_obj, "ggplot")) {
    plot_obj <- cirn_plot_typography(plot_obj)
  }
  if (inherits(plot_obj, c("gtable", "grob", "gTree"))) {
    grid::grid.draw(plot_obj)
  } else {
    print(plot_obj)
  }
  invisible(plot_obj)
}

plot_selected_model_posteriors <- function(model, max_terms = 12) {
  terms <- bayes_model_terms(model, max_terms)
  if (length(terms) == 0) {
    return(plot_empty_message("No non-intercept Bayesian coefficients are available for this model."))
  }
  safe_model_plot(
    plot_all_posterior_effects(model, terms),
    "Could not draw posterior coefficient densities."
  )
}

plot_selected_model_hist_trace <- function(model, max_terms = 8) {
  terms <- bayes_model_terms(model, max_terms)
  if (length(terms) == 0) {
    return(plot_empty_message("No non-intercept MCMC parameters are available for this model."))
  }
  safe_model_plot(
    plot_mcmc_hist_trace(model, pars = paste0("b_", terms)),
    "Could not draw MCMC histogram/trace diagnostics."
  )
}

plot_target_representation_posteriors <- function(res, target) {
  if (is.null(res) || is.null(target) || !target %in% names(res$models)) {
    return(plot_empty_message("Run CIRN Algorithm and choose a target to view Bayesian posterior plots."))
  }
  model_list <- res$models[[target]]
  model_list <- model_list[vapply(model_list, is_fitted_bayes_model, logical(1))]
  if (length(model_list) == 0) {
    return(plot_empty_message("No fitted Bayesian models are available for this target."))
  }
  safe_model_plot(
    plot_all_cirn_posteriors(model_list),
    "Could not draw cross-representation posterior densities."
  )
}

plot_target_trace_with_burnin <- function(res, target) {
  if (is.null(res) || is.null(target) || !target %in% names(res$models)) {
    return(plot_empty_message("Run CIRN Algorithm and choose a target to view MCMC traces."))
  }
  model_list <- res$models[[target]]
  model_list <- model_list[vapply(model_list, is_fitted_bayes_model, logical(1))]
  if (length(model_list) == 0) {
    return(plot_empty_message("No fitted Bayesian models are available for this target."))
  }
  safe_model_plot(
    plot_all_traces_with_burnin(model_list, target),
    "Could not draw target-level MCMC traces with warm-up."
  )
}

plot_time_series <- function(df, time_col, vars) {
  req(length(vars) > 0)
  df %>%
    select(all_of(c(time_col, vars))) %>%
    pivot_longer(cols = all_of(vars), names_to = "variable", values_to = "value") %>%
    ggplot(aes(.data[[time_col]], value, color = variable)) +
    geom_line(linewidth = 0.8, alpha = 0.9) +
    geom_point(size = 1.4, alpha = 0.75) +
    scale_color_manual(values = scales::hue_pal()(length(vars))) +
    labs(x = time_col, y = "Value", color = NULL) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

plot_missingness <- function(df) {
  tibble(row = seq_len(nrow(df))) %>%
    bind_cols(as_tibble(is.na(df))) %>%
    pivot_longer(cols = -row, names_to = "variable", values_to = "missing") %>%
    ggplot(aes(row, variable, fill = missing)) +
    geom_tile() +
    scale_fill_manual(values = c("FALSE" = "#E8EEF4", "TRUE" = "#C33A3A"), labels = c("Observed", "Missing")) +
    labs(x = "Row", y = NULL, fill = NULL) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

plot_correlation_heatmap <- function(df, vars) {
  req(length(vars) >= 2)
  corr <- stats::cor(df[vars], use = "pairwise.complete.obs")
  as.data.frame(as.table(corr)) %>%
    setNames(c("var1", "var2", "correlation")) %>%
    ggplot(aes(var1, var2, fill = correlation)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", correlation)), size = 3) +
    scale_fill_gradient2(low = CIRN_INHIBITION_COLOR, mid = "#F7F7F7", high = CIRN_ACTIVATION_COLOR, limits = c(-1, 1)) +
    labs(x = NULL, y = NULL, fill = "r") +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

plot_phase_plane <- function(df, time_col, phase_x, phase_y) {
  req(df, time_col, phase_x, phase_y)
  plot_df <- tibble(
    panel = paste(phase_x, "vs", phase_y),
    time = df[[time_col]],
    x = df[[phase_x]],
    y = df[[phase_y]]
  )
  direction_df <- trajectory_direction_segments(plot_df, max_arrows_per_panel = 12)
  ggplot(plot_df, aes(.data$x, .data$y)) +
    geom_path(color = "#168A93", linewidth = 0.8, alpha = 0.9, na.rm = TRUE) +
    geom_segment(
      data = direction_df,
      aes(xend = .data$xend, yend = .data$yend, color = .data$time),
      arrow = grid::arrow(length = grid::unit(0.07, "inches"), type = "closed"),
      linewidth = 0.55,
      alpha = 0.9,
      na.rm = TRUE
    ) +
    geom_point(aes(color = .data$time), size = 1.7, na.rm = TRUE) +
    scale_color_viridis_c(option = "C") +
    labs(
      x = phase_x,
      y = phase_y,
      color = time_col,
      subtitle = "Sparse arrowheads indicate the observed direction of time."
    ) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE)
}

eda_vars_available <- function(df, time_col, vars, max_vars = Inf) {
  candidates <- intersect(vars %||% character(), state_columns(df, time_col))
  if (length(candidates) == 0) {
    candidates <- state_columns(df, time_col)
  }
  if (is.finite(max_vars)) {
    candidates <- head(candidates, max_vars)
  }
  candidates
}

scale_numeric <- function(x) {
  s <- stats::sd(x, na.rm = TRUE)
  m <- mean(x, na.rm = TRUE)
  if (!is.finite(s) || s <= 0) {
    return(x - m)
  }
  (x - m) / s
}

plot_time_series_small_multiples <- function(df, time_col, vars) {
  vars <- eda_vars_available(df, time_col, vars)
  if (length(vars) == 0) {
    return(plot_empty_message("No numeric variables are available for time-series EDA."))
  }
  df %>%
    select(all_of(c(time_col, vars))) %>%
    pivot_longer(cols = all_of(vars), names_to = "variable", values_to = "value") %>%
    ggplot(aes(.data[[time_col]], value)) +
    geom_line(color = "#168A93", linewidth = 0.65, alpha = 0.9, na.rm = TRUE) +
    geom_point(color = "#1D4E89", size = 1.1, alpha = 0.55, na.rm = TRUE) +
    facet_wrap(~ variable, scales = "free_y") +
    labs(x = time_col, y = "Value") +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE)
}

plot_normalized_overlay <- function(df, time_col, vars) {
  vars <- eda_vars_available(df, time_col, vars)
  if (length(vars) == 0) {
    return(plot_empty_message("No numeric variables are available for normalized overlay."))
  }
  df %>%
    select(all_of(c(time_col, vars))) %>%
    pivot_longer(cols = all_of(vars), names_to = "variable", values_to = "value") %>%
    group_by(.data$variable) %>%
    mutate(z_value = scale_numeric(.data$value)) %>%
    ungroup() %>%
    ggplot(aes(.data[[time_col]], .data$z_value, color = .data$variable)) +
    geom_hline(yintercept = 0, color = "#D5DEE8") +
    geom_line(linewidth = 0.75, alpha = 0.9, na.rm = TRUE) +
    scale_color_manual(values = scales::hue_pal()(length(vars))) +
    labs(x = time_col, y = "Standardized value", color = NULL) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

plot_variable_distributions <- function(df, time_col, vars) {
  vars <- eda_vars_available(df, time_col, vars)
  if (length(vars) == 0) {
    return(plot_empty_message("No numeric variables are available for distribution plots."))
  }
  df %>%
    select(all_of(vars)) %>%
    pivot_longer(cols = everything(), names_to = "variable", values_to = "value") %>%
    ggplot(aes(value)) +
    geom_histogram(aes(y = after_stat(density)), bins = 28, fill = "#B9DDE0", color = "white", na.rm = TRUE) +
    geom_density(color = "#0F766E", linewidth = 0.8, na.rm = TRUE) +
    facet_wrap(~ variable, scales = "free") +
    labs(x = "Value", y = "Density") +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE)
}

detect_mad_outliers <- function(df, time_col, vars, threshold = 3.5) {
  vars <- eda_vars_available(df, time_col, vars)
  if (length(vars) == 0) {
    return(tibble())
  }
  purrr::map_dfr(vars, function(v) {
    x <- df[[v]]
    center <- stats::median(x, na.rm = TRUE)
    scale <- stats::mad(x, center = center, na.rm = TRUE)
    if (!is.finite(scale) || scale <= 0) {
      scale <- stats::IQR(x, na.rm = TRUE) / 1.349
    }
    if (!is.finite(scale) || scale <= 0) {
      scale <- stats::sd(x, na.rm = TRUE)
    }
    is_outlier <- if (is.finite(scale) && scale > 0) abs(x - center) > threshold * scale else rep(FALSE, length(x))
    tibble(
      time = df[[time_col]],
      variable = v,
      value = x,
      center = center,
      robust_scale = scale,
      outlier = coalesce(is_outlier, FALSE)
    )
  })
}

plot_outlier_timeline <- function(df, time_col, vars, threshold = 3.5) {
  tbl <- detect_mad_outliers(df, time_col, vars, threshold)
  if (nrow(tbl) == 0) {
    return(plot_empty_message("No numeric variables are available for outlier screening."))
  }
  ggplot(tbl, aes(.data$time, .data$value)) +
    geom_line(color = "#A8B7C7", linewidth = 0.55, na.rm = TRUE) +
    geom_point(aes(color = .data$outlier), size = 1.7, alpha = 0.85, na.rm = TRUE) +
    facet_wrap(~ variable, scales = "free_y") +
    scale_color_manual(values = c("FALSE" = "#1D4E89", "TRUE" = "#B2182B"), labels = c("Typical", "Potential outlier")) +
    labs(x = time_col, y = "Value", color = NULL) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

plot_time_gap_profile <- function(df, time_col) {
  time_values <- sort(unique(df[[time_col]]))
  gaps <- diff(time_values)
  if (length(gaps) == 0) {
    return(plot_empty_message("At least two unique time points are needed for a time-gap plot."))
  }
  gap_df <- tibble(interval = seq_along(gaps), gap = gaps)
  ggplot(gap_df, aes(.data$interval, .data$gap)) +
    geom_hline(yintercept = stats::median(gaps, na.rm = TRUE), color = "#B7791F", linetype = 2) +
    geom_line(color = "#1D4E89", linewidth = 0.75) +
    geom_point(color = "#168A93", size = 1.8) +
    labs(x = "Ordered interval", y = paste("Spacing in", time_col), subtitle = "Dashed line marks the median spacing.") +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE)
}

missing_run_length <- function(x) {
  r <- rle(is.na(x))
  if (!any(r$values)) {
    return(0L)
  }
  max(r$lengths[r$values])
}

missingness_summary_table <- function(df) {
  tibble(
    variable = names(df),
    missing_n = vapply(df, function(x) sum(is.na(x)), integer(1)),
    missing_pct = round(100 * vapply(df, function(x) mean(is.na(x)), numeric(1)), 2),
    longest_missing_run = vapply(df, missing_run_length, integer(1))
  ) %>%
    arrange(desc(.data$missing_pct), desc(.data$longest_missing_run), .data$variable)
}

plot_missingness_summary <- function(df) {
  tbl <- missingness_summary_table(df)
  if (nrow(tbl) == 0) {
    return(plot_empty_message("No variables are available for missingness summary."))
  }
  tbl %>%
    select(variable, `Missing percent` = missing_pct, `Longest missing run` = longest_missing_run) %>%
    pivot_longer(cols = -variable, names_to = "metric", values_to = "value") %>%
    ggplot(aes(.data$value, reorder(.data$variable, .data$value), fill = .data$metric)) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~ metric, scales = "free_x") +
    scale_fill_manual(values = c("Missing percent" = "#B2182B", "Longest missing run" = "#1D4E89")) +
    labs(x = NULL, y = NULL) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE)
}

plot_pairwise_scatter_matrix <- function(df, time_col, vars, max_vars = 6) {
  vars <- eda_vars_available(df, time_col, vars, max_vars = max_vars)
  if (length(vars) < 2) {
    return(plot_empty_message("Select at least two numeric variables for pairwise scatter plots."))
  }
  pairs <- t(utils::combn(vars, 2))
  plot_df <- purrr::map_dfr(seq_len(nrow(pairs)), function(i) {
    x_name <- pairs[i, 1]
    y_name <- pairs[i, 2]
    tibble(
      pair = paste(x_name, "vs", y_name),
      x = df[[x_name]],
      y = df[[y_name]],
      x_name = x_name,
      y_name = y_name
    )
  })
  ggplot(plot_df, aes(.data$x, .data$y)) +
    geom_point(color = "#168A93", alpha = 0.65, size = 1.3, na.rm = TRUE) +
    geom_smooth(method = "lm", se = FALSE, color = "#1D4E89", linewidth = 0.55, na.rm = TRUE) +
    facet_wrap(~ pair, scales = "free") +
    labs(x = "Variable on x-axis", y = "Variable on y-axis") +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE)
}

lagged_correlation_table <- function(df, time_col, vars, max_lag = 5, method = "spearman", max_vars = 6) {
  vars <- eda_vars_available(df, time_col, vars, max_vars = max_vars)
  if (length(vars) < 2) {
    return(tibble())
  }
  df <- df[order(df[[time_col]]), , drop = FALSE]
  max_lag <- max(0L, as.integer(max_lag %||% 0L))
  max_lag <- min(max_lag, max(0L, nrow(df) - 2L))
  if (max_lag < 0 || nrow(df) < 3) {
    return(tibble())
  }
  pairs <- expand.grid(source = vars, target = vars, stringsAsFactors = FALSE) %>%
    filter(.data$source != .data$target)
  purrr::map_dfr(seq_len(nrow(pairs)), function(i) {
    source <- pairs$source[[i]]
    target <- pairs$target[[i]]
    purrr::map_dfr(0:max_lag, function(lag_k) {
      if (lag_k == 0) {
        x <- df[[source]]
        y <- df[[target]]
      } else {
        x <- df[[source]][seq_len(nrow(df) - lag_k)]
        y <- df[[target]][(lag_k + 1):nrow(df)]
      }
      ok <- stats::complete.cases(x, y)
      r <- if (sum(ok) >= 4 && stats::sd(x[ok]) > 0 && stats::sd(y[ok]) > 0) {
        suppressWarnings(stats::cor(x[ok], y[ok], method = method))
      } else {
        NA_real_
      }
      tibble(source = source, target = target, lag = lag_k, n = sum(ok), correlation = r)
    })
  })
}

plot_lagged_correlation_heatmap <- function(df, time_col, vars, max_lag = 5, method = "spearman") {
  tbl <- lagged_correlation_table(df, time_col, vars, max_lag, method)
  if (nrow(tbl) == 0) {
    return(plot_empty_message("Select at least two numeric variables for lagged correlation EDA."))
  }
  tbl %>%
    mutate(edge = paste0(.data$source, "(t-lag) -> ", .data$target, "(t)")) %>%
    ggplot(aes(factor(.data$lag), reorder(.data$edge, .data$correlation), fill = .data$correlation)) +
    geom_tile(color = "white") +
    scale_fill_gradient2(low = CIRN_INHIBITION_COLOR, mid = "#F7F7F7", high = CIRN_ACTIVATION_COLOR, limits = c(-1, 1), na.value = "#EEF3F7") +
    labs(x = "Lag in observation steps", y = NULL, fill = paste0(method, " r")) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(panel.grid = element_blank())
}

acf_pacf_table <- function(df, time_col, vars, max_lag = 12, max_vars = 6) {
  vars <- eda_vars_available(df, time_col, vars, max_vars = max_vars)
  if (length(vars) == 0) {
    return(tibble())
  }
  df <- df[order(df[[time_col]]), , drop = FALSE]
  purrr::map_dfr(vars, function(v) {
    x <- df[[v]]
    x <- x[is.finite(x)]
    if (length(x) < 4 || stats::sd(x) <= 0) {
      return(tibble())
    }
    lag_max <- min(as.integer(max_lag), length(x) - 2)
    acf_obj <- stats::acf(x, lag.max = lag_max, plot = FALSE, na.action = stats::na.pass)
    pacf_obj <- stats::pacf(x, lag.max = lag_max, plot = FALSE, na.action = stats::na.pass)
    bind_rows(
      tibble(variable = v, type = "ACF", lag = as.integer(acf_obj$lag[-1]), value = as.numeric(acf_obj$acf[-1])),
      tibble(variable = v, type = "PACF", lag = as.integer(pacf_obj$lag), value = as.numeric(pacf_obj$acf))
    )
  })
}

plot_acf_pacf <- function(df, time_col, vars, max_lag = 12) {
  tbl <- acf_pacf_table(df, time_col, vars, max_lag)
  if (nrow(tbl) == 0) {
    return(plot_empty_message("No finite non-constant series are available for ACF/PACF."))
  }
  ggplot(tbl, aes(.data$lag, .data$value)) +
    geom_hline(yintercept = 0, color = "#A8B7C7") +
    geom_col(fill = "#168A93", width = 0.75) +
    facet_grid(type ~ variable) +
    labs(x = "Lag", y = "Correlation") +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE)
}

plot_phase_pairs <- function(df, time_col, vars, max_vars = 5) {
  vars <- eda_vars_available(df, time_col, vars, max_vars = max_vars)
  if (length(vars) < 2) {
    return(plot_empty_message("Select at least two numeric variables for pairwise phase portraits."))
  }
  pairs <- t(utils::combn(vars, 2))
  plot_df <- purrr::map_dfr(seq_len(nrow(pairs)), function(i) {
    x_name <- pairs[i, 1]
    y_name <- pairs[i, 2]
    tibble(
      panel = paste(x_name, "vs", y_name),
      time = df[[time_col]],
      x = df[[x_name]],
      y = df[[y_name]]
    )
  })
  direction_df <- trajectory_direction_segments(plot_df, max_arrows_per_panel = 10)
  ggplot(plot_df, aes(.data$x, .data$y)) +
    geom_path(aes(color = .data$time), linewidth = 0.7, alpha = 0.85, na.rm = TRUE) +
    geom_segment(
      data = direction_df,
      aes(xend = .data$xend, yend = .data$yend, color = .data$time),
      arrow = grid::arrow(length = grid::unit(0.065, "inches"), type = "closed"),
      linewidth = 0.5,
      alpha = 0.88,
      na.rm = TRUE
    ) +
    geom_point(aes(color = .data$time), size = 1.1, alpha = 0.65, na.rm = TRUE) +
    scale_color_viridis_c(option = "C") +
    facet_wrap(~ panel, scales = "free") +
    labs(
      x = "First variable",
      y = "Second variable",
      color = time_col,
      subtitle = "Sparse arrowheads indicate the observed direction of time."
    ) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

eda_pair_matrix <- function(vars, max_vars = 6, max_pairs = 12) {
  vars <- unique(vars)
  vars <- head(vars, max_vars)
  if (length(vars) < 2) {
    return(matrix(character(), ncol = 2))
  }
  pairs <- t(utils::combn(vars, 2))
  pairs[seq_len(min(nrow(pairs), max_pairs)), , drop = FALSE]
}

evenly_spaced_indices <- function(n, max_n = 80) {
  if (n <= 0) {
    return(integer())
  }
  if (n <= max_n) {
    return(seq_len(n))
  }
  unique(pmax(1L, pmin(n, round(seq(1, n, length.out = max_n)))))
}

trajectory_direction_segments <- function(plot_df, max_arrows_per_panel = 10) {
  required <- c("panel", "time", "x", "y")
  if (is.null(plot_df) || !all(required %in% names(plot_df)) || nrow(plot_df) < 2) {
    return(tibble())
  }
  plot_df <- as_tibble(plot_df) %>%
    mutate(panel = as.character(.data$panel))
  purrr::map_dfr(unique(plot_df$panel), function(panel_name) {
    tmp <- plot_df %>%
      filter(.data$panel == panel_name) %>%
      arrange(.data$time) %>%
      mutate(
        xend = dplyr::lead(.data$x),
        yend = dplyr::lead(.data$y),
        timeend = dplyr::lead(.data$time)
      ) %>%
      filter(stats::complete.cases(.data$x, .data$y, .data$xend, .data$yend, .data$time, .data$timeend))
    if (nrow(tmp) == 0) {
      return(tibble())
    }
    idx <- evenly_spaced_indices(nrow(tmp), max_arrows_per_panel)
    tmp[idx, c("panel", "time", "x", "y", "xend", "yend"), drop = FALSE]
  })
}

plot_directed_phase_portraits <- function(df, time_col, vars, max_vars = 6, max_pairs = 12, max_segments = 80) {
  vars <- eda_vars_available(df, time_col, vars, max_vars = max_vars)
  pairs <- eda_pair_matrix(vars, max_vars = max_vars, max_pairs = max_pairs)
  if (nrow(pairs) == 0) {
    return(plot_empty_message("Select at least two numeric variables for directed phase portraits."))
  }
  df <- df[order(df[[time_col]]), , drop = FALSE]
  plot_df <- purrr::map_dfr(seq_len(nrow(pairs)), function(i) {
    x_name <- pairs[i, 1]
    y_name <- pairs[i, 2]
    tmp <- tibble(
      panel = paste(x_name, "vs", y_name),
      time = df[[time_col]],
      x = df[[x_name]],
      y = df[[y_name]]
    ) %>%
      filter(stats::complete.cases(.data$x, .data$y, .data$time))
    if (nrow(tmp) < 2) {
      return(tibble())
    }
    idx <- evenly_spaced_indices(nrow(tmp) - 1L, max_segments)
    tibble(
      panel = tmp$panel[idx],
      time = tmp$time[idx],
      x = tmp$x[idx],
      y = tmp$y[idx],
      xend = tmp$x[idx + 1L],
      yend = tmp$y[idx + 1L]
    )
  })
  if (nrow(plot_df) == 0) {
    return(plot_empty_message("No complete consecutive observations are available for directed phase portraits."))
  }
  ggplot(plot_df, aes(.data$x, .data$y)) +
    geom_segment(
      aes(xend = .data$xend, yend = .data$yend, color = .data$time),
      arrow = grid::arrow(length = grid::unit(0.08, "inches"), type = "closed"),
      linewidth = 0.55,
      alpha = 0.8,
      na.rm = TRUE
    ) +
    geom_point(aes(color = .data$time), size = 0.9, alpha = 0.65, na.rm = TRUE) +
    scale_color_viridis_c(option = "C") +
    facet_wrap(~ panel, scales = "free") +
    labs(
      x = "First variable",
      y = "Second variable",
      color = time_col,
      subtitle = "Arrows follow the observed time direction in each pairwise state-space projection."
    ) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

plot_derivative_vector_field <- function(df_derivs, time_col, vars, max_vars = 6, max_pairs = 12, max_arrows = 70) {
  if (is.null(df_derivs) || !time_col %in% names(df_derivs)) {
    return(plot_empty_message("Derivative data are not available yet."))
  }
  df_derivs <- df_derivs[order(df_derivs[[time_col]]), , drop = FALSE]
  vars <- intersect(vars %||% character(), sub("_d1$", "", grep("_d1$", names(df_derivs), value = TRUE)))
  if (length(vars) == 0) {
    vars <- unique(sub("_d1$", "", grep("_d1$", names(df_derivs), value = TRUE)))
  }
  pairs <- eda_pair_matrix(vars, max_vars = max_vars, max_pairs = max_pairs)
  if (nrow(pairs) == 0) {
    return(plot_empty_message("Select at least two variables with first derivatives for a derivative vector field."))
  }
  plot_df <- purrr::map_dfr(seq_len(nrow(pairs)), function(i) {
    x_name <- pairs[i, 1]
    y_name <- pairs[i, 2]
    dx_name <- paste0(x_name, "_d1")
    dy_name <- paste0(y_name, "_d1")
    if (!all(c(x_name, y_name, dx_name, dy_name) %in% names(df_derivs))) {
      return(tibble())
    }
    tmp <- tibble(
      panel = paste(x_name, "vs", y_name),
      time = df_derivs[[time_col]],
      x = df_derivs[[x_name]],
      y = df_derivs[[y_name]],
      dx = df_derivs[[dx_name]],
      dy = df_derivs[[dy_name]]
    ) %>%
      filter(stats::complete.cases(.data$x, .data$y, .data$dx, .data$dy, .data$time))
    if (nrow(tmp) < 2) {
      return(tibble())
    }
    idx <- evenly_spaced_indices(nrow(tmp), max_arrows)
    tmp <- tmp[idx, , drop = FALSE]
    xr <- diff(range(tmp$x, na.rm = TRUE))
    yr <- diff(range(tmp$y, na.rm = TRUE))
    max_dx <- max(abs(tmp$dx), na.rm = TRUE)
    max_dy <- max(abs(tmp$dy), na.rm = TRUE)
    scale_x <- if (is.finite(xr) && xr > 0 && is.finite(max_dx) && max_dx > 0) xr / max_dx else NA_real_
    scale_y <- if (is.finite(yr) && yr > 0 && is.finite(max_dy) && max_dy > 0) yr / max_dy else NA_real_
    arrow_scale <- 0.10 * min(c(scale_x, scale_y), na.rm = TRUE)
    if (!is.finite(arrow_scale) || arrow_scale <= 0) {
      arrow_scale <- 1
    }
    tmp %>%
      mutate(
        xend = .data$x + arrow_scale * .data$dx,
        yend = .data$y + arrow_scale * .data$dy,
        speed = sqrt(.data$dx^2 + .data$dy^2)
      )
  })
  if (nrow(plot_df) == 0) {
    return(plot_empty_message("No complete state-derivative pairs are available for vector-field EDA."))
  }
  ggplot(plot_df, aes(.data$x, .data$y)) +
    geom_path(aes(group = .data$panel), color = "#A8B7C7", linewidth = 0.45, alpha = 0.55, na.rm = TRUE) +
    geom_segment(
      aes(xend = .data$xend, yend = .data$yend, color = .data$speed),
      arrow = grid::arrow(length = grid::unit(0.075, "inches"), type = "closed"),
      linewidth = 0.55,
      alpha = 0.85,
      na.rm = TRUE
    ) +
    scale_color_viridis_c(option = "D") +
    facet_wrap(~ panel, scales = "free") +
    labs(
      x = "First state variable",
      y = "Second state variable",
      color = "Local speed",
      subtitle = "Arrows use estimated first derivatives; this is an exploratory projected vector field."
    ) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

plot_delay_embedding <- function(df, time_col, vars, lag_units = 1, max_vars = 5, max_panels = 18) {
  vars <- eda_vars_available(df, time_col, vars, max_vars = max_vars)
  if (length(vars) == 0 || !time_col %in% names(df)) {
    return(plot_empty_message("Select at least one numeric variable for delay embedding."))
  }
  df <- df[order(df[[time_col]]), , drop = FALSE]
  lag_units <- max(1L, as.integer(lag_units %||% 1L))
  lag_units <- min(lag_units, max(1L, nrow(df) - 2L))
  if (nrow(df) <= lag_units + 2L) {
    return(plot_empty_message("More observations are needed for delay embedding at the selected lag."))
  }
  self_panels <- purrr::map_dfr(vars, function(v) {
    idx_now <- (lag_units + 1L):nrow(df)
    idx_lag <- seq_len(nrow(df) - lag_units)
    tibble(
      panel = paste0(v, "(t-", lag_units, ") vs ", v, "(t)"),
      source = v,
      target = v,
      time = df[[time_col]][idx_now],
      x = df[[v]][idx_lag],
      y = df[[v]][idx_now],
      type = "self"
    )
  })
  cross_pairs <- expand.grid(source = vars, target = vars, stringsAsFactors = FALSE) %>%
    filter(.data$source != .data$target) %>%
    slice_head(n = max(0L, max_panels - length(vars)))
  cross_panels <- purrr::map_dfr(seq_len(nrow(cross_pairs)), function(i) {
    source_var <- as.character(cross_pairs[["source"]][i])
    target_var <- as.character(cross_pairs[["target"]][i])
    idx_now <- (lag_units + 1L):nrow(df)
    idx_lag <- seq_len(nrow(df) - lag_units)
    tibble(
      panel = paste0(source_var, "(t-", lag_units, ") -> ", target_var, "(t)"),
      source = source_var,
      target = target_var,
      time = df[[time_col]][idx_now],
      x = df[[source_var]][idx_lag],
      y = df[[target_var]][idx_now],
      type = "cross"
    )
  })
  plot_df <- bind_rows(self_panels, cross_panels) %>%
    filter(stats::complete.cases(.data$x, .data$y, .data$time)) %>%
    mutate(panel = factor(.data$panel, levels = unique(.data$panel)))
  if (nrow(plot_df) == 0) {
    return(plot_empty_message("No complete lagged observations are available for delay embedding."))
  }
  direction_df <- trajectory_direction_segments(plot_df, max_arrows_per_panel = 9)
  ggplot(plot_df, aes(.data$x, .data$y)) +
    geom_path(aes(color = .data$time), linewidth = 0.55, alpha = 0.7, na.rm = TRUE) +
    geom_segment(
      data = direction_df,
      aes(xend = .data$xend, yend = .data$yend, color = .data$time),
      arrow = grid::arrow(length = grid::unit(0.06, "inches"), type = "closed"),
      linewidth = 0.48,
      alpha = 0.82,
      na.rm = TRUE
    ) +
    geom_point(aes(color = .data$time, shape = .data$type), size = 1.0, alpha = 0.7, na.rm = TRUE) +
    scale_color_viridis_c(option = "C") +
    scale_shape_manual(values = c(self = 16, cross = 17), labels = c(self = "Self delay", cross = "Cross delay")) +
    facet_wrap(~ panel, scales = "free") +
    labs(
      x = "Lagged coordinate",
      y = "Current coordinate",
      color = time_col,
      shape = NULL,
      subtitle = "Self-delay panels reveal memory; cross-delay panels screen delayed interactions. Arrowheads follow time."
    ) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

plot_recurrence_map <- function(df, time_col, vars, max_vars = 5, max_pairs = 6, max_points = 120) {
  vars <- eda_vars_available(df, time_col, vars, max_vars = max_vars)
  if (length(vars) == 0 || !time_col %in% names(df)) {
    return(plot_empty_message("Select at least one numeric variable for recurrence analysis."))
  }
  df <- df[order(df[[time_col]]), , drop = FALSE]
  idx <- evenly_spaced_indices(nrow(df), max_points)
  df <- df[idx, , drop = FALSE]
  recurrence_panel <- function(distance_matrix, time_i, time_j, panel) {
    finite_dist <- distance_matrix[is.finite(distance_matrix)]
    finite_dist <- finite_dist[finite_dist > 0]
    if (length(finite_dist) == 0) {
      return(tibble())
    }
    radius <- stats::quantile(finite_dist, probs = 0.10, na.rm = TRUE, names = FALSE)
    grid <- expand.grid(i = seq_along(time_i), j = seq_along(time_j))
    tibble(
      panel = panel,
      time_i = time_i[grid$i],
      time_j = time_j[grid$j],
      recurrent = as.vector(distance_matrix) <= radius,
      radius = radius
    )
  }
  panels <- list()
  complete_state <- df %>%
    select(all_of(c(time_col, vars))) %>%
    filter(stats::complete.cases(.))
  if (nrow(complete_state) >= 3) {
    x <- as.matrix(complete_state[, vars, drop = FALSE])
    keep <- vapply(as.data.frame(x), function(z) stats::sd(z, na.rm = TRUE) > 0, logical(1))
    if (any(keep)) {
      x <- scale(x[, keep, drop = FALSE])
      d <- as.matrix(stats::dist(x))
      panels[[length(panels) + 1L]] <- recurrence_panel(d, complete_state[[time_col]], complete_state[[time_col]], "System state recurrence")
    }
  }
  pairs <- eda_pair_matrix(vars, max_vars = max_vars, max_pairs = max_pairs)
  if (nrow(pairs) > 0) {
    pair_panels <- purrr::map(seq_len(nrow(pairs)), function(i) {
      a <- pairs[i, 1]
      b <- pairs[i, 2]
      tmp <- df %>%
        select(all_of(c(time_col, a, b))) %>%
        filter(stats::complete.cases(.))
      if (nrow(tmp) < 3 || stats::sd(tmp[[a]], na.rm = TRUE) <= 0 || stats::sd(tmp[[b]], na.rm = TRUE) <= 0) {
        return(tibble())
      }
      xa <- as.numeric(scale(tmp[[a]]))
      yb <- as.numeric(scale(tmp[[b]]))
      d <- abs(outer(xa, yb, "-"))
      recurrence_panel(d, tmp[[time_col]], tmp[[time_col]], paste(a, "cross-recurrence", b))
    })
    panels <- c(panels, pair_panels)
  }
  plot_df <- bind_rows(panels) %>%
    mutate(recurrent = coalesce(.data$recurrent, FALSE))
  if (nrow(plot_df) == 0) {
    return(plot_empty_message("No finite recurrence structure is available for the selected variables."))
  }
  ggplot(plot_df, aes(.data$time_i, .data$time_j, fill = .data$recurrent)) +
    geom_raster() +
    facet_wrap(~ panel, scales = "free") +
    scale_fill_manual(values = c("FALSE" = "#EEF3F7", "TRUE" = "#102033"), labels = c("Not recurrent", "Recurrent")) +
    labs(
      x = paste(time_col, "index/time"),
      y = paste(time_col, "index/time"),
      fill = NULL,
      subtitle = "Dark cells mark times where states are close under an adaptive 10% distance threshold."
    ) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(panel.grid = element_blank(), legend.position = "bottom")
}

plot_estimated_nullclines <- function(df_derivs, time_col, vars, max_vars = 5, max_pairs = 8, grid_n = 45) {
  if (is.null(df_derivs) || !time_col %in% names(df_derivs)) {
    return(plot_empty_message("Derivative data are not available yet."))
  }
  df_derivs <- df_derivs[order(df_derivs[[time_col]]), , drop = FALSE]
  vars <- intersect(vars %||% character(), sub("_d1$", "", grep("_d1$", names(df_derivs), value = TRUE)))
  if (length(vars) == 0) {
    vars <- unique(sub("_d1$", "", grep("_d1$", names(df_derivs), value = TRUE)))
  }
  pairs <- eda_pair_matrix(vars, max_vars = max_vars, max_pairs = max_pairs)
  if (nrow(pairs) == 0) {
    return(plot_empty_message("Select at least two variables with first derivatives for estimated nullclines."))
  }
  path_df <- purrr::map_dfr(seq_len(nrow(pairs)), function(i) {
    x_name <- pairs[i, 1]
    y_name <- pairs[i, 2]
    tibble(
      panel = paste(x_name, "vs", y_name),
      time = df_derivs[[time_col]],
      x = df_derivs[[x_name]],
      y = df_derivs[[y_name]]
    ) %>%
      filter(stats::complete.cases(.data$x, .data$y, .data$time))
  })
  contour_df <- purrr::map_dfr(seq_len(nrow(pairs)), function(i) {
    x_name <- pairs[i, 1]
    y_name <- pairs[i, 2]
    dx_name <- paste0(x_name, "_d1")
    dy_name <- paste0(y_name, "_d1")
    if (!all(c(x_name, y_name, dx_name, dy_name) %in% names(df_derivs))) {
      return(tibble())
    }
    dat <- tibble(
      x = df_derivs[[x_name]],
      y = df_derivs[[y_name]],
      dx = df_derivs[[dx_name]],
      dy = df_derivs[[dy_name]]
    ) %>%
      filter(stats::complete.cases(.))
    if (nrow(dat) < 12 || stats::sd(dat$x) <= 0 || stats::sd(dat$y) <= 0) {
      return(tibble())
    }
    x_grid <- seq(min(dat$x, na.rm = TRUE), max(dat$x, na.rm = TRUE), length.out = grid_n)
    y_grid <- seq(min(dat$y, na.rm = TRUE), max(dat$y, na.rm = TRUE), length.out = grid_n)
    pred_grid <- expand.grid(x = x_grid, y = y_grid)
    fit_surface <- function(response_col, label) {
      fit <- tryCatch(
        stats::loess(
          stats::as.formula(paste(response_col, "~ x + y")),
          data = dat,
          span = 0.75,
          degree = 1,
          control = stats::loess.control(surface = "direct")
        ),
        error = function(e) NULL
      )
      if (is.null(fit)) {
        return(tibble())
      }
      z <- tryCatch(stats::predict(fit, newdata = pred_grid), error = function(e) rep(NA_real_, nrow(pred_grid)))
      tibble(
        panel = paste(x_name, "vs", y_name),
        x = pred_grid$x,
        y = pred_grid$y,
        z = as.numeric(z),
        nullcline = label
      )
    }
    bind_rows(
      fit_surface("dx", paste0("d", x_name, " = 0")),
      fit_surface("dy", paste0("d", y_name, " = 0"))
    )
  })
  if (nrow(path_df) == 0 && nrow(contour_df) == 0) {
    return(plot_empty_message("No complete pairwise state-derivative data are available for estimated nullclines."))
  }
  p <- ggplot() +
    geom_path(data = path_df, aes(.data$x, .data$y), color = "#A8B7C7", linewidth = 0.55, alpha = 0.75, na.rm = TRUE) +
    geom_point(data = path_df, aes(.data$x, .data$y), color = "#102033", size = 0.65, alpha = 0.35, na.rm = TRUE)
  if (nrow(contour_df) > 0) {
    nullcline_levels <- sort(unique(contour_df$nullcline))
    nullcline_colors <- rep(c("#1D4E89", "#B7791F", "#6B7280", "#7C3AED", "#0F766E", "#8A1538"), length.out = length(nullcline_levels))
    names(nullcline_colors) <- nullcline_levels
    p <- p +
      geom_contour(
        data = contour_df,
        aes(.data$x, .data$y, z = .data$z, color = .data$nullcline, linetype = .data$nullcline),
        breaks = 0,
        linewidth = 0.85,
        na.rm = TRUE
      ) +
      scale_color_manual(values = nullcline_colors)
  }
  p +
    facet_wrap(~ panel, scales = "free") +
    labs(
      x = "First state variable",
      y = "Second state variable",
      color = NULL,
      linetype = NULL,
      subtitle = "Exploratory pairwise nullcline projections estimated from smoothed first derivatives."
    ) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

plot_pca_trajectory <- function(df, time_col, vars) {
  vars <- eda_vars_available(df, time_col, vars)
  if (length(vars) < 2) {
    return(plot_empty_message("Select at least two numeric variables for PCA trajectory EDA."))
  }
  pca_df <- df %>%
    select(all_of(c(time_col, vars))) %>%
    filter(stats::complete.cases(.))
  if (nrow(pca_df) < 3) {
    return(plot_empty_message("At least three complete rows are needed for PCA trajectory EDA."))
  }
  x <- pca_df[, vars, drop = FALSE]
  keep <- vapply(x, function(z) stats::sd(z, na.rm = TRUE) > 0, logical(1))
  x <- x[, keep, drop = FALSE]
  if (ncol(x) < 2) {
    return(plot_empty_message("At least two non-constant variables are needed for PCA trajectory EDA."))
  }
  pca <- stats::prcomp(x, center = TRUE, scale. = TRUE)
  var_exp <- 100 * (pca$sdev^2 / sum(pca$sdev^2))
  score_df <- tibble(
    panel = "PCA trajectory",
    time = pca_df[[time_col]],
    PC1 = pca$x[, 1],
    PC2 = pca$x[, 2]
  )
  direction_df <- score_df %>%
    transmute(panel = .data$panel, time = .data$time, x = .data$PC1, y = .data$PC2) %>%
    trajectory_direction_segments(max_arrows_per_panel = 12)
  ggplot(score_df, aes(.data$PC1, .data$PC2)) +
    geom_path(color = "#A8B7C7", linewidth = 0.7, na.rm = TRUE) +
    geom_segment(
      data = direction_df,
      aes(x = .data$x, y = .data$y, xend = .data$xend, yend = .data$yend, color = .data$time),
      inherit.aes = FALSE,
      arrow = grid::arrow(length = grid::unit(0.07, "inches"), type = "closed"),
      linewidth = 0.55,
      alpha = 0.88,
      na.rm = TRUE
    ) +
    geom_point(aes(color = .data$time), size = 2, alpha = 0.85, na.rm = TRUE) +
    scale_color_viridis_c(option = "C") +
    labs(
      x = paste0("PC1 (", round(var_exp[1], 1), "%)"),
      y = paste0("PC2 (", round(var_exp[2], 1), "%)"),
      color = time_col,
      subtitle = "Sparse arrowheads indicate multivariate movement through time."
    ) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE)
}

plot_derivative_state_relationships <- function(df_derivs, time_col, vars, targets, max_panels = 12) {
  vars <- intersect(vars %||% character(), names(df_derivs))
  targets <- intersect(targets %||% character(), sub("_d1$", "", grep("_d1$", names(df_derivs), value = TRUE)))
  if (length(vars) == 0 || length(targets) == 0) {
    return(plot_empty_message("Select predictors and targets to view derivative-state relationships."))
  }
  pairs <- expand.grid(predictor = vars, target = targets, stringsAsFactors = FALSE) %>%
    slice_head(n = max_panels)
  plot_df <- purrr::map_dfr(seq_len(nrow(pairs)), function(i) {
    predictor <- pairs$predictor[[i]]
    target <- pairs$target[[i]]
    d1 <- paste0(target, "_d1")
    if (!d1 %in% names(df_derivs)) {
      return(tibble())
    }
    tibble(
      panel = paste0(predictor, " vs d", target),
      x = df_derivs[[predictor]],
      derivative = df_derivs[[d1]],
      predictor = predictor,
      target = target
    )
  })
  if (nrow(plot_df) == 0) {
    return(plot_empty_message("No derivative-state pairs are available."))
  }
  ggplot(plot_df, aes(.data$x, .data$derivative)) +
    geom_hline(yintercept = 0, color = "#A8B7C7") +
    geom_point(color = "#168A93", size = 1.4, alpha = 0.65, na.rm = TRUE) +
    geom_smooth(method = "lm", se = FALSE, color = "#1D4E89", linewidth = 0.55, na.rm = TRUE) +
    facet_wrap(~ panel, scales = "free") +
    labs(x = "Candidate predictor value", y = "Target first derivative") +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE)
}

plot_derivative_distributions <- function(df_derivs, time_col, vars) {
  vars <- intersect(vars %||% character(), sub("_d[12]$", "", grep("_d[12]$", names(df_derivs), value = TRUE)))
  if (length(vars) == 0) {
    return(plot_empty_message("No derivative columns are available for derivative distribution EDA."))
  }
  dvars <- intersect(c(paste0(vars, "_d1"), paste0(vars, "_d2")), names(df_derivs))
  df_derivs %>%
    select(all_of(dvars)) %>%
    pivot_longer(cols = everything(), names_to = "series", values_to = "value") %>%
    mutate(
      order = ifelse(grepl("_d2$", .data$series), "Second derivative", "First derivative"),
      variable = sub("_d[12]$", "", .data$series)
    ) %>%
    ggplot(aes(.data$value)) +
    geom_vline(xintercept = 0, color = "#6B7280", linetype = 2) +
    geom_histogram(aes(y = after_stat(density)), bins = 28, fill = "#B9DDE0", color = "white", na.rm = TRUE) +
    geom_density(color = "#0F766E", linewidth = 0.8, na.rm = TRUE) +
    facet_grid(order ~ variable, scales = "free") +
    labs(x = "Derivative value", y = "Density") +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE)
}

epsilon_sensitivity_table <- function(df_derivs, targets, log_min = -10, log_max = -1, grid_n = 45) {
  targets <- intersect(targets %||% character(), sub("_d1$", "", grep("_d1$", names(df_derivs), value = TRUE)))
  if (length(targets) == 0) {
    return(tibble())
  }
  logs <- seq(log_min, log_max, length.out = grid_n)
  purrr::map_dfr(targets, function(target) {
    d1 <- df_derivs[[paste0(target, "_d1")]]
    purrr::map_dfr(logs, function(log_eps) {
      eps <- 10^log_eps
      class_1 <- sum(d1 > eps, na.rm = TRUE)
      class_0 <- sum(d1 < -eps, na.rm = TRUE)
      blank <- sum(abs(d1) <= eps | is.na(d1), na.rm = TRUE)
      total <- length(d1)
      tibble(
        target = target,
        log10_epsilon = log_eps,
        epsilon = eps,
        decreasing = class_0,
        increasing = class_1,
        blank = blank,
        usable = class_0 + class_1,
        minority_class = pmin(class_0, class_1),
        blank_pct = 100 * blank / total
      )
    })
  })
}

plot_epsilon_sensitivity <- function(df_derivs, targets, current_log_eps = -6) {
  tbl <- epsilon_sensitivity_table(df_derivs, targets)
  if (nrow(tbl) == 0) {
    return(plot_empty_message("No target derivatives are available for epsilon sensitivity."))
  }
  plot_df <- tbl %>%
    select(target, log10_epsilon, `Blank percent` = blank_pct, `Minority class count` = minority_class) %>%
    pivot_longer(cols = c("Blank percent", "Minority class count"), names_to = "metric", values_to = "value")
  ggplot(plot_df, aes(.data$log10_epsilon, .data$value, color = .data$metric)) +
    geom_vline(xintercept = current_log_eps, linetype = 2, color = "#102033") +
    geom_line(linewidth = 0.8, na.rm = TRUE) +
    facet_grid(metric ~ target, scales = "free_y") +
    scale_color_manual(values = c("Blank percent" = "#B2182B", "Minority class count" = "#1D4E89")) +
    labs(x = "log10 epsilon", y = NULL, color = NULL, subtitle = "Vertical line marks the current epsilon setting.") +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

rolling_window_tbl <- function(df, time_col, vars, window = 15, max_vars = 6) {
  vars <- eda_vars_available(df, time_col, vars, max_vars = max_vars)
  if (length(vars) == 0 || !time_col %in% names(df)) {
    return(tibble())
  }
  df <- df[order(df[[time_col]]), , drop = FALSE]
  n <- nrow(df)
  window <- max(3L, as.integer(window %||% 15L))
  window <- min(window, n)
  if (n < 3 || window < 3) {
    return(tibble())
  }
  idx <- window:n
  purrr::map_dfr(vars, function(v) {
    x <- df[[v]]
    purrr::map_dfr(idx, function(i) {
      w <- x[(i - window + 1):i]
      finite_w <- w[is.finite(w)]
      tibble(
        variable = v,
        time = df[[time_col]][i],
        window_n = length(finite_w),
        rolling_mean = if (length(finite_w) > 0) mean(finite_w) else NA_real_,
        rolling_variance = if (length(finite_w) > 1) stats::var(finite_w) else NA_real_
      )
    })
  })
}

plot_rolling_mean_variance <- function(df, time_col, vars, window = 15) {
  tbl <- rolling_window_tbl(df, time_col, vars, window)
  if (nrow(tbl) == 0) {
    return(plot_empty_message("At least one numeric variable and three ordered observations are needed for rolling mean/variance."))
  }
  tbl %>%
    select(variable, time, `Rolling mean` = rolling_mean, `Rolling variance` = rolling_variance) %>%
    pivot_longer(cols = c("Rolling mean", "Rolling variance"), names_to = "metric", values_to = "value") %>%
    ggplot(aes(.data$time, .data$value)) +
    geom_line(color = "#168A93", linewidth = 0.75, na.rm = TRUE) +
    geom_point(color = "#1D4E89", size = 0.9, alpha = 0.55, na.rm = TRUE) +
    facet_grid(metric ~ variable, scales = "free_y") +
    labs(
      x = time_col,
      y = NULL,
      subtitle = paste("Rolling window:", min(max(3L, as.integer(window %||% 15L)), nrow(df)), "observations")
    ) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE)
}

rolling_correlation_table <- function(df, time_col, vars, window = 15, method = "spearman", max_vars = 5) {
  vars <- eda_vars_available(df, time_col, vars, max_vars = max_vars)
  if (length(vars) < 2 || !time_col %in% names(df)) {
    return(tibble())
  }
  df <- df[order(df[[time_col]]), , drop = FALSE]
  n <- nrow(df)
  window <- max(4L, as.integer(window %||% 15L))
  window <- min(window, n)
  if (n < 4 || window < 4) {
    return(tibble())
  }
  pairs <- t(utils::combn(vars, 2))
  idx <- window:n
  purrr::map_dfr(seq_len(nrow(pairs)), function(j) {
    a <- pairs[j, 1]
    b <- pairs[j, 2]
    purrr::map_dfr(idx, function(i) {
      x <- df[[a]][(i - window + 1):i]
      y <- df[[b]][(i - window + 1):i]
      ok <- stats::complete.cases(x, y)
      r <- if (sum(ok) >= 4 && stats::sd(x[ok]) > 0 && stats::sd(y[ok]) > 0) {
        suppressWarnings(stats::cor(x[ok], y[ok], method = method))
      } else {
        NA_real_
      }
      tibble(
        pair = paste(a, "vs", b),
        variable_x = a,
        variable_y = b,
        time = df[[time_col]][i],
        n = sum(ok),
        correlation = r
      )
    })
  })
}

plot_rolling_correlation <- function(df, time_col, vars, window = 15, method = "spearman") {
  tbl <- rolling_correlation_table(df, time_col, vars, window, method)
  if (nrow(tbl) == 0) {
    return(plot_empty_message("Select at least two numeric variables for rolling correlation."))
  }
  ggplot(tbl, aes(.data$time, .data$correlation)) +
    geom_hline(yintercept = 0, color = "#A8B7C7") +
    geom_line(color = "#168A93", linewidth = 0.75, na.rm = TRUE) +
    geom_point(aes(color = .data$correlation), size = 0.9, alpha = 0.7, na.rm = TRUE) +
    facet_wrap(~ pair) +
    scale_y_continuous(limits = c(-1, 1)) +
    scale_color_gradient2(low = CIRN_INHIBITION_COLOR, mid = "#F7F7F7", high = CIRN_ACTIVATION_COLOR, limits = c(-1, 1), na.value = "#A8B7C7") +
    labs(
      x = time_col,
      y = paste("Rolling", method, "correlation"),
      color = "r",
      subtitle = paste("Window:", min(max(4L, as.integer(window %||% 15L)), nrow(df)), "observations")
    ) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

cross_correlation_table <- function(df, time_col, vars, max_lag = 8, method = "spearman", max_vars = 6) {
  vars <- eda_vars_available(df, time_col, vars, max_vars = max_vars)
  if (length(vars) < 2 || !time_col %in% names(df)) {
    return(tibble())
  }
  df <- df[order(df[[time_col]]), , drop = FALSE]
  n <- nrow(df)
  max_lag <- max(0L, as.integer(max_lag %||% 0L))
  max_lag <- min(max_lag, max(0L, n - 3L))
  if (n < 4) {
    return(tibble())
  }
  pairs <- expand.grid(source = vars, target = vars, stringsAsFactors = FALSE) %>%
    filter(.data$source != .data$target)
  purrr::map_dfr(seq_len(nrow(pairs)), function(i) {
    source <- pairs$source[[i]]
    target <- pairs$target[[i]]
    purrr::map_dfr((-max_lag):max_lag, function(lag_k) {
      if (lag_k >= 0) {
        x <- df[[source]][seq_len(n - lag_k)]
        y <- df[[target]][(lag_k + 1):n]
      } else {
        lead_k <- abs(lag_k)
        x <- df[[source]][(lead_k + 1):n]
        y <- df[[target]][seq_len(n - lead_k)]
      }
      ok <- stats::complete.cases(x, y)
      r <- if (sum(ok) >= 4 && stats::sd(x[ok]) > 0 && stats::sd(y[ok]) > 0) {
        suppressWarnings(stats::cor(x[ok], y[ok], method = method))
      } else {
        NA_real_
      }
      tibble(
        source = source,
        target = target,
        lag = lag_k,
        n = sum(ok),
        correlation = r
      )
    })
  })
}

plot_cross_correlation_lead_lag <- function(df, time_col, vars, max_lag = 8, method = "spearman") {
  tbl <- cross_correlation_table(df, time_col, vars, max_lag, method)
  if (nrow(tbl) == 0) {
    return(plot_empty_message("Select at least two numeric variables for cross-correlation lead-lag EDA."))
  }
  tbl %>%
    mutate(edge = paste0(.data$source, " -> ", .data$target)) %>%
    ggplot(aes(.data$lag, reorder(.data$edge, .data$correlation), fill = .data$correlation)) +
    geom_tile(color = "white") +
    geom_vline(xintercept = 0, color = "#102033", linewidth = 0.35) +
    scale_fill_gradient2(low = CIRN_INHIBITION_COLOR, mid = "#F7F7F7", high = CIRN_ACTIVATION_COLOR, limits = c(-1, 1), na.value = "#EEF3F7") +
    labs(
      x = "Lag in observation steps; positive means source earlier than target",
      y = NULL,
      fill = paste0(method, " r")
    ) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(panel.grid = element_blank())
}

plot_derivative_phase_portraits <- function(df_derivs, time_col, vars, max_vars = 6) {
  if (is.null(df_derivs) || !time_col %in% names(df_derivs)) {
    return(plot_empty_message("Derivative data are not available yet."))
  }
  vars <- intersect(vars %||% character(), sub("_d[12]$", "", grep("_d[12]$", names(df_derivs), value = TRUE)))
  if (length(vars) == 0) {
    vars <- unique(sub("_d[12]$", "", grep("_d[12]$", names(df_derivs), value = TRUE)))
  }
  vars <- head(vars, max_vars)
  if (length(vars) == 0) {
    return(plot_empty_message("No derivative columns are available for derivative phase portraits."))
  }
  plot_df <- purrr::map_dfr(vars, function(v) {
    d1 <- paste0(v, "_d1")
    d2 <- paste0(v, "_d2")
    bind_rows(
      if (all(c(v, d1) %in% names(df_derivs))) {
        tibble(
          variable = v,
          panel = paste0(v, " vs d", v),
          x = df_derivs[[v]],
          y = df_derivs[[d1]],
          x_label = v,
          y_label = paste0("d", v),
          time = df_derivs[[time_col]]
        )
      } else {
        tibble()
      },
      if (all(c(d1, d2) %in% names(df_derivs))) {
        tibble(
          variable = v,
          panel = paste0("d", v, " vs d2", v),
          x = df_derivs[[d1]],
          y = df_derivs[[d2]],
          x_label = paste0("d", v),
          y_label = paste0("d2", v),
          time = df_derivs[[time_col]]
        )
      } else {
        tibble()
      }
    )
  })
  if (nrow(plot_df) == 0) {
    return(plot_empty_message("No derivative phase portraits are available."))
  }
  direction_df <- trajectory_direction_segments(plot_df, max_arrows_per_panel = 10)
  ggplot(plot_df, aes(.data$x, .data$y)) +
    geom_hline(yintercept = 0, color = "#D5DEE8") +
    geom_vline(xintercept = 0, color = "#D5DEE8") +
    geom_path(aes(color = .data$time), linewidth = 0.65, alpha = 0.85, na.rm = TRUE) +
    geom_segment(
      data = direction_df,
      aes(xend = .data$xend, yend = .data$yend, color = .data$time),
      arrow = grid::arrow(length = grid::unit(0.065, "inches"), type = "closed"),
      linewidth = 0.5,
      alpha = 0.88,
      na.rm = TRUE
    ) +
    geom_point(aes(color = .data$time), size = 1.0, alpha = 0.65, na.rm = TRUE) +
    scale_color_viridis_c(option = "C") +
    facet_wrap(~ panel, scales = "free") +
    labs(
      x = "Horizontal phase coordinate",
      y = "Vertical phase coordinate",
      color = time_col,
      subtitle = "Sparse arrowheads indicate the observed direction of time."
    ) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

plot_cross_derivative_phase_portraits <- function(df_derivs, time_col, vars, max_pairs = 4, max_panels = 18) {
  if (is.null(df_derivs) || !time_col %in% names(df_derivs)) {
    return(plot_empty_message("Derivative data are not available yet."))
  }
  vars <- intersect(vars %||% character(), sub("_d[12]$", "", grep("_d[12]$", names(df_derivs), value = TRUE)))
  if (length(vars) == 0) {
    vars <- unique(sub("_d[12]$", "", grep("_d[12]$", names(df_derivs), value = TRUE)))
  }
  if (length(vars) < 2) {
    return(plot_empty_message("Select at least two numeric variables for cross-derivative phase portraits."))
  }
  pairs <- t(utils::combn(head(vars, max_pairs), 2))
  feature_value <- function(variable, order) {
    col <- switch(
      order,
      state = variable,
      d1 = paste0(variable, "_d1"),
      d2 = paste0(variable, "_d2"),
      variable
    )
    if (!col %in% names(df_derivs)) {
      return(NULL)
    }
    df_derivs[[col]]
  }
  feature_label <- function(variable, order) {
    switch(
      order,
      state = variable,
      d1 = paste0("d", variable),
      d2 = paste0("d2", variable),
      variable
    )
  }
  add_panel <- function(a, a_order, b, b_order) {
    x <- feature_value(a, a_order)
    y <- feature_value(b, b_order)
    if (is.null(x) || is.null(y)) {
      return(tibble())
    }
    tibble(
      panel = paste0(feature_label(a, a_order), " vs ", feature_label(b, b_order)),
      x = x,
      y = y,
      time = df_derivs[[time_col]]
    )
  }
  plot_df <- purrr::map_dfr(seq_len(nrow(pairs)), function(i) {
    a <- pairs[i, 1]
    b <- pairs[i, 2]
    bind_rows(
      add_panel(a, "state", b, "d1"),
      add_panel(b, "state", a, "d1"),
      add_panel(a, "d1", b, "d1"),
      add_panel(a, "state", b, "d2"),
      add_panel(b, "state", a, "d2"),
      add_panel(a, "d2", b, "d2")
    )
  })
  if (nrow(plot_df) == 0) {
    return(plot_empty_message("No cross-derivative phase portraits are available."))
  }
  panel_order <- unique(plot_df$panel)
  plot_df <- plot_df %>%
    filter(.data$panel %in% head(panel_order, max_panels)) %>%
    mutate(panel = factor(.data$panel, levels = head(panel_order, max_panels)))
  direction_df <- trajectory_direction_segments(plot_df, max_arrows_per_panel = 9)
  ggplot(plot_df, aes(.data$x, .data$y)) +
    geom_hline(yintercept = 0, color = "#D5DEE8") +
    geom_vline(xintercept = 0, color = "#D5DEE8") +
    geom_path(aes(color = .data$time), linewidth = 0.65, alpha = 0.85, na.rm = TRUE) +
    geom_segment(
      data = direction_df,
      aes(xend = .data$xend, yend = .data$yend, color = .data$time),
      arrow = grid::arrow(length = grid::unit(0.06, "inches"), type = "closed"),
      linewidth = 0.48,
      alpha = 0.86,
      na.rm = TRUE
    ) +
    geom_point(aes(color = .data$time), size = 1.0, alpha = 0.65, na.rm = TRUE) +
    scale_color_viridis_c(option = "C") +
    facet_wrap(~ panel, scales = "free") +
    labs(
      x = "Cross-variable phase coordinate",
      y = "Derivative/state coordinate",
      color = time_col,
      subtitle = "Arrowheads follow time. These are exploratory interaction screens, not CIRN edge estimates."
    ) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

plot_epsilon_class_balance_stack <- function(df_derivs, targets, current_log_eps = -6) {
  tbl <- epsilon_sensitivity_table(df_derivs, targets)
  if (nrow(tbl) == 0) {
    return(plot_empty_message("No target derivatives are available for epsilon class-balance EDA."))
  }
  tbl %>%
    select(target, log10_epsilon, decreasing, increasing, blank) %>%
    pivot_longer(cols = c(decreasing, increasing, blank), names_to = "class", values_to = "count") %>%
    mutate(class = factor(.data$class, levels = c("increasing", "decreasing", "blank"))) %>%
    ggplot(aes(.data$log10_epsilon, .data$count, fill = .data$class)) +
    geom_area(alpha = 0.92, color = "white", linewidth = 0.15, na.rm = TRUE) +
    geom_vline(xintercept = current_log_eps, linetype = 2, color = "#102033") +
    facet_wrap(~ target, scales = "free_y") +
    scale_fill_manual(values = c(increasing = CIRN_ACTIVATION_COLOR, decreasing = CIRN_INHIBITION_COLOR, blank = "#CBD5E1")) +
    labs(x = "log10 epsilon", y = "Observation count", fill = NULL, subtitle = "Vertical line marks the current epsilon setting.") +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

variable_attention_table <- function(df, df_derivs, time_col, vars, eps = 1e-6, outlier_threshold = 3.5) {
  vars <- eda_vars_available(df, time_col, vars)
  if (length(vars) == 0) {
    return(tibble())
  }
  outlier_tbl <- detect_mad_outliers(df, time_col, vars, outlier_threshold)
  derivative_tbl <- derivative_summary_table(df_derivs, vars, eps)
  purrr::map_dfr(vars, function(v) {
    x <- df[[v]]
    n <- length(x)
    missing_pct <- 100 * mean(is.na(x))
    outlier_pct <- outlier_tbl %>%
      filter(.data$variable == v) %>%
      summarise(value = 100 * mean(.data$outlier, na.rm = TRUE), .groups = "drop") %>%
      pull(value)
    if (length(outlier_pct) == 0 || !is.finite(outlier_pct)) {
      outlier_pct <- 0
    }
    d1 <- derivative_tbl %>% filter(.data$variable == v, .data$derivative == "first derivative")
    near_zero_pct <- if (nrow(d1) > 0) d1$near_zero_pct[[1]] else NA_real_
    sign_change_rate <- if (nrow(d1) > 0 && is.finite(d1$n_finite[[1]]) && d1$n_finite[[1]] > 0) {
      100 * d1$sign_changes[[1]] / d1$n_finite[[1]]
    } else {
      NA_real_
    }
    sd_x <- stats::sd(x, na.rm = TRUE)
    range_x <- diff(range(x, na.rm = TRUE))
    flatness_score <- if (is.finite(range_x) && range_x > 0 && is.finite(sd_x)) {
      pmax(0, pmin(100, 100 * (1 - sd_x / range_x)))
    } else {
      NA_real_
    }
    tibble(
      variable = v,
      metric = c("Missing %", "MAD outlier %", "Derivative near-zero %", "Derivative sign-change rate", "Low-variation screen"),
      value = c(missing_pct, outlier_pct, near_zero_pct, sign_change_rate, flatness_score)
    )
  })
}

plot_variable_attention_summary <- function(df, df_derivs, time_col, vars, eps = 1e-6, outlier_threshold = 3.5) {
  tbl <- variable_attention_table(df, df_derivs, time_col, vars, eps, outlier_threshold)
  if (nrow(tbl) == 0) {
    return(plot_empty_message("No variables are available for variable attention summary."))
  }
  tbl <- tbl %>% mutate(value_for_order = ifelse(is.finite(.data$value), .data$value, 0))
  ggplot(tbl, aes(.data$metric, reorder(.data$variable, .data$value_for_order, FUN = max), fill = .data$value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = ifelse(is.finite(.data$value), sprintf("%.1f", .data$value), "")), size = 3) +
    scale_fill_gradient(low = "#F8FAFC", high = CIRN_INHIBITION_COLOR, limits = c(0, 100), na.value = "#EEF3F7") +
    labs(x = NULL, y = NULL, fill = "attention") +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(axis.text.x = element_text(angle = 25, hjust = 1), panel.grid = element_blank())
}

safe_test <- function(expr) {
  tryCatch(
    suppressWarnings(suppressMessages(force(expr))),
    error = function(e) NULL
  )
}

time_spacing_diagnostic_table <- function(df, time_col) {
  if (is.null(df) || !time_col %in% names(df)) {
    return(tibble())
  }
  time_values <- df[[time_col]]
  finite_time <- time_values[is.finite(time_values)]
  unique_time <- sort(unique(finite_time))
  gaps <- diff(unique_time)
  gap_min <- if (length(gaps) > 0) min(gaps, na.rm = TRUE) else NA_real_
  gap_median <- if (length(gaps) > 0) stats::median(gaps, na.rm = TRUE) else NA_real_
  gap_max <- if (length(gaps) > 0) max(gaps, na.rm = TRUE) else NA_real_
  gap_sd <- if (length(gaps) > 1) stats::sd(gaps, na.rm = TRUE) else NA_real_
  gap_mean <- if (length(gaps) > 0) mean(gaps, na.rm = TRUE) else NA_real_
  irregularity_ratio <- if (is.finite(gap_min) && gap_min > 0) gap_max / gap_min else NA_real_
  gap_cv <- if (is.finite(gap_mean) && gap_mean > 0) gap_sd / gap_mean else NA_real_
  duplicate_count <- length(time_values) - length(unique(time_values[!is.na(time_values)]))
  ordered_ok <- all(diff(time_values[!is.na(time_values)]) >= 0)
  
  tibble(
    metric = c(
      "Rows",
      "Finite time values",
      "Unique finite time values",
      "Duplicate time values",
      "Time column ordered",
      "Minimum spacing",
      "Median spacing",
      "Maximum spacing",
      "Spacing SD",
      "Spacing CV",
      "Max/min spacing ratio",
      "Interpretation"
    ),
    value = c(
      as.character(nrow(df)),
      as.character(length(finite_time)),
      as.character(length(unique_time)),
      as.character(duplicate_count),
      if (ordered_ok) "yes" else "no",
      as.character(signif(gap_min, 6)),
      as.character(signif(gap_median, 6)),
      as.character(signif(gap_max, 6)),
      as.character(signif(gap_sd, 6)),
      as.character(signif(gap_cv, 6)),
      as.character(signif(irregularity_ratio, 6)),
      dplyr::case_when(
        length(gaps) == 0 ~ "At least two unique time points are needed.",
        duplicate_count > 0 ~ "Duplicate time values are present; inspect aggregation or repeated measurements.",
        is.finite(irregularity_ratio) && irregularity_ratio > 3 ~ "Irregular sampling may affect derivative estimates; inspect time gaps and derivative plots.",
        !ordered_ok ~ "Rows are not ordered by time; the app sorts internally, but inspect the input table.",
        TRUE ~ "Time spacing looks reasonably regular for initial EDA."
      )
    )
  )
}

trend_screen_table <- function(df, time_col, vars) {
  vars <- eda_vars_available(df, time_col, vars)
  if (length(vars) == 0 || !time_col %in% names(df)) {
    return(tibble())
  }
  df <- df[order(df[[time_col]]), , drop = FALSE]
  t <- df[[time_col]]
  purrr::map_dfr(vars, function(v) {
    x <- df[[v]]
    ok <- is.finite(t) & is.finite(x)
    n <- sum(ok)
    if (n < 4 || stats::sd(x[ok]) <= 0 || stats::sd(t[ok]) <= 0) {
      return(tibble(
        variable = v,
        n = n,
        linear_slope = NA_real_,
        linear_p = NA_real_,
        spearman_time_r = NA_real_,
        spearman_time_p = NA_real_,
        first_value = if (n > 0) x[ok][1] else NA_real_,
        last_value = if (n > 0) utils::tail(x[ok], 1) else NA_real_,
        net_change = NA_real_,
        direction_note = "Too few finite, non-constant observations."
      ))
    }
    dat <- tibble(t = t[ok], x = x[ok])
    fit <- safe_test(stats::lm(x ~ t, data = dat))
    slope <- NA_real_
    slope_p <- NA_real_
    if (!is.null(fit)) {
      coef_tbl <- summary(fit)$coefficients
      if ("t" %in% rownames(coef_tbl)) {
        slope <- coef_tbl["t", "Estimate"]
        slope_p <- coef_tbl["t", "Pr(>|t|)"]
      }
    }
    sp <- safe_test(stats::cor.test(dat$t, dat$x, method = "spearman", exact = FALSE))
    sp_r <- if (!is.null(sp)) unname(sp$estimate) else NA_real_
    sp_p <- if (!is.null(sp)) sp$p.value else NA_real_
    net_change <- utils::tail(dat$x, 1) - dat$x[1]
    note <- dplyr::case_when(
      is.finite(slope_p) && slope_p < 0.05 && slope > 0 ~ "Screen suggests upward drift over time.",
      is.finite(slope_p) && slope_p < 0.05 && slope < 0 ~ "Screen suggests downward drift over time.",
      is.finite(sp_p) && sp_p < 0.05 && abs(sp_r) > 0.6 ~ "Screen suggests monotone temporal association.",
      TRUE ~ "No strong simple trend detected by this screen."
    )
    tibble(
      variable = v,
      n = n,
      linear_slope = slope,
      linear_p = slope_p,
      spearman_time_r = sp_r,
      spearman_time_p = sp_p,
      first_value = dat$x[1],
      last_value = utils::tail(dat$x, 1),
      net_change = net_change,
      direction_note = note
    )
  })
}

derivative_summary_table <- function(df_derivs, vars, eps = 1e-6) {
  if (is.null(df_derivs)) {
    return(tibble())
  }
  vars <- intersect(vars %||% character(), sub("_d[12]$", "", grep("_d[12]$", names(df_derivs), value = TRUE)))
  if (length(vars) == 0) {
    vars <- unique(sub("_d[12]$", "", grep("_d[12]$", names(df_derivs), value = TRUE)))
  }
  if (length(vars) == 0) {
    return(tibble())
  }
  derivative_cols <- intersect(c(paste0(vars, "_d1"), paste0(vars, "_d2")), names(df_derivs))
  purrr::map_dfr(derivative_cols, function(col) {
    x <- df_derivs[[col]]
    finite_x <- x[is.finite(x)]
    n <- length(finite_x)
    near_zero_fraction <- if (n > 0) mean(abs(finite_x) <= eps) else NA_real_
    signs <- dplyr::case_when(
      finite_x > eps ~ 1L,
      finite_x < -eps ~ -1L,
      TRUE ~ 0L
    )
    nonzero_signs <- signs[signs != 0L]
    sign_changes <- if (length(nonzero_signs) >= 2) sum(diff(nonzero_signs) != 0L) else 0L
    tibble(
      variable = sub("_d[12]$", "", col),
      derivative = if (grepl("_d2$", col)) "second derivative" else "first derivative",
      n_finite = n,
      mean = if (n > 0) mean(finite_x) else NA_real_,
      sd = if (n > 1) stats::sd(finite_x) else NA_real_,
      median = if (n > 0) stats::median(finite_x) else NA_real_,
      min = if (n > 0) min(finite_x) else NA_real_,
      max = if (n > 0) max(finite_x) else NA_real_,
      increasing_pct = if (n > 0) 100 * mean(finite_x > eps) else NA_real_,
      decreasing_pct = if (n > 0) 100 * mean(finite_x < -eps) else NA_real_,
      near_zero_pct = 100 * near_zero_fraction,
      sign_changes = sign_changes,
      interpretation = dplyr::case_when(
        n == 0 ~ "No finite derivative values.",
        is.finite(near_zero_fraction) && near_zero_fraction > 0.5 ~ "Many near-zero derivative values; epsilon choice may strongly affect class balance.",
        sign_changes >= max(3, floor(n / 10)) ~ "Frequent sign changes; inspect smoothing and outliers.",
        TRUE ~ "Derivative summary looks usable for initial screening."
      )
    )
  })
}

variable_test_table <- function(df, time_col, vars, max_lag = 10) {
  vars <- eda_vars_available(df, time_col, vars)
  if (length(vars) == 0) {
    return(tibble())
  }
  df <- df[order(df[[time_col]]), , drop = FALSE]
  purrr::map_dfr(vars, function(v) {
    x <- df[[v]]
    x_complete <- x[is.finite(x)]
    n <- length(x_complete)
    shapiro_p <- NA_real_
    if (n >= 3 && n <= 5000 && stats::sd(x_complete) > 0) {
      shapiro_p <- safe_test(stats::shapiro.test(x_complete)$p.value) %||% NA_real_
    }
    lb_p <- NA_real_
    lb_lag <- NA_integer_
    if (n >= 5 && stats::sd(x_complete) > 0) {
      lb_lag <- max(1L, min(as.integer(max_lag), floor(n / 4)))
      lb_p <- safe_test(stats::Box.test(x_complete, lag = lb_lag, type = "Ljung-Box")$p.value) %||% NA_real_
    }
    adf_p <- NA_real_
    kpss_p <- NA_real_
    stationarity_note <- "ADF/KPSS unavailable unless package 'tseries' is installed."
    has_tseries <- suppressWarnings(suppressMessages(requireNamespace("tseries", quietly = TRUE)))
    if (has_tseries && n >= 8 && stats::sd(x_complete) > 0) {
      adf_p <- safe_test(tseries::adf.test(x_complete)$p.value) %||% NA_real_
      kpss_p <- safe_test(tseries::kpss.test(x_complete, null = "Level")$p.value) %||% NA_real_
      stationarity_note <- "ADF null: unit root; KPSS null: level stationarity."
    }
    tibble(
      variable = v,
      n_complete = n,
      missing_pct = round(100 * mean(is.na(x)), 2),
      mean = mean(x_complete, na.rm = TRUE),
      sd = stats::sd(x_complete, na.rm = TRUE),
      mad = stats::mad(x_complete, na.rm = TRUE),
      shapiro_p = shapiro_p,
      ljung_box_lag = lb_lag,
      ljung_box_p = lb_p,
      adf_p = adf_p,
      kpss_p = kpss_p,
      note = stationarity_note
    )
  })
}

correlation_test_table <- function(df, time_col, vars, max_vars = 8) {
  vars <- eda_vars_available(df, time_col, vars, max_vars = max_vars)
  if (length(vars) < 2) {
    return(tibble())
  }
  pairs <- t(utils::combn(vars, 2))
  purrr::map_dfr(seq_len(nrow(pairs)), function(i) {
    a <- pairs[i, 1]
    b <- pairs[i, 2]
    x <- df[[a]]
    y <- df[[b]]
    ok <- stats::complete.cases(x, y)
    if (sum(ok) < 4 || stats::sd(x[ok]) <= 0 || stats::sd(y[ok]) <= 0) {
      return(tibble(variable_x = a, variable_y = b, n = sum(ok), pearson_r = NA_real_, pearson_p = NA_real_, spearman_r = NA_real_, spearman_p = NA_real_))
    }
    pearson <- safe_test(stats::cor.test(x[ok], y[ok], method = "pearson"))
    spearman <- safe_test(stats::cor.test(x[ok], y[ok], method = "spearman", exact = FALSE))
    tibble(
      variable_x = a,
      variable_y = b,
      n = sum(ok),
      pearson_r = if (!is.null(pearson)) unname(pearson$estimate) else NA_real_,
      pearson_p = if (!is.null(pearson)) pearson$p.value else NA_real_,
      spearman_r = if (!is.null(spearman)) unname(spearman$estimate) else NA_real_,
      spearman_p = if (!is.null(spearman)) spearman$p.value else NA_real_
    )
  }) %>%
    arrange(.data$spearman_p, .data$pearson_p)
}

make_lagged_design <- function(df, target, source, lag_order) {
  n <- nrow(df)
  if (n <= lag_order + 3) {
    return(tibble())
  }
  out <- tibble(y = df[[target]][(lag_order + 1):n])
  for (lag_k in seq_len(lag_order)) {
    idx <- (lag_order + 1 - lag_k):(n - lag_k)
    out[[paste0("target_lag", lag_k)]] <- df[[target]][idx]
    out[[paste0("source_lag", lag_k)]] <- df[[source]][idx]
  }
  out %>% filter(stats::complete.cases(.))
}

granger_screen_table <- function(df, time_col, vars, lag_order = 2, max_vars = 6) {
  vars <- eda_vars_available(df, time_col, vars, max_vars = max_vars)
  if (length(vars) < 2) {
    return(tibble())
  }
  lag_order <- max(1L, as.integer(lag_order %||% 1L))
  df <- df[order(df[[time_col]]), , drop = FALSE]
  pairs <- expand.grid(source = vars, target = vars, stringsAsFactors = FALSE) %>%
    filter(.data$source != .data$target)
  purrr::map_dfr(seq_len(nrow(pairs)), function(i) {
    source <- pairs$source[[i]]
    target <- pairs$target[[i]]
    dat <- make_lagged_design(df, target, source, lag_order)
    if (nrow(dat) <= (2 * lag_order + 2)) {
      return(tibble(source = source, target = target, lag_order = lag_order, n = nrow(dat), f_statistic = NA_real_, p_value = NA_real_, note = "Too few complete lagged rows."))
    }
    target_terms <- paste0("target_lag", seq_len(lag_order))
    source_terms <- paste0("source_lag", seq_len(lag_order))
    restricted <- stats::lm(stats::as.formula(paste("y ~", paste(target_terms, collapse = " + "))), data = dat)
    full <- stats::lm(stats::as.formula(paste("y ~", paste(c(target_terms, source_terms), collapse = " + "))), data = dat)
    rss_restricted <- sum(stats::resid(restricted)^2)
    rss_full <- sum(stats::resid(full)^2)
    q <- length(source_terms)
    df2 <- stats::df.residual(full)
    f_stat <- if (df2 > 0 && rss_full > 0) ((rss_restricted - rss_full) / q) / (rss_full / df2) else NA_real_
    p_value <- if (is.finite(f_stat)) stats::pf(f_stat, q, df2, lower.tail = FALSE) else NA_real_
    tibble(
      source = source,
      target = target,
      lag_order = lag_order,
      n = nrow(dat),
      f_statistic = f_stat,
      p_value = p_value,
      note = "Exploratory linear Granger screen; not proof of causality."
    )
  }) %>%
    arrange(.data$p_value)
}

plot_derivative_orders <- function(df_derivs, time_col, vars) {
  dvars <- c(vars, paste0(vars, "_d1"), paste0(vars, "_d2"))
  dvars <- intersect(dvars, names(df_derivs))
  req(length(dvars) > 0)
  df_derivs %>%
    select(all_of(c(time_col, dvars))) %>%
    pivot_longer(cols = all_of(dvars), names_to = "series", values_to = "value") %>%
    mutate(
      order = case_when(
        grepl("_d2$", series) ~ "Second derivative",
        grepl("_d1$", series) ~ "First derivative",
        TRUE ~ "State"
      ),
      variable = sub("_d[12]$", "", series)
    ) %>%
    ggplot(aes(.data[[time_col]], value, color = variable)) +
    geom_line(linewidth = 0.75, alpha = 0.9, na.rm = TRUE) +
    facet_wrap(~ order, scales = "free_y", ncol = 1) +
    labs(x = time_col, y = NULL, color = NULL) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

plot_class_balance <- function(class_tbl) {
  req(nrow(class_tbl) > 0)
  class_tbl %>%
    select(target, class_0, class_1, blank) %>%
    pivot_longer(cols = -target, names_to = "class", values_to = "count") %>%
    mutate(class = dplyr::recode(class, class_0 = "decreasing", class_1 = "increasing", blank = "blank")) %>%
    ggplot(aes(target, count, fill = class)) +
    geom_col(position = "stack") +
    scale_fill_manual(
      values = c(decreasing = CIRN_INHIBITION_COLOR, increasing = CIRN_ACTIVATION_COLOR, blank = "#B8C4D0"),
      breaks = c("increasing", "decreasing", "blank")
    ) +
    labs(x = "Target", y = "Count", fill = NULL, subtitle = "Green = increasing class; red = decreasing class; grey = blank near-zero derivative region.") +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

plot_hdi_coefficients <- function(coef_tbl, retained_only = FALSE, max_terms = 80, title = "Coefficient HDI Plot") {
  tbl <- as_result_tbl(coef_tbl)
  if (nrow(tbl) == 0 || !all(c("term", "target", "omega", "hdi_lower95", "hdi_upper95") %in% names(tbl))) {
    return(plot_empty_message("No coefficient HDI summaries are available for this selection."))
  }
  if (retained_only && "retained" %in% names(tbl)) {
    tbl <- filter(tbl, .data$retained)
  }
  if (nrow(tbl) == 0) {
    return(plot_empty_message("No coefficients match the current retained-only filter."))
  }
  tbl <- tbl %>%
    mutate(
      label = paste(format_node(.data$term), "->", format_node(.data$target)),
      color_group = case_when(
        .data$hdi_lower95 > 0 ~ "activation",
        .data$hdi_upper95 < 0 ~ "inhibition",
        TRUE ~ "uncertain"
      )
    ) %>%
    arrange(desc(abs(.data$omega))) %>%
    slice_head(n = max_terms)

  ggplot(tbl, aes(x = omega, y = reorder(label, omega), color = color_group)) +
    geom_vline(xintercept = 0, linetype = 2, color = "#6B7280") +
    geom_errorbar(aes(xmin = hdi_lower95, xmax = hdi_upper95), orientation = "y", width = 0.15, linewidth = 0.75) +
    geom_point(size = 2.2) +
    scale_color_manual(values = c(activation = CIRN_ACTIVATION_COLOR, inhibition = CIRN_INHIBITION_COLOR, uncertain = CIRN_UNCERTAIN_COLOR)) +
    labs(title = title, x = "Posterior mean coefficient with 95% HDI", y = NULL, color = NULL) +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(
      plot.title = element_text(face = "bold", size = 15, color = "#102033"),
      legend.position = "bottom"
    )
}

plot_jitter_magnitude <- function(diag_tbl) {
  req(!is.null(diag_tbl), nrow(diag_tbl) > 0)
  tbl <- diag_tbl %>%
    mutate(
      jitter_used = coalesce(as.logical(.data$jitter_used), FALSE),
      label = paste(.data$target, .data$predictor_set, sep = " / ")
    )

  if (!any(tbl$jitter_used, na.rm = TRUE)) {
    return(
      ggplot() +
        annotate("text", x = 0, y = 0, label = "No adaptive jitter was used in the fitted models.") +
        theme_void()
    )
  }

  tbl %>%
    filter(.data$jitter_used) %>%
    ggplot(aes(reorder(label, jitter_scale_factor), jitter_scale_factor, fill = target)) +
    geom_col() +
    coord_flip() +
    scale_y_log10() +
    labs(x = NULL, y = "Jitter scale factor (log scale)", fill = "Target") +
    theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
    theme(legend.position = "bottom")
}

truth_for_system <- function(system) {
  mat <- NULL
  if (system == "predator_prey") {
    vars <- c("x1", "x2")
    mat <- matrix(0, 2, 2, dimnames = list(vars, vars))
    mat["x1", "x1"] <- 1
    mat["x2", "x1"] <- -1
    mat["x1", "x2"] <- 1
    mat["x2", "x2"] <- -1
  }
  if (system == "sir") {
    vars <- c("S", "I", "R")
    mat <- matrix(0, 3, 3, dimnames = list(vars, vars))
    mat["I", "S"] <- -1
    mat["S", "I"] <- 1
    mat["I", "R"] <- 1
  }
  if (system == "competition") {
    vars <- c("x1", "x2")
    mat <- matrix(-1, 2, 2, dimnames = list(vars, vars))
  }
  if (system == "mutualism") {
    vars <- c("x1", "x2")
    mat <- matrix(0, 2, 2, dimnames = list(vars, vars))
    mat["x1", "x1"] <- -1
    mat["x2", "x2"] <- -1
    mat["x1", "x2"] <- 1
    mat["x2", "x1"] <- 1
  }
  if (system == "oscillator") {
    vars <- c("x1", "x2")
    mat <- matrix(0, 2, 2, dimnames = list(vars, vars))
    mat["x2", "x1"] <- 1
    mat["x1", "x2"] <- -1
  }
  mat
}

parse_key_values <- function(text) {
  lines <- trimws(unlist(strsplit(text %||% "", "\n|,")))
  lines <- lines[nzchar(lines)]
  out <- list()
  for (line in lines) {
    pieces <- strsplit(line, "=", fixed = TRUE)[[1]]
    if (length(pieces) != 2) {
      next
    }
    key <- trimws(pieces[[1]])
    val <- suppressWarnings(as.numeric(trimws(pieces[[2]])))
    if (nzchar(key) && is.finite(val)) {
      out[[key]] <- val
    }
  }
  out
}

format_simulation_value <- function(value, digits = 5) {
  value <- suppressWarnings(as.numeric(value)[1])
  if (!is.finite(value)) {
    return("not set")
  }
  format(signif(value, digits), scientific = FALSE, trim = TRUE)
}

simulation_equation_row <- function(lhs, rhs) {
  tags$div(
    class = "simulation-equation-row",
    tags$span(class = "simulation-equation-lhs", lhs),
    tags$span(class = "simulation-equation-equals", "="),
    tags$span(class = "simulation-equation-rhs", rhs)
  )
}

simulation_value_grid <- function(values) {
  if (length(values) == 0) {
    return(tags$p(class = "simulation-equation-empty", "No numeric values have been defined."))
  }
  tags$div(
    class = "simulation-value-grid",
    lapply(names(values), function(name) {
      tags$div(
        class = "simulation-value-chip",
        tags$span(class = "simulation-value-name", name),
        tags$strong(class = "simulation-value-number", format_simulation_value(values[[name]]))
      )
    })
  )
}

simulation_definition_list <- function(items) {
  tags$dl(
    class = "simulation-definition-list",
    lapply(items, function(item) {
      tagList(
        tags$dt(item[[1]]),
        tags$dd(item[[2]])
      )
    })
  )
}

simulate_builtin <- function(input) {
  system <- input$sim_system
  times <- seq(input$sim_t_start, input$sim_t_end, by = input$sim_dt)

  if (system == "predator_prey") {
    state <- c(x1 = input$pp_x1, x2 = input$pp_x2)
    parms <- c(a = input$pp_a, b = input$pp_b, c = input$pp_c, d = input$pp_d)
    derivs <- function(t, state, parms) {
      with(as.list(c(state, parms)), list(c(a * x1 - b * x1 * x2, c * x1 * x2 - d * x2)))
    }
  } else if (system == "sir") {
    state <- c(S = input$sir_S, I = input$sir_I, R = input$sir_R)
    parms <- c(beta = input$sir_beta, gamma = input$sir_gamma, N = input$sir_N)
    derivs <- function(t, state, parms) {
      with(as.list(c(state, parms)), list(c(-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I)))
    }
  } else if (system == "competition") {
    state <- c(x1 = input$comp_x1, x2 = input$comp_x2)
    parms <- c(r1 = input$comp_r1, r2 = input$comp_r2, K1 = input$comp_K1, K2 = input$comp_K2, a12 = input$comp_a12, a21 = input$comp_a21)
    derivs <- function(t, state, parms) {
      with(as.list(c(state, parms)), list(c(
        r1 * x1 * (1 - (x1 + a12 * x2) / K1),
        r2 * x2 * (1 - (x2 + a21 * x1) / K2)
      )))
    }
  } else if (system == "mutualism") {
    state <- c(x1 = input$mut_x1, x2 = input$mut_x2)
    parms <- c(r1 = input$mut_r1, r2 = input$mut_r2, K1 = input$mut_K1, K2 = input$mut_K2, m12 = input$mut_m12, m21 = input$mut_m21)
    derivs <- function(t, state, parms) {
      with(as.list(c(state, parms)), list(c(
        r1 * x1 * (1 - x1 / K1) + m12 * x1 * x2 / (1 + x2),
        r2 * x2 * (1 - x2 / K2) + m21 * x1 * x2 / (1 + x1)
      )))
    }
  } else if (system == "oscillator") {
    state <- c(x1 = input$osc_x1, x2 = input$osc_x2)
    parms <- c(alpha = input$osc_alpha, beta = input$osc_beta)
    derivs <- function(t, state, parms) {
      with(as.list(c(state, parms)), list(c(alpha * x2, -beta * x1)))
    }
  } else {
    return(simulate_custom(input))
  }

  out <- deSolve::ode(y = state, times = times, func = derivs, parms = parms)
  df <- as.data.frame(out)
  names(df)[1] <- "t"
  df
}

simulate_custom <- function(input) {
  vars <- trimws(unlist(strsplit(input$custom_vars, ",")))
  vars <- vars[nzchar(vars)]
  if (length(vars) == 0) {
    stop("Custom simulation needs at least one variable.")
  }

  init <- parse_key_values(input$custom_initials)
  params <- parse_key_values(input$custom_params)
  missing_init <- setdiff(vars, names(init))
  if (length(missing_init) > 0) {
    stop("Missing initial values for: ", paste(missing_init, collapse = ", "))
  }

  eq_lines <- trimws(unlist(strsplit(input$custom_equations, "\n")))
  eq_lines <- eq_lines[nzchar(eq_lines)]
  eq_map <- list()
  for (line in eq_lines) {
    pieces <- strsplit(line, "=", fixed = TRUE)[[1]]
    if (length(pieces) != 2) {
      stop("Custom equation must look like x1 = expression.")
    }
    lhs <- trimws(pieces[[1]])
    rhs <- trimws(pieces[[2]])
    if (!lhs %in% vars) {
      stop("Equation left side is not in custom variables: ", lhs)
    }
    if (grepl("::|system|file|source|library|require|assign|get\\(|function|<-", rhs)) {
      stop("Custom equations allow only arithmetic/math expressions using variables, parameters, and t.")
    }
    eq_map[[lhs]] <- rhs
  }
  if (!all(vars %in% names(eq_map))) {
    stop("Provide one custom equation for every variable.")
  }

  safe_math <- list(
    sin = sin, cos = cos, tan = tan, exp = exp, log = log, sqrt = sqrt,
    abs = abs, min = min, max = max, pmin = pmin, pmax = pmax
  )

  derivs <- function(t, state, parms) {
    values <- c(as.list(state), as.list(parms), safe_math, list(t = t))
    env <- list2env(values, parent = baseenv())
    dx <- vapply(vars, function(v) {
      eval(parse(text = eq_map[[v]]), envir = env)
    }, numeric(1))
    list(dx)
  }

  times <- seq(input$sim_t_start, input$sim_t_end, by = input$sim_dt)
  out <- deSolve::ode(
    y = unlist(init[vars]),
    times = times,
    func = derivs,
    parms = unlist(params)
  )
  df <- as.data.frame(out)
  names(df)[1] <- "t"
  df
}

apply_simulation_artifacts <- function(df, input) {
  set.seed(input$sim_seed)
  vars <- setdiff(names(df), "t")

  if (input$sim_irregular > 0 && nrow(df) > 4) {
    dt <- stats::median(diff(df$t), na.rm = TRUE)
    jitter <- stats::rnorm(nrow(df), 0, input$sim_irregular * dt)
    jitter[1] <- 0
    jitter[nrow(df)] <- 0
    df$t <- sort(df$t + jitter)
  }

  if (input$sim_noise > 0) {
    for (v in vars) {
      s <- stats::sd(df[[v]], na.rm = TRUE)
      if (is.finite(s) && s > 0) {
        df[[v]] <- df[[v]] + stats::rnorm(nrow(df), 0, input$sim_noise * s)
      }
    }
  }

  if (input$sim_outlier_fraction > 0) {
    n_cells <- nrow(df) * length(vars)
    n_out <- floor(n_cells * input$sim_outlier_fraction)
    if (n_out > 0) {
      cell_ids <- sample(seq_len(n_cells), n_out)
      for (id in cell_ids) {
        row <- ((id - 1) %% nrow(df)) + 1
        var <- vars[((id - 1) %/% nrow(df)) + 1]
        s <- stats::sd(df[[var]], na.rm = TRUE)
        if (is.finite(s) && s > 0) {
          df[[var]][row] <- df[[var]][row] + stats::rnorm(1, 0, input$sim_outlier_size * s)
        }
      }
    }
  }

  if (input$sim_missing > 0) {
    n_cells <- nrow(df) * length(vars)
    n_missing <- floor(n_cells * input$sim_missing)
    if (n_missing > 0) {
      cell_ids <- sample(seq_len(n_cells), n_missing)
      for (id in cell_ids) {
        row <- ((id - 1) %% nrow(df)) + 1
        var <- vars[((id - 1) %/% nrow(df)) + 1]
        df[[var]][row] <- NA_real_
      }
    }
  }

  df
}

build_rf_support <- function(df_derivs, targets, predictors, time_col, eps, trees, boot, threshold, seed) {
  terms <- unique(c(predictors, paste0(predictors, "_d1"), paste0(predictors, "_d2")))
  terms <- intersect(terms, names(df_derivs))
  rows <- list()
  set.seed(seed)

  for (target in targets) {
    d1 <- paste0(target, "_d1")
    if (!d1 %in% names(df_derivs)) {
      next
    }
    target_terms <- setdiff(terms, d1)

    for (term in target_terms) {
      dat <- tibble(
        Class = case_when(
          df_derivs[[d1]] > eps ~ "increasing",
          df_derivs[[d1]] < -eps ~ "decreasing",
          TRUE ~ NA_character_
        ),
        x = dplyr::lag(df_derivs[[term]])
      ) %>%
        filter(!is.na(.data$Class), is.finite(.data$x))

      if (nrow(dat) < 10 || length(unique(dat$Class)) < 2) {
        rows[[length(rows) + 1L]] <- tibble(
          target = paste0("d", target),
          term = term,
          n = nrow(dat),
          boot = boot,
          support_rate = NA_real_,
          mean_importance = NA_real_,
          supported = FALSE,
          status = "insufficient_two_class_data"
        )
        next
      }

      importances <- numeric(boot)
      for (b in seq_len(boot)) {
        idx <- sample(seq_len(nrow(dat)), nrow(dat), replace = TRUE)
        fit <- tryCatch(
          randomForest::randomForest(
            as.factor(Class) ~ x,
            data = dat[idx, , drop = FALSE],
            ntree = trees,
            importance = TRUE
          ),
          error = function(e) e
        )
        if (inherits(fit, "error")) {
          importances[b] <- NA_real_
        } else {
          imp <- randomForest::importance(fit, type = 1)
          importances[b] <- as.numeric(imp[1, 1])
        }
      }

      support_rate <- mean(importances > 0, na.rm = TRUE)
      rows[[length(rows) + 1L]] <- tibble(
        target = paste0("d", target),
        term = term,
        n = nrow(dat),
        boot = boot,
        support_rate = support_rate,
        mean_importance = mean(importances, na.rm = TRUE),
        supported = is.finite(support_rate) && support_rate >= threshold,
        status = "completed"
      )
    }
  }

  bind_rows(rows)
}

build_latent_z <- function(df_derivs, targets, predictors, time_col, eps, min_gain) {
  terms <- unique(c(predictors, paste0(predictors, "_d1"), paste0(predictors, "_d2")))
  terms <- intersect(terms, names(df_derivs))

  purrr::map_dfr(targets, function(target) {
    d1 <- paste0(target, "_d1")
    target_terms <- setdiff(terms, d1)
    if (!d1 %in% names(df_derivs) || length(target_terms) < 2) {
      return(tibble(target = paste0("d", target), status = "insufficient_predictors"))
    }

    dat <- df_derivs %>%
      transmute(
        Class = case_when(
          .data[[d1]] > eps ~ 1,
          .data[[d1]] < -eps ~ 0,
          TRUE ~ NA_real_
        ),
        across(all_of(target_terms), ~ dplyr::lag(.x))
      ) %>%
      filter(!is.na(.data$Class), stats::complete.cases(.))

    if (nrow(dat) < 12 || length(unique(dat$Class)) < 2) {
      return(tibble(target = paste0("d", target), status = "insufficient_two_class_data"))
    }

    x <- scale(as.matrix(dat[target_terms]))
    pca <- stats::prcomp(x, center = FALSE, scale. = FALSE)
    z <- as.numeric(pca$x[, 1])
    dat_base <- data.frame(Class = dat$Class, x)
    dat_z <- cbind(dat_base, Z = z)

    base_fit <- suppressWarnings(tryCatch(stats::glm(Class ~ ., data = dat_base, family = binomial()), error = function(e) e))
    z_fit <- suppressWarnings(tryCatch(stats::glm(Class ~ ., data = dat_z, family = binomial()), error = function(e) e))

    if (inherits(base_fit, "error") || inherits(z_fit, "error")) {
      return(tibble(target = paste0("d", target), status = "glm_failed"))
    }

    pred_base <- as.numeric(stats::predict(base_fit, type = "response") >= 0.5)
    pred_z <- as.numeric(stats::predict(z_fit, type = "response") >= 0.5)
    acc_base <- mean(pred_base == dat$Class)
    acc_z <- mean(pred_z == dat$Class)
    var_exp <- (pca$sdev[1]^2) / sum(pca$sdev^2)

    tibble(
      target = paste0("d", target),
      n = nrow(dat),
      predictors = length(target_terms),
      variance_explained_pc1 = var_exp,
      baseline_accuracy = acc_base,
      latent_z_accuracy = acc_z,
      accuracy_gain = acc_z - acc_base,
      useful = (acc_z - acc_base) >= min_gain,
      status = "completed"
    )
  })
}

make_settings_list <- function(input, eps, jitter_grid) {
  list(
    generated_at = as.character(Sys.time()),
    data_mode = input$data_mode,
    time_col = input$time_col,
    targets = input$targets,
    predictors = input$predictors,
    normalization = input$normalization,
    points_per_interval = input$points_per_interval,
    spar = if (isTRUE(input$auto_spar)) NULL else input$spar,
    response_eps = eps,
    representation_mode = input$representation_mode,
    run_pairwise = input$run_pairwise,
    pairwise_representation_mode = input$pairwise_representation_mode,
    sensitivity_inference_scope = input$sensitivity_scope,
    model_iter = input$model_iter,
    model_warmup = input$model_warmup,
    model_chains = input$model_chains,
    model_cores = input$model_cores,
    prior_mean = input$prior_mean,
    prior_sd = input$prior_sd,
    adapt_delta = input$adapt_delta,
    compute_loo = input$compute_loo,
    adaptive_jitter = input$adaptive_jitter,
    jitter_predictors = input$jitter_predictors,
    jitter_min_class_count = input$jitter_min_class_count,
    jitter_scale_grid = jitter_grid,
    jitter_scale_basis = input$jitter_scale_basis
  )
}

feedback_contact_email <- Sys.getenv(
  "CIRN_FEEDBACK_EMAIL",
  unset = "gmentero@up.edu.ph,jfrabajante@up.edu.ph"
)
feedback_default_dir <- Sys.getenv("CIRN_FEEDBACK_DIR", unset = file.path(app_dir, "feedback"))

feedback_areas <- c(
  "Start / onboarding",
  "Data upload",
  "Simulation Lab",
  "Equation builder",
  "EDA",
  "Preprocess",
  "Run CIRN Algorithm",
  "Results",
  "CIRN Figures",
  "Diagnostics",
  "Sensitivity",
  "Benchmark",
  "Export",
  "User Guide / documentation",
  "General usability"
)

feedback_value <- function(x, collapse = ", ") {
  if (is.null(x) || length(x) == 0) {
    return("")
  }
  x <- as.character(x)
  x[is.na(x)] <- ""
  paste(x, collapse = collapse)
}

feedback_text_block <- function(label, value) {
  value <- trimws(feedback_value(value, collapse = "\n"))
  if (!nzchar(value)) {
    value <- "Not provided"
  }
  paste0(label, ":\n", value)
}

format_feedback_report <- function(record) {
  context <- record$context %||% list()
  context_lines <- vapply(
    names(context),
    function(nm) paste0(nm, ": ", feedback_value(context[[nm]])),
    character(1)
  )

  paste(
    c(
      "CIRN Studio Feedback Report",
      paste0("Generated: ", record$timestamp %||% as.character(Sys.time())),
      "",
      paste0("Type: ", feedback_value(record$type)),
      paste0("Severity: ", feedback_value(record$severity)),
      paste0("App area: ", feedback_value(record$area)),
      paste0("Follow-up allowed: ", if (isTRUE(record$follow_up)) "yes" else "no"),
      "",
      feedback_text_block("Summary", record$summary),
      "",
      feedback_text_block("What happened / suggestion", record$details),
      "",
      feedback_text_block("Steps to reproduce", record$steps),
      "",
      feedback_text_block("Expected behavior", record$expected),
      "",
      feedback_text_block("Actual behavior", record$actual),
      "",
      "Reporter",
      paste0("Name: ", feedback_value(record$name)),
      paste0("Email: ", feedback_value(record$email)),
      paste0("Affiliation: ", feedback_value(record$affiliation)),
      "",
      "Automatic Context",
      if (length(context_lines) > 0) context_lines else "No context attached."
    ),
    collapse = "\n"
  )
}

feedback_mailto_uri <- function(record, to = feedback_contact_email) {
  recipients <- trimws(unlist(strsplit(to, "[,;]")))
  recipients <- recipients[nzchar(recipients)]
  recipient_text <- paste(recipients, collapse = ",")

  subject_summary <- trimws(feedback_value(record$summary))
  if (!nzchar(subject_summary)) {
    subject_summary <- "CIRN Studio feedback"
  }
  if (nchar(subject_summary) > 80) {
    subject_summary <- paste0(substr(subject_summary, 1, 77), "...")
  }

  body <- format_feedback_report(record)
  if (nchar(body) > 5500) {
    body <- paste0(substr(body, 1, 5500), "\n\n[Message truncated for email-link length. Please attach the downloaded feedback report if needed.]")
  }

  paste0(
    "mailto:", recipient_text,
    "?subject=", utils::URLencode(paste("CIRN Studio feedback:", subject_summary), reserved = TRUE),
    "&body=", utils::URLencode(body, reserved = TRUE)
  )
}

feedback_record_row <- function(record) {
  context <- record$context %||% list()
  tibble(
    timestamp = record$timestamp %||% as.character(Sys.time()),
    type = feedback_value(record$type),
    severity = feedback_value(record$severity),
    area = feedback_value(record$area),
    summary = feedback_value(record$summary),
    details = feedback_value(record$details, collapse = "\n"),
    steps = feedback_value(record$steps, collapse = "\n"),
    expected = feedback_value(record$expected, collapse = "\n"),
    actual = feedback_value(record$actual, collapse = "\n"),
    reporter_name = feedback_value(record$name),
    reporter_email = feedback_value(record$email),
    reporter_affiliation = feedback_value(record$affiliation),
    follow_up_allowed = isTRUE(record$follow_up),
    context_json = jsonlite::toJSON(context, auto_unbox = TRUE, null = "null")
  )
}

ui <- tagList(
  tags$head(
    tags$link(rel = "stylesheet", type = "text/css", href = "cirn-studio.css?v=20260724-diagnostics-scroll-viewer-13"),
    tags$script(HTML("
      (function() {
        function decoratePrimaryNavigation() {
          var nav = document.querySelector('.navbar-nav');
          if (!nav) return;
          nav.setAttribute('aria-label', 'Primary CIRN Studio navigation');

          var items = Array.prototype.slice.call(nav.children || []);
          function itemByText(label) {
            return items.find(function(item) {
              var link = item.querySelector('a, button');
              return link && link.textContent.trim() === label;
            });
          }

          var what = itemByText('What Is CIRN?');
          var guide = itemByText('User Guide');
          if (what && guide && what.nextElementSibling !== guide) {
            what.after(guide);
          }

          ['Data Source', 'Export'].forEach(function(label) {
            var item = itemByText(label);
            if (item) item.classList.add('nav-group-start');
          });

          var run = itemByText('Run CIRN Algorithm') || itemByText('Run CIRN');
          if (run) {
            var runLink = run.querySelector('a, button');
            if (runLink) {
              runLink.setAttribute('title', 'Run CIRN Algorithm');
              runLink.setAttribute('aria-label', 'Run CIRN Algorithm');

              var walker = document.createTreeWalker(
                runLink,
                NodeFilter.SHOW_TEXT
              );
              var textNode;
              while ((textNode = walker.nextNode())) {
                if (textNode.nodeValue.trim() === 'Run CIRN Algorithm') {
                  textNode.nodeValue = 'Run CIRN';
                  break;
                }
              }
            }
          }
        }

        document.addEventListener('DOMContentLoaded', decoratePrimaryNavigation);
        document.addEventListener('shiny:connected', decoratePrimaryNavigation);
        setTimeout(decoratePrimaryNavigation, 250);
      })();
    "))
  ),
  navbarPage(
    title = tags$span("CIRN Studio"),
    id = "main_nav",
    theme = bslib::bs_theme(
      version = 5,
      primary = "#0F766E",
      secondary = "#1D4E89",
      success = "#168A93",
      warning = "#B7791F",
      danger = "#B2182B",
      base_font = bslib::font_google("Inter")
    ),

    tabPanel(
      "Start",
      fluidPage(
        tags$div(
          class = "hero-band hero-band-compact",
          tags$div(
            tags$h1(
              class = "hero-title",
              tags$span(class = "hero-title-main", "Classification-Based Inference of Regulatory Networks"),
              actionLink(
                "hero_cirn_link",
                label = "CIRN",
                class = "hero-title-acronym hero-title-acronym-link",
                title = "Open the What Is CIRN? tab"
              )
            ),
            tags$div(
              class = "hero-authors",
              "by Giovannie M. Entero, Jomar F. Rabajante, Neil Jerome A. Egarguin, ",
              "Mark Jayson V. Cortez, Maica Krizna A. Gavina, & Patricia Ann J. Sanchez"
            ),
            tags$div(
              class = "hero-affiliation",
              "Institute of Mathematical Sciences | School of Environmental Science and Management | Graduate School | University of the Philippines Los Baños"
            ),
            tags$p(
              class = "hero-method",
              "CIRN reconstructs exploratory signed regulatory networks from multivariate time-series data by modeling the direction of each target's local change. ",
              "For each target response, the algorithm encodes the first-derivative sign as increase versus decrease and fits Bayesian logistic regression using temporally lagged state, first-derivative, and second-derivative features. ",
              "Edges are retained when the coefficient 95% HDI excludes zero, while diagnostics, Random Forest support, latent-Z screening, sensitivity checks, and benchmarking qualify how strongly each signed edge should be interpreted."
            ),
            tags$div(
              class = "hero-question",
              "Main inferential question: Which earlier state or derivative features provide credible evidence that a target is more likely to increase or decrease?"
            ),
            tags$div(
              class = "hero-badges",
              tags$span(class = "hero-badge", title = "CIRN reports cautious directed activation/inhibition edge hypotheses, not final causal proof.", "Signed regulatory hypotheses"),
              tags$span(class = "hero-badge", title = "Primary model for classifying whether each target's first derivative is positive or negative.", "Bayesian logistic regression"),
              tags$span(class = "hero-badge", title = "Retains edges whose 95% posterior highest density interval excludes zero.", "95% HDI edge retention"),
              tags$span(class = "hero-badge", title = "Candidate regulators may enter as state values, first derivatives, or second derivatives.", "State, d, and d2 predictors"),
              tags$span(class = "hero-badge", title = "Random Forest support is a robustness diagnostic, not the sign-defining CIRN model.", "RF support diagnostics"),
              tags$span(class = "hero-badge", title = "Sensitivity checks perturb settings/data; benchmarking compares against known ground truth when available.", "Sensitivity and benchmarking")
            ),
            tags$div(
              class = "hero-legend",
              tags$span(class = "legend-key activation-key"),
              tags$span("activation (+)"),
              tags$span(class = "legend-key inhibition-key"),
              tags$span("inhibition (-)"),
              tags$span(class = "legend-separator"),
              tags$span("Retained edges are exploratory evidence for signed regulation.")
            )
          )
        ),
        tags$div(
          class = "hero-panel-divider",
          role = "presentation",
          `aria-hidden` = "true"
        ),
        tags$div(
          class = "row start-dashboard-row",
          tags$div(
            class = "col-sm-8 start-info-column",
            tags$div(
              class = "workflow-panel",
              tags$div(class = "start-section-label", "Guided analysis workflow"),
              tags$p(
                class = "start-panel-description",
                "Follow these ten stages from data selection through export. Optional stages are labeled, and each status badge updates as your analysis progresses."
              ),
              uiOutput("workflow_status")
            ),
            panel_box(
              "Current Analysis Status",
              tags$p(
                class = "start-panel-description",
                "A live summary of the active dataset, selected variables, derivative readiness, CIRN fitting state, and unique retained signed edges."
              ),
              div(class = "metric-grid",
                  metric_box("Data rows", "metric_rows"),
                  metric_box("State variables", "metric_vars"),
                  metric_box("Targets", "metric_targets"),
                  metric_box("Unique retained edges", "metric_edges")),
              uiOutput("status_pills"),
              uiOutput("edge_representation_status"),
              class = "studio-panel analysis-status-panel"
            ),
            panel_box(
              "Scientific Guardrail",
              div(
                class = "method-box",
                tags$p(
                  strong("CIRN output: "),
                  "a signed, directed, uncertainty-aware candidate regulatory network inferred from whether lagged state and derivative features help classify the direction of each target's first derivative."
                ),
                tags$p(
                  strong("Edge retention rule: "),
                  "an edge is retained only when the posterior 95% HDI for its Bayesian logistic coefficient excludes zero; green indicates positive association with the target increasing, red indicates negative association."
                ),
                tags$p(
                  strong("Interpretation limit: "),
                  "CIRN supports exploratory regulatory hypotheses, not definitive interventional causality, exact ODE/SDE equations, parameter values, or final mechanism without domain validation, diagnostics, and sensitivity checks."
                )
              ),
              class = "studio-panel scientific-guardrail-panel"
            ),
            panel_box(
              "Citation and Reproducibility",
              div(
                class = "repro-box",
                tags$p(
                  strong("Report with every CIRN result: "),
                  "data source, selected targets and predictors, preprocessing settings, inference modes, Bayesian settings, retention rule, diagnostics, and sensitivity scope."
                ),
                tags$p(
                  strong("Method: "),
                  "Classification-Based Inference of Regulatory Networks (CIRN)."
                ),
                tags$p(
                  strong("CIRN authors: "),
                  "Giovannie M. Entero, Jomar F. Rabajante, Neil Jerome A. Egarguin, Mark Jayson V. Cortez, Maica Krizna A. Gavina, & Patricia Ann J. Sanchez."
                ),
                tags$p(
                  strong("Contact: "),
                  "gmentero@up.edu.ph; jfrabajante@up.edu.ph"
                )
              ),
              class = "studio-panel citation-repro-panel"
            )
          ),
          tags$div(
            class = "col-sm-4 start-setup-column",
            panel_box(
              "Are You a Beginner? Try This First",
              div(
                class = "beginner-start",
                tags$div(class = "beginner-kicker", "Automatic beginner setup"),
                tags$p(
                  strong("One click prepares the demonstration for you. "),
                  "The app loads the built-in Predator-Prey data and applies the Script-matched preset automatically."
                ),
                tags$p(
                  class = "subtle",
                  "No upload or manual setup is needed. When Example or Uploaded Data opens, continue by following Steps 1-10 in the Guided Analysis Workflow."
                ),
                actionButton(
                  "beginner_example",
                  button_label(icon("graduation-cap"), "Use Predator-Prey Example"),
                  class = "btn-primary studio-action-button beginner-start-button",
                  style = "color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;"
                ),
                tags$p(
                  class = "beginner-settings-note",
                  icon("sliders"),
                  " Want to try different settings? Load the example here first, then use Set Up Your Analysis below."
                )
              ),
              class = "studio-panel beginner-start-panel"
            ),
            panel_box(
              "Set Up Your Analysis",
              div(
                class = "start-analysis-intro",
                strong("Use your own data or create a new dataset. "),
                "Choose Upload, Simulator, or Equation Builder, then select an optional analysis preset. For the bundled Predator-Prey data, use the beginner route above."
              ),
              tags$div(
                style = "display:none;",
                radioButtons(
                  "data_mode",
                  "Choose data source",
                  choices = c("Upload file" = "upload", "Built-in example" = "example", "Simulation lab" = "simulation"),
                  selected = "example"
                )
              ),
              tags$div(class = "start-action-kicker", "Choose one data source"),
              div(
                class = "start-action-grid",
                div(
                  class = "start-action-card",
                  strong("Upload Data"),
                  span(class = "start-action-description", "Use your own CSV or Excel time-series table."),
                  actionButton("launch_upload", button_label(icon("upload"), "Start Upload"), class = "btn-primary studio-action-button", style = "color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;")
                ),
                div(
                  class = "start-action-card",
                  strong("Simulate System"),
                  span(class = "start-action-description", "Generate data from built-in dynamical systems."),
                  actionButton("launch_simulation", button_label(icon("play"), "Open Simulator"), class = "btn-primary studio-action-button", style = "color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;")
                ),
                div(
                  class = "start-action-card",
                  strong("Build From Equations"),
                  span(class = "start-action-description", "Enter variables, parameters, and derivative equations."),
                  actionButton("go_equation_builder", button_label(icon("code"), "Open Builder"), class = "btn-primary studio-action-button", style = "color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;")
                )
              ),
              tags$hr(),
              tags$div(
                class = "advanced-options-label",
                tags$span("Advanced Analysis Options"),
                tags$span(class = "optional-chip", "Optional")
              ),
              tags$p(
                class = "advanced-options-description",
                "Choose a preset to adjust preprocessing, Bayesian fitting, diagnostics, and sensitivity settings for your purpose. Keep Script-matched run for the recommended first analysis; a new preset affects results only after you run CIRN again."
              ),
              selectInput(
                "analysis_preset",
                label = NULL,
                choices = c(
                  "Script-matched run" = "script",
                  "Teaching demo" = "teaching",
                  "Fast exploration" = "fast",
                  "Careful analysis" = "careful",
                  "Publication quality" = "publication",
                  "Benchmark mode" = "benchmark"
                ),
                selected = "script"
              ),
              div(class = "preset-note", uiOutput("preset_note")),
              uiOutput("preset_applied_settings"),
              class = "studio-panel start-analysis-panel"
            )
          )
        )
      )
    ),

    tabPanel(
      "What Is CIRN?",
      fluidPage(
        div(
          class = "guide-hero",
          tags$h2("What Is Classification-Based Inference of Regulatory Networks?"),
          tags$p(
            "CIRN is an exploratory framework for reconstructing directed and signed regulatory hypotheses from multivariate time-series data. ",
            "Instead of trying to recover the full differential equation of a system, CIRN asks whether earlier state and derivative features help explain whether each target variable is locally increasing or decreasing."
          ),
          div(
            class = "guide-pill-row",
            span(class = "guide-pill activation", "Activation: omega > 0"),
            span(class = "guide-pill inhibition", "Inhibition: omega < 0"),
            span(class = "guide-pill neutral", "95% HDI retention rule"),
            span(class = "guide-pill neutral", "State, d, and d2 features"),
            span(class = "guide-pill neutral", "Exploratory"),
            span(class = "guide-pill neutral", "Hypothesis Generation")
          )
        ),
        fluidRow(
          column(
            6,
            panel_box(
              "CIRN In Plain Language",
              tags$p(
                "Suppose you observe several variables through time. CIRN treats each variable, one at a time, as a possible target response. ",
                "For that target, the app estimates whether the target is increasing or decreasing at each usable time point. ",
                "Then CIRN asks which earlier variables, earlier rates of change, or earlier curvature patterns provide credible evidence for that direction of change."
              ),
              tags$p(
                "The result is a candidate regulatory network. An edge does not mean the app has proven a biological, ecological, or physical mechanism. ",
                "It means the selected predictor had posterior-supported evidence for shifting the probability that the target was increasing under the chosen preprocessing, lag, model mode, and diagnostic context."
              )
            )
          ),
          column(
            6,
            panel_box(
              "Main Inferential Question",
              div(
                class = "method-box",
                tags$p(
                  strong("CIRN asks: "),
                  "Which earlier state or derivative features provide credible evidence that a target variable is more likely to be locally increasing or decreasing?"
                ),
                tags$p(
                  strong("It does not primarily ask: "),
                  "Can we forecast the exact future value of the target? The target being modeled is the direction of local change, not the full future trajectory."
                ),
                tags$p(
                  strong("Why this is useful: "),
                  "many regulatory questions are about whether a source variable supports or suppresses a target response, even when the exact governing equations are unknown."
                )
              )
            )
          )
        ),
        panel_box(
          "Where CIRN Sits Among Existing Methods",
          tags$p(
            "CIRN is best understood as a complementary, middle-ground method for regulatory network reconstruction. ",
            "It is more structured than simple association screening because it uses temporal ordering, derivative-based target responses, signed edges, and Bayesian uncertainty. ",
            "At the same time, it is less assumption-heavy than full mechanistic equation recovery because it does not require the analyst to know the governing ODE form, kinetic law, or exact interaction function in advance."
          ),
          fluidRow(
            column(
              3,
              div(
                class = "guide-card",
                tags$h4("Compared With Correlation"),
                tags$p(
                  "Correlation can show variables moving together, but it is usually static and unsigned in the regulatory sense. ",
                  "CIRN asks a more dynamical question: whether an earlier predictor representation supports target increase or decrease."
                ),
                tags$p(strong("CIRN adds: "), "direction, sign, lagging, and uncertainty.")
              )
            ),
            column(
              3,
              div(
                class = "guide-card",
                tags$h4("Compared With Granger-Type Screens"),
                tags$p(
                  "Granger-style analyses test whether past values improve prediction under a chosen time-series model. ",
                  "CIRN focuses on the sign of local target change and returns activation or inhibition hypotheses on the target-response scale."
                ),
                tags$p(strong("CIRN adds: "), "derivative-response encoding and representation-aware signed edges.")
              )
            ),
            column(
              3,
              div(
                class = "guide-card",
                tags$h4("Compared With CCM/PCM"),
                tags$p(
                  "State-space causality approaches can be powerful for nonlinear dynamical dependence, but interpretation may be less direct for signed regulatory edges. ",
                  "CIRN is designed to produce readable signed network hypotheses with coefficient uncertainty."
                ),
                tags$p(strong("CIRN adds: "), "posterior HDIs, omega signs, and coefficient-level diagnostics.")
              )
            ),
            column(
              3,
              div(
                class = "guide-card",
                tags$h4("Compared With ODE Discovery"),
                tags$p(
                  "ODE fitting or equation discovery can seek explicit governing equations, but often needs dense data, candidate functional forms, and identifiable parameters. ",
                  "CIRN instead asks for credible temporal-response evidence without claiming full equation recovery."
                ),
                tags$p(strong("CIRN adds: "), "accessible hypothesis generation when full mechanism is unknown.")
              )
            )
          )
        ),
        panel_box(
          "Why CIRN Can Be Especially Helpful",
          fluidRow(
            column(
              4,
              div(
                class = "guide-good",
                tags$h4("It Reports Signed Direction"),
                tags$p(
                  "Many exploratory methods stop at association or predictive relevance. CIRN reports whether a retained predictor supports a higher or lower probability of target increase, giving activation or inhibition on the target-response scale."
                )
              )
            ),
            column(
              4,
              div(
                class = "guide-good",
                tags$h4("It Uses Dynamical Features"),
                tags$p(
                  "CIRN does not only use state values. It can evaluate state, first-derivative, and second-derivative representations, allowing the user to detect level-dependent, velocity-dependent, and curvature-dependent temporal-response signals."
                )
              )
            ),
            column(
              4,
              div(
                class = "guide-good",
                tags$h4("It Quantifies Uncertainty"),
                tags$p(
                  "Edges are not retained from a single point estimate. CIRN uses posterior coefficient distributions and retains an edge only when the 95% HDI excludes zero, then asks the user to read that edge with diagnostics."
                )
              )
            )
          ),
          fluidRow(
            column(
              4,
              div(
                class = "guide-good",
                tags$h4("It Is Multi-Resolution"),
                tags$p(
                  "Pairwise, sublevel, all-predictors, and consistency views let users compare marginal, representation-specific, and joint conditional evidence rather than relying on one fragile network view."
                )
              )
            ),
            column(
              4,
              div(
                class = "guide-good",
                tags$h4("It Is Diagnostic-Rich"),
                tags$p(
                  "Class balance, derivative plots, epsilon checks, MCMC diagnostics, VIF, RF support, latent-Z screening, sensitivity, and benchmarking help users see when a result is strong, cautious, or not reportable."
                )
              )
            ),
            column(
              4,
              div(
                class = "guide-good",
                tags$h4("It Is System-Agnostic"),
                tags$p(
                  "CIRN can be used across biological, ecological, environmental, synthetic, and other dynamical systems as long as the data are time-ordered and the selected variables have interpretable temporal variation."
                )
              )
            )
          )
        ),
        panel_box(
          "How CIRN Turns Time-Series Data Into Edges",
          tags$ol(
            class = "guide-steps",
            tags$li(strong("Start with a time-series table. "), "The app expects a time column, usually named t, and one or more numeric system variables."),
            tags$li(strong("Select targets and allowed predictors. "), "A target is a variable whose local direction of change will be modeled. A predictor is a variable allowed to explain that target response."),
            tags$li(strong("Smooth and differentiate the trajectories. "), "CIRN estimates first derivatives and second derivatives so the model can use state, velocity-like, and curvature-like information."),
            tags$li(strong("Build the response from the target first derivative. "), "Positive derivative values become the increasing class; negative derivative values become the decreasing class; near-zero values inside the epsilon blank region are excluded from the binary response."),
            tags$li(strong("Lag the candidate predictors. "), "Predictor features are taken from earlier time points so the model is temporally ordered and avoids using future information to explain the present response."),
            tags$li(strong("Fit Bayesian logistic regression. "), "For each target and selected inference mode, CIRN estimates posterior distributions for the predictor coefficients."),
            tags$li(strong("Retain uncertainty-supported coefficients. "), "A coefficient becomes a CIRN edge only when its 95% highest-density interval excludes zero."),
            tags$li(strong("Assign regulatory sign. "), "The posterior mean coefficient, omega, determines sign: positive omega is activation on the target-response scale; negative omega is inhibition on the target-response scale."),
            tags$li(strong("Interpret with diagnostics. "), "Class balance, derivative quality, MCMC diagnostics, collinearity, jitter records, Random Forest support, latent-Z screening, sensitivity, and benchmarking qualify how strongly the edge should be trusted.")
          )
        ),
        fluidRow(
          column(
            4,
            div(
              class = "guide-card",
              tags$h4("State Predictors"),
              tags$p(
                "A state predictor uses the earlier level of a variable, such as X at a previous time point. ",
                "A retained state edge suggests that the prior level of the source contains information about whether the target tends to increase or decrease."
              ),
              tags$p(
                strong("Example interpretation: "),
                "higher lagged X supports a higher or lower probability that dY is positive, depending on the sign of omega."
              )
            )
          ),
          column(
            4,
            div(
              class = "guide-card",
              tags$h4("First-Derivative Predictors"),
              tags$p(
                "A first-derivative predictor uses the earlier rate or direction of change of a variable, such as dX. ",
                "This can reveal whether recent growth, decline, or velocity-like behavior carries information about the next target response."
              ),
              tags$p(
                strong("Use with care: "),
                "derivative estimates can be more noise-sensitive than state values, so smoothing and derivative diagnostics matter."
              )
            )
          ),
          column(
            4,
            div(
              class = "guide-card",
              tags$h4("Second-Derivative Predictors"),
              tags$p(
                "A second-derivative predictor uses curvature or change in rate, such as d2X. ",
                "It can be informative in oscillatory, delayed, transition-like, or turning-point systems where acceleration or concavity carries regulatory signal."
              ),
              tags$p(
                strong("Use with care: "),
                "second derivatives are often the most sensitive to smoothing, sampling density, and noise."
              )
            )
          )
        ),
        fluidRow(
          column(
            6,
            panel_box(
              "Advantages Of CIRN",
              div(
                class = "guide-good",
                tags$ul(
                  tags$li(strong("Directed output: "), "lagged predictors are aligned before the target response, so the network is temporally ordered rather than purely simultaneous."),
                  tags$li(strong("Signed interpretation: "), "positive omega supports target increase; negative omega suppresses target increase."),
                  tags$li(strong("Uncertainty-aware retention: "), "the 95% HDI rule avoids relying only on point estimates or p-values."),
                  tags$li(strong("Representation-aware inference: "), "state, first-derivative, and second-derivative predictors can reveal different temporal mechanisms."),
                  tags$li(strong("Flexible analysis depth: "), "users can run quick exploratory settings or more careful publication-quality settings."),
                  tags$li(strong("Strong reporting trail: "), "tables, figures, settings, diagnostics, sensitivity summaries, and benchmarks support reproducibility.")
                )
              )
            )
          ),
          column(
            6,
            panel_box(
              "Disadvantages And Tradeoffs",
              div(
                class = "guide-warning",
                tags$ul(
                  tags$li(strong("Derivative sensitivity: "), "first and especially second derivatives can be affected by noise, sparse sampling, and smoothing choices."),
                  tags$li(strong("Class-balance dependence: "), "a target must have enough increasing and decreasing observations after blank-region removal and lagging."),
                  tags$li(strong("Computational cost: "), "Bayesian fitting, pairwise modes, sensitivity analysis, and everything-mode runs can become expensive."),
                  tags$li(strong("Collinearity: "), "state, d, and d2 features from the same trajectories may compete or redistribute support across modes."),
                  tags$li(strong("Conditional interpretation: "), "all retained edges are conditional on the selected data, preprocessing, lag, prior, inference mode, and predictor library."),
                  tags$li(strong("No automatic mechanism recovery: "), "a retained edge is not the same as a biochemical rate law, ecological functional response, or exact equation term.")
                )
              )
            )
          )
        ),
        fluidRow(
          column(
            6,
            panel_box(
              "Inference Modes",
              tags$ul(
                tags$li(strong("Sublevel mode: "), "fits state, first-derivative, and second-derivative predictor groups separately. This is useful for representation-specific evidence."),
                tags$li(strong("All-predictors mode: "), "fits eligible state, d, and d2 predictors together in one target-wise model. This asks which features remain supported after competing with other features."),
                tags$li(strong("Pairwise mode: "), "fits one source-target relation at a time. This is useful as a robustness or screening view, but it is interpreted separately from multivariable conditioning."),
                tags$li(strong("Consistency views: "), "compare which signed edges recur across modes. Edges with stable sign across multiple modes are often more compelling than edges that appear in only one fragile setting.")
              )
            )
          ),
          column(
            6,
            panel_box(
              "How To Read A Signed Edge",
              div(
                class = "guide-good",
                tags$h4("Activation"),
                tags$p(
                  "If omega is positive and the 95% HDI excludes zero, larger values of the lagged predictor are associated with a higher posterior probability that the target derivative is positive."
                )
              ),
              div(
                class = "guide-warning",
                tags$h4("Inhibition"),
                tags$p(
                  "If omega is negative and the 95% HDI excludes zero, larger values of the lagged predictor are associated with a lower posterior probability that the target derivative is positive."
                )
              ),
              tags$p(
                strong("Important: "),
                "the sign is defined on the target-response scale. It should be read as support for target increase or decrease, not automatically as a complete mechanistic claim."
              )
            )
          )
        ),
        panel_box(
          "Evidence Layers Available In CIRN Studio",
          fluidRow(
            column(
              4,
              div(
                class = "guide-card",
                tags$h4("Primary Evidence"),
                tags$ul(
                  tags$li("Bayesian logistic coefficients"),
                  tags$li("95% HDI edge retention"),
                  tags$li("Posterior omega and odds ratio"),
                  tags$li("Mode-specific CIRN networks"),
                  tags$li("Edge consistency grids")
                )
              )
            ),
            column(
              4,
              div(
                class = "guide-card",
                tags$h4("Diagnostic Evidence"),
                tags$ul(
                  tags$li("Response class balance"),
                  tags$li("Derivative and epsilon checks"),
                  tags$li("MCMC diagnostics and trace plots"),
                  tags$li("VIF and collinearity checks"),
                  tags$li("Adaptive jitter records")
                )
              )
            ),
            column(
              4,
              div(
                class = "guide-card",
                tags$h4("Robustness Evidence"),
                tags$ul(
                  tags$li("Random Forest support diagnostics"),
                  tags$li("Latent-Z screening"),
                  tags$li("Sensitivity stability"),
                  tags$li("Feature-level and state-level stability"),
                  tags$li("Ground-truth benchmarking when available")
                )
              )
            )
          )
        ),
        panel_box(
          "Limitations, Delimitations, And Assumptions",
          fluidRow(
            column(
              4,
              div(
                class = "guide-warning",
                tags$h4("Limitations"),
                tags$ul(
                  tags$li("CIRN does not prove interventional causality from observational data."),
                  tags$li("CIRN does not recover complete governing equations, kinetic parameters, or final mechanisms by itself."),
                  tags$li("CIRN cannot identify omitted regulators, hidden confounders, or unmeasured drivers unless they leave detectable indirect signatures."),
                  tags$li("A missing edge does not prove that no real relation exists; it may reflect weak signal, poor sampling, unsuitable lag, collinearity, or conservative HDI retention."),
                  tags$li("Results can change when smoothing, epsilon, lag, predictors, priors, or inference scope change.")
                )
              )
            ),
            column(
              4,
              div(
                class = "guide-card",
                tags$h4("Delimitations"),
                tags$ul(
                  tags$li("CIRN Studio is designed for time-ordered numeric trajectories, not non-temporal cross-sectional data."),
                  tags$li("The app reconstructs regulatory hypotheses for user-selected targets and predictors; it does not automatically define the scientific system boundary."),
                  tags$li("The primary response is binary derivative direction: increasing versus decreasing after near-zero derivative observations are removed."),
                  tags$li("The implemented predictor library is state, first derivative, and second derivative, with user-selected inference modes."),
                  tags$li("RF support, latent-Z screening, EDA tests, and benchmarks qualify interpretation; they do not replace the Bayesian HDI edge rule.")
                )
              )
            ),
            column(
              4,
              div(
                class = "guide-card",
                tags$h4("Core Assumptions"),
                tags$ul(
                  tags$li("The observed trajectories contain meaningful temporal variation."),
                  tags$li("The selected lag is scientifically interpretable as prior information for the target response."),
                  tags$li("The declared predictor set contains plausible regulatory information for the question being asked."),
                  tags$li("Smoothing and derivative estimation produce usable local direction and curvature information."),
                  tags$li("The target response has enough increasing and decreasing observations after preprocessing."),
                  tags$li("The Bayesian logistic model is an interpretable approximation on the log-odds scale, not the true governing equation."),
                  tags$li("Diagnostics are reviewed before assigning strong interpretation to retained edges.")
                )
              )
            )
          )
        ),
        fluidRow(
          column(
            6,
            panel_box(
              "What CIRN Can Support",
              div(
                class = "guide-good",
                tags$ul(
                  tags$li("Exploratory directed and signed regulatory hypotheses."),
                  tags$li("Uncertainty-aware edge retention using posterior HDIs."),
                  tags$li("Representation-aware evidence from state, d, and d2 features."),
                  tags$li("Comparisons across sublevel, all-predictors, and pairwise views."),
                  tags$li("Sensitivity and benchmark summaries that help qualify trust.")
                )
              )
            )
          ),
          column(
            6,
            panel_box(
              "What CIRN Does Not Prove By Itself",
              div(
                class = "guide-warning",
                tags$ul(
                  tags$li("It does not prove interventional causality without independent validation."),
                  tags$li("It does not recover exact ODEs, SDEs, or parameter values by itself."),
                  tags$li("It does not guarantee that an absent edge means no real relation exists."),
                  tags$li("It does not make noisy, sparse, or one-class responses automatically reliable."),
                  tags$li("It does not replace domain expertise, experimental follow-up, or careful diagnostics.")
                )
              )
            )
          )
        ),
        panel_box(
          "Sources Behind This In-App Summary",
          tags$p(
            "This tab summarizes the CIRN manuscript materials, the CIRN user manual, and the current CIRN_Algorithm.R implementation used in this dissertation app build. ",
            "The wording is intentionally user-facing, but the scientific meaning follows the implementation-level workflow: derivative construction, response encoding, lag alignment, Bayesian logistic regression, HDI-based edge retention, and cautious diagnostic interpretation."
          ),
          tags$p(
            strong("Recommended citation language inside reports: "),
            "retained edges are CIRN-inferred directed and signed regulatory hypotheses on the target-response scale, supported under the declared data, preprocessing, lag, inference mode, Bayesian settings, and diagnostic context."
          )
        )
      )
    ),

    tabPanel(
      "User Guide",
      fluidPage(
        div(
          class = "guide-hero",
          tags$h2("CIRN Studio User Guide"),
          tags$p(
            "This user guide walks a new user from data entry to final exported evidence. ",
            "CIRN Studio is designed for exploratory signed regulatory network reconstruction from multivariate time-series data. ",
            "Use the app to generate cautious, uncertainty-aware edge hypotheses, then interpret those edges together with diagnostics, sensitivity checks, and domain knowledge."
          ),
          div(
            class = "guide-pill-row",
            span(class = "guide-pill activation", "Green = activation / increasing class"),
            span(class = "guide-pill inhibition", "Red = inhibition / decreasing class"),
            span(class = "guide-pill neutral", "Edge thickness = larger |omega|"),
            span(class = "guide-pill neutral", "95% HDI excludes zero = retained edge")
          )
        ),
        tabsetPanel(
          tabPanel(
            "Quick Start",
            fluidRow(
              column(
                3,
                div(
                  class = "guide-card",
                  tags$h4("Guided Predator-Prey Example"),
                  tags$ol(
                    tags$li("Open Start."),
                    tags$li("Click Use Predator-Prey Example to load the built-in time series and apply Script-matched settings."),
                    tags$li("Confirm the Active source badge says built-in example or Predator_Prey.csv."),
                    tags$li("Use this route first when learning the interface, testing the workflow, or demonstrating CIRN.")
                  ),
                  tags$p(
                    class = "subtle",
                    "This is the fastest route because variables, time index, and a familiar two-variable system are already available."
                  )
                )
              ),
              column(
                3,
                div(
                  class = "guide-card",
                  tags$h4("Upload Data"),
                  tags$ol(
                    tags$li("Open Start and click Start Upload, or go directly to Data Source > Example or Uploaded Data."),
                    tags$li("Upload a CSV, XLSX, or XLS file in wide time-series format."),
                    tags$li("Select the time column, target variables, and allowed base predictors."),
                    tags$li("Check that every selected target and predictor is numeric and has enough complete observations."),
                    tags$li("Use this route for real experimental, observational, simulation, or dissertation datasets.")
                  )
                )
              ),
              column(
                3,
                div(
                  class = "guide-card",
                  tags$h4("Simulate System"),
                  tags$ol(
                    tags$li("Open Start and click Open Simulator, or choose Data Source > Simulation Lab."),
                    tags$li("Choose a built-in dynamical system."),
                    tags$li("Review the live Model Equation Reference beside the controls. It shows the selected system's generating equations, parameter meanings, and current values; these equations generate the demonstration data but are not supplied to CIRN during inference."),
                    tags$li("Set time span, initial values, parameters, noise, missingness, and outlier options."),
                    tags$li("Click Simulate and Use Data."),
                    tags$li("Confirm the Active source badge says Simulation Lab output."),
                    tags$li("Use this route to test whether CIRN recovers known qualitative network structure under controlled conditions.")
                  )
                )
              ),
              column(
                3,
                div(
                  class = "guide-card",
                  tags$h4("Open Builder"),
                  tags$ol(
                    tags$li("Open Start and click Open Builder."),
                    tags$li("The app will take you to Simulation Lab with User-defined dynamical system selected."),
                    tags$li("Enter variable names, parameter names and values, initial conditions, and right-hand-side equations."),
                    tags$li("Simulate and inspect the generated trajectories before using them for CIRN."),
                    tags$li("Use this route when you want to create synthetic data from your own proposed model.")
                  )
                )
              )
            ),
            panel_box(
              "Recommended First Full Run",
              fluidRow(
                column(
                  4,
                  div(
                    class = "guide-good",
                    tags$h4("1. Start with safe defaults"),
                    tags$ul(
                      tags$li("Use the built-in example or a clean uploaded dataset first."),
                      tags$li("Choose Script-matched run when comparing with CIRN_Algorithm.R."),
                      tags$li("Choose Fast exploration only for checking whether the pipeline runs."),
                      tags$li("Do not change many sliders at once during your first successful run.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("2. Inspect before modeling"),
                    tags$ul(
                      tags$li("In EDA, inspect raw trajectories, normalized overlay, phase planes, missingness, outliers, correlations, lagged correlations, and derivative-state relationships."),
                      tags$li("In Preprocess, check raw versus processed values, derivative preview, epsilon blanking, and response class balance."),
                      tags$li("If the class balance is almost one-sided, adjust epsilon, smoothing, or jitter settings before fitting.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("3. Fit, then qualify"),
                    tags$ul(
                      tags$li("Run CIRN Algorithm with the selected representation mode."),
                      tags$li("Read retained edges together with 95% HDIs, coefficient signs, mode consistency, RF support, latent-Z screening, sensitivity stability, and benchmark results when available."),
                      tags$li("Export the workbook, settings, figures, report, and ZIP bundle for any analysis you plan to cite.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "Tab-By-Tab Workflow",
              tags$ol(
                class = "guide-steps",
                tags$li(strong("Start. "), "Use the guided Predator-Prey example, upload a dataset, simulate a system, or build data from equations. Review the current analysis status and recommended next step."),
                tags$li(strong("Example or Uploaded Data. "), "Load the bundled example or upload and confirm your own table, choose the time column, select target variables, select allowed base predictors, and optionally upload ground-truth or structural adjacency matrices."),
                tags$li(strong("Data Source > Simulation Lab. "), "Generate synthetic data from built-in systems or user-defined equations. Click Simulate and Use Data when the simulated dataset should become the active dataset for EDA, Preprocess, Run CIRN Algorithm, Sensitivity, Benchmark, and Export."),
                tags$li(strong("EDA. "), "Inspect raw time series, normalized overlays, distributions, outliers, time gaps, missingness, correlations, lagged relationships, phase portraits, PCA trajectory, derivative-state patterns, and basic statistical screens."),
                tags$li(strong("Preprocess. "), "Set normalization, spline smoothing, outlier handling, epsilon blank threshold, and adaptive jitter. Use derivative preview and response class balance to verify that each target has usable increasing and decreasing classes."),
                tags$li(strong("Run CIRN Algorithm. "), "Set representation mode, pairwise option, lag units, priors, iterations, warmup, chains, cores, seed, and optional LOO. Click Run CIRN Algorithm only after reviewing these settings."),
                tags$li(strong("Results. "), "Review retained edges, all coefficients, HDI plots, mode-specific network figures, and the edge consistency grid."),
                tags$li(strong("CIRN Figures. "), "Use this gallery for publication-style CIRN networks, Bayesian posterior density plots, trace plots, coefficient HDI figures, consistency plots, sensitivity plots, benchmark plots, and diagnostic figures."),
                tags$li(strong("Diagnostics. "), "Read response construction diagnostics, VIF, jitter details, RF support summaries, latent-Z screening results, and warnings that may weaken edge interpretation."),
                tags$li(strong("Sensitivity. "), "Rerun selected CIRN scopes under perturbations such as noise, missingness, lag changes, downsampling, or target sample-size changes. Use state-level and feature-level stability to judge robustness."),
                tags$li(strong("Benchmark. "), "If a true signed adjacency is available, compare inferred edges with known activation, inhibition, and absent relations."),
                tags$li(strong("Export. "), "Download reproducible evidence: CSV tables, workbook, settings, report, figures, and ZIP bundle."),
                tags$li(strong("Feedback. "), "Send bug reports, usability notes, documentation questions, and improvement suggestions to the CIRN authors.")
              )
            ),
            panel_box(
              "What To Check Before Trusting A Run",
              fluidRow(
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Data checks"),
                    tags$ul(
                      tags$li("The active source is the dataset you intend to analyze."),
                      tags$li("The time column is correctly selected and ordered."),
                      tags$li("Targets and predictors are numeric state variables."),
                      tags$li("Missingness and outliers are understood, not ignored.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Model checks"),
                    tags$ul(
                      tags$li("Derivative curves look plausible for the system."),
                      tags$li("Increasing and decreasing classes are both represented for each target."),
                      tags$li("Bayesian settings are appropriate for the goal: quick demo, careful analysis, publication-quality run, or benchmark."),
                      tags$li("Pairwise mode is treated as support or robustness evidence, not automatically as the main result.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Evidence checks"),
                    tags$ul(
                      tags$li("Retained edge signs match the color legend: green activation, red inhibition."),
                      tags$li("HDIs, omega magnitudes, diagnostics, and consistency agree well enough to support interpretation."),
                      tags$li("Sensitivity results do not contradict the main retained edges."),
                      tags$li("Benchmark metrics are used only when the ground-truth matrix is valid.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "What CIRN Answers And What It Does Not Answer",
              fluidRow(
                column(
                  6,
                  div(
                    class = "guide-good",
                    tags$h4("CIRN can support statements like"),
                    tags$ul(
                      tags$li("This lagged state or derivative feature has posterior evidence for increasing the probability that a target derivative is positive."),
                      tags$li("This predictor-target relation appears as activation or inhibition under the selected representation mode."),
                      tags$li("This edge is stable or unstable across inference modes, sensitivity scenarios, or benchmark comparisons.")
                    )
                  )
                ),
                column(
                  6,
                  div(
                    class = "guide-warning",
                    tags$h4("CIRN does not prove by itself"),
                    tags$ul(
                      tags$li("Interventional causality."),
                      tags$li("The exact differential equation of the system."),
                      tags$li("A final biological, ecological, epidemiological, or social mechanism without validation."),
                      tags$li("That RF support or latent-Z screening assigns regulatory sign.")
                    )
                  )
                )
              )
            )
          ),
          tabPanel(
            "Data Source",
            panel_box(
              "Accepted Data Structure",
              tags$p(
                "CIRN Studio expects multivariate time-series data in wide format. ",
                "Each row is one observation time and each measured system component has its own numeric column."
              ),
              fluidRow(
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("Required table shape"),
                    tags$ul(
                      tags$li(strong("Rows: "), "time points or ordered observations."),
                      tags$li(strong("Columns: "), "one time/index column plus one column for each state variable."),
                      tags$li(strong("Format: "), "CSV, XLSX, or XLS files are accepted for uploaded data."),
                      tags$li(strong("Wide, not long: "), "use columns such as t, X, Y, Z rather than rows such as variable = X, value = 2.1."),
                      tags$li(strong("Ordering: "), "rows should be ordered by time or should have a time column that can be ordered.")
                    )
                  )
                ),
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("Column requirements"),
                    tags$ul(
                      tags$li(strong("Time column: "), "numeric or ordered time index. If a column named t exists, the app selects it by default."),
                      tags$li(strong("State variables: "), "numeric measurements of system components."),
                      tags$li(strong("Targets: "), "numeric variables whose derivative direction will be modeled."),
                      tags$li(strong("Allowed predictors: "), "numeric state variables allowed to generate lagged state, first-derivative, and second-derivative predictor features."),
                      tags$li(strong("Missing values: "), "allowed for inspection, but inference uses complete cases for the selected analysis variables.")
                    )
                  )
                )
              ),
              tags$table(
                class = "preset-settings-table guide-data-example-table",
                tags$thead(
                  tags$tr(
                    tags$th("t"),
                    tags$th("X"),
                    tags$th("Y"),
                    tags$th("Z")
                  )
                ),
                tags$tbody(
                  tags$tr(tags$td("0"), tags$td("2.00"), tags$td("1.50"), tags$td("0.80")),
                  tags$tr(tags$td("1"), tags$td("2.31"), tags$td("1.42"), tags$td("0.91")),
                  tags$tr(tags$td("2"), tags$td("2.58"), tags$td("1.37"), tags$td("1.02"))
                )
              ),
              div(
                class = "note-box",
                "Do not upload ordinary CIRN-derived columns such as dX, dY, d2X, or d2Y as base predictors unless those quantities are genuinely observed variables. The app constructs derivative features internally from the selected state variables."
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "Data Sources",
                  tags$p("The active data source is the table that EDA, Preprocess, Run CIRN Algorithm, Sensitivity, Benchmark, and Export will use."),
                  tags$ul(
                    tags$li(strong("Use Predator-Prey Example: "), "loads Predator_Prey.csv and applies Script-matched settings. This is the friendly first route for learning the app."),
                    tags$li(strong("Upload Data: "), "uses your CSV or Excel table after you choose the time column, targets, and allowed base predictors."),
                    tags$li(strong("Simulate System: "), "creates synthetic data from a built-in dynamical system. The simulated table becomes active only after clicking Simulate and Use Data."),
                    tags$li(strong("Open Builder: "), "opens the user-defined equation builder inside Simulation Lab. Enter variables, parameters, initial conditions, and equations, then click Simulate and Use Data to make the generated table active."),
                    tags$li(strong("Different settings for the example: "), "load the guided Predator-Prey example first, return to Start, and choose another preset under Advanced Analysis Options before rerunning CIRN.")
                  ),
                  div(class = "note-box", "Always check the Active source badge before interpreting EDA, Preprocess, CIRN results, Sensitivity results, Benchmark results, or exported files.")
                )
              ),
              column(
                6,
                panel_box(
                  "Targets And Predictors",
                  tags$ul(
                    tags$li(strong("Target variables: "), "variables whose local direction of change is converted to a binary response: increasing versus decreasing."),
                    tags$li(strong("Allowed base predictors: "), "state variables that may contribute candidate predictors after lagging and derivative construction."),
                    tags$li(strong("State predictor: "), "the lagged value of a selected base variable, such as X."),
                    tags$li(strong("First-derivative predictor: "), "the lagged local rate of change of a selected base variable, such as dX."),
                    tags$li(strong("Second-derivative predictor: "), "the lagged change in rate of change, such as d2X."),
                    tags$li(strong("Self terms: "), "a target may use its own lagged state or derivative features when scientifically allowed and selected."),
                    tags$li(strong("Non-specialist tip: "), "start with all numeric state variables as targets and predictors, then remove variables only when there is a scientific or data-quality reason.")
                  )
                )
              )
            ),
            panel_box(
              "What Happens After Selection",
              fluidRow(
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("1. Preview"),
                    tags$ul(
                      tags$li("Data Source > Example or Uploaded Data previews the active example or uploaded table."),
                      tags$li("Check the number of rows, variable names, and obvious import problems."),
                      tags$li("If an upload looks wrong, fix the source file before modeling.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("2. Variable mapping"),
                    tags$ul(
                      tags$li("The selected time column defines temporal order."),
                      tags$li("Targets define the response variables for CIRN."),
                      tags$li("Allowed predictors define which base variables can produce state, d, and d2 features.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("3. Downstream use"),
                    tags$ul(
                      tags$li("EDA describes the active table."),
                      tags$li("Preprocess smooths and differentiates the active table."),
                      tags$li("Run CIRN Algorithm fits models using the active table and current selections.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "Optional Matrices",
              tags$p("Optional matrices are useful for benchmarking, filtering, or comparing inferred networks with known structure. They are not required for ordinary exploratory CIRN runs."),
              fluidRow(
                column(
                  6,
                  div(
                    class = "guide-good",
                    tags$h4("Ground-truth signed adjacency"),
                    tags$ul(
                      tags$li("Use when the true signed regulatory network is known, usually from a simulation or carefully validated reference."),
                      tags$li("Rows should represent source or regulator variables."),
                      tags$li("Columns should represent target variables."),
                      tags$li("Values may be +1 for activation, -1 for inhibition, and 0 for no true edge."),
                      tags$li("Weighted values are allowed, but the sign is the key benchmark information.")
                    )
                  )
                ),
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("Topology / structural adjacency"),
                    tags$ul(
                      tags$li("Use when you know which source-target connections are structurally possible."),
                      tags$li("This matrix may be unsigned: 1 for possible connection and 0 for no known connection."),
                      tags$li("Use it as structural context, not as signed regulatory truth unless signs are known."),
                      tags$li("A topology matrix can help explain why some inferred edges are plausible or implausible.")
                    )
                  )
                )
              ),
              div(
                class = "note-box",
                "Best practice: use row and column names that exactly match the state-variable names in the data. Benchmark metrics are meaningful only when the ground-truth matrix is scientifically valid for the active dataset."
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "Common Data Mistakes",
                  div(
                    class = "guide-warning",
                    tags$ul(
                      tags$li("Uploading long-format data instead of wide-format data."),
                      tags$li("Selecting an ID, treatment group, replicate label, or category column as a numeric predictor."),
                      tags$li("Leaving the wrong data source active after simulating or uploading a different dataset."),
                      tags$li("Selecting precomputed derivative columns as base predictors when they are not truly observed measurements."),
                      tags$li("Using a truth matrix whose row or column names do not match the data variables."),
                      tags$li("Interpreting benchmark metrics when no defensible ground truth exists.")
                    )
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "Data-Ready Checklist",
                  div(
                    class = "guide-good",
                    tags$ul(
                      tags$li("The Active source badge shows the dataset you intend to analyze."),
                      tags$li("The time column is correct and ordered."),
                      tags$li("Selected targets and predictors are numeric state variables."),
                      tags$li("There are enough complete observations after selecting variables."),
                      tags$li("EDA has been checked for missingness, outliers, time gaps, and unusual trajectories."),
                      tags$li("Optional truth or topology matrices use matching variable names and the correct source-to-target orientation.")
                    )
                  )
                )
              )
            )
          ),
          tabPanel(
            "EDA",
            panel_box(
              "Why EDA Comes Before Preprocess",
              tags$p(
                "Exploratory Data Analysis, or EDA, is the pre-model inspection stage of CIRN Studio. It asks whether the active multivariate time-series data are scientifically and numerically reasonable before derivative smoothing, response-class construction, Bayesian logistic regression, sensitivity analysis, benchmarking, and export. ",
                "EDA is especially important for CIRN because the algorithm learns signed regulatory hypotheses from the direction of target derivatives. If time ordering, sampling gaps, missingness, outliers, scale, or derivative behavior are problematic, the inferred signed edges can become fragile even when the software runs without an error."
              ),
              div(
                class = "note-box",
                "Use EDA to ask: Is this the intended active dataset? Are the selected columns real numeric state variables? Is the time column correct and ordered? Are trajectories plausible? Are gaps, missingness, and outliers manageable? Do first-derivative directions form usable increasing and decreasing classes?"
              ),
              div(
                class = "guide-warning",
                "EDA screens describe the data and suggest modeling risks. They do not prove regulation, causality, or the final CIRN network by themselves."
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "Recommended EDA Reading Order",
                  tags$ol(
                    tags$li(strong("Confirm the active source: "), "make sure the badge says Uploaded, Built-in example, Simulated, or Equation-built data according to what you intend to analyze."),
                    tags$li(strong("Check raw trajectories: "), "start with Raw Small Multiples and Processed Time Series to verify units, trends, cycles, spikes, flat variables, and unexpected values."),
                    tags$li(strong("Check data quality: "), "inspect Time Gaps, Missingness Map, Missingness Summary, Outlier Timeline, and Variable Attention before interpreting associations."),
                    tags$li(strong("Check scale and distribution: "), "use Normalized Overlay, Rolling Mean/Variance, and Distributions to see whether variables have comparable behavior, changing variance, skewness, or heavy tails."),
                    tags$li(strong("Screen associations and temporal dependence: "), "use Correlation Heatmap, Pairwise Scatter, Lagged Correlation, Rolling Correlation, Lead-Lag Cross-Correlation, and ACF / PACF."),
                    tags$li(strong("Inspect dynamics: "), "use Phase Plane, Pairwise Phase Portraits, Directed Phase Portraits, Derivative Vector Field, Delay Embedding, Recurrence Map, Estimated Nullclines, PCA Trajectory, Derivative vs State, Derivative Phase Portraits, and Cross-Derivative Phase Portraits to look for cycles, coupled movement, memory, repeated states, turning regions, and derivative structure."),
                    tags$li(strong("Check response construction: "), "use Derivative Distributions, Epsilon Class Balance, and Epsilon Sensitivity to decide whether the increasing/decreasing target classes are usable."),
                    tags$li(strong("Read the statistical tables: "), "use Basic Statistical Tests and Summaries to document missingness, time spacing, drift, serial dependence, derivative balance, and lag screens before running CIRN.")
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "EDA Controls",
                  tags$ul(
                    tags$li(strong("Variables to display: "), "select the state variables to include in EDA plots. In large systems, inspect related groups instead of plotting every column at once. Dynamic plots remain system-agnostic by showing selected pairwise or faceted projections rather than assuming that the system has only two variables."),
                    tags$li(strong("Horizontal and vertical variables: "), "choose two variables for the focused Phase Plane plot. This is useful for cycles, predator-prey loops, oscillations, and coupled state-space movement."),
                    tags$li(strong("Maximum lag: "), "sets the largest lag used in Lagged Correlation, Lead-Lag Cross-Correlation, and ACF / PACF. Larger lags explore longer memory but use fewer paired observations."),
                    tags$li(strong("Rolling window size: "), "sets the local window for Rolling Mean/Variance and Rolling Correlation. Smaller windows reveal local changes; larger windows give smoother long-run summaries."),
                    tags$li(strong("Granger screen lag order: "), "sets the lag order for the exploratory Granger Screen table. Treat it as lagged predictability only, not regulatory proof."),
                    tags$li(strong("MAD outlier threshold: "), "controls outlier flags based on median absolute deviation. Smaller thresholds flag more possible artifacts; larger thresholds flag only stronger extremes.")
                  ),
                  div(class = "note-box", "Changing EDA controls does not refit CIRN. It changes only the diagnostic views used to understand the active dataset.")
                )
              )
            ),
            panel_box(
              "Exploratory Data Analysis Plot Gallery Reference",
              tags$p(
                "The Exploratory Data Analysis Plot Gallery is organized into Data Quality, Trajectories & Distributions, Relationships & Lags, System Dynamics, and Derivative Diagnostics. Use the plots together: one plot can raise a concern, but repeated evidence across several plots is stronger. For multivariable systems, pairwise and faceted plots show projections of the selected variables; they are exploratory views of a higher-dimensional system, not a restriction of CIRN to two variables. Arrowheads in time-ordered trajectory views indicate observed temporal progression only; they are not inferred CIRN regulatory edges."
              ),
              fluidRow(
                column(
                  3,
                  div(
                    class = "guide-card",
                    tags$h4("Trajectory and source checks"),
                    tags$ul(
                      tags$li(strong("Raw Small Multiples: "), "one panel per variable using the active data source. Use it to see trends, cycles, flat variables, spikes, measurement breaks, or obvious import mistakes."),
                      tags$li(strong("Processed Time Series: "), "the same variables after the current preprocessing choices. Use it to compare the analysis-ready series with the raw trajectories."),
                      tags$li(strong("Normalized Overlay: "), "rescales selected variables so their timing can be compared on one plot. Useful for synchrony, delay, phase shift, opposing movement, and shared cycles."),
                      tags$li(strong("Rolling Mean/Variance: "), "local mean and variance across a moving window. Use it to detect drift, changing variability, regime shifts, or sections where one smoothing choice may not fit the whole series."),
                      tags$li(strong("Distributions: "), "histograms or density summaries for selected variables. Use it to detect skewness, heavy tails, bounded variables, multimodality, and extreme values."),
                      tags$li(strong("Outlier Timeline: "), "marks observations flagged by the MAD threshold. Use it to see whether outliers are isolated points or time-localized events.")
                    )
                  )
                ),
                column(
                  3,
                  div(
                    class = "guide-card",
                    tags$h4("Data quality and missingness"),
                    tags$ul(
                      tags$li(strong("Variable Attention: "), "summary heatmap for missingness, outliers, near-zero derivatives, sign changes, and low variation. Use it to quickly identify variables needing closer inspection."),
                      tags$li(strong("Time Gaps: "), "time-step diagnostic plot. Use it to detect duplicated times, uneven sampling, long gaps, or time-ordering problems that can distort derivative estimates."),
                      tags$li(strong("Missingness Map: "), "shows where missing values occur across variables and time. Blocks or runs of missingness are more concerning than isolated blanks."),
                      tags$li(strong("Missingness Summary: "), "summarizes missing counts and percentages by variable. Use it to decide whether imputation, variable exclusion, or sensitivity analysis is needed."),
                      tags$li(strong("Pairwise Scatter: "), "scatterplots for selected variable pairs. Use it to inspect nonlinear association, clusters, saturation, heteroscedasticity, and outlier leverage."),
                      tags$li(strong("Correlation Heatmap: "), "same-time association matrix. Use it to screen redundancy and strong associations, not to claim regulation.")
                    )
                  )
                ),
                column(
                  3,
                  div(
                    class = "guide-card",
                    tags$h4("Temporal association and dynamics"),
                    tags$ul(
                      tags$li(strong("Lagged Correlation: "), "correlations between earlier source values and later target values across lags. Use it to screen possible time-delay relationships."),
                      tags$li(strong("Rolling Correlation: "), "pairwise correlation inside moving windows. Use it to see whether relationships are stable, transient, or regime-dependent."),
                      tags$li(strong("Lead-Lag Cross-Correlation: "), "checks both directions of temporal offset. Positive lag means the source is earlier than the target; negative lag means the target is earlier than the source."),
                      tags$li(strong("ACF / PACF: "), "autocorrelation and partial autocorrelation screens for each variable. Use them to assess serial dependence, memory, oscillation, and whether very large lag choices are plausible."),
                      tags$li(strong("Phase Plane: "), "plots one selected variable against another through time. Sparse arrowheads show temporal direction; use the view for cycles, loops, attraction, divergence, and state-space structure."),
                      tags$li(strong("Pairwise Phase Portraits: "), "phase-plane panels across several variable pairs with sparse directional arrowheads. Useful when the system has more than two variables."),
                      tags$li(strong("Directed Phase Portraits: "), "draws more detailed arrows along sampled trajectory segments so local time direction is explicit. Use it to see whether a trajectory circulates clockwise or counterclockwise, approaches a region, leaves a region, spirals, or moves through repeated loops."),
                      tags$li(strong("Delay Embedding: "), "plots lagged values against current values, with sparse arrows following time. Self-delay panels screen memory, oscillation, or repeated attractor-like shape within one variable; cross-delay panels screen whether one variable's earlier value aligns with another variable's later value."),
                      tags$li(strong("Recurrence Map: "), "marks times when the selected system state returns close to a previous state. The system-state panel uses selected variables jointly; cross-recurrence panels compare selected variable pairs. Use it to detect repetition, cycles, regime returns, or irregular recurrence."),
                      tags$li(strong("PCA Trajectory: "), "low-dimensional summary of multivariate movement with sparse time-direction arrows. Use it to see clusters, regimes, loops, or directional transitions in the whole selected system.")
                    )
                  )
                ),
                column(
                  3,
                  div(
                    class = "guide-card",
                    tags$h4("Derivative and response diagnostics"),
                    tags$ul(
                      tags$li(strong("Derivative Vector Field: "), "uses estimated first derivatives to draw small arrows in pairwise state planes. It helps reveal local flow direction and speed, but it is an exploratory projection and not a fitted mechanistic ODE."),
                      tags$li(strong("Estimated Nullclines: "), "estimates where a selected variable's first derivative is approximately zero in a pairwise state projection. Use it to locate turning regions, possible equilibrium neighborhoods, or state combinations where growth changes direction. Treat it cautiously when data are sparse, noisy, or cover only a narrow part of state space."),
                      tags$li(strong("Derivative vs State: "), "state predictor versus target first derivative. With two predictors and two targets, it shows four panels. This is a state-to-derivative screen, not the full CIRN model."),
                      tags$li(strong("Derivative Phase Portraits: "), "within-variable state-rate and rate-acceleration portraits, such as X vs dX and dX vs d2X. Sparse arrowheads show how the trajectory moves through each projected phase space."),
                      tags$li(strong("Cross-Derivative Phase Portraits: "), "cross-variable state, first-derivative, and second-derivative portraits, such as X vs dY, dX vs dY, and X vs d2Y. Sparse arrows follow time; use the panels as exploratory interaction screens."),
                      tags$li(strong("Derivative Distributions: "), "distribution of first and second derivatives. Use it to see whether derivative signs are balanced, noisy, centered near zero, or dominated by extremes."),
                      tags$li(strong("Epsilon Class Balance: "), "green is increasing, red is decreasing, and grey is blank near-zero derivative region. Use it to see whether target response classes are usable."),
                      tags$li(strong("Epsilon Sensitivity: "), "shows how changing the blank threshold changes usable counts and class balance. Use it before choosing epsilon for CIRN.")
                    )
                  )
                )
              ),
              div(
                class = "note-box",
                "Important: EDA plots may suggest possible relationships, but CIRN edges are retained only after the Bayesian CIRN model and 95% HDI rule are applied in the Run CIRN Algorithm step. Directed phase portraits, vector fields, delay embeddings, recurrence maps, and nullcline views are descriptive dynamics diagnostics; they do not prove regulatory causality or replace the CIRN inference results."
              )
            ),
            panel_box(
              "Basic Statistical Tests and Summaries Reference",
              tags$p(
                "These tables provide numeric EDA evidence behind the plots. They are designed for screening, reporting, and preprocessing decisions. A small p-value or large correlation is not automatically a CIRN edge."
              ),
              fluidRow(
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Variable-level diagnostics"),
                    tags$ul(
                      tags$li(strong("Variable Diagnostics: "), "reports complete observations, missing percentage, mean, SD, MAD, Shapiro normality screen, Ljung-Box serial-dependence screen, and optional ADF/KPSS stationarity screens when available. Use it to identify sparse, noisy, non-normal, serially dependent, or drifting variables."),
                      tags$li(strong("Time Spacing: "), "reports time-ordering and spacing diagnostics such as unique time points, duplicate times, minimum/median/maximum spacing, and irregularity. Use it to decide whether derivative estimation is trustworthy."),
                      tags$li(strong("Trend Screen: "), "summarizes simple time trends and monotone association with time. Use it to detect drift that may dominate derivative signs or create apparent relationships."),
                      tags$li(strong("Missingness Runs: "), "lists runs of missing values rather than only totals. Long consecutive missing blocks can be more damaging to derivatives than the same number of isolated missing cells.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Association and lag screens"),
                    tags$ul(
                      tags$li(strong("Correlation Tests: "), "reports Pearson and Spearman pairwise association among selected variables. Pearson screens linear association; Spearman screens monotone association. Neither implies regulation by itself."),
                      tags$li(strong("Granger Screen: "), "tests whether past values of a source improve prediction of a target beyond the target's own past at the selected lag order. It is a lagged-predictability screen, not proof of causality."),
                      tags$li(strong("Lagged Correlations: "), "lists source-target-lag correlations used by lagged plots. Use the sign, magnitude, lag, and sample size together."),
                      tags$li(strong("ACF/PACF context: "), "interpret lag screens together with ACF/PACF plots. Strong autocorrelation can make naive lag relationships look stronger than they really are.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Derivative and response summaries"),
                    tags$ul(
                      tags$li(strong("Derivative Summary: "), "reports first- and second-derivative spread, sign counts, near-zero percentage, and sign changes. Use it to assess whether the derivative signal is informative or mostly noise/flatness."),
                      tags$li(strong("Epsilon Grid: "), "reports increasing, decreasing, blank, usable, and minority-class counts across epsilon choices. Use it to choose a blank threshold that avoids near-zero noise while preserving enough usable observations."),
                      tags$li(strong("Class-balance meaning: "), "CIRN models classify whether the target derivative is increasing or decreasing. If one class is nearly absent, Bayesian logistic fits can become unstable or require adaptive jitter."),
                      tags$li(strong("Reporting tip: "), "for serious analyses, report the active data source, selected variables, time spacing, missingness, derivative class balance, epsilon, smoothing choice, and any major EDA concern.")
                    )
                  )
                )
              ),
              div(
                class = "guide-warning",
                "Do not choose preprocessing settings only to obtain a preferred edge. Use EDA to justify defensible data handling, then use diagnostics and sensitivity analysis to test whether the resulting network is stable."
              )
            ),
            panel_box(
              "Practical EDA Decision Rules",
              fluidRow(
                column(
                  4,
                  div(
                    class = "guide-good",
                    tags$h4("Looks ready to proceed when"),
                    tags$ul(
                      tags$li("The active source is correct."),
                      tags$li("Time is ordered and reasonably regular."),
                      tags$li("Variables have enough complete observations."),
                      tags$li("Trajectories and derivatives look plausible."),
                      tags$li("Increasing and decreasing derivative classes are both represented after preprocessing.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-warning",
                    tags$h4("Pause and inspect when"),
                    tags$ul(
                      tags$li("There are duplicate or highly irregular time values."),
                      tags$li("Missing values occur in long runs."),
                      tags$li("A few outliers dominate the trajectory or derivative plot."),
                      tags$li("A variable is almost constant."),
                      tags$li("Derivative summary shows many near-zero values or excessive sign changes.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Useful actions"),
                    tags$ul(
                      tags$li("Return to Data if the wrong source or variables are selected."),
                      tags$li("Return to Data Source > Simulation Lab if synthetic data do not match the intended model."),
                      tags$li("Use Preprocess to adjust smoothing, outlier handling, normalization, epsilon, or jitter."),
                      tags$li("Use Sensitivity later to test whether results survive plausible perturbations.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "Important Cautions",
              div(
                class = "guide-warning",
                tags$ul(
                  tags$li("Correlation, lagged correlation, and Granger screens are not CIRN edges and do not prove regulatory causality."),
                  tags$li("Stationarity screens are diagnostic only and may be unavailable if optional packages are not installed."),
                  tags$li("A clean EDA does not guarantee a correct network; it only reduces avoidable data-quality and preprocessing mistakes."),
                  tags$li("A problematic EDA does not automatically make the data unusable, but it should be reported and tested with sensitivity analysis.")
                )
              )
            ),
            div(class = "note-box", "Use EDA to find data quality issues before Bayesian fitting. It is cheaper to fix data-source or preprocessing mistakes before running CIRN than after generating final figures.")
          ),
          tabPanel(
            "Preprocess",
            panel_box(
              "What Preprocess Does",
              tags$p(
                "Preprocessing is the bridge between EDA and the CIRN Algorithm. It turns the active time-series data into an analysis-ready table, estimates first and second derivatives, constructs increasing/decreasing target response classes, and checks whether those response classes are usable for Bayesian logistic CIRN fitting."
              ),
              tags$p(
                "For non-specialist users: think of this tab as the quality-control step before modeling. EDA tells you what the data look like; Preprocess decides how the data will be cleaned, smoothed, differentiated, and converted into the binary response used by CIRN."
              ),
              div(
                class = "note-box",
                "CIRN does not model raw values alone. Its response is based on whether each target variable is locally increasing or decreasing. That is why smoothing, derivatives, epsilon, blanks, and class balance matter so much."
              )
            ),
            panel_box(
              "Recommended Preprocess Workflow",
              tags$ol(
                tags$li(strong("Confirm the active source: "), "make sure the processed preview is based on the dataset you intend to analyze, especially after using Simulation Lab, the guided Predator-Prey example, Upload Data, or Build From Equations."),
                tags$li(strong("Start with defaults: "), "use the default smoothing, epsilon, normalization, and jitter settings first. They are meant to provide a reasonable starting point before expert tuning."),
                tags$li(strong("Inspect the processed preview: "), "confirm the time column, selected variables, and row count look correct."),
                tags$li(strong("Read quality warnings: "), "resolve or document warnings about missingness, outliers, derivative problems, or class imbalance before fitting CIRN."),
                tags$li(strong("Check derivative preview: "), "derivative curves should be informative but not dominated by noise, spikes, or excessive flattening."),
                tags$li(strong("Check response class balance: "), "each target should have enough increasing and decreasing observations after blank near-zero derivative points are excluded."),
                tags$li(strong("Adjust one setting at a time: "), "if something looks wrong, change one preprocessing control, re-check the plots and tables, then continue."),
                tags$li(strong("Record final settings: "), "for dissertation or manuscript use, report normalization, smoothing method, smoothing value if manual, outlier action, epsilon, derivative grid, and whether adaptive jitter was triggered.")
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "Core Preprocessing Controls",
                  tags$ul(
                    tags$li(strong("Normalization: "), "controls variable scaling before modeling. None keeps original units. Z-score centers variables and scales by standard deviation. Min-max rescales values to a common range. Use normalization when variables have very different magnitudes."),
                    tags$li(strong("Derivative grid points per interval: "), "controls how finely derivatives are evaluated between observed time points. Larger values can make derivative previews smoother and more resolved, but may increase computation and should still reflect the sampling density."),
                    tags$li(strong("Choose smoothing by GCV: "), "lets the app choose the spline smoothing level using generalized cross-validation. This is the best starting choice for most users."),
                    tags$li(strong("Spline smoothing spar: "), "manual smoothing strength used when GCV is not selected. Lower values follow the data more closely; higher values smooth more strongly. Too low can create noisy derivatives; too high can erase real changes."),
                    tags$li(strong("Outlier action: "), "controls what happens to potential extreme values. Keep leaves them unchanged, Winsorize caps extremes without removing rows, and Remove drops flagged observations. Winsorize is often safer than removing rows when time order matters."),
                    tags$li(strong("MAD outlier threshold: "), "controls how extreme a point must be before it is flagged. Smaller values flag more points; larger values flag only more extreme observations."),
                    tags$li(strong("log10 epsilon blank threshold: "), "sets the near-zero derivative region. Derivatives close to zero are labeled blank and are not used as increasing or decreasing response cases. For example, -6 means epsilon is 1e-6.")
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "Adaptive Jitter and Class Construction",
                  tags$ul(
                    tags$li(strong("Increasing class: "), "target derivative is positive after excluding the blank region. In app figures, this class is shown in green."),
                    tags$li(strong("Decreasing class: "), "target derivative is negative after excluding the blank region. In app figures, this class is shown in red."),
                    tags$li(strong("Blank class: "), "target derivative is very close to zero. These points are not used in the binary increasing/decreasing response."),
                    tags$li(strong("Use adaptive response jitter when needed: "), "allows the app to make a minimal adjustment when a target response is too sparse or nearly one-class. This is a diagnostic rescue tool, not something to hide."),
                    tags$li(strong("Minimum class count after jitter: "), "sets the minimum number of increasing and decreasing cases expected after jitter. If this cannot be met, the target may be weakly informative for CIRN."),
                    tags$li(strong("Predictor-jitter sensitivity: "), "expert option for checking how stable the results are when jitter is needed. Use it when final results depend on jitter."),
                    tags$li(strong("Jitter scale basis: "), "controls the scale used for jitter. Keep the default unless you have a specific methodological reason to change it.")
                  ),
                  div(
                    class = "guide-warning",
                    "If adaptive jitter is triggered, report it. A jitter-assisted fit can still be useful, but it should be interpreted more cautiously than a fit with naturally balanced derivative classes."
                  )
                )
              )
            ),
            panel_box(
              "How To Read Preprocess Outputs",
              fluidRow(
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Tables and warnings"),
                    tags$ul(
                      tags$li(strong("Processed Preview: "), "shows the current analysis-ready data. Use it to confirm the row count, time column, variable names, and processed values."),
                      tags$li(strong("Quality Warnings: "), "summarizes problems found during preprocessing. Read this before running CIRN."),
                      tags$li(strong("Class Balance Table: "), "lists increasing, decreasing, blank, usable, and minority-class counts by target. Low usable counts or very small minority class counts are warning signs."),
                      tags$li(strong("Current Epsilon: "), "shows the actual epsilon value implied by the log10 slider.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Derivative preview"),
                    tags$ul(
                      tags$li(strong("State curve: "), "the processed target trajectory."),
                      tags$li(strong("First derivative: "), "the local rate and direction of change. This determines increasing versus decreasing response classes."),
                      tags$li(strong("Second derivative: "), "the local change in rate. It can represent acceleration, deceleration, curvature, and concavity-related features."),
                      tags$li(strong("Good sign: "), "curves are smooth enough to suppress measurement noise but still preserve meaningful turning points."),
                      tags$li(strong("Warning sign: "), "derivatives are extremely jagged, dominated by spikes, almost perfectly flat, or inconsistent with the raw trajectory.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Response class balance"),
                    tags$ul(
                      tags$li(strong("Green bars: "), "increasing derivative cases."),
                      tags$li(strong("Red bars: "), "decreasing derivative cases."),
                      tags$li(strong("Grey bars: "), "blank near-zero derivative cases."),
                      tags$li(strong("Usable cases: "), "increasing plus decreasing cases used by the binary CIRN response."),
                      tags$li(strong("Minority class: "), "the smaller of increasing and decreasing counts. Very small minority classes can make logistic regression unstable.")
                    )
                  )
                )
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "Common Problems and Fixes",
                  tags$ul(
                    tags$li(strong("Derivatives look too jagged: "), "inspect outliers, increase smoothing, use GCV, or check whether the time step is too sparse."),
                    tags$li(strong("Derivatives look too flat: "), "reduce manual smoothing, inspect whether GCV over-smoothed, or check whether the variable has little variation."),
                    tags$li(strong("Too many blank points: "), "epsilon may be too large, the target may be nearly constant, or smoothing may have flattened the derivative."),
                    tags$li(strong("One class dominates: "), "try a smaller epsilon, inspect the trajectory for monotone drift, consider whether the observation window is too short, and check whether adaptive jitter is triggered."),
                    tags$li(strong("Many outliers are flagged: "), "return to EDA, inspect the Outlier Timeline, and decide whether they are measurement artifacts or real events."),
                    tags$li(strong("Processed data are not the expected data: "), "return to Data Source > Example or Uploaded Data or Data Source > Simulation Lab and check the active source badge before continuing.")
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "Non-Specialist Decision Rules",
                  tags$ul(
                    tags$li(strong("Change settings for data reasons, not for desired edges: "), "a preprocessing choice should be justified by the trajectory, derivative preview, quality warnings, or class balance."),
                    tags$li(strong("Prefer simple first runs: "), "start with GCV smoothing, default epsilon, and conservative outlier handling before expert tuning."),
                    tags$li(strong("Keep enough usable data: "), "if epsilon or outlier removal leaves too few observations, the CIRN fit may be unreliable."),
                    tags$li(strong("Preserve scientific meaning: "), "do not normalize, smooth, or remove points in a way that destroys the interpretation of the variables."),
                    tags$li(strong("Use sensitivity analysis later: "), "if results depend strongly on smoothing, epsilon, outlier action, or jitter, report this and test stability.")
                  ),
                  div(
                    class = "note-box",
                    "A good preprocessing choice makes derivatives interpretable and response classes usable. It should not be chosen only because it gives a cleaner or more expected network."
                  )
                )
              )
            ),
            panel_box(
              "Preprocess Reporting Checklist",
              div(
                class = "guide-good",
                tags$ul(
                  tags$li("Active data source and selected time column."),
                  tags$li("Selected target variables and allowed base predictors."),
                  tags$li("Normalization method."),
                  tags$li("Smoothing method: GCV or manual spline spar value."),
                  tags$li("Derivative grid points per interval."),
                  tags$li("Outlier action and MAD threshold."),
                  tags$li("Epsilon value or log10 epsilon slider setting."),
                  tags$li("Response class counts for each target."),
                  tags$li("Whether adaptive response jitter was used."),
                  tags$li("Any quality warnings that remain before CIRN fitting.")
                )
              )
            ),
            panel_box(
              "Important Cautions",
              div(
                class = "guide-warning",
                tags$ul(
                  tags$li("Preprocessing can change derivative signs and therefore can change CIRN edges."),
                  tags$li("Removing rows can affect time spacing and derivative estimation. Use removal carefully for time-series data."),
                  tags$li("Very smooth derivatives may hide real short-term regulation; very jagged derivatives may create artificial sign changes."),
                  tags$li("Near-zero derivative blanks are useful because tiny numerical fluctuations should not be forced into increasing or decreasing classes."),
                  tags$li("If a target has too few increasing or decreasing cases, CIRN may be inappropriate for that target even if the app can still run.")
                )
              )
            )
          ),
          tabPanel(
            "Run CIRN",
            panel_box(
              "What Run CIRN Algorithm Does",
              tags$p(
                "This tab is where CIRN Studio fits the Bayesian logistic regression models that produce signed regulatory hypotheses. For each selected target variable, the app converts the target's first derivative into an increasing-versus-decreasing response, then asks whether earlier state, first-derivative, or second-derivative predictor features help classify that response."
              ),
              tags$p(
                "In plain language: CIRN asks whether a candidate predictor provides credible evidence that a target variable is more likely to be locally increasing or locally decreasing. A retained positive coefficient is reported as activation; a retained negative coefficient is reported as inhibition."
              ),
              div(
                class = "note-box",
                "The Run CIRN Algorithm button should be clicked after Data, EDA, and Preprocess have been reviewed. The default slider values are recommended first-try settings; increase computational settings later for final reporting."
              ),
              div(
                class = "guide-warning",
                "CIRN edges are exploratory, signed, directed hypotheses. They should be interpreted with HDIs, diagnostics, sensitivity analysis, benchmark results when available, and scientific domain knowledge."
              )
            ),
            panel_box(
              "Before Clicking Run CIRN Algorithm",
              tags$ol(
                tags$li(strong("Confirm active data source: "), "the Run tab should show the intended uploaded, simulated, equation-built, or example dataset."),
                tags$li(strong("Confirm targets and predictors: "), "targets are the variables whose derivative direction will be modeled; allowed base predictors generate state, first-derivative, and second-derivative candidate features."),
                tags$li(strong("Check preprocessing: "), "derivatives should look plausible and target response class balance should have enough increasing and decreasing cases."),
                tags$li(strong("Choose representation mode: "), "decide whether to run sublevel models, all-predictor models, or both."),
                tags$li(strong("Decide whether pairwise CIRN is needed: "), "pairwise runs are useful for robustness/support views but can add runtime."),
                tags$li(strong("Review Bayesian sliders: "), "warmup must be smaller than total iterations, prior SD must be positive, and chains/cores should match your machine."),
                tags$li(strong("Set seed for reproducibility: "), "use a fixed model seed when you want repeatable results.")
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "Representation Modes",
                  tags$ul(
                    tags$li(strong("Sublevel models: "), "fits separate models for state predictors, first-derivative predictors, and second-derivative predictors. This is easier to explain because each feature level is inspected separately."),
                    tags$li(strong("All predictors: "), "fits one joint model containing all candidate state, first-derivative, and second-derivative features. This can reveal combined evidence, but it is more sensitive to collinearity and overfitting."),
                    tags$li(strong("Both: "), "runs sublevel and all-predictor modes so you can compare whether retained edges are stable across model representations."),
                    tags$li(strong("Lag units: "), "sets how far back predictor features are shifted before being used to classify the target derivative direction. A lag of 1 means the immediately previous time step; larger lags test longer memory."),
                    tags$li(strong("Recommended first choice: "), "use Both with lag units = 1 for an initial full comparison, then simplify or expand only when the science and diagnostics justify it.")
                  ),
                  div(
                    class = "note-box",
                    "Sublevel results are often more interpretable for explanation. All-predictor results are useful for checking whether evidence remains when candidate features compete in one joint model."
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "Pairwise CIRN",
                  tags$ul(
                    tags$li(strong("Run pairwise CIRN: "), "fits one target-regulator relation at a time instead of fitting all selected predictors together in one multivariable model."),
                    tags$li(strong("Pairwise representation: "), "chooses whether pairwise fits use sublevel, all-predictor, or both representations."),
                    tags$li(strong("Why use it: "), "pairwise CIRN can reveal whether a source-target relation has support when considered alone. It is helpful as a robustness or screening view."),
                    tags$li(strong("Why not use it alone: "), "pairwise fits do not adjust for all other candidate predictors at the same time. A pairwise edge can disappear in multivariable CIRN if its signal is explained by another feature."),
                    tags$li(strong("Runtime warning: "), "pairwise mode can become expensive when many predictors and targets are selected because it creates many additional models.")
                  ),
                  div(
                    class = "guide-warning",
                    "For final interpretation, read pairwise results together with sublevel, all-predictor, HDI, VIF, RF support, sensitivity, and benchmark outputs."
                  )
                )
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "Bayesian Sliders",
                  tags$ul(
                    tags$li(strong("Prior mean: "), "the center of the coefficient prior. The default 0 says that, before seeing the data, positive and negative effects are treated symmetrically."),
                    tags$li(strong("Prior SD: "), "the prior spread. Smaller values shrink coefficients more strongly toward 0; larger values allow larger effects. Very large values may make weak-data fits less stable."),
                    tags$li(strong("Total iterations: "), "total MCMC draws per chain, including warmup. More iterations can improve posterior estimation but increase runtime."),
                    tags$li(strong("Warmup iterations: "), "initial iterations used to tune the sampler. Warmup must be smaller than total iterations and is not used as final posterior evidence."),
                    tags$li(strong("Chains: "), "independent MCMC runs. More chains help diagnose convergence. Four chains are typical for serious Bayesian reporting."),
                    tags$li(strong("Cores: "), "parallel CPU workers. More cores can speed fitting but may slow your computer if too many are used."),
                    tags$li(strong("adapt_delta: "), "sampler tuning target. Higher values can reduce difficult-sampling warnings but can make models slower."),
                    tags$li(strong("Model seed: "), "controls reproducibility of Bayesian sampling. Keep it fixed when comparing settings.")
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "LOO and Model Diagnostics",
                  tags$ul(
                    tags$li(strong("Compute LOO diagnostics: "), "adds approximate leave-one-out predictive diagnostics for Bayesian model comparison and reporting."),
                    tags$li(strong("When to enable LOO: "), "use it for careful analysis, publication-quality runs, or when comparing model representations."),
                    tags$li(strong("When to disable LOO: "), "leave it off for fast exploration, teaching demos, or very large runs where speed matters."),
                    tags$li(strong("Run Log: "), "records start, finish, target, representation, status, and progress details. Use it to see what actually ran."),
                    tags$li(strong("Pre-Run Summary: "), "shows the current data, target, predictor, epsilon, representation, and iteration settings before fitting."),
                    tags$li(strong("Progress behavior: "), "Bayesian models may update progress only between completed model fits. A model can be working even if the progress bar appears paused for a while.")
                  ),
                  div(
                    class = "note-box",
                    "If the app feels slow, first reduce pairwise/everything-style runs, lower iterations, reduce chains, or use a faster preset before increasing settings again."
                  )
                )
              )
            ),
            panel_box(
              "Analysis Presets",
              tags$p(
                "Presets change several Bayesian and workflow settings at once. They are designed to help users choose a reasonable analysis scale without manually tuning every slider."
              ),
              tags$ul(
                tags$li(strong("Script-matched run: "), "closest to the main CIRN_Algorithm.R defaults. Use this when checking whether the Shiny app and script-based workflow agree."),
                tags$li(strong("Teaching demo: "), "fast, light settings for classroom, meeting, or demonstration use. Good for showing the workflow, not for final evidence."),
                tags$li(strong("Fast exploration: "), "quick pipeline check. Use it to make sure data, preprocessing, and outputs are working before spending time on heavier runs."),
                tags$li(strong("Careful analysis: "), "balanced setting for serious exploratory inference when you want more stable results than a quick test."),
                tags$li(strong("Publication quality: "), "heavier setting with stronger reporting intent. Use for dissertation figures, manuscript-ready analysis, or final retained-edge evidence."),
                tags$li(strong("Benchmark mode: "), "for simulations or known-truth data where you plan to compare inferred edges with ground truth.")
              ),
              div(class = "note-box", "Recommended workflow: run Fast exploration or Script-matched first, confirm outputs look sensible, then use Careful analysis or Publication quality for final evidence.")
            ),
            panel_box(
              "Optional Exploratory Diagnostics",
              fluidRow(
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("Random Forest support check"),
                    tags$ul(
                      tags$li(strong("Purpose: "), "checks whether retained Bayesian candidate predictors also show predictive support in a nonlinear Random Forest classifier."),
                      tags$li(strong("RF trees: "), "number of trees in each Random Forest. More trees are more stable but slower."),
                      tags$li(strong("RF bootstrap repetitions: "), "number of repeated RF checks. More repetitions give a more stable support rate."),
                      tags$li(strong("RF support threshold: "), "minimum support rate needed to call a term RF-supported."),
                      tags$li(strong("Interpretation: "), "RF support is a robustness diagnostic. It does not assign the signed activation/inhibition direction; Bayesian CIRN coefficients remain the source of signs.")
                    )
                  )
                ),
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("Latent-Z screening"),
                    tags$ul(
                      tags$li(strong("Purpose: "), "screens whether adding a low-dimensional latent component improves derivative-direction classification."),
                      tags$li(strong("Latent-Z useful gain threshold: "), "minimum accuracy gain needed to flag latent-Z as potentially useful."),
                      tags$li(strong("Interpretation: "), "a useful latent-Z result may suggest hidden collective structure or shared variation not captured by individual predictors."),
                      tags$li(strong("Caution: "), "latent-Z does not prove a hidden regulator. It is an exploratory diagnostic that should be reported cautiously.")
                    )
                  )
                )
              ),
              div(
                class = "guide-warning",
                "RF support and latent-Z screening can be run only after CIRN has been fitted. Their results appear in the Diagnostics tab."
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "How To Choose a Run Strategy",
                  tags$ul(
                    tags$li(strong("First-time user: "), "use the built-in example, keep defaults, choose Both, leave pairwise off, and run Fast exploration."),
                    tags$li(strong("Script comparison: "), "choose Script-matched run and use the same representation mode as CIRN_Algorithm.R."),
                    tags$li(strong("Small system: "), "Both plus pairwise can be useful because runtime is manageable and mode consistency is easy to inspect."),
                    tags$li(strong("Large system: "), "start with sublevel or all-predictor modes without pairwise, then add pairwise only for selected variables if needed."),
                    tags$li(strong("Final dissertation analysis: "), "use Careful analysis or Publication quality, fixed seed, multiple chains, diagnostics, sensitivity analysis, and exported settings.")
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "Common Run Problems",
                  tags$ul(
                    tags$li(strong("Button is unavailable: "), "CIRN needs active data, selected targets, selected predictors, and valid preprocessing."),
                    tags$li(strong("Warmup error: "), "warmup iterations must be smaller than total iterations."),
                    tags$li(strong("Slow run: "), "reduce pairwise mode, reduce targets/predictors, lower iterations, reduce chains, or disable LOO."),
                    tags$li(strong("Few or no retained edges: "), "check class balance, smoothing, epsilon, prior strength, selected predictors, and whether the system truly contains detectable signed structure."),
                    tags$li(strong("Many retained edges: "), "inspect VIF, collinearity, RF support, sensitivity stability, and whether preprocessing created noisy derivative sign changes."),
                    tags$li(strong("Different mode results: "), "sublevel, all-predictor, and pairwise modes answer related but different questions. Consistency across modes is stronger evidence than a single-mode edge.")
                  )
                )
              )
            ),
            panel_box(
              "Run CIRN Reporting Checklist",
              div(
                class = "guide-good",
                tags$ul(
                  tags$li("Active data source, selected targets, and allowed base predictors."),
                  tags$li("Representation mode: sublevel, all predictors, or both."),
                  tags$li("Whether pairwise CIRN was run and which pairwise representation was used."),
                  tags$li("Lag units."),
                  tags$li("Prior mean and prior SD."),
                  tags$li("Total iterations, warmup iterations, chains, cores, adapt_delta, and seed."),
                  tags$li("Whether LOO diagnostics were computed."),
                  tags$li("Number of retained signed edges and which mode retained them."),
                  tags$li("Any sampler warnings, convergence concerns, VIF/collinearity concerns, jitter use, or class-balance concerns."),
                  tags$li("Whether RF support, latent-Z screening, sensitivity analysis, and benchmark checks were performed.")
                )
              )
            ),
            panel_box(
              "Important Cautions",
              div(
                class = "guide-warning",
                tags$ul(
                  tags$li("A retained HDI edge means the fitted Bayesian coefficient credibly excludes zero under the chosen settings. It is not automatic proof of mechanism."),
                  tags$li("Changing preprocessing, representation mode, lag, prior, or pairwise setting can change retained edges."),
                  tags$li("A fast preset is useful for workflow testing but should not be cited as final evidence."),
                  tags$li("Pairwise evidence is not the same as multivariable evidence because pairwise models do not adjust for all other predictors at once."),
                  tags$li("If results will be reported publicly, always inspect Results, CIRN Figures, Diagnostics, Sensitivity, Benchmark when available, and Export settings together.")
                )
              )
            )
          ),
          tabPanel(
            "Interpret",
            panel_box(
              "What Interpretation Means",
              tags$p(
                "The Interpret section explains what the fitted CIRN outputs mean after the Bayesian models have run. CIRN reports signed, directed, uncertainty-aware regulatory hypotheses from multivariate time-series data. The central evidence is a Bayesian logistic regression coefficient, called omega in the output tables and figures."
              ),
              tags$p(
                "For each target variable, CIRN models whether the target's first derivative is positive or negative. Therefore, an edge is about evidence for the target becoming more likely to increase or decrease locally, not simply about whether the target's raw value is large or small."
              ),
              div(
                class = "note-box",
                "Non-specialist translation: a retained edge means the predictor feature helped classify the local direction of target change, and the model was sufficiently uncertain-aware that the 95% HDI did not include zero."
              ),
              div(
                class = "guide-warning",
                "Interpret CIRN edges as exploratory hypotheses. They can support scientific reasoning, but they do not by themselves prove intervention-level causality or a final biological, ecological, epidemiological, or physical mechanism."
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "Reading a Retained Edge",
                  tags$ul(
                    tags$li(strong("Source / predictor: "), "the variable or derivative feature placed at the start of the arrow."),
                    tags$li(strong("Target / response: "), "the variable whose first-derivative direction is being classified as increasing or decreasing."),
                    tags$li(strong("Omega: "), "the Bayesian logistic regression coefficient for the predictor feature. Larger absolute omega means a stronger fitted association with target derivative direction."),
                    tags$li(strong("95% HDI: "), "the high-density interval summarizing coefficient uncertainty. If the 95% HDI excludes zero, the edge is retained."),
                    tags$li(strong("Retained edge: "), "a source-target relation whose coefficient has credible nonzero evidence under the selected model settings."),
                    tags$li(strong("Edge thickness: "), "larger absolute omega is drawn as a thicker arrow in CIRN network figures.")
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "Activation and Inhibition",
                  tags$ul(
                    tags$li(strong("Positive omega / activation: "), "higher predictor-feature values increase the fitted probability that the target derivative is positive. In figures, this is shown in green."),
                    tags$li(strong("Negative omega / inhibition: "), "higher predictor-feature values decrease the fitted probability that the target derivative is positive. In figures, this is shown in red."),
                    tags$li(strong("Important wording: "), "activation does not mean the predictor always raises the target's raw value at every time point. It means the predictor is associated with a higher probability of local target increase in the fitted model."),
                    tags$li(strong("Likewise: "), "inhibition means the predictor is associated with a lower probability of local target increase, or equivalently more evidence toward local target decrease."),
                    tags$li(strong("Scientific interpretation: "), "use activation/inhibition labels only as model-supported signed hypotheses, then compare them with known system biology, ecology, epidemiology, or domain theory.")
                  ),
                  div(
                    class = "note-box",
                    "Green and red are sign conventions for retained CIRN coefficients: green = activation-positive omega; red = inhibition-negative omega."
                  )
                )
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "Figure Color Code",
                  tags$ul(
                    tags$li(strong("Green edge: "), "activation: positive omega."),
                    tags$li(strong("Red edge: "), "inhibition: negative omega."),
                    tags$li(strong("Thicker edge: "), "larger absolute omega."),
                    tags$li(strong("Orange node border: "), "state predictor such as X."),
                    tags$li(strong("Blue node border: "), "first-derivative predictor such as dX."),
                    tags$li(strong("Magenta node border: "), "second-derivative predictor such as d2X."),
                    tags$li(strong("Black node border: "), "target response node, such as dY when Y is the target.")
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "Feature Names and Arrows",
                  tags$ul(
                    tags$li(strong("X -> dY: "), "state variable X helps classify whether target Y is increasing or decreasing."),
                    tags$li(strong("dX -> dY: "), "the first derivative of X helps classify whether Y is increasing or decreasing."),
                    tags$li(strong("d2X -> dY: "), "the second derivative of X helps classify whether Y is increasing or decreasing."),
                    tags$li(strong("Self-edge: "), "an edge such as X -> dX or dX -> dX means a variable's own state or derivative feature helps classify its own derivative direction."),
                    tags$li(strong("Direction: "), "the arrow points from the candidate predictor feature to the target derivative response being classified."),
                    tags$li(strong("Do not reverse the arrow: "), "X -> dY and Y -> dX are different fitted questions.")
                  )
                )
              )
            ),
            panel_box(
              "CIRN Figure Tabs",
              fluidRow(
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("Network figures"),
                    tags$ul(
                      tags$li(strong("Interactive Combined: "), "an explorable retained-edge network. Use it for hovering, selecting, zooming, and quickly seeing which edges were retained."),
                      tags$li(strong("Sublevel: "), "static publication-style network from separate state, first-derivative, and second-derivative sublevel fits."),
                      tags$li(strong("All Predictors: "), "static network from the joint model where all candidate predictor features compete together."),
                      tags$li(strong("Pairwise: "), "static network from one-source-one-target fits. Useful as a support view, but not the same as multivariable evidence."),
                      tags$li(strong("Consistent >=2: "), "edges retained with the same sign in at least two fitted inference-mode sources."),
                      tags$li(strong("Consistent All 3: "), "edges retained with the same sign across sublevel, all-predictor, and pairwise sources when all are available.")
                    )
                  )
                ),
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("Consistency figures"),
                    tags$ul(
                      tags$li(strong("Edge Consistency Grid: "), "rows are inferred edges and columns are inference modes. Green means activation, red means inhibition, and grey means the edge was not retained in that mode."),
                      tags$li(strong("Mode Consistency: "), "summarizes which retained edges appear across sublevel, all-predictor, and pairwise sources."),
                      tags$li(strong("Consistent sign: "), "stronger evidence occurs when the same source-target edge appears with the same sign across multiple modes."),
                      tags$li(strong("Mode disagreement: "), "if an edge appears only in one mode, changes sign, or disappears in the joint all-predictor fit, interpret it more cautiously."),
                      tags$li(strong("No edge in a mode: "), "absence in one mode does not prove no regulation; it means the selected model and thresholds did not retain that relation.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "Coefficient and Bayesian Plots",
              fluidRow(
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("Coefficient evidence"),
                    tags$ul(
                      tags$li(strong("Coefficient HDI plot: "), "shows omega estimates and 95% HDIs. Retained edges have HDIs entirely above or below zero."),
                      tags$li(strong("HDI crossing zero: "), "the model is not confident enough about the sign. Do not report it as a retained activation or inhibition edge."),
                      tags$li(strong("Large absolute omega: "), "stronger fitted association with target derivative direction, but still check uncertainty, diagnostics, and sensitivity."),
                      tags$li(strong("Small HDI width: "), "more precise coefficient estimate. Wide HDIs indicate uncertainty."),
                      tags$li(strong("Retained-only filter: "), "use this to focus on edges that passed the HDI retention rule, but still inspect the full coefficient table for context.")
                    )
                  )
                ),
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("Posterior and trace plots"),
                    tags$ul(
                      tags$li(strong("Posterior Densities: "), "show the uncertainty distribution of coefficients. A density mostly on the positive side supports activation; mostly negative supports inhibition."),
                      tags$li(strong("Cross-Representation Posteriors: "), "compare the same target or relation across sublevel, all-predictor, and pairwise-style representations."),
                      tags$li(strong("Histogram + Trace: "), "shows posterior samples and chain behavior. Use it to detect poor mixing or unstable sampling."),
                      tags$li(strong("Trace With Warm-Up: "), "checks whether chains stabilize after warmup and sample the same posterior region."),
                      tags$li(strong("Diagnostics Summary: "), "summarizes model health indicators that should be reviewed before trusting retained edges.")
                    )
                  )
                )
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "What Strengthens an Interpretation",
                  div(
                    class = "guide-good",
                    tags$ul(
                      tags$li("The 95% HDI excludes zero clearly."),
                      tags$li("The same signed edge appears across multiple inference modes."),
                      tags$li("MCMC diagnostics look healthy, with no major convergence or sampling concerns."),
                      tags$li("VIF and collinearity diagnostics are acceptable."),
                      tags$li("Response class balance is adequate and jitter was not needed, or jitter sensitivity is stable."),
                      tags$li("RF support, sensitivity stability, and benchmark results, when available, agree with the edge."),
                      tags$li("The edge is scientifically plausible for the system being studied.")
                    )
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "What Weakens an Interpretation",
                  div(
                    class = "guide-warning",
                    tags$ul(
                      tags$li("The edge appears in only one mode and disappears in other modes."),
                      tags$li("The sign changes across modes or sensitivity scenarios."),
                      tags$li("The HDI is close to zero or very wide."),
                      tags$li("Pairwise evidence exists but multivariable evidence does not."),
                      tags$li("Preprocessing choices strongly change the retained edge."),
                      tags$li("Class balance is poor or adaptive jitter is essential."),
                      tags$li("Diagnostics show convergence, collinearity, or low effective-sample-size concerns.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "Non-Specialist Interpretation Workflow",
              tags$ol(
                tags$li(strong("Start with the retained-edge table: "), "identify the source, target, sign, omega, HDI, and inference mode."),
                tags$li(strong("Open the CIRN figure: "), "confirm the same edge appears visually with the correct color and direction."),
                tags$li(strong("Read the HDI plot: "), "check whether uncertainty clearly excludes zero."),
                tags$li(strong("Compare modes: "), "look for the same sign in sublevel, all-predictor, and pairwise outputs when available."),
                tags$li(strong("Check diagnostics: "), "review MCMC diagnostics, class balance, VIF, jitter, RF support, and latent-Z if run."),
                tags$li(strong("Check sensitivity: "), "ask whether the edge remains under plausible perturbations."),
                tags$li(strong("Check benchmark: "), "if ground truth is known, compare inferred edges with the signed adjacency matrix."),
                tags$li(strong("Write cautiously: "), "say 'CIRN inferred a signed regulatory hypothesis' rather than 'CIRN proved causation'.")
              )
            ),
            panel_box(
              "Interpretation Reporting Checklist",
              div(
                class = "guide-good",
                tags$ul(
                  tags$li("Source feature and target response for each retained edge."),
                  tags$li("Sign: activation-positive omega or inhibition-negative omega."),
                  tags$li("Omega estimate and 95% HDI."),
                  tags$li("Inference mode that retained the edge: sublevel, all predictors, pairwise, or consistent mode."),
                  tags$li("Whether the edge appears consistently across modes."),
                  tags$li("Bayesian diagnostics and any sampler warnings."),
                  tags$li("Class balance and whether adaptive jitter was used."),
                  tags$li("VIF or collinearity concerns."),
                  tags$li("Sensitivity stability and benchmark performance when available."),
                  tags$li("Scientific interpretation and limitations.")
                )
              )
            ),
            panel_box(
              "Important Cautions",
              div(
                class = "guide-warning",
                tags$ul(
                  tags$li("CIRN signs refer to the probability that the target derivative is positive, not direct changes in the raw target value under intervention."),
                  tags$li("A retained edge is conditional on the chosen active data, preprocessing settings, lag, prior, representation mode, and retention rule."),
                  tags$li("Pairwise and multivariable results can differ because they answer different modeling questions."),
                  tags$li("Self-edges and derivative-feature edges can be meaningful, but they require careful system-specific interpretation."),
                  tags$li("Do not interpret absence of a retained edge as proof of no relationship."),
                  tags$li("For dissertation or publication use, always interpret results together with diagnostics, sensitivity analysis, and domain knowledge.")
                )
              )
            )
          ),
          tabPanel(
            "Diagnostics",
            panel_box(
              "What Diagnostics Are For",
              tags$p(
                "Diagnostics help you decide how much confidence to place in retained CIRN edges. They do not create new signed edges by themselves. Instead, they check whether the fitted Bayesian models, derivative response construction, collinearity, adaptive jitter, pairwise fits, Random Forest support, and latent-Z screening give reasons to trust, qualify, or question the reported network."
              ),
              tags$p(
                "For non-specialist users: a CIRN edge is stronger when the coefficient is retained by HDI, the model diagnostics look healthy, the response classes are usable, collinearity is not severe, the result is stable across modes or sensitivity scenarios, and optional support checks agree."
              ),
              div(
                class = "note-box",
                "Read diagnostics after Run CIRN Algorithm and before making conclusions from Results or CIRN Figures."
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "MCMC / Model Diagnostics",
                  tags$ul(
                    tags$li(strong("Purpose: "), "checks whether Bayesian sampling behaved well enough for posterior summaries, HDIs, and retained edges to be trusted."),
                    tags$li(strong("Rhat: "), "chain agreement diagnostic. Values very close to 1 are reassuring. Values clearly above 1 suggest chains may not have converged."),
                    tags$li(strong("Effective sample size, or ESS: "), "amount of useful posterior information after accounting for chain autocorrelation. Larger ESS is better; very small ESS means the posterior summary may be unreliable."),
                    tags$li(strong("Divergences or sampler warnings: "), "signals that the sampler had difficulty exploring the posterior. Retained edges from models with serious warnings should be interpreted cautiously."),
                    tags$li(strong("Iterations and warmup context: "), "low iterations can make diagnostics weak. Final analyses should use stronger settings than quick demonstrations."),
                    tags$li(strong("Action if poor: "), "increase iterations, increase adapt_delta, check class balance, simplify the model, reduce collinearity, or rerun with a more careful preset.")
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "VIF and Collinearity Diagnostics",
                  tags$ul(
                    tags$li(strong("Purpose: "), "checks whether predictors are highly redundant with one another, especially in all-predictor models."),
                    tags$li(strong("VIF: "), "variance inflation factor. Higher VIF means a predictor is strongly explained by other predictors, making individual coefficients harder to interpret."),
                    tags$li(strong("Why it matters: "), "when predictors are highly collinear, an edge may appear, disappear, or change sign depending on which related features are included."),
                    tags$li(strong("Sublevel vs all-predictor context: "), "all-predictor models are usually more vulnerable to collinearity because state, first-derivative, and second-derivative features compete together."),
                    tags$li(strong("Action if high: "), "compare sublevel and all-predictor results, inspect correlation plots, reduce redundant predictors if scientifically justified, and emphasize mode consistency."),
                    tags$li(strong("Interpretation rule: "), "a retained edge with severe collinearity should be described as tentative unless it is also stable across modes and sensitivity checks.")
                  ),
                  div(
                    class = "guide-warning",
                    "Collinearity does not mean the app is wrong. It means individual edge attribution is harder because related predictors carry overlapping information."
                  )
                )
              )
            ),
            panel_box(
              "Response-Class and Jitter Diagnostics",
              fluidRow(
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Class balance"),
                    tags$ul(
                      tags$li(strong("Purpose: "), "checks whether each target has enough increasing and decreasing derivative cases for binary classification."),
                      tags$li(strong("Usable observations: "), "the non-blank cases used in the increasing-versus-decreasing response."),
                      tags$li(strong("Minority class: "), "the smaller of increasing and decreasing cases. Very small minority counts can make logistic regression unstable."),
                      tags$li(strong("Warning sign: "), "almost all observations are increasing, almost all are decreasing, or many observations are blank near-zero derivatives.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Adaptive jitter magnitude"),
                    tags$ul(
                      tags$li(strong("Purpose: "), "shows whether response construction needed adaptive jitter to avoid sparse or one-class derivative responses."),
                      tags$li(strong("No jitter used: "), "usually reassuring because response classes were naturally usable."),
                      tags$li(strong("Jitter used: "), "not automatically bad, but the target should be interpreted more cautiously and reported transparently."),
                      tags$li(strong("Larger jitter: "), "stronger intervention in response construction and therefore a stronger caution flag.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Adaptive jitter detail"),
                    tags$ul(
                      tags$li(strong("Purpose: "), "shows how different jitter amounts affect the increasing/decreasing response construction for a selected target."),
                      tags$li(strong("Stable pattern: "), "small jitter changes do not strongly alter class balance."),
                      tags$li(strong("Unstable pattern: "), "small jitter changes strongly alter response classes; treat retained edges for that target as fragile."),
                      tags$li(strong("Action if unstable: "), "inspect smoothing, epsilon, outliers, target trajectory, and sensitivity analysis.")
                    )
                  )
                )
              ),
              div(
                class = "note-box",
                "CIRN is a derivative-direction classifier. If the derivative response classes are weak, the network evidence is weak even if some coefficients are retained."
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "Pairwise Diagnostics",
                  tags$ul(
                    tags$li(strong("Purpose: "), "summarizes the health of pairwise CIRN fits when pairwise mode was run."),
                    tags$li(strong("How pairwise differs: "), "pairwise fits inspect one source-target relation at a time. They do not adjust for all other predictors simultaneously."),
                    tags$li(strong("Useful when: "), "you want a robustness or screening view for individual source-target relations."),
                    tags$li(strong("Caution: "), "pairwise support alone is weaker than agreement between pairwise and multivariable modes."),
                    tags$li(strong("Mode conflict: "), "if pairwise and multivariable results disagree, report the conflict and inspect collinearity, preprocessing, and sensitivity stability.")
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "Random Forest Support",
                  tags$ul(
                    tags$li(strong("Purpose: "), "checks whether candidate predictors have nonlinear predictive support for classifying increasing versus decreasing target derivatives."),
                    tags$li(strong("support_rate: "), "the fraction of repeated Random Forest checks in which the predictor was supported."),
                    tags$li(strong("supported: "), "TRUE when the support rate meets or exceeds the selected RF support threshold."),
                    tags$li(strong("mean_importance: "), "average RF variable importance across bootstrap repetitions."),
                    tags$li(strong("Interpretation: "), "RF support strengthens confidence that a predictor is useful for classification, but it does not define the sign of the regulatory edge."),
                    tags$li(strong("Sign source: "), "activation or inhibition still comes from the Bayesian CIRN coefficient omega, not the Random Forest.")
                  )
                )
              )
            ),
            panel_box(
              "Latent-Z Screening",
              fluidRow(
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("What it checks"),
                    tags$ul(
                      tags$li(strong("Purpose: "), "tests whether a low-dimensional latent component improves derivative-direction classification."),
                      tags$li(strong("variance_explained_pc1: "), "how much variation is summarized by the first latent component."),
                      tags$li(strong("baseline_accuracy: "), "classification accuracy without the latent-Z enhancement."),
                      tags$li(strong("latent_z_accuracy: "), "classification accuracy with the latent-Z component included."),
                      tags$li(strong("accuracy_gain: "), "improvement from adding latent-Z.")
                    )
                  )
                ),
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("How to interpret it"),
                    tags$ul(
                      tags$li(strong("useful: "), "TRUE when the accuracy gain meets the selected useful-gain threshold."),
                      tags$li(strong("Possible meaning: "), "latent-Z may suggest shared hidden structure, collective system movement, or unmeasured common variation."),
                      tags$li(strong("Important limit: "), "latent-Z does not prove a hidden regulator, hidden variable, or mechanism."),
                      tags$li(strong("Action if useful: "), "report it as exploratory, inspect PCA/EDA plots, and consider whether unmeasured structure is scientifically plausible."),
                      tags$li(strong("Action if not useful: "), "do not overinterpret; it simply means this diagnostic did not find useful latent-Z improvement under the selected settings.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "How Diagnostics Should Change Interpretation",
              fluidRow(
                column(
                  6,
                  div(
                    class = "guide-good",
                    tags$h4("Stronger evidence when"),
                    tags$ul(
                      tags$li("Rhat is close to 1 and ESS is adequate."),
                      tags$li("No serious sampler warnings are present."),
                      tags$li("Response classes are balanced enough for each target."),
                      tags$li("No adaptive jitter was needed, or jitter sensitivity looks stable."),
                      tags$li("VIF and collinearity concerns are mild."),
                      tags$li("Pairwise and multivariable modes agree in sign when both were run."),
                      tags$li("RF support agrees with Bayesian retained edges."),
                      tags$li("Latent-Z results do not indicate unaccounted structure that would undermine interpretation.")
                    )
                  )
                ),
                column(
                  6,
                  div(
                    class = "guide-warning",
                    tags$h4("Weaker evidence when"),
                    tags$ul(
                      tags$li("Rhat is high, ESS is low, or sampler warnings appear."),
                      tags$li("Usable sample size is low after lagging, blank removal, missingness, or outlier handling."),
                      tags$li("One class dominates the target response."),
                      tags$li("Adaptive jitter is essential or large."),
                      tags$li("VIF is severe in all-predictor models."),
                      tags$li("Pairwise and multivariable signs conflict."),
                      tags$li("RF support does not support a retained Bayesian edge."),
                      tags$li("Sensitivity analysis later shows unstable signs or low detection rates.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "Non-Specialist Diagnostic Workflow",
              tags$ol(
                tags$li(strong("Start with MCMC / Model Diagnostics: "), "check whether Bayesian fitting is healthy enough to trust coefficient summaries."),
                tags$li(strong("Check response construction: "), "review class balance and jitter diagnostics because CIRN depends on increasing/decreasing derivative classes."),
                tags$li(strong("Check VIF: "), "decide whether collinearity makes individual predictors hard to interpret."),
                tags$li(strong("If pairwise was run: "), "compare pairwise evidence with sublevel and all-predictor evidence."),
                tags$li(strong("If RF was run: "), "see whether retained Bayesian predictors also have predictive support."),
                tags$li(strong("If latent-Z was run: "), "see whether hidden shared structure may be relevant."),
                tags$li(strong("Then interpret Results and CIRN Figures: "), "use diagnostics to decide whether each retained edge is strong, tentative, or fragile.")
              )
            ),
            panel_box(
              "Diagnostics Reporting Checklist",
              div(
                class = "guide-good",
                tags$ul(
                  tags$li("MCMC diagnostics: Rhat, ESS, and any sampler warnings."),
                  tags$li("Response class balance for each target."),
                  tags$li("Whether adaptive jitter was used and how large it was."),
                  tags$li("VIF or collinearity concerns, especially for all-predictor models."),
                  tags$li("Pairwise diagnostics when pairwise CIRN was run."),
                  tags$li("Random Forest settings and support results if RF support check was run."),
                  tags$li("Latent-Z gain threshold and screening results if latent-Z was run."),
                  tags$li("How diagnostics changed the interpretation of retained edges.")
                )
              )
            ),
            panel_box(
              "Important Cautions",
              div(
                class = "guide-warning",
                tags$ul(
                  tags$li("Diagnostics do not replace the Bayesian HDI retention rule; they qualify the reliability of retained edges."),
                  tags$li("Good diagnostics do not prove causality. They only reduce technical reasons to distrust the fitted model."),
                  tags$li("Poor diagnostics do not automatically invalidate all results, but they require cautious wording and additional checks."),
                  tags$li("RF support is unsigned. It cannot convert an inhibition edge into activation or vice versa."),
                  tags$li("Latent-Z screening is exploratory. Treat it as a clue about hidden structure, not proof of a hidden regulator."),
                  tags$li("When diagnostics disagree, report the disagreement rather than hiding it.")
                )
              )
            )
          ),
          tabPanel(
            "Sensitivity",
            panel_box(
              "What Sensitivity Analysis Is For",
              tags$p(
                "Sensitivity analysis asks whether retained CIRN edges remain stable when the data or modeling settings are perturbed. It reruns CIRN under modified conditions such as added noise, different lags, downsampling, smaller sample sizes, or missing rows."
              ),
              tags$p(
                "For non-specialist users: a sensitivity run is a stress test. If an edge appears only in the original run but disappears under small reasonable changes, interpret it cautiously. If an edge keeps the same sign across many scenarios, it is stronger exploratory evidence."
              ),
              div(
                class = "note-box",
                "Sensitivity analysis is slower than a single CIRN run because every row of the sensitivity plan can rerun one or more CIRN models."
              ),
              div(
                class = "guide-warning",
                "Sensitivity stability does not prove causality. It tells you whether the inferred signed edge is robust to the perturbations you asked the app to test."
              )
            ),
            panel_box(
              "Recommended Sensitivity Workflow",
              tags$ol(
                tags$li(strong("Run CIRN Algorithm first: "), "sensitivity analysis needs an active fitted CIRN result and uses the selected data, targets, predictors, preprocessing, and Bayesian settings."),
                tags$li(strong("Choose sensitivity inference scope: "), "decide which CIRN modes should be rerun during sensitivity analysis."),
                tags$li(strong("Start small: "), "use few replicates and a narrow set of scenarios first to confirm the workflow is correct."),
                tags$li(strong("Preview Plan: "), "inspect how many scenario rows will be run before starting the expensive computation."),
                tags$li(strong("Run Sensitivity: "), "execute the previewed scenarios. Large plans may take time."),
                tags$li(strong("Read the progress log: "), "confirm which scenarios completed, failed, or were skipped."),
                tags$li(strong("Read stability outputs: "), "use the feature-level table and heatmap to decide which retained edges are stable, unstable, or sign-inconsistent."),
                tags$li(strong("Report the scope and scenarios: "), "sensitivity results are meaningful only when readers know what perturbations and inference modes were tested.")
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "Sensitivity Sliders",
                  tags$ul(
                    tags$li(strong("Stochastic replicates: "), "repeats random perturbations. More replicates give a more reliable stability estimate but increase runtime."),
                    tags$li(strong("Noise fractions: "), "adds observation noise at the listed fractions. Use this to ask whether edges survive measurement noise."),
                    tags$li(strong("Lag units: "), "reruns CIRN with different lag choices. Use this to ask whether an edge depends on one specific temporal delay."),
                    tags$li(strong("Downsample intervals: "), "keeps every kth observation, such as every 2nd or 5th time point. Use this to ask whether edges survive coarser sampling."),
                    tags$li(strong("Target sample sizes: "), "reruns CIRN on smaller numbers of observations. Use this to ask how much sample size affects retained edges."),
                    tags$li(strong("Missing row fractions: "), "randomly removes rows at the listed fractions. Use this to ask whether edges survive missing observations."),
                    tags$li(strong("Save sensitivity CSVs to temp folder: "), "writes sensitivity output files during the run. Useful for debugging or advanced audit trails.")
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "Inference Scope",
                  tags$ul(
                    tags$li(strong("Use Run CIRN settings: "), "uses the same representation mode and pairwise choices currently selected in the Run CIRN Algorithm tab."),
                    tags$li(strong("Sublevel only: "), "reruns only sublevel multivariable CIRN. This is usually the fastest serious scope and is easy to interpret."),
                    tags$li(strong("All predictors only: "), "reruns only all-predictor multivariable CIRN. Use this when you care about the joint model where all candidate features compete."),
                    tags$li(strong("Sublevel + all predictors: "), "reruns both multivariable modes without pairwise. This is a good serious default when pairwise is not needed."),
                    tags$li(strong("Pairwise only: "), "focuses stability summaries on pairwise retained edges. Useful for source-target screening but not a substitute for multivariable evidence."),
                    tags$li(strong("Everything: "), "reruns sublevel, all-predictor, pairwise sublevel, and pairwise all-predictor CIRN. This gives the broadest sensitivity view and is the most expensive option.")
                  ),
                  div(
                    class = "guide-warning",
                    "Runtime depends on both the sensitivity plan and the inference scope. Everything is much slower than Sublevel only."
                  )
                )
              )
            ),
            panel_box(
              "Sensitivity Tables",
              fluidRow(
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("Plan and progress tables"),
                    tags$ul(
                      tags$li(strong("Sensitivity Plan: "), "the list of scenario-replicate runs the app will execute. Preview this before running so you know the computational cost."),
                      tags$li(strong("Sensitivity Progress Log: "), "records which scenario is starting, running, completed, failed, or skipped."),
                      tags$li(strong("Sensitivity Runs: "), "summarizes completed runs, scenario names, scenario values, run status, messages, row counts, and retained-edge counts."),
                      tags$li(strong("Failed run: "), "one scenario failed. Read the message and decide whether to rerun with lighter settings, fewer variables, or a smaller plan.")
                    )
                  )
                ),
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("Feature-level stability table"),
                    tags$ul(
                      tags$li(strong("Feature-level edge stability: "), "preserves exact CIRN feature labels, such as X -> dY, dX -> dY, or d2X -> dY."),
                      tags$li(strong("completed_runs: "), "number of successful sensitivity runs contributing to that edge/scenario summary."),
                      tags$li(strong("detection_rate: "), "fraction of completed runs where the edge was retained, regardless of sign details."),
                      tags$li(strong("activation_rate: "), "fraction of runs retaining the edge as activation."),
                      tags$li(strong("inhibition_rate: "), "fraction of runs retaining the edge as inhibition."),
                      tags$li(strong("mixed sign: "), "activation and inhibition both appear across runs. Treat sign interpretation as unstable.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "State-Level vs Feature-Level Stability",
              fluidRow(
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("State-level stability"),
                    tags$ul(
                      tags$li(strong("Meaning: "), "collapses derivative representations to a base regulator-to-target relation."),
                      tags$li(strong("Example: "), "X -> Y groups evidence from X, dX, and d2X-style predictors pointing to target Y."),
                      tags$li(strong("Question answered: "), "is the general base regulator-to-target relation stable after ignoring which representation of the regulator was used?"),
                      tags$li(strong("Use when: "), "you want a simple biological, ecological, or system-level summary.")
                    )
                  )
                ),
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("Feature-level stability"),
                    tags$ul(
                      tags$li(strong("Meaning: "), "keeps the exact predictor representation used by CIRN."),
                      tags$li(strong("Example: "), "X -> dY, dX -> dY, and d2X -> dY are treated as different feature-level edges."),
                      tags$li(strong("Question answered: "), "is this specific state, first-derivative, or second-derivative predictor relation stable?"),
                      tags$li(strong("Use when: "), "you want to know whether the stable signal comes from the state, rate of change, or acceleration/curvature representation.")
                    )
                  )
                )
              ),
              div(
                class = "note-box",
                "Key idea: state-level stability is broader and easier to summarize; feature-level stability is more precise and closer to the actual CIRN model features."
              )
            ),
            panel_box(
              "Reading the Feature-Level Edge Stability Heatmap",
              tags$ul(
                tags$li(strong("Rows: "), "feature-level CIRN edges, using the same feature labels as CIRN edge-consistency plots."),
                tags$li(strong("Columns: "), "sensitivity scenarios such as baseline, noise, lag, downsampling, sample size, or missing-row perturbations."),
                tags$li(strong("Green: "), "the edge is retained as activation in that scenario."),
                tags$li(strong("Red: "), "the edge is retained as inhibition in that scenario."),
                tags$li(strong("Grey: "), "the edge is not detected or not retained in that scenario."),
                tags$li(strong("Purple: "), "mixed sign: the same feature-level edge appears as activation in some runs and inhibition in others."),
                tags$li(strong("Darker or stronger color: "), "higher signed detection rate across completed runs."),
                tags$li(strong("Best pattern: "), "same color across many scenarios, especially baseline plus perturbations."),
                tags$li(strong("Weak pattern: "), "edge appears only in baseline, disappears under small perturbations, or changes sign.")
              ),
              div(
                class = "guide-warning",
                "A blank or grey cell does not prove no relation exists. It means that edge was not retained under that sensitivity scenario and selected inference scope."
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "How To Interpret Stability",
                  div(
                    class = "guide-good",
                    tags$h4("Stronger sensitivity evidence"),
                    tags$ul(
                      tags$li("The same edge is retained at baseline and under multiple perturbations."),
                      tags$li("The sign stays the same across scenarios."),
                      tags$li("Detection rate is high across completed runs."),
                      tags$li("The edge is stable at feature level, not only after state-level collapsing."),
                      tags$li("Results remain stable under reasonable noise, lag, missingness, and sample-size changes."),
                      tags$li("The edge is also supported by diagnostics, mode consistency, and benchmark results when available.")
                    )
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "When To Be Cautious",
                  div(
                    class = "guide-warning",
                    tags$h4("Weaker sensitivity evidence"),
                    tags$ul(
                      tags$li("The edge appears only in the original run or baseline scenario."),
                      tags$li("The sign changes from activation to inhibition across scenarios."),
                      tags$li("Detection rate is low or based on very few completed runs."),
                      tags$li("Only state-level stability is high but feature-level stability is scattered."),
                      tags$li("Results fail under small noise or modest missing-row perturbations."),
                      tags$li("Many sensitivity scenarios fail, leaving too little evidence for stable conclusions.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "Practical Run Strategy",
              tags$ol(
                tags$li(strong("First check: "), "use Sublevel only or Use Run CIRN settings with a small plan and 1-2 replicates."),
                tags$li(strong("Serious multivariable check: "), "use Sublevel + all predictors if runtime is acceptable."),
                tags$li(strong("Pairwise-focused check: "), "use Pairwise only when your question is about pairwise retained edges."),
                tags$li(strong("Full stress test: "), "use Everything only for small systems or final analyses where runtime is acceptable."),
                tags$li(strong("Increase gradually: "), "add more replicates and more scenario values after the first small run succeeds."),
                tags$li(strong("Export results: "), "include sensitivity tables and settings in the final export bundle for reproducibility.")
              )
            ),
            panel_box(
              "Sensitivity Reporting Checklist",
              div(
                class = "guide-good",
                tags$ul(
                  tags$li("Sensitivity inference scope."),
                  tags$li("Number of stochastic replicates."),
                  tags$li("Noise fractions, lag units, downsample intervals, target sample sizes, and missing-row fractions tested."),
                  tags$li("Number of completed, failed, or skipped runs."),
                  tags$li("State-level edge stability summary, if used."),
                  tags$li("Feature-level edge stability summary."),
                  tags$li("Whether pairwise-only or everything mode was used."),
                  tags$li("Which retained edges were stable, unstable, or mixed sign."),
                  tags$li("How sensitivity results changed the interpretation of the main CIRN network.")
                )
              )
            ),
            panel_box(
              "Important Cautions",
              div(
                class = "guide-warning",
                tags$ul(
                  tags$li("Sensitivity analysis can only test the perturbations included in the plan. It cannot guarantee robustness to every possible data problem."),
                  tags$li("A large sensitivity plan with many failed runs is not stronger evidence than a smaller plan with clean completed runs."),
                  tags$li("Everything mode is powerful but expensive; use it deliberately."),
                  tags$li("Pairwise-only sensitivity should not be reported as if it were full multivariable stability."),
                  tags$li("If sensitivity and main CIRN results disagree, report the disagreement and interpret the affected edges as tentative.")
                )
              )
            )
          ),
          tabPanel(
            "Benchmark",
            panel_box(
              "When Benchmarking Is Valid",
              tags$p(
                "Benchmarking compares the inferred CIRN network with a known signed ground-truth network. It is valid only for simulated systems or empirical datasets where the true signed regulatory network is defensible. If there is no scientifically valid truth matrix, benchmark numbers should not be reported."
              ),
              tags$p(
                "For non-specialist users: benchmarking answers, 'Did CIRN recover the known activation and inhibition edges?' It does not discover the truth by itself. The benchmark is only as reliable as the ground-truth matrix you provide."
              ),
              div(
                class = "note-box",
                "Use benchmarking for built-in simulations, custom simulations with a known true network, or empirical systems with a carefully justified reference network."
              ),
              div(
                class = "guide-warning",
                "Do not benchmark against a guess, a literature sketch with uncertain signs, or a topology-only matrix unless you clearly state that it is not signed ground truth."
              )
            ),
            panel_box(
              "Recommended Benchmark Workflow",
              tags$ol(
                tags$li(strong("Prepare or confirm ground truth: "), "upload a signed ground-truth adjacency matrix in Data Source > Example or Uploaded Data, or use a built-in simulated system with a known true network."),
                tags$li(strong("Check names and orientation: "), "rows must be sources/regulators and columns must be targets/responses."),
                tags$li(strong("Run CIRN Algorithm: "), "benchmarking needs fitted retained CIRN edges before it can compare inferred and true networks."),
                tags$li(strong("Open Benchmark: "), "click Evaluate Against Ground Truth."),
                tags$li(strong("Compare heatmaps: "), "read True Adjacency beside Inferred State-Level Adjacency."),
                tags$li(strong("Read metrics: "), "use the Metrics table to summarize recovery, missed edges, extra edges, and wrong-sign edges."),
                tags$li(strong("Report limitations: "), "state the truth matrix source, whether it is signed, and whether the benchmark is simulation-based or empirical.")
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "Ground Truth Matrix Format",
                  tags$ul(
                    tags$li(strong("Rows: "), "source or regulator variables."),
                    tags$li(strong("Columns: "), "target or response variables."),
                    tags$li(strong("+1: "), "true activation edge from the row variable to the column variable."),
                    tags$li(strong("-1: "), "true inhibition edge from the row variable to the column variable."),
                    tags$li(strong("0: "), "no true signed edge from the row variable to the column variable."),
                    tags$li(strong("Weighted signed values: "), "allowed when scientifically justified, but the sign is the most important benchmark information."),
                    tags$li(strong("Names: "), "row and column names should match the state-variable names used by CIRN, such as X and Y, not ordinary derivative labels unless those derivative quantities are truly the benchmark nodes.")
                  ),
                  div(
                    class = "guide-warning",
                    "The most common benchmark mistake is reversing the matrix orientation. X in a row and Y in a column means X -> Y, not Y -> X."
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "What the App Compares",
                  tags$ul(
                    tags$li(strong("True Adjacency: "), "the uploaded or built-in signed ground-truth matrix."),
                    tags$li(strong("Inferred State-Level Adjacency: "), "the retained CIRN edges collapsed to base state-variable source and target names."),
                    tags$li(strong("State-level collapse: "), "feature-level edges such as X -> dY, dX -> dY, and d2X -> dY are compared as the base relation X -> Y."),
                    tags$li(strong("Sign comparison: "), "positive inferred omega is compared with true activation; negative inferred omega is compared with true inhibition."),
                    tags$li(strong("Wrong-sign rule: "), "an inferred activation where truth says inhibition, or inferred inhibition where truth says activation, is a serious error. In signed evaluation it can count as both a false positive and a false negative."),
                    tags$li(strong("No ground truth: "), "if no signed truth matrix is available, use Results, Diagnostics, and Sensitivity instead of benchmark metrics.")
                  ),
                  div(
                    class = "note-box",
                    "Benchmarking in this app is primarily a state-level signed adjacency comparison. Use feature-level CIRN figures separately to understand which representation produced an inferred relation."
                  )
                )
              )
            ),
            panel_box(
              "Benchmark Outputs",
              fluidRow(
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Benchmark controls"),
                    tags$ul(
                      tags$li(strong("Evaluate Against Ground Truth: "), "runs the benchmark comparison after CIRN has been fitted and a truth matrix is available."),
                      tags$li(strong("Metrics table: "), "summarizes signed recovery performance."),
                      tags$li(strong("Benchmark explanation: "), "reminds users that benchmarking is valid only with known signed truth and explains wrong-sign handling."),
                      tags$li(strong("If button errors: "), "check that CIRN has been run and that a truth matrix exists.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Heatmaps"),
                    tags$ul(
                      tags$li(strong("True Adjacency heatmap: "), "shows the signed ground-truth network. Green means true activation, red means true inhibition, and neutral means no true edge."),
                      tags$li(strong("Inferred State-Level Adjacency heatmap: "), "shows the signed network inferred by retained CIRN edges after feature-level collapse."),
                      tags$li(strong("Cell position: "), "row is source and column is target."),
                      tags$li(strong("Visual check: "), "the best case is matching color and sign in the same cells across the two heatmaps.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Metrics"),
                    tags$ul(
                      tags$li(strong("True positive: "), "CIRN inferred an edge with the correct source, target, and sign."),
                      tags$li(strong("False positive: "), "CIRN inferred an edge that is absent from truth or has the wrong sign."),
                      tags$li(strong("False negative: "), "truth contains an edge that CIRN missed or recovered with the wrong sign."),
                      tags$li(strong("True negative: "), "truth has no edge and CIRN also inferred no edge."),
                      tags$li(strong("Precision: "), "among inferred edges, the fraction that are correct."),
                      tags$li(strong("Recall / sensitivity: "), "among true edges, the fraction recovered correctly."),
                      tags$li(strong("Specificity: "), "among true non-edges, the fraction correctly left absent."),
                      tags$li(strong("F1 score: "), "single summary balancing precision and recall, when available.")
                    )
                  )
                )
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "How To Interpret Benchmark Results",
                  div(
                    class = "guide-good",
                    tags$h4("Stronger benchmark evidence"),
                    tags$ul(
                      tags$li("Most true signed edges are recovered with the correct sign."),
                      tags$li("Few extra inferred edges appear outside the truth matrix."),
                      tags$li("Wrong-sign recoveries are rare or absent."),
                      tags$li("Benchmark performance remains reasonable across sensitivity analysis."),
                      tags$li("Recovered edges also have good HDI, diagnostics, and mode-consistency support."),
                      tags$li("The truth matrix is from a simulation or a defensible reference network.")
                    )
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "When To Be Cautious",
                  div(
                    class = "guide-warning",
                    tags$h4("Weaker benchmark evidence"),
                    tags$ul(
                      tags$li("Many inferred edges are not in the truth matrix."),
                      tags$li("Many true edges are missed."),
                      tags$li("Wrong-sign recoveries occur."),
                      tags$li("The truth matrix is incomplete, uncertain, unsigned, or based on a rough literature diagram."),
                      tags$li("Node names do not match exactly and some true relations may not be compared."),
                      tags$li("Feature-level CIRN edges are being overinterpreted even though the benchmark is state-level.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "Ground Truth vs Topology Matrix",
              fluidRow(
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("Ground-truth signed adjacency"),
                    tags$ul(
                      tags$li("Use this for benchmark metrics."),
                      tags$li("Contains direction and sign: activation, inhibition, or absence."),
                      tags$li("Should represent a known or defensible true network."),
                      tags$li("Rows are sources and columns are targets.")
                    )
                  )
                ),
                column(
                  6,
                  div(
                    class = "guide-card",
                    tags$h4("Topology / structural adjacency"),
                    tags$ul(
                      tags$li("Use this as structural context, not signed truth unless signs are known."),
                      tags$li("Often indicates whether a connection is possible, without saying activation or inhibition."),
                      tags$li("Do not treat an unsigned topology matrix as a signed benchmark."),
                      tags$li("If you use topology only, report it qualitatively rather than as signed recovery performance.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "Benchmark Reporting Checklist",
              div(
                class = "guide-good",
                tags$ul(
                  tags$li("Source of the ground-truth matrix."),
                  tags$li("Whether the truth matrix is simulated, empirical, curated literature, or expert-defined."),
                  tags$li("Whether the matrix is signed and directed."),
                  tags$li("Matrix orientation: rows are sources, columns are targets."),
                  tags$li("Node-name matching procedure."),
                  tags$li("Whether benchmark comparison is state-level after feature-level CIRN collapse."),
                  tags$li("Metrics table values: true positives, false positives, false negatives, true negatives, and summary rates when available."),
                  tags$li("Wrong-sign recoveries, if any."),
                  tags$li("Any missing, uncertain, or excluded truth edges."),
                  tags$li("How benchmark results affect interpretation of the CIRN network.")
                )
              )
            ),
            panel_box(
              "Important Cautions",
              div(
                class = "guide-warning",
                tags$ul(
                  tags$li("Benchmark metrics are meaningful only when the truth matrix is meaningful."),
                  tags$li("A high benchmark score on one simulated system does not guarantee performance on all systems."),
                  tags$li("A low benchmark score may reflect poor data, insufficient sampling, preprocessing choices, model settings, or a truth matrix that does not match the observed variables."),
                  tags$li("If the empirical truth network is incomplete, false positives may include real but undocumented relations."),
                  tags$li("If the benchmark is state-level, do not claim that it separately validates X, dX, and d2X feature-level mechanisms."),
                  tags$li("Always interpret benchmark evidence together with Results, CIRN Figures, Diagnostics, and Sensitivity.")
                )
              )
            )
          ),
          tabPanel(
            "Export",
            panel_box(
              "What Export Is For",
              tags$p(
                "Export is the reproducibility step. It saves the data, settings, inferred edges, coefficients, diagnostics, sensitivity results, and benchmark results needed to understand and reproduce a CIRN Studio analysis."
              ),
              tags$p(
                "For non-specialist users: exporting is how you avoid the problem of having a nice figure but not remembering which data source, targets, predictors, smoothing, epsilon, Bayesian settings, or sensitivity scope produced it."
              ),
              div(
                class = "note-box",
                "For any analysis you plan to cite, present, include in a dissertation, or share with collaborators, export the complete ZIP bundle and settings JSON."
              )
            ),
            panel_box(
              "Recommended Export Workflow",
              tags$ol(
                tags$li(strong("Confirm active data and results: "), "make sure the app is showing the data source and fitted CIRN run you intend to save."),
                tags$li(strong("Check the Export Manifest: "), "verify which result types are available before downloading."),
                tags$li(strong("Download the Settings JSON: "), "save the exact analysis settings for reproducibility."),
                tags$li(strong("Download the Excel workbook: "), "use this as the most convenient multi-table record for review."),
                tags$li(strong("Download key CSV files: "), "save edges, all coefficients, and diagnostics as separate files if you will analyze them elsewhere."),
                tags$li(strong("Download the HTML report: "), "use this as a lightweight readable summary for sharing."),
                tags$li(strong("Download the Complete ZIP bundle: "), "use this as the archival copy containing the main CSV records, settings, workbook, and figure pack."),
                tags$li(strong("Store exports with date and dataset version: "), "keep exported files in a folder named for the analysis date, dataset, and run type.")
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "Download Buttons",
                  tags$ul(
                    tags$li(strong("Edges CSV: "), "exports retained signed CIRN edges across the fitted modes. Use this for the final edge list and network interpretation."),
                    tags$li(strong("All coefficients CSV: "), "exports all modeled coefficients, not only retained edges. Use this to inspect HDIs, non-retained terms, and uncertainty."),
                    tags$li(strong("Diagnostics CSV: "), "exports model diagnostics for fitted CIRN runs. Use this to document Rhat, ESS, jitter, class-balance-related fields, and model health."),
                    tags$li(strong("Excel workbook: "), "exports many related tables in one workbook, including selected data, derivatives, class balance, edges, consistency summaries, coefficients, diagnostics, VIF, optional RF, optional latent-Z, optional sensitivity, optional benchmark, and settings."),
                    tags$li(strong("Settings JSON: "), "exports the exact configuration used by the app. This is critical for reproducing the run."),
                    tags$li(strong("HTML report: "), "exports a lightweight readable summary of the analysis. Use it for quick sharing, but keep the workbook and settings JSON for full reproducibility."),
                    tags$li(strong("Complete ZIP bundle: "), "exports the main CSV records, settings JSON, workbook, and figure pack in one archive. Use this as the safest archival copy.")
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "Export Manifest and Reproducibility Panel",
                  tags$ul(
                    tags$li(strong("Export Manifest: "), "shows which output groups are currently available, such as selected data, derivative preview, CIRN edges, edge consistency, coefficients, diagnostics, VIF, RF support, latent-Z, sensitivity, benchmark, and settings."),
                    tags$li(strong("Available = TRUE: "), "that output exists in the current app session and can be included in relevant exports."),
                    tags$li(strong("Available = FALSE: "), "that output has not been generated yet. For example, RF support is unavailable until the RF support check is run."),
                    tags$li(strong("Reproducibility panel: "), "prints the current settings object so you can visually inspect the active configuration before downloading files."),
                    tags$li(strong("Why it matters: "), "two CIRN runs can differ if the data source, preprocessing, representation mode, lag, pairwise option, prior, seed, or sensitivity scope differs.")
                  ),
                  div(
                    class = "guide-warning",
                    "Before final export, confirm that the manifest marks the outputs you actually need as available."
                  )
                )
              )
            ),
            panel_box(
              "What To Save for Different Uses",
              fluidRow(
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Quick exploration"),
                    tags$ul(
                      tags$li("Edges CSV."),
                      tags$li("All coefficients CSV."),
                      tags$li("Diagnostics CSV."),
                      tags$li("Settings JSON."),
                      tags$li("Optional HTML report for easy reading.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Collaboration"),
                    tags$ul(
                      tags$li("Excel workbook for multi-table review."),
                      tags$li("HTML report for a readable summary."),
                      tags$li("Settings JSON so collaborators can see exact settings."),
                      tags$li("Complete ZIP bundle if collaborators need all records.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Dissertation or publication"),
                    tags$ul(
                      tags$li("Complete ZIP bundle."),
                      tags$li("Settings JSON."),
                      tags$li("Excel workbook."),
                      tags$li("Edges and all coefficients CSVs."),
                      tags$li("Diagnostics, sensitivity, and benchmark tables when used."),
                      tags$li("Final figures from Results or CIRN Figures.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "Minimum Reporting Checklist",
              fluidRow(
                column(
                  6,
                  div(
                    class = "guide-good",
                    tags$h4("Always report"),
                    tags$ul(
                      tags$li("Data source and date/version of dataset."),
                      tags$li("Active source type: uploaded, example, simulation, or equation-built data."),
                      tags$li("Time column, targets, and allowed base predictors."),
                      tags$li("Preprocessing choices: normalization, smoothing method, smoothing value if manual, derivative grid, outlier action, MAD threshold, epsilon, and jitter settings."),
                      tags$li("Representation mode and pairwise setting."),
                      tags$li("Lag units."),
                      tags$li("Bayesian settings: prior mean, prior SD, total iterations, warmup, chains, cores, adapt_delta, seed, and LOO choice."),
                      tags$li("Retained-edge rule: 95% HDI excludes zero.")
                    )
                  )
                ),
                column(
                  6,
                  div(
                    class = "guide-good",
                    tags$h4("Report when used"),
                    tags$ul(
                      tags$li("Diagnostics: class balance, VIF/correlation, Rhat/ESS, sampler warnings, jitter diagnostics, RF support, and latent-Z screening."),
                      tags$li("Sensitivity settings: inference scope, replicates, noise fractions, lags, downsampling, sample sizes, missing-row fractions, completed runs, and feature-level stability."),
                      tags$li("Benchmark settings: truth matrix source, source-target orientation, state-level comparison, and benchmark metrics."),
                      tags$li("Mode consistency: whether edges were retained in sublevel, all-predictor, pairwise, or consistency views."),
                      tags$li("The CIRN Studio version and relevant release notes if an app correction or version change affected the analysis.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "File Interpretation Guide",
              fluidRow(
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Core inference files"),
                    tags$ul(
                      tags$li(strong("CIRN_edges.csv / edges.csv: "), "retained signed edges used for network interpretation."),
                      tags$li(strong("CIRN_all_coefficients.csv / all_coefficients.csv: "), "full coefficient table with retained and non-retained terms."),
                      tags$li(strong("diagnostics.csv: "), "Bayesian and model diagnostics."),
                      tags$li(strong("vif.csv and vif_pairs.csv: "), "collinearity diagnostics."),
                      tags$li(strong("mode_consistency.csv and edge_consistency.csv: "), "mode agreement and edge-consistency summaries.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Data and preprocessing files"),
                    tags$ul(
                      tags$li(strong("selected_data.csv: "), "the active dataset used by the app."),
                      tags$li(strong("derivatives.csv: "), "derivative table generated during preprocessing."),
                      tags$li(strong("class_balance.csv: "), "increasing, decreasing, blank, usable, and minority-class counts."),
                      tags$li(strong("settings.json: "), "full app settings needed to reproduce the run."),
                      tags$li(strong("CIRN_Studio_Workbook.xlsx: "), "multi-sheet workbook collecting the main tables.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("Optional evidence files"),
                    tags$ul(
                      tags$li(strong("rf_support.csv: "), "Random Forest support diagnostics, available only after RF support check."),
                      tags$li(strong("latent_z.csv: "), "latent-Z screening results, available only after latent-Z screening."),
                      tags$li(strong("sensitivity_runs.csv: "), "scenario-level sensitivity run records."),
                      tags$li(strong("feature_edge_stability.csv: "), "feature-level sensitivity stability preserving X, dX, and d2X-style labels."),
                      tags$li(strong("state_edge_stability.csv: "), "state-level sensitivity stability after collapsing derivative representations."),
                      tags$li(strong("benchmark table: "), "ground-truth comparison metrics, available only after benchmark evaluation.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "Feedback",
              tags$p(
                "Use the Feedback tab to report bugs, confusing behavior, feature requests, documentation questions, or scientific-method questions directly to the CIRN authors."
              ),
              tags$ul(
                tags$li(strong("Feedback report: "), "a human-readable record of one submitted issue or suggestion."),
                tags$li(strong("Feedback CSV record: "), "a structured record that can be collected across users."),
                tags$li(strong("Email draft: "), "opens an addressed message containing the report after a one-sentence summary is provided. CIRN Studio does not send it automatically; review the draft and click Send in your email application."),
                tags$li(strong("Best practice: "), "include clear reproduction steps, the affected app area, and downloaded context when reporting a problem.")
              )
            ),
            panel_box(
              "Important Cautions",
              div(
                class = "guide-warning",
                tags$ul(
                  tags$li("An exported figure without settings is not reproducible."),
                  tags$li("A CSV file alone may not contain enough context to reproduce the run."),
                  tags$li("If sensitivity, RF, latent-Z, or benchmark outputs are unavailable, they were not run in the current session."),
                  tags$li("Do not rename or edit exported columns before archiving the original files."),
                  tags$li("For final work, save a complete bundle before changing parameters or starting a new analysis."),
                  tags$li("If you compare multiple runs, keep each run in a separate dated folder with its own settings JSON.")
                )
              )
            )
          ),
          tabPanel(
            "Troubleshooting",
            panel_box(
              "How To Troubleshoot CIRN Studio",
              tags$p(
                "Troubleshooting means identifying whether the problem comes from the active data source, selected variables, preprocessing, model settings, app state, computational cost, or interpretation of results."
              ),
              tags$ol(
                tags$li(strong("First check the active source badge: "), "confirm the app is using the uploaded, example, simulated, or equation-built data you intend to analyze."),
                tags$li(strong("Then check the workflow status: "), "Data, EDA, Preprocess, Run CIRN Algorithm, Interpret, and Export should progress in order."),
                tags$li(strong("Read warnings before rerunning: "), "quality warnings, class balance warnings, run log messages, and diagnostics usually explain why a result looks unusual."),
                tags$li(strong("Change one setting at a time: "), "if you adjust smoothing, epsilon, lag, prior, or pairwise settings all at once, it becomes hard to know what fixed or caused the issue."),
                tags$li(strong("Save evidence for bugs: "), "record the data source, targets, predictors, settings JSON, screenshot, run log, and steps to reproduce.")
              ),
              div(
                class = "note-box",
                "Most problems can be diagnosed by asking: Am I using the intended data? Are the target response classes usable? Did CIRN actually run? Are the requested outputs available?"
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "Data and Workflow Problems",
                  tags$details(
                    tags$summary("I simulated data, but EDA still shows Predator_Prey.csv."),
                    tags$p("Click Simulate and Use Data in Data Source > Simulation Lab. Then confirm the Active source badge says Simulation Lab output. If the badge still shows Built-in example or Uploaded data, open Data Source > Example or Uploaded Data and check the selected source. Restart the app if you may be seeing an older cached session.")
                  ),
                  tags$details(
                    tags$summary("I loaded an example, but it looks like I did not upload anything."),
                    tags$p("That is expected. The Use Predator-Prey Example button loads the app's bundled data, so no file upload is needed. Confirm the Active source badge says Built-in example and check the row count and variable names.")
                  ),
                  tags$details(
                    tags$summary("The app is using the wrong dataset."),
                    tags$p("Go back to Start or Data Source and choose the intended source: the guided Predator-Prey example, Upload Data, Simulate System, or Build From Equations. Then confirm the Active source badge in EDA, Preprocess, and Run CIRN Algorithm.")
                  ),
                  tags$details(
                    tags$summary("My uploaded file does not look correct."),
                    tags$p("Check that the file is wide-format time-series data: one time column named t or another selected time column, plus one column per state variable. Do not upload long-format data with variable names stacked in one column unless you reshape it first.")
                  ),
                  tags$details(
                    tags$summary("The time column is wrong."),
                    tags$p("Open Data Source > Example or Uploaded Data and select the correct time column. For CIRN examples and many simulations, the time column is usually lowercase t. The time column should be ordered and numeric.")
                  ),
                  tags$details(
                    tags$summary("The workflow says something is pending."),
                    tags$p("A pending badge usually means the step has not been completed yet. For example, CIRN not fitted means Run CIRN Algorithm has not completed in the current session. Follow the workflow cards from Data to EDA to Preprocess to Run CIRN Algorithm.")
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "Preprocess and Run Problems",
                  tags$details(
                    tags$summary("The Run CIRN Algorithm button is not ready."),
                    tags$p("CIRN needs active data, a valid time column, selected targets, selected predictors, usable derivative response classes, and valid Bayesian settings. Check Data selections, Preprocess class balance, and whether warmup is smaller than total iterations.")
                  ),
                  tags$details(
                    tags$summary("The app says no retained edges."),
                    tags$p("This may be a valid result. Check class balance, usable sample size, smoothing, epsilon, prior settings, and whether HDIs include zero.")
                  ),
                  tags$details(
                    tags$summary("The response class balance looks poor."),
                    tags$p("A target may have too few increasing or decreasing derivative cases. Inspect smoothing, epsilon, outliers, and whether the trajectory is mostly monotone or nearly flat. Adaptive jitter may help, but jitter-assisted results should be reported cautiously.")
                  ),
                  tags$details(
                    tags$summary("The derivative preview looks too jagged."),
                    tags$p("Inspect outliers and time gaps. Try GCV smoothing, increase manual smoothing if GCV is off, or reduce extreme outlier influence. Jagged derivatives can create artificial sign changes.")
                  ),
                  tags$details(
                    tags$summary("The derivative preview looks too flat."),
                    tags$p("Smoothing may be too strong, the variable may have little variation, or epsilon may be too large. Reduce manual smoothing if applicable and inspect the EDA trajectory.")
                  ),
                  tags$details(
                    tags$summary("RF support or latent-Z buttons cannot be used."),
                    tags$p("Run CIRN Algorithm first. RF support and latent-Z screening are optional diagnostics that depend on fitted CIRN outputs. After they run, read their tables under Diagnostics.")
                  ),
                  tags$details(
                    tags$summary("The app is slow during sensitivity analysis."),
                    tags$p("Sensitivity reruns CIRN many times. Reduce replicates, fewer scenarios, or use Sublevel only before trying Everything.")
                  ),
                  tags$details(
                    tags$summary("The app exits or RStudio becomes unstable."),
                    tags$p("Restart R, reduce model_cores, reduce chains, disable pairwise/everything modes, and try a smaller preset first.")
                  )
                )
              )
            ),
            fluidRow(
              column(
                6,
                panel_box(
                  "Plot and Output Questions",
                  tags$details(
                    tags$summary("Interactive Combined looks different from static figures."),
                    tags$p("Interactive Combined is for exploration and layout can move. Use the static Sublevel, All predictors, Pairwise, and consistency figures for publication-style reporting.")
                  ),
                  tags$details(
                    tags$summary("A plot or table is missing."),
                    tags$p("Most plots and tables require a previous step. CIRN figures and coefficient HDI plots require Run CIRN Algorithm. RF tables require Run RF Support Check. Latent-Z tables require Run Latent-Z Screening. Sensitivity plots require Run Sensitivity. Benchmark plots require a truth matrix and benchmark evaluation.")
                  ),
                  tags$details(
                    tags$summary("The network figure has no edges."),
                    tags$p("No retained edges were available for that selected mode or consistency filter. Check Results, coefficient HDIs, and whether you ran the mode you are viewing.")
                  ),
                  tags$details(
                    tags$summary("Consistent all 3 is empty."),
                    tags$p("This means no edge was retained with the same sign across all three sources being compared. It is not necessarily an error. Check Consistent >=2, Sublevel, All predictors, and Pairwise separately.")
                  ),
                  tags$details(
                    tags$summary("The colors confuse me."),
                    tags$p("Green means activation or increasing class. Red means inhibition or decreasing class. Grey usually means blank, absent, or not detected. Purple in sensitivity means mixed sign across runs.")
                  ),
                  tags$details(
                    tags$summary("I cannot interpret edge thickness."),
                    tags$p("In CIRN network figures, thicker arrows correspond to larger absolute omega values. Thickness shows fitted coefficient magnitude, not certainty by itself. Always read HDIs and diagnostics too.")
                  )
                )
              ),
              column(
                6,
                panel_box(
                  "Result Questions",
                  tags$details(
                    tags$summary("Why do pairwise and multivariable results differ?"),
                    tags$p("Pairwise fits one regulator-target relation at a time. Multivariable modes condition on multiple predictors. Differences can reveal confounding, redundancy, collinearity, or support that is not stable across model contexts.")
                  ),
                  tags$details(
                    tags$summary("What if RF support disagrees with Bayesian CIRN?"),
                    tags$p("Bayesian CIRN remains the signed inference model. RF is a nonlinear support diagnostic and should be reported as additional evidence, not as a replacement for HDI-based retention.")
                  ),
                  tags$details(
                    tags$summary("What if latent-Z screening shows a gain?"),
                    tags$p("Treat it as exploratory evidence that hidden collective structure may improve classification. It does not identify a specific hidden regulator by itself.")
                  ),
                  tags$details(
                    tags$summary("What if sensitivity shows mixed signs?"),
                    tags$p("Mixed signs mean the same feature-target relation is not sign-stable under perturbations. Report this caution and avoid strong mechanistic claims for that edge.")
                  ),
                  tags$details(
                    tags$summary("What if benchmark results look poor?"),
                    tags$p("Check whether the truth matrix is valid, signed, directed, correctly oriented, and named consistently with the data. Poor benchmark performance can also reflect sparse data, preprocessing choices, lag choice, or a system where the available time series does not contain enough information.")
                  ),
                  tags$details(
                    tags$summary("What if an edge appears only in pairwise mode?"),
                    tags$p("Pairwise evidence means the relation appears when tested alone. If it disappears in multivariable modes, other predictors may explain the same signal. Treat pairwise-only edges as tentative support rather than final multivariable evidence.")
                  ),
                  tags$details(
                    tags$summary("What if all-predictor mode disagrees with sublevel mode?"),
                    tags$p("All-predictor mode lets all candidate features compete together, so collinearity and redundancy can change which edges are retained. Check VIF, coefficient HDIs, and mode consistency before deciding how strongly to interpret the edge.")
                  ),
                  tags$details(
                    tags$summary("What should I cite or save after a serious run?"),
                    tags$p("Export the ZIP bundle, settings JSON, report, edge tables, diagnostics, sensitivity scope, sensitivity stability tables, and benchmark files if applicable.")
                  )
                )
              )
            ),
            panel_box(
              "Performance and Stability Fixes",
              fluidRow(
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("If the app is slow"),
                    tags$ul(
                      tags$li("Use Fast exploration or Teaching demo first."),
                      tags$li("Reduce total iterations and chains for testing."),
                      tags$li("Turn off LOO diagnostics during exploratory runs."),
                      tags$li("Disable pairwise CIRN for large systems."),
                      tags$li("Use Sublevel only for sensitivity before trying Everything."),
                      tags$li("Reduce sensitivity replicates and scenario values.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("If results look wrong"),
                    tags$ul(
                      tags$li("Confirm active source badge."),
                      tags$li("Confirm time column, targets, and predictors."),
                      tags$li("Check processed preview and derivative preview."),
                      tags$li("Check class balance and blank counts."),
                      tags$li("Check whether adaptive jitter was used."),
                      tags$li("Compare sublevel, all-predictor, and pairwise modes.")
                    )
                  )
                ),
                column(
                  4,
                  div(
                    class = "guide-card",
                    tags$h4("If RStudio freezes"),
                    tags$ul(
                      tags$li("Stop the app and restart R."),
                      tags$li("Use fewer cores than your maximum CPU count."),
                      tags$li("Reduce chains and iterations."),
                      tags$li("Run fewer targets or predictors first."),
                      tags$li("Avoid Everything sensitivity mode until smaller plans succeed."),
                      tags$li("Save exports before trying heavier runs.")
                    )
                  )
                )
              )
            ),
            panel_box(
              "What To Include in a Feedback Report",
              tags$p(
                "Use the Feedback tab when you encounter bugs, confusing behavior, incorrect-looking results, missing outputs, or suggestions for improvement."
              ),
              div(
                class = "guide-good",
                tags$ul(
                  tags$li("Feedback type and area where it happened."),
                  tags$li("One-sentence summary of the problem."),
                  tags$li("Steps to reproduce the issue."),
                  tags$li("Expected behavior and actual behavior."),
                  tags$li("Screenshot, if available."),
                  tags$li("Active data source and whether the data were uploaded, example, simulated, or equation-built."),
                  tags$li("Targets, predictors, representation mode, pairwise setting, and analysis preset."),
                  tags$li("Whether Run CIRN, RF support, latent-Z, sensitivity, or benchmark had been run."),
                  tags$li("Settings JSON and exported ZIP bundle for serious bugs.")
                )
              )
            ),
            panel_box(
              "Important Cautions",
              div(
                class = "guide-warning",
                tags$ul(
                  tags$li("Do not keep rerunning with random changes until you get the network you expect. Diagnose the data, preprocessing, and settings first."),
                  tags$li("A software warning is not automatically a scientific failure, but it must be understood before reporting results."),
                  tags$li("No retained edges can be a valid scientific result."),
                  tags$li("Conflicting modes are informative; report them instead of hiding them."),
                  tags$li("For public or dissertation results, save the complete ZIP bundle before changing settings or restarting the app.")
                )
              )
            )
          )
        )
      )
    ),

    navbarMenu(
      "Data Source",
      tabPanel(
        "Example or Uploaded Data",
        value = "Data",
        fluidPage(
        contextual_tab_header(
          "Example or Uploaded Data",
          "Load the bundled demonstration or your own observed time series, assign each column's analysis role, and optionally supply reference network matrices.",
          "The bundled dataset or a wide-format CSV/Excel table containing one ordered time column and at least one numeric state-variable column. Signed ground-truth and structural adjacency matrices are optional.",
          "A validated active time series with assigned time, target, and predictor columns, initial quality checks, and any reference matrices ready for downstream analysis."
        ),
        fluidRow(
          column(
            3,
            panel_box(
              "Input Data",
              conditionalPanel(
                "input.data_mode == 'upload'",
                fileInput("data_file", "Upload CSV or Excel data", accept = c(".csv", ".xlsx", ".xls"))
              ),
              conditionalPanel(
                "input.data_mode == 'example'",
                div(class = "note-box", "Using the built-in Predator_Prey.csv demonstration from the app data folder.")
              ),
              conditionalPanel(
                "input.data_mode == 'simulation'",
                div(class = "note-box", "Simulation data comes from Data Source > Simulation Lab.")
              ),
              uiOutput("active_data_source_ui"),
              selectInput("time_col", "Time column", choices = character()),
              checkboxGroupInput("targets", "Targets to model", choices = character()),
              checkboxGroupInput("predictors", "Allowed base predictors", choices = character())
            ),
            panel_box(
              "Optional Matrices",
              fileInput("truth_file", "Ground-truth signed adjacency", accept = c(".csv", ".xlsx", ".xls")),
              fileInput("topology_file", "Topology/structural adjacency", accept = c(".csv", ".xlsx", ".xls")),
              help_line("Rows are source nodes, columns are target nodes. Values may be -1, 0, +1, or weighted.")
            )
          ),
          column(
            9,
            panel_box("Active Data Source", uiOutput("active_data_source_panel")),
            panel_box("Data Preview", DTOutput("data_preview")),
            fluidRow(
              column(6, panel_box("Missingness", plotOutput("missing_plot", height = "330px"))),
              column(6, panel_box("Time Spacing", DTOutput("time_spacing_table")))
            ),
            panel_box("Variable Summary", DTOutput("variable_summary_table")),
            panel_box("Uploaded Matrix Preview", tabsetPanel(
              tabPanel("Ground truth", DTOutput("truth_preview")),
              tabPanel("Topology", DTOutput("topology_preview"))
            ))
          )
        )
        )
      ),

      tabPanel(
        "Simulation Lab",
        fluidPage(
        contextual_tab_header(
          "Simulation Lab",
          "Generate synthetic time-series data from a built-in dynamical system or user-entered differential equations, then use the simulation as the active CIRN data source.",
          "A selected system or derivative equations, initial conditions, parameter values, a time grid, and a random seed. Noise, missing values, irregular sampling, and outliers are optional.",
          "A simulated time-series table and, for supported built-in systems, a known signed network that can serve as benchmark ground truth."
        ),
        fluidRow(
          column(
            3,
            panel_box(
              "System",
              selectInput(
                "sim_system",
                "System type",
                choices = c(
                  "Predator-prey" = "predator_prey",
                  "SIR epidemic" = "sir",
                  "Competition" = "competition",
                  "Mutualism" = "mutualism",
                  "Coupled oscillator" = "oscillator",
                  "User-defined dynamical system" = "custom"
                )
              ),
              sliderInput("sim_t_start", "Start time", min = 0, max = 50, value = 0, step = 1),
              sliderInput("sim_t_end", "End time", min = 10, max = 500, value = 100, step = 5),
              sliderInput("sim_dt", "Time step", min = 0.01, max = 5, value = 1, step = 0.01),
              sliderInput("sim_seed", "Simulation seed", min = 1, max = 9999, value = 123, step = 1)
            ),
            panel_box(
              "Observation Artifacts",
              sliderInput("sim_noise", "Observation noise fraction", min = 0, max = 0.5, value = 0, step = 0.01),
              sliderInput("sim_missing", "Missing cell fraction", min = 0, max = 0.5, value = 0, step = 0.01),
              sliderInput("sim_irregular", "Irregular sampling strength", min = 0, max = 0.75, value = 0, step = 0.01),
              sliderInput("sim_outlier_fraction", "Outlier cell fraction", min = 0, max = 0.2, value = 0, step = 0.005),
              sliderInput("sim_outlier_size", "Outlier size multiplier", min = 1, max = 10, value = 4, step = 0.5),
              actionButton(
                "simulate_btn",
                button_label(icon("play"), "Simulate and Use Data"),
                class = "btn-primary studio-action-button section-action-button"
              ),
              help_line("This makes the simulated time series the active data source for EDA, Preprocess, Run CIRN Algorithm, Sensitivity, Benchmark, and Export.")
            )
          ),
          column(
            9,
            panel_box(
              "System Parameters and Equation Input",
              div(
                class = "note-box",
                strong("To input your own dynamical system: "),
                "choose User-defined dynamical system, then enter variables, initial values, parameters, and one equation for each derivative."
              ),
              fluidRow(
                class = "simulation-builder-layout",
                column(
                  7,
                  class = "simulation-parameter-column",
                  conditionalPanel(
                    "input.sim_system == 'predator_prey'",
                    sliderInput("pp_x1", "Initial prey x1", min = 0.1, max = 10, value = 2, step = 0.1),
                    sliderInput("pp_x2", "Initial predator x2", min = 0.1, max = 10, value = 2, step = 0.1),
                    sliderInput("pp_a", "Prey growth a", min = 0.01, max = 3, value = 1.2, step = 0.01),
                    sliderInput("pp_b", "Predation b", min = 0.01, max = 3, value = 0.6, step = 0.01),
                    sliderInput("pp_c", "Predator gain c", min = 0.01, max = 3, value = 0.35, step = 0.01),
                    sliderInput("pp_d", "Predator mortality d", min = 0.01, max = 3, value = 0.8, step = 0.01)
                  ),
                  conditionalPanel(
                    "input.sim_system == 'sir'",
                    sliderInput("sir_S", "Initial susceptible S", min = 1, max = 1000, value = 990, step = 1),
                    sliderInput("sir_I", "Initial infected I", min = 1, max = 500, value = 10, step = 1),
                    sliderInput("sir_R", "Initial recovered R", min = 0, max = 500, value = 0, step = 1),
                    sliderInput("sir_N", "Population N", min = 100, max = 2000, value = 1000, step = 10),
                    sliderInput("sir_beta", "Transmission beta", min = 0.01, max = 2, value = 0.35, step = 0.01),
                    sliderInput("sir_gamma", "Recovery gamma", min = 0.01, max = 1, value = 0.08, step = 0.01)
                  ),
                  conditionalPanel(
                    "input.sim_system == 'competition'",
                    sliderInput("comp_x1", "Initial x1", min = 0.1, max = 100, value = 20, step = 0.1),
                    sliderInput("comp_x2", "Initial x2", min = 0.1, max = 100, value = 15, step = 0.1),
                    sliderInput("comp_r1", "Growth r1", min = 0.01, max = 2, value = 0.5, step = 0.01),
                    sliderInput("comp_r2", "Growth r2", min = 0.01, max = 2, value = 0.45, step = 0.01),
                    sliderInput("comp_K1", "Carrying capacity K1", min = 1, max = 200, value = 80, step = 1),
                    sliderInput("comp_K2", "Carrying capacity K2", min = 1, max = 200, value = 70, step = 1),
                    sliderInput("comp_a12", "Effect x2 on x1", min = 0, max = 3, value = 0.8, step = 0.05),
                    sliderInput("comp_a21", "Effect x1 on x2", min = 0, max = 3, value = 1.1, step = 0.05)
                  ),
                  conditionalPanel(
                    "input.sim_system == 'mutualism'",
                    sliderInput("mut_x1", "Initial x1", min = 0.1, max = 100, value = 10, step = 0.1),
                    sliderInput("mut_x2", "Initial x2", min = 0.1, max = 100, value = 12, step = 0.1),
                    sliderInput("mut_r1", "Growth r1", min = 0.01, max = 2, value = 0.25, step = 0.01),
                    sliderInput("mut_r2", "Growth r2", min = 0.01, max = 2, value = 0.22, step = 0.01),
                    sliderInput("mut_K1", "Carrying capacity K1", min = 1, max = 200, value = 80, step = 1),
                    sliderInput("mut_K2", "Carrying capacity K2", min = 1, max = 200, value = 70, step = 1),
                    sliderInput("mut_m12", "Mutual benefit x2 to x1", min = 0, max = 2, value = 0.08, step = 0.01),
                    sliderInput("mut_m21", "Mutual benefit x1 to x2", min = 0, max = 2, value = 0.07, step = 0.01)
                  ),
                  conditionalPanel(
                    "input.sim_system == 'oscillator'",
                    sliderInput("osc_x1", "Initial x1", min = -10, max = 10, value = 2, step = 0.1),
                    sliderInput("osc_x2", "Initial x2", min = -10, max = 10, value = 0, step = 0.1),
                    sliderInput("osc_alpha", "alpha", min = 0.01, max = 3, value = 1, step = 0.01),
                    sliderInput("osc_beta", "beta", min = 0.01, max = 3, value = 1, step = 0.01)
                  ),
                  conditionalPanel(
                    "input.sim_system == 'custom'",
                    textInput("custom_vars", "Variables", value = "x1,x2"),
                    textAreaInput("custom_initials", "Initial values", value = "x1=2\nx2=2", rows = 3),
                    textAreaInput("custom_params", "Parameters", value = "a=1.2\nb=0.6\nc=0.35\nd=0.8", rows = 5),
                    textAreaInput("custom_equations", "Equations: left side is the variable, right side is d(variable)/dt", value = "x1 = a*x1 - b*x1*x2\nx2 = c*x1*x2 - d*x2", rows = 5)
                  )
                ),
                column(
                  5,
                  class = "simulation-equation-column",
                  uiOutput("simulation_equation_panel")
                )
              )
            ),
            fluidRow(
              column(8, panel_box("Simulated Time Series", plotOutput("simulation_plot", height = "430px"))),
              column(4, panel_box("Built-in True Network", visNetworkOutput("simulation_truth_network", height = "430px")))
            ),
            panel_box("Simulated Data", DTOutput("simulation_table"))
          )
          )
        )
      )
    ),

    tabPanel(
      "EDA",
      fluidPage(
        contextual_tab_header(
          "Exploratory Data Analysis",
          "Examine data quality, temporal behavior, variable relationships, and system dynamics before constructing CIRN features. This step is optional but strongly recommended.",
          "An active raw time series and selected variables, together with exploratory settings for lags, rolling windows, state-space axes, and robust outlier screening.",
          "Exploratory plots and statistical screens describing missingness, sampling, distributions, lags, phase-space structure, and derivative behavior. These findings guide analysis choices but do not constitute retained CIRN edges."
        ),
        div(
          class = "eda-workspace",
          panel_box(
            "EDA Workspace Setup",
            div(
              class = "eda-controls-layout",
              tags$section(
                class = "eda-control-section eda-control-data",
                tags$h4("Data and variables"),
                tags$p("Confirm the active source and choose which state variables appear in the exploratory views."),
                uiOutput("active_data_source_eda_ui"),
                checkboxGroupInput("eda_vars", "Variables to display", choices = character(), inline = TRUE)
              ),
              tags$section(
                class = "eda-control-section eda-control-axes",
                tags$h4("State-space axes"),
                tags$p("Choose the two variables shown in the focused Phase Plane view."),
                div(
                  class = "eda-axis-grid",
                  selectInput("phase_x", "Horizontal variable", choices = character()),
                  selectInput("phase_y", "Vertical variable", choices = character())
                )
              ),
              tags$section(
                class = "eda-control-section eda-control-temporal",
                tags$h4("Temporal settings"),
                tags$p("Set the lag horizon and rolling window used by temporal-association plots and screens."),
                div(
                  class = "eda-temporal-grid",
                  div(
                    class = "eda-temporal-control",
                    sliderInput("eda_lag_max", "Maximum lag", min = 1, max = 30, value = 8, step = 1)
                  ),
                  div(
                    class = "eda-temporal-control",
                    sliderInput("eda_rolling_window", "Rolling window size", min = 3, max = 60, value = 15, step = 1)
                  ),
                  div(
                    class = "eda-temporal-control",
                    sliderInput("eda_granger_lag", "Granger screen lag order", min = 1, max = 10, value = 2, step = 1)
                  )
                )
              ),
              tags$section(
                class = "eda-control-section eda-control-quality",
                tags$h4("Quality screening"),
                tags$p("Adjust the robust threshold used to flag potentially unusual observations."),
                sliderInput("eda_outlier_thresh", "MAD outlier threshold", min = 1, max = 8, value = 3.5, step = 0.1)
              )
            ),
            div(
              class = "eda-controls-footer",
              div(
                class = "eda-screening-caution",
                strong("Interpretation boundary: "),
                "EDA describes the active time series; it does not establish regulatory causality."
              ),
              div(
                class = "eda-recommended-order",
                strong("Recommended exploration order"),
                tags$span("Data Quality"),
                tags$span(class = "eda-order-arrow", "→"),
                tags$span("Trajectories & Distributions"),
                tags$span(class = "eda-order-arrow", "→"),
                tags$span("Relationships & Lags"),
                tags$span(class = "eda-order-arrow", "→"),
                tags$span("System Dynamics"),
                tags$span(class = "eda-order-arrow", "→"),
                tags$span("Derivative Diagnostics")
              )
            ),
            class = "studio-panel eda-controls-panel"
          ),
            panel_box(
              "Exploratory Data Analysis Plot Gallery",
              div(
                class = "eda-scope-banner",
                strong("Exploratory scope. "),
                "Exploratory data analysis is not part of the core CIRN inference algorithm. ",
                "Use these views to characterize data quality and system dynamics before preprocessing and fitting. They generate insights and hypotheses, but not retained CIRN edges."
              ),
              div(
                class = "eda-gallery-navigation",
                div(
                  class = "eda-gallery-navigation-heading",
                  tags$span(class = "eda-gallery-navigation-eyebrow", "EDA analysis categories"),
                  tags$strong("Choose a category to reveal its available plots and diagnostics.")
                ),
                tabsetPanel(
                  id = "eda_gallery_category",
                tabPanel(
                  "Data Quality",
                  div(
                    class = "eda-gallery-category",
                    tags$p(
                      class = "eda-gallery-category-note",
                      "Begin here to verify completeness, sampling, unusual observations, and variables that need attention."
                    ),
                    div(class = "eda-gallery-subnav-label", "Available plots in this category"),
                    tabsetPanel(
                      type = "pills",
                      tabPanel("Raw Small Multiples", plotOutput("eda_raw_small_multiples_plot", height = "520px")),
                      tabPanel("Variable Attention", plotOutput("eda_variable_attention_plot", height = "520px")),
                      tabPanel("Time Gaps", plotOutput("eda_time_gap_plot", height = "380px")),
                      tabPanel("Missingness Map", plotOutput("eda_missing_plot", height = "430px")),
                      tabPanel("Missingness Summary", plotOutput("eda_missing_summary_plot", height = "430px")),
                      tabPanel("Outlier Timeline", plotOutput("eda_outlier_timeline_plot", height = "520px"))
                    )
                  )
                ),
                tabPanel(
                  "Trajectories & Distributions",
                  div(
                    class = "eda-gallery-category",
                    tags$p(
                      class = "eda-gallery-category-note",
                      "Inspect temporal shape, scale, local variation, and marginal distributions before studying interactions."
                    ),
                    div(class = "eda-gallery-subnav-label", "Available plots in this category"),
                    tabsetPanel(
                      type = "pills",
                      tabPanel("Processed Time Series", plotOutput("eda_time_plot", height = "460px")),
                      tabPanel("Normalized Overlay", plotOutput("eda_normalized_overlay_plot", height = "460px")),
                      tabPanel("Rolling Mean/Variance", plotOutput("eda_rolling_mean_variance_plot", height = "620px")),
                      tabPanel("Distributions", plotOutput("eda_distribution_plot", height = "520px"))
                    )
                  )
                ),
                tabPanel(
                  "Relationships & Lags",
                  div(
                    class = "eda-gallery-category",
                    tags$p(
                      class = "eda-gallery-category-note",
                      "Screen same-time association, changing relationships, serial dependence, and possible temporal offsets."
                    ),
                    div(class = "eda-gallery-subnav-label", "Available plots in this category"),
                    tabsetPanel(
                      type = "pills",
                      tabPanel("Correlation Heatmap", plotOutput("corr_plot", height = "460px")),
                      tabPanel("Pairwise Scatter", plotOutput("eda_pairwise_scatter_plot", height = "620px")),
                      tabPanel("Rolling Correlation", plotOutput("eda_rolling_correlation_plot", height = "620px")),
                      tabPanel("Lagged Correlation", plotOutput("eda_lagged_corr_plot", height = "620px")),
                      tabPanel("Lead-Lag Cross-Correlation", plotOutput("eda_cross_correlation_plot", height = "620px")),
                      tabPanel("ACF / PACF", plotOutput("eda_acf_pacf_plot", height = "540px"))
                    )
                  )
                ),
                tabPanel(
                  "System Dynamics",
                  div(
                    class = "eda-gallery-category",
                    tags$p(
                      class = "eda-gallery-category-note",
                      "Explore state-space movement, recurrence, memory, local flow, and multivariate transitions. Arrowheads mark time direction where applicable."
                    ),
                    div(class = "eda-gallery-subnav-label", "Available plots in this category"),
                    tabsetPanel(
                      type = "pills",
                      tabPanel("Phase Plane", plotOutput("phase_plot", height = "460px")),
                      tabPanel("Pairwise Phase Portraits", plotOutput("eda_phase_pairs_plot", height = "620px")),
                      tabPanel("Directed Phase Portraits", plotOutput("eda_directed_phase_plot", height = "720px")),
                      tabPanel("Derivative Vector Field", plotOutput("eda_vector_field_plot", height = "720px")),
                      tabPanel("Delay Embedding", plotOutput("eda_delay_embedding_plot", height = "720px")),
                      tabPanel("Recurrence Map", plotOutput("eda_recurrence_plot", height = "720px")),
                      tabPanel("Estimated Nullclines", plotOutput("eda_nullcline_plot", height = "720px")),
                      tabPanel("PCA Trajectory", plotOutput("eda_pca_plot", height = "520px"))
                    )
                  )
                ),
                tabPanel(
                  "Derivative Diagnostics",
                  div(
                    class = "eda-gallery-category",
                    tags$p(
                      class = "eda-gallery-category-note",
                      "Examine rate-of-change structure, derivative representations, response balance, and the effect of the epsilon threshold."
                    ),
                    div(class = "eda-gallery-subnav-label", "Available plots in this category"),
                    tabsetPanel(
                      type = "pills",
                      tabPanel("Derivative vs State", plotOutput("eda_derivative_state_plot", height = "620px")),
                      tabPanel("Derivative Phase Portraits", plotOutput("eda_derivative_phase_plot", height = "620px")),
                      tabPanel("Cross-Derivative Phase Portraits", plotOutput("eda_cross_derivative_phase_plot", height = "720px")),
                      tabPanel("Derivative Distributions", plotOutput("eda_derivative_distribution_plot", height = "620px")),
                      tabPanel("Epsilon Class Balance", plotOutput("eda_epsilon_class_balance_plot", height = "620px")),
                      tabPanel("Epsilon Sensitivity", plotOutput("eda_epsilon_sensitivity_plot", height = "620px"))
                    )
                  )
                )
                )
              ),
              class = "studio-panel eda-gallery-panel"
            ),
            panel_box(
              "Basic Statistical Tests and Summaries",
              div(
                class = "eda-scope-banner eda-statistics-scope",
                strong("Statistical screening. "),
                "These descriptive summaries and screening tests complement the plot gallery but are not part of CIRN's Bayesian edge-retention rule. ",
                "Use them to quantify time spacing, serial dependence, trends, missingness, derivative balance, and lagged associations. P-values and screening results do not independently identify regulation."
              ),
              div(
                class = "eda-statistics-navigation",
                tabsetPanel(
                  id = "eda_statistics_category",
                  tabPanel(
                    "Data Integrity",
                    div(
                      class = "eda-statistics-group",
                      tags$p("Assess variable completeness, sampling structure, and patterns of missing observations."),
                      tabsetPanel(
                        type = "pills",
                        tabPanel("Variable Diagnostics", DTOutput("eda_variable_tests_table")),
                        tabPanel("Time Spacing", DTOutput("eda_time_spacing_diagnostics_table")),
                        tabPanel("Missingness Runs", DTOutput("eda_missing_summary_table"))
                      )
                    )
                  ),
                  tabPanel(
                    "Temporal & Association",
                    div(
                      class = "eda-statistics-group",
                      tags$p("Screen contemporaneous association, trends, serial dependence, lag structure, and predictive timing."),
                      tabsetPanel(
                        type = "pills",
                        tabPanel("Correlation Tests", DTOutput("eda_correlation_tests_table")),
                        tabPanel("Granger Screen", DTOutput("eda_granger_tests_table")),
                        tabPanel("Trend Screen", DTOutput("eda_trend_screen_table")),
                        tabPanel("Lagged Correlations", DTOutput("eda_lagged_corr_table"))
                      )
                    )
                  ),
                  tabPanel(
                    "Derivative & Response",
                    div(
                      class = "eda-statistics-group",
                      tags$p("Review derivative behavior and how the epsilon blank-region threshold changes usable response classes."),
                      tabsetPanel(
                        type = "pills",
                        tabPanel("Derivative Summary", DTOutput("eda_derivative_summary_table")),
                        tabPanel("Epsilon Grid", DTOutput("eda_epsilon_sensitivity_table"))
                      )
                    )
                  )
                )
              ),
              class = "studio-panel eda-statistics-panel"
            )
        )
      )
    ),

    tabPanel(
      "Preprocess",
      fluidPage(
        contextual_tab_header(
          "Preprocess",
          "Transform the active time series into the state, derivative, and derivative-direction variables required by CIRN.",
          "An active dataset plus choices for normalization, spline smoothing, outlier handling, the epsilon blank region, and optional adaptive response jitter.",
          "Processed state trajectories, first- and second-derivative features, increasing/decreasing response classes, excluded near-zero regions, class-balance summaries, and quality warnings."
        ),
        fluidRow(
          column(
            3,
            panel_box(
              "Preprocessing Controls",
              uiOutput("active_data_source_preprocess_ui"),
              selectInput("normalization", "Normalization", choices = c("None" = "none", "Z-score" = "zscore", "Min-max" = "minmax")),
              sliderInput("points_per_interval", "Derivative grid points per interval", min = 1, max = 25, value = 1, step = 1),
              checkboxInput("auto_spar", "Choose smoothing by GCV", TRUE),
              sliderInput("spar", "Spline smoothing spar", min = 0, max = 1, value = 0.55, step = 0.01),
              selectInput("outlier_action", "Outlier action", choices = c("Keep" = "keep", "Winsorize" = "winsorize", "Remove" = "remove"), selected = "keep"),
              sliderInput("outlier_thresh", "MAD outlier threshold", min = 1, max = 8, value = 3.5, step = 0.1),
              sliderInput("response_eps_log", "log10 epsilon blank threshold", min = -10, max = -1, value = -6, step = 0.25),
              uiOutput("epsilon_label")
            ),
            panel_box(
              "Adaptive Jitter",
              checkboxInput("adaptive_jitter", "Use adaptive response jitter when needed", TRUE),
              checkboxInput("jitter_predictors", "Expert: predictor-jitter sensitivity when response jitter is triggered", FALSE),
              sliderInput("jitter_min_class_count", "Minimum class count after jitter", min = 2, max = 30, value = 5, step = 1),
              selectInput("jitter_scale_basis", "Jitter scale basis", choices = c("state_sd", "state_range", "derivative_sd", "derivative_max_abs", "absolute")),
              sliderInput("jitter_log_min", "Minimum log10 jitter factor", min = -12, max = -2, value = -8, step = 0.5),
              sliderInput("jitter_log_max", "Maximum log10 jitter factor", min = -8, max = -1, value = -2, step = 0.5),
              sliderInput("jitter_grid_n", "Jitter grid size", min = 4, max = 30, value = 13, step = 1)
            )
          ),
          column(
            9,
            panel_box("Raw vs Processed Data", tabsetPanel(
              tabPanel("Processed preview", DTOutput("processed_preview")),
              tabPanel("Quality warnings", uiOutput("quality_warnings"))
            )),
            fluidRow(
              column(7, panel_box("Derivative Preview", plotOutput("derivative_plot", height = "520px"))),
              column(5, panel_box("Response Class Balance", plotOutput("class_balance_plot", height = "310px"), DTOutput("class_balance_table")))
            )
          )
        )
      )
    ),

    tabPanel(
      "Run CIRN Algorithm",
      fluidPage(
        contextual_tab_header(
          "Run CIRN Algorithm",
          "Fit Bayesian logistic models that relate earlier state and derivative features to the direction of each target's local change. Run sublevel, all-predictor, or both representations, with optional pairwise models.",
          "Processed data, selected targets and predictors, representation and lag choices, prior settings, MCMC controls, and any requested pairwise configuration.",
          "Posterior coefficient estimates and 95% HDIs, signed feature-to-target edges retained only when their HDIs exclude zero, model diagnostics, and a reproducible run log."
        ),
        fluidRow(
          column(
            3,
            panel_box(
              "Run Control",
              uiOutput("active_data_source_run_ui"),
              div(
                class = "main-run-button",
                actionButton("run_cirn", button_label(icon("play"), "Run CIRN Algorithm"), class = "btn-primary studio-action-button")
              ),
              tags$div(
                class = "subtle run-control-guidance",
                "Before running, set or review the parameter values below. The slider values currently shown are recommended default starting values; try them first before making expert adjustments."
              )
            ),
            panel_box(
              "Model Structure",
              selectInput(
                "representation_mode",
                "Representation mode",
                choices = c("Sublevel models" = "sublevel", "All predictors" = "all_predictors", "Both" = "both"),
                selected = "both"
              ),
              sliderInput("lag_units", "Lag units", min = 1, max = 5, value = 1, step = 1),
              checkboxInput("run_pairwise", "Run pairwise CIRN", FALSE),
              selectInput("pairwise_representation_mode", "Pairwise representation", choices = c("Sublevel" = "sublevel", "All predictors" = "all_predictors", "Both" = "both"), selected = "sublevel")
            ),
            panel_box(
              "Bayesian Sliders",
              sliderInput("prior_mean", "Prior mean", min = -5, max = 5, value = 0, step = 0.1),
              sliderInput("prior_sd", "Prior SD", min = 0.1, max = 10, value = 2, step = 0.1),
              sliderInput("model_iter", "Total iterations", min = 500, max = 8000, value = 3000, step = 100),
              sliderInput("model_warmup", "Warmup iterations", min = 100, max = 4000, value = 1000, step = 100),
              sliderInput("model_chains", "Chains", min = 1, max = 4, value = 4, step = 1),
              sliderInput("model_cores", "Cores", min = 1, max = 4, value = 2, step = 1),
              sliderInput("adapt_delta", "adapt_delta", min = 0.8, max = 0.999, value = 0.95, step = 0.001),
              sliderInput("model_seed", "Model seed", min = 1, max = 99999, value = 123, step = 1),
              checkboxInput("compute_loo", "Compute LOO diagnostics", FALSE)
            )
          ),
          column(
            9,
            panel_box("Pre-Run Summary", uiOutput("run_summary")),
            panel_box("Run Log", DTOutput("progress_log_table")),
            panel_box(
              "Optional Exploratory Diagnostics",
              fluidRow(
                column(6,
                       sliderInput("rf_trees", "RF trees", min = 50, max = 1000, value = 300, step = 50),
                       sliderInput("rf_boot", "RF bootstrap repetitions", min = 5, max = 100, value = 20, step = 5),
                       sliderInput("rf_threshold", "RF support threshold", min = 0.5, max = 1, value = 0.95, step = 0.01),
                       uiOutput("rf_button_ui"),
                       div(class = "diagnostic-help", "Checks whether candidate predictors show stable Random Forest importance for classifying increasing versus decreasing target derivatives."),
                       uiOutput("rf_status_ui")),
                column(6,
                       sliderInput("latent_gain_threshold", "Latent-Z useful gain threshold", min = 0, max = 0.5, value = 0.05, step = 0.01),
                       uiOutput("latent_button_ui"),
                       div(class = "diagnostic-help", "Screens whether a first latent component improves derivative-direction classification, suggesting possible hidden collective structure."),
                       uiOutput("latent_status_ui"),
                       div(class = "method-box", "RF and latent-Z here are app-level diagnostics. Bayesian CIRN remains the source of signed edges."))
              )
            )
          )
        )
      )
    ),

    tabPanel(
      "Results",
      fluidPage(
        contextual_tab_header(
          "Results",
          "Interpret the fitted CIRN evidence across inference modes by comparing edge direction, coefficient magnitude, posterior uncertainty, and agreement among modes.",
          "A completed CIRN run. Random Forest support is shown when that optional diagnostic has also been completed.",
          "Retained-edge and coefficient tables, mode-specific regulatory networks, coefficient HDI plots, cross-mode consistency summaries, and interpretation cautions."
        ),
        fluidRow(
          column(
            8,
            panel_box(
              "Mode-Specific CIRN Figures",
              tabsetPanel(
                tabPanel("Interactive combined", visNetworkOutput("cirn_network", height = "620px")),
                tabPanel("Sublevel", plotOutput("sublevel_network_plot", height = "620px")),
                tabPanel("All predictors", plotOutput("all_predictors_network_plot", height = "620px")),
                tabPanel("Pairwise", plotOutput("pairwise_network_plot", height = "620px")),
                tabPanel("Consistent >=2", plotOutput("consistent_network_plot", height = "620px")),
                tabPanel("Consistent all 3", plotOutput("three_mode_network_plot", height = "620px"))
              )
            ),
            panel_box(
              "Coefficient HDI Figures",
              tabsetPanel(
                tabPanel("Combined", plotOutput("coef_plot", height = "680px")),
                tabPanel("Sublevel", plotOutput("sublevel_coef_plot", height = "680px")),
                tabPanel("All predictors", plotOutput("all_predictors_coef_plot", height = "680px")),
                tabPanel("Pairwise", plotOutput("pairwise_coef_plot", height = "680px"))
              )
            )
          ),
          column(
            4,
            panel_box("Result Filters",
                      checkboxInput("coef_retained_only", "Show retained coefficients only", FALSE),
                      sliderInput("coef_max_terms", "Maximum coefficients plotted", min = 10, max = 200, value = 80, step = 10)),
            panel_box("Interpretation Note", uiOutput("interpretation_note")),
            panel_box(
              "Edge Consistency Grid",
              div(
                class = "note-box",
                "Rows are retained edges and columns are inference modes. Green = activation, red = inhibition, grey = not detected; labels show the strongest retained omega in that mode."
              ),
              plotOutput("consistency_heatmap", height = "560px"),
              DTOutput("consistency_table")
            ),
            panel_box("RF-Supported Bayesian Edges", DTOutput("rf_supported_edges"))
          )
        ),
        panel_box("Retained Edges", DTOutput("edge_table")),
        panel_box("All Coefficients", DTOutput("coef_table"))
      )
    ),

    tabPanel(
      "CIRN Figures",
      fluidPage(
        contextual_tab_header(
          "CIRN Figure Gallery",
          "Review the principal scientific figures from the current session in one organized workspace before reporting or publication.",
          "Available results from CIRN fitting and any completed diagnostic, sensitivity, or benchmark analyses.",
          "Posterior and trace plots, regulatory networks, coefficient HDI and consistency figures, and available diagnostic or robustness plots. Downloadable files are prepared in Export."
        ),
        fluidRow(
          column(
            3,
            panel_box(
              "CIRN Figure Controls",
              selectInput("allplots_bayes_target", "Bayesian target", choices = character()),
              selectInput("allplots_bayes_representation", "Bayesian representation", choices = character()),
              sliderInput("allplots_bayes_max_terms", "Maximum posterior/MCMC parameters", min = 1, max = 30, value = 8, step = 1),
              checkboxInput("allplots_coef_retained_only", "Show retained coefficients only", FALSE),
              sliderInput("allplots_coef_max_terms", "Maximum coefficients plotted", min = 10, max = 200, value = 80, step = 10),
              selectInput("allplots_jitter_target", "Target for jitter detail", choices = character()),
              div(
                class = "note-box",
                "This figure gallery contains the main CIRN regulatory networks, coefficient HDI figures, Bayesian logistic posterior/MCMC diagnostics, edge-consistency summaries, and benchmark/sensitivity plots."
              )
            )
          ),
          column(
            9,
            panel_box(
              "Bayesian Logistic Regression Plots",
              tabsetPanel(
                tabPanel("Posterior densities", plotOutput("allplots_bayes_posterior_density_plot", height = "620px")),
                tabPanel("Cross-representation posteriors", plotOutput("allplots_bayes_cross_posterior_plot", height = "720px")),
                tabPanel("Histogram + trace", plotOutput("allplots_bayes_hist_trace_plot", height = "760px")),
                tabPanel("Trace with warm-up", plotOutput("allplots_bayes_trace_burnin_plot", height = "760px")),
                tabPanel("Diagnostics summary", plotOutput("allplots_bayes_diagnostics_summary_plot", height = "460px"))
              )
            ),
            panel_box(
              "CIRN Network Figures",
              tabsetPanel(
                tabPanel("Interactive combined", visNetworkOutput("allplots_cirn_network", height = "650px")),
                tabPanel("Sublevel", plotOutput("allplots_sublevel_network_plot", height = "650px")),
                tabPanel("All predictors", plotOutput("allplots_all_predictors_network_plot", height = "650px")),
                tabPanel("Pairwise", plotOutput("allplots_pairwise_network_plot", height = "650px")),
                tabPanel("Consistent >=2", plotOutput("allplots_consistent_network_plot", height = "650px")),
                tabPanel("Consistent all 3", plotOutput("allplots_three_mode_network_plot", height = "650px"))
              )
            ),
            panel_box(
              "Coefficient HDI Figures",
              tabsetPanel(
                tabPanel("Combined", plotOutput("allplots_coef_plot", height = "700px")),
                tabPanel("Sublevel", plotOutput("allplots_sublevel_coef_plot", height = "700px")),
                tabPanel("All predictors", plotOutput("allplots_all_predictors_coef_plot", height = "700px")),
                tabPanel("Pairwise", plotOutput("allplots_pairwise_coef_plot", height = "700px"))
              )
            ),
            panel_box(
              "Diagnostics, Sensitivity, and Benchmark Figures",
              tabsetPanel(
                tabPanel("Edge consistency grid", plotOutput("allplots_consistency_heatmap", height = "640px")),
                tabPanel("Jitter magnitude", plotOutput("allplots_jitter_magnitude_plot", height = "430px")),
                tabPanel("Jitter detail", plotOutput("allplots_jitter_detail_plot", height = "540px")),
                tabPanel("Sensitivity heatmap", plotOutput("allplots_sensitivity_heatmap", height = "540px")),
                tabPanel("Ground truth adjacency", plotOutput("allplots_truth_heatmap", height = "430px")),
                tabPanel("Inferred adjacency", plotOutput("allplots_inferred_heatmap", height = "430px"))
              )
            )
          )
        )
      )
    ),

    tabPanel(
      "Diagnostics",
      fluidPage(
        contextual_tab_header(
          "Diagnostics",
          "Evaluate computational reliability, usable information, collinearity, adaptive-jitter involvement, and optional predictive or latent-structure support before interpreting retained edges.",
          "A completed CIRN run. Random Forest and latent-Z results appear after their optional checks are run.",
          "MCMC, data-sufficiency, VIF, pairwise, jitter, Random Forest, and latent-Z summaries that qualify the reliability and strength of CIRN evidence. These checks do not assign regulatory sign."
        ),
        fluidRow(
          column(
            12,
            panel_box(
              "MCMC / Model Diagnostics",
              DTOutput("diagnostics_table"),
              class = "studio-panel diagnostics-table-panel"
            )
          )
        ),
        fluidRow(
          column(
            12,
            panel_box(
              "Data Sufficiency / Effective Sample Size",
              DTOutput("effective_sample_size_table"),
              class = "studio-panel diagnostics-table-panel"
            )
          )
        ),
        fluidRow(
          column(
            12,
            panel_box(
              "VIF Diagnostics",
              DTOutput("vif_table"),
              class = "studio-panel diagnostics-table-panel"
            )
          )
        ),
        fluidRow(
          column(
            12,
            panel_box(
              "VIF Pair Diagnostics",
              DTOutput("vif_pairs_table"),
              class = "studio-panel diagnostics-table-panel"
            )
          )
        ),
        fluidRow(
          column(
            12,
            panel_box(
              "Pairwise Diagnostics",
              DTOutput("pairwise_diagnostics_table"),
              class = "studio-panel diagnostics-table-panel"
            )
          )
        ),
        fluidRow(
          column(
            12,
            panel_box(
              "Pairwise VIF Pair Diagnostics",
              DTOutput("pairwise_vif_pairs_table"),
              class = "studio-panel diagnostics-table-panel"
            )
          )
        ),
        fluidRow(
          column(6, panel_box("Adaptive Jitter Magnitude", plotOutput("jitter_magnitude_plot", height = "360px"))),
          column(
            6,
            panel_box(
              "Adaptive Jitter Detail",
              selectInput("jitter_target", "Target for jitter plot", choices = character()),
              plotOutput("jitter_detail_plot", height = "520px")
            )
          )
        ),
        fluidRow(
          column(
            12,
            panel_box(
              "Random Forest Support",
              DTOutput("rf_table"),
              class = "studio-panel diagnostics-table-panel"
            )
          )
        ),
        fluidRow(
          column(
            12,
            panel_box(
              "Latent-Z Screening",
              DTOutput("latent_table"),
              class = "studio-panel diagnostics-table-panel"
            )
          )
        )
      )
    ),

    tabPanel(
      "Sensitivity",
      fluidPage(
        contextual_tab_header(
          "Sensitivity",
          "Test robustness by rerunning a chosen CIRN inference scope under controlled changes to the data or analysis settings.",
          "A fitted baseline analysis, a sensitivity inference scope, selected noise, lag, downsampling, sample-size, or missingness scenarios, and the number of stochastic replicates.",
          "Scenario run records, diagnostics, signed stability heatmaps, state-level stability after collapsing derivative representations, and feature-level stability that preserves the specific state or derivative predictor."
        ),
        fluidRow(
          column(
            3,
            panel_box(
              "Sensitivity Sliders",
              selectInput(
                "sensitivity_scope",
                "Sensitivity inference scope",
                choices = c(
                  "Use Run CIRN settings" = "use_run_settings",
                  "Sublevel only" = "sublevel",
                  "All predictors only" = "all_predictors",
                  "Sublevel + all predictors" = "sublevel_all_predictors",
                  "Pairwise only" = "pairwise_only",
                  "Everything" = "everything"
                ),
                selected = "use_run_settings"
              ),
              uiOutput("sensitivity_scope_note"),
              div(
                class = "note-box compact",
                tags$p(strong("Scenario-field tip: "), "enter values only for the sensitivity factor you want to test. Leave the other scenario fields blank."),
                tags$p(
                  class = "subtle",
                  "For lag-only sensitivity, clear Noise fractions, Downsample intervals, Target sample sizes, and Missing row fractions; put the lag values in Lag units, such as 1,2,3,4,5. Use 1 stochastic replicate for deterministic settings like lag."
                )
              ),
              div(
                class = "sensitivity-field",
                sliderInput("sens_replicates", "Stochastic replicates", min = 1, max = 10, value = 2, step = 1),
                tags$p(
                  class = "sensitivity-field-help",
                  "Number of independent random realizations used for Gaussian-noise and missing-row scenarios. Deterministic lag, downsampling, and target-size scenarios run once."
                )
              ),
              div(
                class = "sensitivity-field",
                textInput("sens_noise", "Noise fractions", value = "0.01,0.05,0.10"),
                tags$p(
                  class = "sensitivity-field-help",
                  "Adds independent, zero-mean Gaussian measurement noise to each state variable before preprocessing. A value of 0.05 sets the noise standard deviation to 5% of that variable's observed standard deviation."
                )
              ),
              div(
                class = "sensitivity-field",
                textInput("sens_lags", "Lag units", value = "1,2"),
                tags$p(
                  class = "sensitivity-field-help",
                  "Tests alternative time lags measured in observation steps. Lag 2 uses predictor values from observation k - 2 to classify the target's change at observation k."
                )
              ),
              div(
                class = "sensitivity-field",
                textInput("sens_downsample", "Downsample intervals", value = "2,5"),
                tags$p(
                  class = "sensitivity-field-help",
                  "Tests lower sampling frequency by keeping one row every n observations, beginning with the first row. Interval 2 keeps rows 1, 3, 5, and so on."
                )
              ),
              div(
                class = "sensitivity-field",
                textInput("sens_sample_sizes", "Target sample sizes", value = "25,50,75"),
                tags$p(
                  class = "sensitivity-field-help",
                  "Tests shorter datasets by selecting approximately n evenly spaced observations across the full time range. Values larger than the available row count are capped at the original size."
                )
              ),
              div(
                class = "sensitivity-field",
                textInput("sens_missing", "Missing row fractions", value = "0.10,0.25"),
                tags$p(
                  class = "sensitivity-field-help",
                  "Randomly removes the stated fraction of complete time-point rows before preprocessing. For example, 0.10 removes 10% of rows, including every variable recorded at those times."
                )
              ),
              checkboxInput("sens_save_outputs", "Save sensitivity CSVs to temp folder", FALSE),
              div(
                class = "paired-action-buttons",
                actionButton(
                  "preview_sensitivity",
                  button_label(icon("eye"), "Preview Plan"),
                  class = "btn-secondary studio-action-button"
                ),
                actionButton(
                  "run_sensitivity",
                  button_label(icon("play"), "Run Sensitivity"),
                  class = "btn-primary studio-action-button"
                )
              ),
              uiOutput("sensitivity_progress_status")
            )
          ),
          column(
            9,
            panel_box("Sensitivity Plan", DTOutput("sensitivity_plan_table")),
            panel_box(
              "Feature-Level Edge Stability Heatmap",
              div(
                class = "note-box",
                "Rows keep the CIRN feature labels used in the edge-consistency plot, including state, first-derivative, and second-derivative predictors. Green means activation, red means inhibition, grey means not detected, and purple means mixed sign."
              ),
              plotOutput("sensitivity_heatmap", height = "620px")
            ),
            panel_box("Sensitivity Progress Log", DTOutput("sensitivity_progress_table")),
            panel_box("Sensitivity Runs", DTOutput("sensitivity_runs_table")),
            panel_box("Feature-Level Edge Stability", DTOutput("sensitivity_stability_table")),
            panel_box(
              "Detailed Sensitivity Records",
              tabsetPanel(
                tabPanel("State-Level Stability", DTOutput("sensitivity_state_stability_table")),
                tabPanel("Retained Edges", DTOutput("sensitivity_edges_table")),
                tabPanel("Diagnostics", DTOutput("sensitivity_diagnostics_table")),
                tabPanel("Data Sufficiency", DTOutput("sensitivity_sample_size_table")),
                tabPanel("Ground-Truth Metrics", DTOutput("sensitivity_truth_metrics_table"))
              )
            )
          )
        )
      )
    ),

    tabPanel(
      "Benchmark",
      fluidPage(
        contextual_tab_header(
          "Benchmark",
          "Measure how well the inferred network recovers a known signed, directed network when valid ground truth is available.",
          "A completed CIRN run and a compatible signed ground-truth adjacency matrix, either uploaded by the user or supplied by a built-in simulation.",
          "State-level activation, inhibition, and absence-recovery metrics; true and inferred adjacency heatmaps; and an interpretation summary. Feature-specific CIRN edges are collapsed only for this state-level comparison."
        ),
        fluidRow(
          column(
            4,
            panel_box(
              "Benchmark Controls",
              div(class = "note-box", "Upload a signed ground-truth adjacency matrix in Data Source > Example or Uploaded Data or use a built-in simulated system with a known true network."),
              actionButton(
                "run_benchmark",
                button_label(icon("check-circle"), "Evaluate Against Ground Truth"),
                class = "btn-primary studio-action-button section-action-button"
              )
            ),
            panel_box("Metrics", DTOutput("benchmark_metrics"))
          ),
          column(
            8,
            fluidRow(
              column(6, panel_box("True Adjacency", plotOutput("truth_heatmap", height = "400px"))),
              column(6, panel_box("Inferred State-Level Adjacency", plotOutput("inferred_heatmap", height = "400px")))
            ),
            panel_box("Benchmark Explanation", uiOutput("benchmark_note"))
          )
        )
      )
    ),

    tabPanel(
      "Export",
      fluidPage(
        contextual_tab_header(
          "Export",
          "Save the current analysis, its settings, and its supporting evidence in formats suitable for review, reporting, and reproducibility.",
          "Any results available in the current session, including the active data configuration, CIRN fits, and completed diagnostic, sensitivity, or benchmark analyses.",
          "CSV tables, an Excel workbook, settings JSON, an HTML report, session information, a figure pack, and a complete reproducibility ZIP bundle."
        ),
        fluidRow(
          column(
            4,
            panel_box(
              "Tables",
              downloadButton("download_edges_csv", button_label(icon("download"), "Edges CSV"), class = "btn-secondary studio-action-button export-action-button"),
              downloadButton("download_coefficients_csv", button_label(icon("download"), "All Coefficients CSV"), class = "btn-secondary studio-action-button export-action-button"),
              downloadButton("download_diagnostics_csv", button_label(icon("download"), "Diagnostics CSV"), class = "btn-secondary studio-action-button export-action-button"),
              downloadButton("download_effective_sample_size_csv", button_label(icon("download"), "Data Sufficiency CSV"), class = "btn-secondary studio-action-button export-action-button"),
              downloadButton("download_vif_pairs_csv", button_label(icon("download"), "VIF Pairs CSV"), class = "btn-secondary studio-action-button export-action-button"),
              downloadButton("download_workbook", button_label(icon("download"), "Excel Workbook"), class = "btn-secondary studio-action-button export-action-button"),
              class = "studio-panel export-action-panel"
            ),
            panel_box(
              "Reports",
              downloadButton("download_settings_json", button_label(icon("download"), "Settings JSON"), class = "btn-secondary studio-action-button export-action-button"),
              downloadButton("download_html_report", button_label(icon("download"), "HTML Report"), class = "btn-secondary studio-action-button export-action-button"),
              downloadButton("download_session_info", button_label(icon("download"), "Session Info TXT"), class = "btn-secondary studio-action-button export-action-button"),
              downloadButton("download_figure_pack", button_label(icon("download"), "Figure Pack ZIP"), class = "btn-secondary studio-action-button export-action-button"),
              downloadButton("download_zip", button_label(icon("download"), "Complete ZIP Bundle"), class = "btn-primary studio-action-button export-action-button export-primary-button"),
              class = "studio-panel export-action-panel"
            )
          ),
          column(
            8,
            panel_box("Export Manifest", DTOutput("export_manifest")),
            panel_box("Reproducibility", verbatimTextOutput("settings_text"))
          )
        )
      )
    ),

    tabPanel(
      "Feedback",
      fluidPage(
        contextual_tab_header(
          "Feedback",
          "Document a software problem, confusing workflow, scientific-method question, or improvement suggestion with enough context for the CIRN authors to review it.",
          "A short summary, relevant details, reproduction steps for bugs, expected and observed behavior, and optional reporter information. App context is added automatically without attaching the uploaded raw data.",
          "A reviewable on-screen report, downloadable text and CSV records, an optional local copy, and a draft opened in the user's email application. CIRN Studio never sends email automatically."
        ),
        fluidRow(
          column(
            4,
            panel_box(
              "Send Feedback",
              div(
                class = "note-box",
                "Use this form for bugs, confusing behavior, feature requests, and suggestions that would make CIRN Studio easier to use."
              ),
              selectInput(
                "feedback_type",
                "Feedback type",
                choices = c("Bug / error", "Suggestion", "Usability issue", "Documentation question", "Scientific-method question", "Other"),
                selected = "Bug / error"
              ),
              selectInput(
                "feedback_area",
                "Where did it happen?",
                choices = feedback_areas,
                selected = "General usability"
              ),
              selectInput(
                "feedback_severity",
                "Impact",
                choices = c("Low: minor confusion", "Medium: slows analysis", "High: blocks analysis", "Critical: app exits or results look wrong"),
                selected = "Medium: slows analysis"
              ),
              textInput("feedback_summary", "One-sentence summary", placeholder = "Example: Sublevel run finishes but HDI plots do not appear."),
              textAreaInput("feedback_details", "What happened, or what should improve?", rows = 5, placeholder = "Describe the bug, confusing step, or improvement idea."),
              textAreaInput("feedback_steps", "Steps to reproduce", rows = 4, placeholder = "1. Open Data Source > Example or Uploaded Data\n2. Load Predator_Prey.csv\n3. Run sublevel only\n4. Open CIRN Figures"),
              textAreaInput("feedback_expected", "Expected behavior", rows = 3),
              textAreaInput("feedback_actual", "Actual behavior", rows = 3),
              checkboxInput("feedback_include_context", "Attach app settings and session context", TRUE)
            )
          ),
          column(
            4,
            panel_box(
              "Feedback Preview",
              div(
                class = "warning-box",
                "Review this before sending. The automatic context helps reproduce issues but does not include uploaded raw files."
              ),
              verbatimTextOutput("feedback_preview")
            )
          ),
          column(
            4,
            panel_box(
              "Reporter Details",
              textInput("feedback_name", "Name", placeholder = "Optional"),
              textInput("feedback_email", "Email", placeholder = "Optional, for follow-up"),
              textInput("feedback_affiliation", "Affiliation / role", placeholder = "Optional"),
              checkboxInput("feedback_followup", "I am willing to be contacted for clarification", TRUE),
              div(
                class = "feedback-actions",
                uiOutput("feedback_email_link"),
                downloadButton("download_feedback_txt", button_label(icon("download"), "Download Feedback Report"), class = "btn-secondary studio-action-button"),
                downloadButton("download_feedback_csv", button_label(icon("download"), "Download CSV Record"), class = "btn-secondary studio-action-button"),
                actionButton("save_feedback_local", button_label(icon("save"), "Save Local Copy"), class = "btn-secondary studio-action-button")
              ),
              uiOutput("feedback_contact_note"),
              uiOutput("feedback_log_status")
            ),
            panel_box(
              "What To Include",
              tags$ul(
                tags$li("For bugs: what you clicked, what data mode you used, and the exact error message if visible."),
                tags$li("For result concerns: the selected targets, predictors, inference mode, and whether pairwise/all-predictor/sublevel was enabled."),
                tags$li("For suggestions: describe the workflow problem and what would make the app clearer or faster.")
              )
            )
          )
        )
      )
    )

  )
)

server <- function(input, output, session) {
  observeEvent(input$hero_cirn_link, {
    updateNavbarPage(session, "main_nav", selected = "What Is CIRN?")
  })

  observeEvent(input$launch_upload, {
    updateRadioButtons(session, "data_mode", selected = "upload")
    updateNavbarPage(session, "main_nav", selected = "Data")
  })

  observeEvent(input$beginner_example, {
    updateRadioButtons(session, "data_mode", selected = "example")
    updateSelectInput(session, "analysis_preset", selected = "script")
    updateNavbarPage(session, "main_nav", selected = "Data")
    showNotification(
      "The built-in Predator-Prey dataset and Script-matched settings are ready.",
      type = "message",
      duration = 5
    )
  })

  observeEvent(input$launch_simulation, {
    updateRadioButtons(session, "data_mode", selected = "simulation")
    updateNavbarPage(session, "main_nav", selected = "Simulation Lab")
  })

  observeEvent(input$go_equation_builder, {
    updateRadioButtons(session, "data_mode", selected = "simulation")
    updateSelectInput(session, "sim_system", selected = "custom")
    updateNavbarPage(session, "main_nav", selected = "Simulation Lab")
  })

  observeEvent(input$simulate_btn, {
    updateRadioButtons(session, "data_mode", selected = "simulation")
    showNotification(
      "Simulation Lab output is now the active dataset for EDA, Preprocess, Run CIRN Algorithm, Sensitivity, Benchmark, and Export.",
      type = "message",
      duration = 6
    )
  }, ignoreInit = TRUE)

  observeEvent(input$workflow_go_data, {
    updateNavbarPage(session, "main_nav", selected = "Data")
  })

  observeEvent(input$workflow_go_eda, {
    updateNavbarPage(session, "main_nav", selected = "EDA")
  })

  observeEvent(input$workflow_go_preprocess, {
    updateNavbarPage(session, "main_nav", selected = "Preprocess")
  })

  observeEvent(input$workflow_go_run, {
    updateNavbarPage(session, "main_nav", selected = "Run CIRN Algorithm")
  })

  observeEvent(input$workflow_go_results, {
    updateNavbarPage(session, "main_nav", selected = "Results")
  })

  observeEvent(input$workflow_go_figures, {
    updateNavbarPage(session, "main_nav", selected = "CIRN Figures")
  })

  observeEvent(input$workflow_go_diagnostics, {
    updateNavbarPage(session, "main_nav", selected = "Diagnostics")
  })

  observeEvent(input$workflow_go_sensitivity, {
    updateNavbarPage(session, "main_nav", selected = "Sensitivity")
  })

  observeEvent(input$workflow_go_benchmark, {
    updateNavbarPage(session, "main_nav", selected = "Benchmark")
  })

  observeEvent(input$workflow_go_export, {
    updateNavbarPage(session, "main_nav", selected = "Export")
  })

  observeEvent(input$analysis_preset, {
    preset <- input$analysis_preset
    if (identical(preset, "script")) {
      updateSliderInput(session, "points_per_interval", value = 1)
      updateSelectInput(session, "outlier_action", selected = "keep")
      updateSliderInput(session, "model_iter", value = 3000)
      updateSliderInput(session, "model_warmup", value = 1000)
      updateSliderInput(session, "model_chains", value = 4)
      updateSliderInput(session, "model_cores", value = 2)
      updateSliderInput(session, "adapt_delta", value = 0.95)
      updateCheckboxInput(session, "compute_loo", value = FALSE)
      updateSliderInput(session, "rf_trees", value = 300)
      updateSliderInput(session, "rf_boot", value = 20)
    } else if (identical(preset, "teaching")) {
      updateSliderInput(session, "points_per_interval", value = 3)
      updateSliderInput(session, "model_iter", value = 800)
      updateSliderInput(session, "model_warmup", value = 300)
      updateSliderInput(session, "model_chains", value = 1)
      updateSliderInput(session, "model_cores", value = 1)
      updateSliderInput(session, "adapt_delta", value = 0.9)
      updateCheckboxInput(session, "compute_loo", value = FALSE)
      updateCheckboxInput(session, "run_pairwise", value = FALSE)
      updateSliderInput(session, "rf_trees", value = 100)
      updateSliderInput(session, "rf_boot", value = 10)
    } else if (identical(preset, "fast")) {
      updateSliderInput(session, "points_per_interval", value = 3)
      updateSliderInput(session, "model_iter", value = 1000)
      updateSliderInput(session, "model_warmup", value = 400)
      updateSliderInput(session, "model_chains", value = 1)
      updateSliderInput(session, "model_cores", value = 1)
      updateSliderInput(session, "adapt_delta", value = 0.93)
      updateCheckboxInput(session, "compute_loo", value = FALSE)
      updateCheckboxInput(session, "run_pairwise", value = FALSE)
      updateSliderInput(session, "rf_trees", value = 200)
      updateSliderInput(session, "rf_boot", value = 15)
    } else if (identical(preset, "careful")) {
      updateSliderInput(session, "points_per_interval", value = 5)
      updateSliderInput(session, "model_iter", value = 1500)
      updateSliderInput(session, "model_warmup", value = 500)
      updateSliderInput(session, "model_chains", value = 2)
      updateSliderInput(session, "model_cores", value = 2)
      updateSliderInput(session, "adapt_delta", value = 0.95)
      updateCheckboxInput(session, "compute_loo", value = FALSE)
      updateCheckboxInput(session, "run_pairwise", value = FALSE)
      updateSliderInput(session, "rf_trees", value = 300)
      updateSliderInput(session, "rf_boot", value = 20)
    } else if (identical(preset, "publication")) {
      updateSliderInput(session, "points_per_interval", value = 8)
      updateSliderInput(session, "model_iter", value = 4000)
      updateSliderInput(session, "model_warmup", value = 1000)
      updateSliderInput(session, "model_chains", value = 4)
      updateSliderInput(session, "model_cores", value = 4)
      updateSliderInput(session, "adapt_delta", value = 0.98)
      updateCheckboxInput(session, "compute_loo", value = TRUE)
      updateCheckboxInput(session, "run_pairwise", value = FALSE)
      updateSliderInput(session, "rf_trees", value = 500)
      updateSliderInput(session, "rf_boot", value = 50)
    } else if (identical(preset, "benchmark")) {
      updateSliderInput(session, "points_per_interval", value = 5)
      updateSliderInput(session, "model_iter", value = 1500)
      updateSliderInput(session, "model_warmup", value = 500)
      updateSliderInput(session, "model_chains", value = 2)
      updateSliderInput(session, "model_cores", value = 2)
      updateSliderInput(session, "adapt_delta", value = 0.95)
      updateCheckboxInput(session, "compute_loo", value = FALSE)
      updateCheckboxInput(session, "run_pairwise", value = FALSE)
      updateSliderInput(session, "rf_trees", value = 300)
      updateSliderInput(session, "rf_boot", value = 20)
      updateSliderInput(session, "sens_replicates", value = 2)
    }

    preset_title <- switch(
      preset,
      script = "Script-matched run",
      teaching = "Teaching demo",
      fast = "Fast exploration",
      careful = "Careful analysis",
      publication = "Publication quality",
      benchmark = "Benchmark mode",
      "Advanced Analysis Options"
    )
    showNotification(
      paste0(
        preset_title,
        " applied. It updates preprocessing, Bayesian, diagnostics, and sensitivity controls; rerun CIRN to change results."
      ),
      type = "message",
      duration = 6
    )
  }, ignoreInit = TRUE)

  progress_log <- reactiveVal(tibble())
  sensitivity_progress_log <- reactiveVal(tibble())
  rf_status <- reactiveVal(list(state = "not_run", message = "Run CIRN Algorithm first, then run RF support diagnostics."))
  latent_status <- reactiveVal(list(state = "not_run", message = "Run CIRN Algorithm first, then run latent-Z screening."))
  maybe_value <- function(expr) {
    tryCatch(expr, error = function(e) NULL)
  }

  output$simulation_equation_panel <- renderUI({
    system <- input$sim_system %||% "predator_prey"

    if (identical(system, "custom")) {
      variables <- trimws(unlist(strsplit(input$custom_vars %||% "", ",")))
      variables <- variables[nzchar(variables)]
      initial_values <- parse_key_values(input$custom_initials)
      parameter_values <- parse_key_values(input$custom_params)
      equation_lines <- trimws(unlist(strsplit(input$custom_equations %||% "", "\n")))
      equation_lines <- equation_lines[nzchar(equation_lines)]

      parsed_lhs <- character()
      malformed <- character()
      equation_rows <- lapply(equation_lines, function(line) {
        pieces <- strsplit(line, "=", fixed = TRUE)[[1]]
        if (length(pieces) != 2 || !nzchar(trimws(pieces[[1]])) || !nzchar(trimws(pieces[[2]]))) {
          malformed <<- c(malformed, line)
          return(simulation_equation_row(
            tags$span(class = "simulation-equation-invalid-label", "Invalid line"),
            tags$code(line)
          ))
        }
        lhs <- trimws(pieces[[1]])
        rhs <- trimws(pieces[[2]])
        parsed_lhs <<- c(parsed_lhs, lhs)
        simulation_equation_row(
          tags$span("d", lhs, "/dt"),
          tags$code(rhs)
        )
      })
      if (length(equation_rows) == 0) {
        equation_rows <- list(tags$p(
          class = "simulation-equation-empty",
          "Enter one derivative equation per line to build the live preview."
        ))
      }

      missing_equations <- setdiff(variables, parsed_lhs)
      unknown_equations <- setdiff(parsed_lhs, variables)
      missing_initials <- setdiff(variables, names(initial_values))
      preview_ready <- length(variables) > 0 &&
        length(malformed) == 0 &&
        length(missing_equations) == 0 &&
        length(unknown_equations) == 0 &&
        length(missing_initials) == 0

      issue_parts <- character()
      if (length(variables) == 0) {
        issue_parts <- c(issue_parts, "define at least one variable")
      }
      if (length(malformed) > 0) {
        issue_parts <- c(issue_parts, paste(length(malformed), "malformed equation line(s)"))
      }
      if (length(missing_equations) > 0) {
        issue_parts <- c(issue_parts, paste("missing equations for", paste(missing_equations, collapse = ", ")))
      }
      if (length(unknown_equations) > 0) {
        issue_parts <- c(issue_parts, paste("unknown left sides", paste(unknown_equations, collapse = ", ")))
      }
      if (length(missing_initials) > 0) {
        issue_parts <- c(issue_parts, paste("missing initial values for", paste(missing_initials, collapse = ", ")))
      }

      displayed_initials <- initial_values
      if (length(displayed_initials) > 0) {
        names(displayed_initials) <- paste0(names(displayed_initials), "(0)")
      }
      displayed_values <- c(displayed_initials, parameter_values)

      title <- "User-Defined Dynamical System"
      summary <- "Live preview of the derivative equations that will be parsed when the simulation is run."
      equations <- equation_rows
      values <- displayed_values
      definitions <- list(
        list("Variables", if (length(variables) > 0) paste(variables, collapse = ", ") else "Not yet defined"),
        list("Equation format", "Write one line as variable = derivative expression; the left side names the state variable."),
        list("Allowed operations", "Arithmetic and the supported functions sin, cos, tan, exp, log, sqrt, abs, min, max, pmin, and pmax.")
      )
      status_class <- if (preview_ready) "is-ready" else "needs-attention"
      status_text <- if (preview_ready) {
        "Preview complete. Click Simulate and Use Data to validate and integrate these equations."
      } else {
        paste0("Review before simulation: ", paste(issue_parts, collapse = "; "), ".")
      }
    } else {
      spec <- switch(
        system,
        predator_prey = list(
          title = "Predator-Prey System",
          summary = "Classical Lotka-Volterra interactions between prey abundance and predator abundance.",
          equations = list(
            simulation_equation_row(
              HTML("dx<sub>1</sub>/dt"),
              HTML("a x<sub>1</sub> - b x<sub>1</sub>x<sub>2</sub>")
            ),
            simulation_equation_row(
              HTML("dx<sub>2</sub>/dt"),
              HTML("c x<sub>1</sub>x<sub>2</sub> - d x<sub>2</sub>")
            )
          ),
          values = list(
            "x1(0)" = input$pp_x1,
            "x2(0)" = input$pp_x2,
            a = input$pp_a,
            b = input$pp_b,
            c = input$pp_c,
            d = input$pp_d
          ),
          definitions = list(
            list("x1, x2", "Prey and predator abundance."),
            list("a, d", "Intrinsic prey growth and predator mortality rates."),
            list("b, c", "Predation loss and predator gain coefficients.")
          )
        ),
        sir = list(
          title = "SIR Epidemic System",
          summary = "A frequency-dependent susceptible-infected-recovered transmission model.",
          equations = list(
            simulation_equation_row(
              HTML("dS/dt"),
              HTML("-&beta;SI/N")
            ),
            simulation_equation_row(
              HTML("dI/dt"),
              HTML("&beta;SI/N - &gamma;I")
            ),
            simulation_equation_row(
              HTML("dR/dt"),
              HTML("&gamma;I")
            )
          ),
          values = list(
            "S(0)" = input$sir_S,
            "I(0)" = input$sir_I,
            "R(0)" = input$sir_R,
            N = input$sir_N,
            "\u03b2" = input$sir_beta,
            "\u03b3" = input$sir_gamma
          ),
          definitions = list(
            list("S, I, R", "Susceptible, infected, and recovered population counts."),
            list("\u03b2, \u03b3", "Transmission and recovery rates."),
            list("N", "Population size used to scale frequency-dependent transmission.")
          )
        ),
        competition = list(
          title = "Two-Species Competition System",
          summary = "Logistic growth with reciprocal interspecific competition.",
          equations = list(
            simulation_equation_row(
              HTML("dx<sub>1</sub>/dt"),
              HTML("r<sub>1</sub>x<sub>1</sub>[1 - (x<sub>1</sub> + a<sub>12</sub>x<sub>2</sub>)/K<sub>1</sub>]")
            ),
            simulation_equation_row(
              HTML("dx<sub>2</sub>/dt"),
              HTML("r<sub>2</sub>x<sub>2</sub>[1 - (x<sub>2</sub> + a<sub>21</sub>x<sub>1</sub>)/K<sub>2</sub>]")
            )
          ),
          values = list(
            "x1(0)" = input$comp_x1,
            "x2(0)" = input$comp_x2,
            r1 = input$comp_r1,
            r2 = input$comp_r2,
            K1 = input$comp_K1,
            K2 = input$comp_K2,
            a12 = input$comp_a12,
            a21 = input$comp_a21
          ),
          definitions = list(
            list("x1, x2", "The two competing state variables."),
            list("r1, r2", "Intrinsic growth rates."),
            list("K1, K2", "Carrying capacities."),
            list("a12, a21", "Cross-species competition effects.")
          )
        ),
        mutualism = list(
          title = "Two-Species Mutualism System",
          summary = "Logistic self-regulation with saturating reciprocal benefits.",
          equations = list(
            simulation_equation_row(
              HTML("dx<sub>1</sub>/dt"),
              HTML("r<sub>1</sub>x<sub>1</sub>(1 - x<sub>1</sub>/K<sub>1</sub>) + m<sub>12</sub>x<sub>1</sub>x<sub>2</sub>/(1 + x<sub>2</sub>)")
            ),
            simulation_equation_row(
              HTML("dx<sub>2</sub>/dt"),
              HTML("r<sub>2</sub>x<sub>2</sub>(1 - x<sub>2</sub>/K<sub>2</sub>) + m<sub>21</sub>x<sub>1</sub>x<sub>2</sub>/(1 + x<sub>1</sub>)")
            )
          ),
          values = list(
            "x1(0)" = input$mut_x1,
            "x2(0)" = input$mut_x2,
            r1 = input$mut_r1,
            r2 = input$mut_r2,
            K1 = input$mut_K1,
            K2 = input$mut_K2,
            m12 = input$mut_m12,
            m21 = input$mut_m21
          ),
          definitions = list(
            list("x1, x2", "The two mutually interacting state variables."),
            list("r1, r2; K1, K2", "Intrinsic growth rates and carrying capacities."),
            list("m12, m21", "Saturating reciprocal-benefit coefficients.")
          )
        ),
        oscillator = list(
          title = "Coupled Linear Oscillator",
          summary = "A two-state conservative oscillator with reciprocal signed coupling.",
          equations = list(
            simulation_equation_row(
              HTML("dx<sub>1</sub>/dt"),
              HTML("&alpha;x<sub>2</sub>")
            ),
            simulation_equation_row(
              HTML("dx<sub>2</sub>/dt"),
              HTML("-&beta;x<sub>1</sub>")
            )
          ),
          values = list(
            "x1(0)" = input$osc_x1,
            "x2(0)" = input$osc_x2,
            "\u03b1" = input$osc_alpha,
            "\u03b2" = input$osc_beta
          ),
          definitions = list(
            list("x1, x2", "Coupled oscillator state variables."),
            list("\u03b1", "Positive coupling from x2 to the rate of change of x1."),
            list("\u03b2", "Magnitude of the negative coupling from x1 to the rate of change of x2.")
          )
        )
      )

      title <- spec$title
      summary <- spec$summary
      equations <- spec$equations
      values <- spec$values
      definitions <- spec$definitions
      status_class <- "is-ready"
      status_text <- "The displayed equations are the same equations used by Simulate and Use Data."
    }

    tags$aside(
      class = "simulation-equation-reference",
      tags$div(
        class = "simulation-equation-heading",
        tags$span(class = "simulation-equation-eyebrow", "Selected generating model"),
        tags$h4(title),
        tags$p(summary)
      ),
      tags$section(
        class = "simulation-equation-section",
        tags$h5("Model equations"),
        tags$div(class = "simulation-equation-stack", equations)
      ),
      tags$section(
        class = "simulation-equation-section",
        tags$h5("Current values"),
        simulation_value_grid(values)
      ),
      tags$details(
        class = "simulation-definition-disclosure",
        tags$summary("Variable and parameter definitions"),
        simulation_definition_list(definitions)
      ),
      tags$div(
        class = paste("simulation-equation-status", status_class),
        icon(if (identical(status_class, "is-ready")) "check-circle" else "exclamation-triangle"),
        tags$span(status_text)
      ),
      tags$div(
        class = "simulation-equation-boundary",
        tags$strong("Generating-model boundary. "),
        "These equations are used only to create the synthetic time series. CIRN receives the resulting observations and reconstructs candidate signed edges without being given these equations. The known network is used only when a supported benchmark is requested."
      )
    )
  })

  simulation_data <- eventReactive(input$simulate_btn, {
    df <- simulate_builtin(input)
    apply_simulation_artifacts(df, input)
  }, ignoreNULL = FALSE)

  raw_data <- reactive({
    if (identical(input$data_mode, "upload")) {
      req(input$data_file)
      read_any_table(input$data_file)
    } else if (identical(input$data_mode, "simulation")) {
      simulation_data()
    } else {
      readr::read_csv(example_path, show_col_types = FALSE)
    }
  })

  active_data_source <- reactive({
    mode <- input$data_mode %||% "example"
    if (identical(mode, "upload")) {
      file_name <- if (!is.null(input$data_file$name)) input$data_file$name else "waiting for upload"
      return(paste("Uploaded file:", file_name))
    }
    if (identical(mode, "simulation")) {
      sim_name <- input$sim_system %||% "selected system"
      return(paste("Simulation Lab output:", sim_name))
    }
    "Built-in example: Predator_Prey.csv from the app data folder"
  })

  active_data_source_badge <- reactive({
    mode <- input$data_mode %||% "example"
    if (identical(mode, "upload")) {
      file_name <- if (!is.null(input$data_file$name)) input$data_file$name else "waiting for upload"
      return(paste("Source: Uploaded", file_name))
    }
    if (identical(mode, "simulation")) {
      sim_name <- input$sim_system %||% "selected system"
      return(paste0("Source: Simulation Lab (", sim_name, ")"))
    }
    "Source: Built-in Predator_Prey.csv"
  })

  output$active_data_source_ui <- renderUI({
    div(class = "data-source-note compact", strong("Active source: "), active_data_source())
  })

  output$active_data_source_eda_ui <- renderUI({
    div(class = "data-source-note compact", strong("Active source: "), active_data_source())
  })

  output$active_data_source_preprocess_ui <- renderUI({
    div(class = "data-source-note compact", strong("Active source: "), active_data_source())
  })

  output$active_data_source_run_ui <- renderUI({
    div(class = "data-source-note compact", strong("Active source: "), active_data_source())
  })

  output$active_data_source_panel <- renderUI({
    df <- maybe_value(raw_data())
    div(
      class = "data-source-note",
      strong("Active source: "),
      active_data_source(),
      if (!is.null(df)) {
        tags$div(class = "data-source-meta", nrow(df), " rows; ", ncol(df), " columns currently loaded.")
      },
      if (identical(input$data_mode, "example")) {
        tags$div(class = "data-source-meta", "This is the built-in demonstration dataset. It is loaded automatically; it was not imported from your computer in this session.")
      }
    )
  })

  truth_matrix <- reactive({
    if (!is.null(input$truth_file)) {
      return(read_matrix_file(input$truth_file))
    }
    if (identical(input$data_mode, "simulation")) {
      return(truth_for_system(input$sim_system))
    }
    NULL
  })

  topology_matrix <- reactive({
    if (!is.null(input$topology_file)) {
      return(read_matrix_file(input$topology_file))
    }
    NULL
  })

  observeEvent(raw_data(), {
    df <- raw_data()
    num_cols <- numeric_columns(df)
    default_time <- if ("t" %in% names(df)) "t" else first_or(num_cols, first_or(names(df), ""))
    updateSelectInput(session, "time_col", choices = names(df), selected = default_time)
  }, ignoreInit = FALSE)

  observeEvent(list(raw_data(), input$time_col), {
    req(raw_data(), input$time_col)
    vars <- state_columns(raw_data(), input$time_col)
    selected <- vars[seq_len(min(length(vars), 5))]
    updateCheckboxGroupInput(session, "targets", choices = vars, selected = selected)
    updateCheckboxGroupInput(session, "predictors", choices = vars, selected = vars)
    updateCheckboxGroupInput(session, "eda_vars", choices = vars, selected = selected)
    updateSelectInput(session, "phase_x", choices = vars, selected = first_or(vars, ""))
    updateSelectInput(session, "phase_y", choices = vars, selected = if (length(vars) >= 2) vars[[2]] else first_or(vars, ""))
  }, ignoreInit = FALSE)

  response_eps <- reactive({
    10^input$response_eps_log
  })

  jitter_grid <- reactive({
    lo <- min(input$jitter_log_min, input$jitter_log_max)
    hi <- max(input$jitter_log_min, input$jitter_log_max)
    10^seq(lo, hi, length.out = input$jitter_grid_n)
  })

  processed_data <- reactive({
    req(raw_data(), input$time_col)
    normalize_data(raw_data(), input$time_col, input$normalization)
  })

  selected_data <- reactive({
    req(processed_data(), input$time_col, input$targets, input$predictors)
    vars <- unique(c(input$time_col, input$targets, input$predictors))
    processed_data()[, vars, drop = FALSE]
  })

  derivative_data <- reactive({
    req(selected_data(), input$time_col)
    spar_value <- if (isTRUE(input$auto_spar)) NULL else input$spar
    tryCatch(
      compute_derivatives(
        selected_data(),
        time_col = input$time_col,
        points_per_interval = input$points_per_interval,
        spar = spar_value,
        outlier_method = "MAD",
        outlier_thresh = input$outlier_thresh,
        outlier_action = input$outlier_action
      ),
      error = function(e) {
        fallback_derivatives(selected_data(), input$time_col)
      }
    )
  })

  class_preview <- reactive({
    req(derivative_data(), input$targets)
    response_preview_table(derivative_data(), input$targets, response_eps())
  })

  output$metric_rows <- renderUI({
    df <- maybe_value(raw_data())
    if (is.null(df)) return("0")
    format(nrow(df), big.mark = ",")
  })

  output$metric_vars <- renderUI({
    df <- maybe_value(raw_data())
    if (is.null(df) || is.null(input$time_col)) return("0")
    length(state_columns(df, input$time_col))
  })

  output$metric_targets <- renderUI({
    length(input$targets %||% character())
  })

  output$metric_edges <- renderUI({
    res <- maybe_value(cirn_results())
    if (is.null(res) || is.null(res$edges_combined)) {
      return("0")
    }
    edges <- as_result_tbl(res$edges_combined)
    if (nrow(edges) == 0 || !all(c("term", "target", "omega") %in% names(edges))) {
      return("0")
    }
    edges %>%
      filter(is.finite(.data$omega), .data$omega != 0) %>%
      distinct(.data$term, .data$target) %>%
      nrow() %>%
      format(big.mark = ",")
  })

  output$edge_representation_status <- renderUI({
    representation_levels <- tibble(
      representation = c("state", "first_derivative", "second_derivative"),
      label = c("State predictors", "First-derivative predictors", "Second-derivative predictors"),
      activation = 0L,
      inhibition = 0L,
      conflict = 0L,
      mode_detections = 0L
    )

    res <- maybe_value(cirn_results())
    fitted <- !is.null(res)
    edges <- if (fitted) as_result_tbl(res$edges_combined %||% tibble()) else tibble()

    if (nrow(edges) > 0 && all(c("term", "target", "omega") %in% names(edges))) {
      edge_records <- add_mode_group(edges) %>%
        transmute(
          representation = case_when(
            grepl("_d2$", as.character(.data$term)) ~ "second_derivative",
            grepl("_d1$", as.character(.data$term)) ~ "first_derivative",
            TRUE ~ "state"
          ),
          term = as.character(.data$term),
          target = as.character(.data$target),
          mode_group = as.character(.data$mode_group),
          sign = case_when(
            .data$omega > 0 ~ "activation",
            .data$omega < 0 ~ "inhibition",
            TRUE ~ NA_character_
          )
        ) %>%
        filter(!is.na(.data$sign))

      unique_edge_signs <- edge_records %>%
        distinct(.data$representation, .data$term, .data$target, .data$sign)

      unique_edge_profiles <- unique_edge_signs %>%
        group_by(.data$representation, .data$term, .data$target) %>%
        summarise(
          sign = if_else(n_distinct(.data$sign) > 1L, "conflict", first(.data$sign)),
          .groups = "drop"
        )

      observed <- unique_edge_profiles %>%
        count(.data$representation, .data$sign, name = "n") %>%
        tidyr::complete(
          representation = representation_levels$representation,
          sign = c("activation", "inhibition", "conflict"),
          fill = list(n = 0L)
        ) %>%
        tidyr::pivot_wider(names_from = "sign", values_from = "n", values_fill = 0)

      mode_counts <- edge_records %>%
        distinct(.data$representation, .data$mode_group, .data$term, .data$target) %>%
        count(.data$representation, name = "mode_detections")

      representation_levels <- representation_levels %>%
        select(all_of(c("representation", "label"))) %>%
        left_join(observed, by = "representation") %>%
        left_join(mode_counts, by = "representation") %>%
        mutate(
          activation = as.integer(coalesce(.data$activation, 0)),
          inhibition = as.integer(coalesce(.data$inhibition, 0)),
          conflict = as.integer(coalesce(.data$conflict, 0)),
          mode_detections = as.integer(coalesce(.data$mode_detections, 0))
        )
    }

    cards <- lapply(seq_len(nrow(representation_levels)), function(i) {
      row <- representation_levels[i, ]
      div(
        class = paste("edge-representation-card", row$representation),
        tags$strong(row$label),
        div(
          class = "signed-edge-counts",
          span(
            class = "signed-edge-count activation",
            span(class = "signed-edge-swatch"),
            span("Activation"),
            tags$strong(format(row$activation, big.mark = ","))
          ),
          span(
            class = "signed-edge-count inhibition",
            span(class = "signed-edge-swatch"),
            span("Inhibition"),
            tags$strong(format(row$inhibition, big.mark = ","))
          ),
          span(
            class = "signed-edge-count conflict",
            title = "The same feature-to-target edge was retained with opposite signs across fitted inference modes.",
            span(class = "signed-edge-swatch"),
            span("Sign conflict"),
            tags$strong(format(row$conflict, big.mark = ","))
          )
        ),
        tags$div(
          class = "edge-representation-total",
          "Unique feature edges: ",
          tags$strong(format(row$activation + row$inhibition + row$conflict, big.mark = ",")),
          tags$span(
            class = "edge-mode-detection-count",
            title = "Distinct appearances of these feature edges across Sublevel, All Predictors, and Pairwise inference modes.",
            "Mode detections: ",
            tags$strong(format(row$mode_detections, big.mark = ","))
          )
        )
      )
    })

    div(
      class = "edge-representation-summary",
      div(
        class = "edge-representation-summary-heading",
        tags$strong("Signed retained edges by predictor representation")
      ),
      if (fitted) {
        div(
          class = "edge-representation-summary-note edge-representation-summary-note-ready",
          paste(
            "Activation and inhibition are unique feature-level edge counts across all fitted modes.",
            "The same signed edge is counted once even when retained by Sublevel, All Predictors, and Pairwise.",
            "Opposite signs across modes are reported as a sign conflict; repeated appearances are shown as mode detections."
          )
        )
      } else {
        NULL
      },
      div(class = "edge-representation-grid", cards),
      if (!fitted) {
        div(
          class = "edge-representation-summary-note edge-representation-summary-note-pending edge-representation-summary-note-footer",
          "Run CIRN Algorithm to populate activation and inhibition counts."
        )
      } else {
        NULL
      }
    )
  })

  output$status_pills <- renderUI({
    df <- maybe_value(raw_data())
    class_tbl <- maybe_value(class_preview())
    pills <- list()
    pills[[length(pills) + 1L]] <- status_pill(if (!is.null(df)) "Data loaded" else "Waiting for data", if (!is.null(df)) "ready" else "warn")
    if (!is.null(df)) {
      pills[[length(pills) + 1L]] <- status_pill(active_data_source_badge(), "info", active_data_source())
    }
    pills[[length(pills) + 1L]] <- status_pill(if (!is.null(input$targets) && length(input$targets) > 0) "Targets selected" else "Select targets", if (!is.null(input$targets) && length(input$targets) > 0) "ready" else "warn")
    pills[[length(pills) + 1L]] <- status_pill(
      if (!is.null(class_tbl) && nrow(class_tbl) > 0) "Derivative preview ready" else "Derivative preview pending",
      if (!is.null(class_tbl) && nrow(class_tbl) > 0) "ready" else "warn"
    )
    res <- maybe_value(cirn_results())
    pills[[length(pills) + 1L]] <- status_pill(if (!is.null(res)) "CIRN fitted" else "CIRN not fitted", if (!is.null(res)) "ready" else "pending")
    tagList(pills)
  })

  output$workflow_status <- renderUI({
    class_tbl <- maybe_value(class_preview())
    res <- maybe_value(cirn_results())
    data_ready <- !is.null(maybe_value(raw_data()))
    selected_ready <- length(input$targets %||% character()) > 0 &&
      length(input$predictors %||% character()) > 0
    derivative_ready <- !is.null(class_tbl) && nrow(class_tbl) > 0
    diagnostics_ok <- derivative_ready &&
      all(class_tbl$status == "ready", na.rm = TRUE)
    diagnostics_warn <- derivative_ready && !diagnostics_ok
    fitted <- !is.null(res)
    results_available <- fitted && !is.null(res$all_coefficients_combined)
    figures_available <- fitted && !is.null(res$edges_combined)
    diagnostics_available <- fitted && !is.null(res$diagnostics)
    sensitivity_available <- !is.null(maybe_value(sensitivity_results()))
    truth_available <- !is.null(maybe_value(truth_matrix()))
    benchmark_available <- !is.null(maybe_value(benchmark_results()))

    step <- function(input_id, number, title, body, status, label, optional = FALSE) {
      actionLink(
        input_id,
        label = tagList(
          tags$div(
            class = "step-title-row",
            strong(paste0(number, ". ", title)),
            if (isTRUE(optional)) tags$span(class = "step-optional-badge", "Optional")
          ),
          span(body),
          tags$div(class = "step-status", label)
        ),
        class = paste("step-chip step-action", status),
        title = paste("Open", title, if (isTRUE(optional)) "(optional)" else "")
      )
    }

    div(
      class = "workflow-map",
      step("workflow_go_data", 1, "Data Source", "Choose example, uploaded, simulated, or equation-built data.", if (data_ready) "done" else "warn", if (data_ready) "loaded" else "needed"),
      step("workflow_go_eda", 2, "EDA", "Inspect trajectories, gaps, correlations, and system dynamics.", if (selected_ready) "ready" else "pending", if (selected_ready) "ready" else "select variables", optional = TRUE),
      step("workflow_go_preprocess", 3, "Preprocess", "Set smoothing and verify derivatives, blanks, and response classes.", if (diagnostics_ok) "done" else if (diagnostics_warn) "warn" else if (selected_ready) "ready" else "pending", if (diagnostics_ok) "clear" else if (diagnostics_warn) "review" else if (selected_ready) "ready" else "pending"),
      step("workflow_go_run", 4, "Run CIRN Algorithm", "Fit selected Bayesian modes and optional pairwise models.", if (fitted) "done" else if (diagnostics_ok) "ready" else "pending", if (fitted) "fitted" else if (diagnostics_ok) "ready" else "waiting"),
      step("workflow_go_results", 5, "Results", "Review retained edges, HDIs, coefficients, and mode consistency.", if (results_available) "ready" else "pending", if (results_available) "available" else "pending"),
      step("workflow_go_figures", 6, "CIRN Figures", "Open regulatory networks, posterior plots, and publication figures.", if (figures_available) "ready" else "pending", if (figures_available) "available" else "pending"),
      step("workflow_go_diagnostics", 7, "Diagnostics", "Check sampling, VIF, jitter, RF support, and latent-Z evidence.", if (diagnostics_available) "ready" else "pending", if (diagnostics_available) "available" else "pending"),
      step("workflow_go_sensitivity", 8, "Sensitivity", "Test edge stability under data and setting perturbations.", if (sensitivity_available) "done" else if (fitted) "ready" else "pending", if (sensitivity_available) "completed" else if (fitted) "ready" else "pending", optional = TRUE),
      step("workflow_go_benchmark", 9, "Benchmark", "Compare inferred edges with known signed ground truth when available.", if (benchmark_available) "done" else if (fitted && truth_available) "ready" else "pending", if (benchmark_available) "completed" else if (fitted && truth_available) "ready" else "not configured", optional = TRUE),
      step("workflow_go_export", 10, "Export", "Save tables, figures, settings, workbook, and report bundle.", if (fitted) "ready" else "pending", if (fitted) "available" else "pending")
    )
  })

  output$preset_note <- renderUI({
    note <- switch(
      input$analysis_preset,
      script = list(
        title = "Script-matched run",
        body = "Closest to the main CIRN_Algorithm.R defaults. Use this when you want the Shiny app results to match the script-based results as closely as possible.",
        detail = "Sets derivative grid to 1, keeps outliers, and uses 3000 iterations, 1000 warmup, 4 chains, adapt_delta 0.95, and seed-controlled Bayesian fitting. Use the same representation mode when comparing outputs."
      ),
      teaching = list(
        title = "Teaching demo",
        body = "Lightweight settings for explaining the workflow in class, meetings, or demonstrations.",
        detail = "Fast and easy to show, but not intended for serious inference or publication-quality posterior diagnostics."
      ),
      fast = list(
        title = "Fast exploration",
        body = "A quick first pass for checking whether the data, selected variables, preprocessing, and CIRN pipeline are working.",
        detail = "Useful before spending time on heavier Bayesian runs. Treat results as preliminary screening, not final evidence."
      ),
      careful = list(
        title = "Careful analysis",
        body = "A balanced preset for serious exploratory analysis when you want more reliable fitting than fast exploration without excessive runtime.",
        detail = "Good for inspecting retained edges, HDIs, diagnostics, and sensitivity before deciding whether a publication-quality run is needed."
      ),
      publication = list(
        title = "Publication quality",
        body = "Heavier settings for final dissertation or manuscript analyses where results may be cited, defended, or reported.",
        detail = "Uses longer MCMC settings, higher adapt_delta, and LOO diagnostics for stronger reporting. Slower, but more appropriate for final figures and conclusions."
      ),
      benchmark = list(
        title = "Benchmark mode",
        body = "Designed for simulations or datasets with a known ground-truth signed network.",
        detail = "Use this when comparing inferred CIRN edges against a supplied or simulated adjacency matrix, including signed recovery, false positives, and missed edges."
      ),
      list(
        title = "Careful analysis",
        body = "A balanced preset for ordinary CIRN analysis.",
        detail = "Good when you are unsure which preset to use."
      )
    )
    tags$div(
      class = "preset-note-detail",
      tags$strong(note$title),
      tags$p(note$body),
      tags$p(class = "subtle", note$detail)
    )
  })

  output$preset_applied_settings <- renderUI({
    bool_label <- function(x) {
      if (isTRUE(x)) "Yes" else "No"
    }
    row <- function(area, setting, value) {
      tags$tr(
        tags$td(area),
        tags$td(setting),
        tags$td(value)
      )
    }
    
    tags$details(
      class = "preset-settings-card preset-settings-disclosure",
      tags$summary(
        class = "preset-settings-summary",
        tags$span(class = "preset-settings-title", "Preset-controlled settings now applied"),
        tags$span(class = "preset-settings-summary-hint", "View applied settings")
      ),
      tags$p(
        class = "preset-settings-help",
        "The preset changes controls across Preprocess, Run CIRN Algorithm, Diagnostics, and Sensitivity. ",
        "It does not change existing results until you click Run CIRN Algorithm again."
      ),
      tags$table(
        class = "preset-settings-table",
        tags$thead(
          tags$tr(
            tags$th("Area"),
            tags$th("Setting"),
            tags$th("Current value")
          )
        ),
        tags$tbody(
          row("Preprocess", "Derivative grid points per interval", input$points_per_interval %||% ""),
          row("Preprocess", "Outlier action", input$outlier_action %||% ""),
          row("Run CIRN", "Total iterations", input$model_iter %||% ""),
          row("Run CIRN", "Warmup iterations", input$model_warmup %||% ""),
          row("Run CIRN", "Chains / cores", paste0(input$model_chains %||% "", " / ", input$model_cores %||% "")),
          row("Run CIRN", "adapt_delta", input$adapt_delta %||% ""),
          row("Run CIRN", "Compute LOO diagnostics", bool_label(input$compute_loo)),
          row("Run CIRN", "Run pairwise CIRN", bool_label(input$run_pairwise)),
          row("Diagnostics", "RF trees / bootstrap repetitions", paste0(input$rf_trees %||% "", " / ", input$rf_boot %||% "")),
          row("Sensitivity", "Stochastic replicates", input$sens_replicates %||% "")
        )
      )
    )
  })

  output$epsilon_label <- renderUI({
    div(class = "note-box", paste("Current epsilon:", signif(response_eps(), 4)))
  })

  output$data_preview <- renderDT({
    safe_dt(raw_data(), page_length = 10)
  })

  output$processed_preview <- renderDT({
    safe_dt(processed_data(), page_length = 10)
  })

  output$missing_plot <- renderPlot({
    cirn_plot_typography(plot_missingness(raw_data()))
  })

  output$eda_missing_plot <- renderPlot({
    cirn_plot_typography(plot_missingness(processed_data()))
  })

  output$time_spacing_table <- renderDT({
    req(raw_data(), input$time_col)
    safe_dt(quality_summary(raw_data(), input$time_col)$time, page_length = 5, scroll_x = FALSE)
  })

  output$variable_summary_table <- renderDT({
    req(raw_data(), input$time_col)
    safe_dt(quality_summary(raw_data(), input$time_col)$variables, page_length = 8)
  })

  output$quality_warnings <- renderUI({
    req(raw_data(), input$time_col)
    warnings <- quality_summary(raw_data(), input$time_col)$warnings
    if (length(warnings) == 0) {
      return(div(class = "note-box", "No major pre-run data warnings detected. Still inspect derivatives before interpreting edges."))
    }
    div(class = "warning-box", tags$ul(lapply(warnings, tags$li)))
  })

  output$truth_preview <- renderDT({
    mat <- truth_matrix()
    shiny::validate(shiny::need(!is.null(mat), "No ground-truth matrix available yet."))
    safe_dt(as.data.frame(mat) %>% tibble::rownames_to_column("source"), page_length = 8)
  })

  output$topology_preview <- renderDT({
    mat <- topology_matrix()
    shiny::validate(shiny::need(!is.null(mat), "No topology matrix uploaded yet."))
    safe_dt(as.data.frame(mat) %>% tibble::rownames_to_column("source"), page_length = 8)
  })

  output$simulation_plot <- renderPlot({
    df <- simulation_data()
    cirn_plot_typography(plot_time_series(df, "t", setdiff(names(df), "t")))
  })

  output$simulation_table <- renderDT({
    safe_dt(simulation_data(), page_length = 8)
  })

  output$simulation_truth_network <- renderVisNetwork({
    mat <- truth_for_system(input$sim_system)
    if (is.null(mat)) {
      return(visNetwork(data.frame(id = "Custom", label = "Custom truth not specified"), data.frame()))
    }
    edge_tbl <- as.data.frame(as.table(mat)) %>%
      setNames(c("term", "target", "omega")) %>%
      filter(.data$omega != 0) %>%
      mutate(
        hdi_lower95 = .data$omega,
        hdi_upper95 = .data$omega,
        jitter_used = FALSE
      )
    make_network_widget(edge_tbl, height = "430px", show_legend = FALSE)
  })

  output$derivative_plot <- renderPlot({
    req(derivative_data(), input$targets)
    cirn_plot_typography(plot_derivative_orders(derivative_data(), input$time_col, input$targets))
  })

  output$class_balance_plot <- renderPlot({
    cirn_plot_typography(plot_class_balance(class_preview()))
  })

  output$class_balance_table <- renderDT({
    safe_dt(class_preview(), page_length = 8, scroll_x = TRUE)
  })

  output$eda_time_plot <- renderPlot({
    req(processed_data(), input$time_col, input$eda_vars)
    cirn_plot_typography(plot_time_series(processed_data(), input$time_col, input$eda_vars))
  })

  output$eda_raw_small_multiples_plot <- renderPlot({
    req(raw_data(), input$time_col)
    cirn_plot_typography(plot_time_series_small_multiples(raw_data(), input$time_col, input$eda_vars))
  })

  output$eda_normalized_overlay_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_normalized_overlay(processed_data(), input$time_col, input$eda_vars))
  })

  output$eda_rolling_mean_variance_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_rolling_mean_variance(processed_data(), input$time_col, input$eda_vars, input$eda_rolling_window))
  })

  output$eda_distribution_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_variable_distributions(processed_data(), input$time_col, input$eda_vars))
  })

  output$eda_outlier_timeline_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_outlier_timeline(processed_data(), input$time_col, input$eda_vars, input$eda_outlier_thresh))
  })

  output$eda_variable_attention_plot <- renderPlot({
    req(processed_data(), derivative_data(), input$time_col)
    cirn_plot_typography(plot_variable_attention_summary(processed_data(), derivative_data(), input$time_col, input$eda_vars, response_eps(), input$eda_outlier_thresh))
  })

  output$eda_time_gap_plot <- renderPlot({
    req(raw_data(), input$time_col)
    cirn_plot_typography(plot_time_gap_profile(raw_data(), input$time_col))
  })

  output$eda_missing_summary_plot <- renderPlot({
    req(raw_data())
    cirn_plot_typography(plot_missingness_summary(raw_data()))
  })

  output$corr_plot <- renderPlot({
    req(processed_data(), input$eda_vars)
    cirn_plot_typography(plot_correlation_heatmap(processed_data(), input$eda_vars))
  })

  output$phase_plot <- renderPlot({
    cirn_plot_typography(plot_phase_plane(processed_data(), input$time_col, input$phase_x, input$phase_y))
  })

  output$eda_pairwise_scatter_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_pairwise_scatter_matrix(processed_data(), input$time_col, input$eda_vars))
  })

  output$eda_lagged_corr_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_lagged_correlation_heatmap(processed_data(), input$time_col, input$eda_vars, input$eda_lag_max, "spearman"))
  })

  output$eda_rolling_correlation_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_rolling_correlation(processed_data(), input$time_col, input$eda_vars, input$eda_rolling_window, "spearman"))
  })

  output$eda_cross_correlation_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_cross_correlation_lead_lag(processed_data(), input$time_col, input$eda_vars, input$eda_lag_max, "spearman"))
  })

  output$eda_acf_pacf_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_acf_pacf(processed_data(), input$time_col, input$eda_vars, input$eda_lag_max))
  })

  output$eda_phase_pairs_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_phase_pairs(processed_data(), input$time_col, input$eda_vars))
  })

  output$eda_directed_phase_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_directed_phase_portraits(processed_data(), input$time_col, input$eda_vars))
  })

  output$eda_vector_field_plot <- renderPlot({
    req(derivative_data(), input$time_col)
    cirn_plot_typography(plot_derivative_vector_field(derivative_data(), input$time_col, input$eda_vars))
  })

  output$eda_delay_embedding_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_delay_embedding(processed_data(), input$time_col, input$eda_vars, input$eda_lag_max))
  })

  output$eda_recurrence_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_recurrence_map(processed_data(), input$time_col, input$eda_vars))
  })

  output$eda_nullcline_plot <- renderPlot({
    req(derivative_data(), input$time_col)
    cirn_plot_typography(plot_estimated_nullclines(derivative_data(), input$time_col, input$eda_vars))
  })

  output$eda_pca_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_pca_trajectory(processed_data(), input$time_col, input$eda_vars))
  })

  output$eda_derivative_state_plot <- renderPlot({
    req(derivative_data(), input$time_col, input$eda_vars, input$targets)
    cirn_plot_typography(plot_derivative_state_relationships(derivative_data(), input$time_col, input$eda_vars, input$targets))
  })

  output$eda_derivative_phase_plot <- renderPlot({
    req(derivative_data(), input$time_col, input$eda_vars)
    cirn_plot_typography(plot_derivative_phase_portraits(derivative_data(), input$time_col, input$eda_vars))
  })

  output$eda_cross_derivative_phase_plot <- renderPlot({
    req(derivative_data(), input$time_col, input$eda_vars)
    cirn_plot_typography(plot_cross_derivative_phase_portraits(derivative_data(), input$time_col, input$eda_vars))
  })

  output$eda_derivative_distribution_plot <- renderPlot({
    req(derivative_data(), input$time_col, input$eda_vars)
    cirn_plot_typography(plot_derivative_distributions(derivative_data(), input$time_col, input$eda_vars))
  })

  output$eda_epsilon_class_balance_plot <- renderPlot({
    req(derivative_data(), input$targets)
    cirn_plot_typography(plot_epsilon_class_balance_stack(derivative_data(), input$targets, input$response_eps_log))
  })

  output$eda_epsilon_sensitivity_plot <- renderPlot({
    req(derivative_data(), input$targets)
    cirn_plot_typography(plot_epsilon_sensitivity(derivative_data(), input$targets, input$response_eps_log))
  })

  output$eda_variable_tests_table <- renderDT({
    req(processed_data(), input$time_col)
    safe_dt(variable_test_table(processed_data(), input$time_col, input$eda_vars, input$eda_lag_max), page_length = 12)
  })

  output$eda_correlation_tests_table <- renderDT({
    req(processed_data(), input$time_col)
    safe_dt(correlation_test_table(processed_data(), input$time_col, input$eda_vars), page_length = 12)
  })

  output$eda_granger_tests_table <- renderDT({
    req(processed_data(), input$time_col)
    safe_dt(granger_screen_table(processed_data(), input$time_col, input$eda_vars, input$eda_granger_lag), page_length = 12)
  })

  output$eda_time_spacing_diagnostics_table <- renderDT({
    req(raw_data(), input$time_col)
    safe_dt(time_spacing_diagnostic_table(raw_data(), input$time_col), page_length = 12, scroll_x = FALSE)
  })

  output$eda_trend_screen_table <- renderDT({
    req(processed_data(), input$time_col)
    safe_dt(trend_screen_table(processed_data(), input$time_col, input$eda_vars), page_length = 12)
  })

  output$eda_derivative_summary_table <- renderDT({
    req(derivative_data(), input$eda_vars)
    safe_dt(derivative_summary_table(derivative_data(), input$eda_vars, response_eps()), page_length = 12)
  })

  output$eda_missing_summary_table <- renderDT({
    req(raw_data())
    safe_dt(missingness_summary_table(raw_data()), page_length = 12)
  })

  output$eda_lagged_corr_table <- renderDT({
    req(processed_data(), input$time_col)
    safe_dt(lagged_correlation_table(processed_data(), input$time_col, input$eda_vars, input$eda_lag_max, "spearman"), page_length = 12)
  })

  output$eda_epsilon_sensitivity_table <- renderDT({
    req(derivative_data(), input$targets)
    safe_dt(epsilon_sensitivity_table(derivative_data(), input$targets), page_length = 12)
  })

  output$run_summary <- renderUI({
    req(selected_data(), input$targets, input$predictors)
    class_tbl <- class_preview()
    risky <- class_tbl %>% filter(.data$status != "ready")
    div(
      class = if (nrow(risky) > 0) "warning-box" else "note-box",
      tags$p(strong("Selected data: "), nrow(selected_data()), " rows, ", length(input$targets), " target(s), ", length(input$predictors), " predictor(s)."),
      tags$p(strong("Representation: "), input$representation_mode, "; ", strong("epsilon: "), signif(response_eps(), 4), "; ", strong("iterations: "), input$model_iter, " with warmup ", input$model_warmup, "."),
      tags$p(strong("Preprocessing: "), "grid points per interval = ", input$points_per_interval, "; smoothing = ", if (isTRUE(input$auto_spar)) "GCV" else input$spar, "; outlier action = ", input$outlier_action, "."),
      if (nrow(risky) > 0) tags$p("Some targets have sparse classes or many blanks. Adaptive jitter may be needed and results should be read cautiously.")
    )
  })

  diagnostics_ready <- reactive({
    res <- maybe_value(cirn_results())
    !is.null(res) && !is.null(res$all_coefficients_combined) && nrow(res$all_coefficients_combined) > 0
  })

  output$rf_button_ui <- renderUI({
    ready <- isTRUE(diagnostics_ready())
    actionButton(
      "run_rf",
      button_label(icon(if (ready) "tree" else "lock"), if (ready) "Run RF Support Check" else "Run CIRN First"),
      class = paste("btn-secondary studio-action-button", if (!ready) "diagnostic-button-disabled" else ""),
      disabled = if (!ready) "disabled" else NULL,
      title = if (ready) "Run Random Forest support diagnostics." else "Run CIRN Algorithm first; Shiny cannot process this while the main CIRN run is still executing."
    )
  })

  output$latent_button_ui <- renderUI({
    ready <- isTRUE(diagnostics_ready())
    actionButton(
      "run_latent",
      button_label(icon(if (ready) "search" else "lock"), if (ready) "Run Latent-Z Screening" else "Run CIRN First"),
      class = paste("btn-secondary studio-action-button", if (!ready) "diagnostic-button-disabled" else ""),
      disabled = if (!ready) "disabled" else NULL,
      title = if (ready) "Run latent-Z exploratory screening." else "Run CIRN Algorithm first; Shiny cannot process this while the main CIRN run is still executing."
    )
  })

  diagnostic_status_box <- function(status, result_tbl = NULL, open_button_id = NULL) {
    state <- status$state %||% "not_run"
    message <- status$message %||% ""
    div(
      class = paste("diagnostic-status", paste0("diagnostic-status-", state)),
      tags$div(class = "diagnostic-status-title", status$label %||% "Diagnostic status"),
      tags$div(class = "diagnostic-status-message", message),
      if (!is.null(result_tbl)) {
        tags$div(class = "diagnostic-status-meta", "Rows available in Diagnostics tab: ", nrow(result_tbl))
      },
      if (!is.null(open_button_id) && identical(state, "completed")) {
        actionButton(open_button_id, button_label(icon("chart-bar"), "Open Diagnostics"), class = "btn-secondary studio-action-button diagnostic-open-button")
      }
    )
  }

  output$rf_status_ui <- renderUI({
    rf <- maybe_value(rf_results())
    status <- rf_status()
    if (identical(status$state, "not_run") && isTRUE(diagnostics_ready())) {
      status <- list(
        state = "ready",
        label = "RF support check",
        message = "Ready to run. Click the button above after reviewing the RF tree, bootstrap, and support-threshold settings."
      )
    }
    diagnostic_status_box(status, rf, "open_rf_diagnostics")
  })

  output$latent_status_ui <- renderUI({
    latent <- maybe_value(latent_results())
    status <- latent_status()
    if (identical(status$state, "not_run") && isTRUE(diagnostics_ready())) {
      status <- list(
        state = "ready",
        label = "Latent-Z screening",
        message = "Ready to run. Click the button above after reviewing the useful-gain threshold."
      )
    }
    diagnostic_status_box(status, latent, "open_latent_diagnostics")
  })

  observeEvent(input$open_rf_diagnostics, {
    updateNavbarPage(session, "main_nav", selected = "Diagnostics")
  })

  observeEvent(input$open_latent_diagnostics, {
    updateNavbarPage(session, "main_nav", selected = "Diagnostics")
  })

  cirn_results <- eventReactive(input$run_cirn, {
    req(selected_data(), input$time_col, input$targets, input$predictors)
    shiny::validate(
      shiny::need(length(input$targets) > 0, "Select at least one target."),
      shiny::need(length(input$predictors) > 0, "Select at least one predictor."),
      shiny::need(input$model_warmup < input$model_iter, "Warmup must be smaller than total iterations."),
      shiny::need(input$prior_sd > 0, "Prior SD must be positive.")
    )

    progress_log(tibble())
    rf_status(list(state = "not_run", label = "RF support check", message = "Run CIRN Algorithm first, then run RF support diagnostics."))
    latent_status(list(state = "not_run", label = "Latent-Z screening", message = "Run CIRN Algorithm first, then run latent-Z screening."))
    spar_value <- if (isTRUE(input$auto_spar)) NULL else input$spar
    grid <- jitter_grid()

    tryCatch(
      withProgress(message = "Running CIRN", value = 0, {
        callback <- function(target, predictor_set, stage, status = NA_character_,
                             completed = 0, total = 1, percent_done = 0,
                             n_predictors = NA_integer_, n_edges = NA_integer_, ...) {
          progress_log(bind_rows(
            progress_log(),
            tibble(
              time = format(Sys.time(), "%H:%M:%S"),
              stage = stage,
              target = target,
              predictor_set = predictor_set,
              status = status,
              completed = completed,
              total = total,
              percent_done = percent_done,
              n_predictors = n_predictors,
              n_edges = n_edges
            )
          ))
          setProgress(percent_done / 100, detail = paste("Target", target, "|", predictor_set, "|", stage))
        }

        infer_network(
          df = selected_data(),
          time_col = input$time_col,
          targets = input$targets,
          predictors = input$predictors,
          representation_mode = input$representation_mode,
          lag_units = input$lag_units,
          run_pairwise = isTRUE(input$run_pairwise),
          pairwise_representation_mode = input$pairwise_representation_mode,
          points_per_interval = input$points_per_interval,
          spar = spar_value,
          outlier_method = "MAD",
          outlier_thresh = input$outlier_thresh,
          outlier_action = input$outlier_action,
          response_eps = response_eps(),
          adaptive_jitter = isTRUE(input$adaptive_jitter),
          jitter_predictors = isTRUE(input$jitter_predictors),
          jitter_min_class_count = input$jitter_min_class_count,
          jitter_scale_grid = grid,
          jitter_scale_basis = input$jitter_scale_basis,
          model_iter = input$model_iter,
          model_warmup = input$model_warmup,
          model_chains = input$model_chains,
          model_cores = input$model_cores,
          prior_mean = input$prior_mean,
          prior_sd = input$prior_sd,
          adapt_delta = input$adapt_delta,
          compute_loo = isTRUE(input$compute_loo),
          seed = input$model_seed,
          show_progress = FALSE,
          progress_bar = FALSE,
          progress_callback = callback,
          debug = FALSE,
          assign_debug_to_global = FALSE
        )
      }),
      error = function(e) {
        msg <- conditionMessage(e)
        progress_log(bind_rows(
          progress_log(),
          tibble(
            time = format(Sys.time(), "%H:%M:%S"),
            stage = "run",
            target = NA_character_,
            predictor_set = NA_character_,
            status = paste("failed:", msg),
            completed = NA_real_,
            total = NA_real_,
            percent_done = NA_real_,
            n_predictors = length(input$predictors),
            n_edges = NA_integer_
          )
        ))
        showNotification(paste("CIRN run did not complete:", msg), type = "error", duration = 12)
        empty_cirn_result(msg)
      }
    )
  }, ignoreInit = TRUE)

  rf_results <- eventReactive(input$run_rf, {
    req(diagnostics_ready(), derivative_data(), input$targets, input$predictors)
    rf_status(list(
      state = "running",
      label = "RF support check",
      message = "Running Random Forest bootstrap diagnostics. Please wait until the completion message appears."
    ))
    result <- tryCatch(
      withProgress(message = "Running RF support check", value = 0, {
        setProgress(0.25, detail = "Preparing derivative-direction classes")
        out <- build_rf_support(
          derivative_data(),
          targets = input$targets,
          predictors = input$predictors,
          time_col = input$time_col,
          eps = response_eps(),
          trees = input$rf_trees,
          boot = input$rf_boot,
          threshold = input$rf_threshold,
          seed = input$model_seed
        )
        setProgress(1, detail = "RF support table ready")
        out
      }),
      error = function(e) e
    )
    if (inherits(result, "error")) {
      msg <- conditionMessage(result)
      rf_status(list(state = "failed", label = "RF support check", message = paste("RF support check failed:", msg)))
      showNotification(paste("RF support check failed:", msg), type = "error", duration = 10)
      return(tibble(status = paste("failed:", msg)))
    }
    supported <- if ("supported" %in% names(result)) sum(result$supported, na.rm = TRUE) else 0L
    rf_status(list(
      state = "completed",
      label = "RF support check",
      message = paste0("Completed. Checked ", nrow(result), " candidate feature-target rows; ", supported, " met the RF support threshold. Results are in Diagnostics > Random Forest Support.")
    ))
    showNotification("RF support check completed. Open Diagnostics to view the table.", type = "message", duration = 6)
    result
  }, ignoreInit = TRUE)

  latent_results <- eventReactive(input$run_latent, {
    req(diagnostics_ready(), derivative_data(), input$targets, input$predictors)
    latent_status(list(
      state = "running",
      label = "Latent-Z screening",
      message = "Running latent-Z screening. Please wait until the completion message appears."
    ))
    result <- tryCatch(
      withProgress(message = "Running latent-Z screening", value = 0, {
        setProgress(0.25, detail = "Constructing latent component")
        out <- build_latent_z(
          derivative_data(),
          targets = input$targets,
          predictors = input$predictors,
          time_col = input$time_col,
          eps = response_eps(),
          min_gain = input$latent_gain_threshold
        )
        setProgress(1, detail = "Latent-Z table ready")
        out
      }),
      error = function(e) e
    )
    if (inherits(result, "error")) {
      msg <- conditionMessage(result)
      latent_status(list(state = "failed", label = "Latent-Z screening", message = paste("Latent-Z screening failed:", msg)))
      showNotification(paste("Latent-Z screening failed:", msg), type = "error", duration = 10)
      return(tibble(status = paste("failed:", msg)))
    }
    useful <- if ("useful" %in% names(result)) sum(result$useful, na.rm = TRUE) else 0L
    latent_status(list(
      state = "completed",
      label = "Latent-Z screening",
      message = paste0("Completed. Screened ", nrow(result), " target row(s); ", useful, " met the useful-gain threshold. Results are in Diagnostics > Latent-Z Screening.")
    ))
    showNotification("Latent-Z screening completed. Open Diagnostics to view the table.", type = "message", duration = 6)
    result
  }, ignoreInit = TRUE)

  output$progress_log_table <- renderDT({
    safe_dt(progress_log(), page_length = 10)
  })

  output$cirn_network <- renderVisNetwork({
    res <- maybe_value(cirn_results())
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view the network."))
    make_network_widget(res$edges_combined, show_legend = FALSE)
  })

  output$coef_plot <- renderPlot({
    res <- maybe_value(cirn_results())
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view coefficient HDIs."))
    plot_hdi_coefficients(
      res$all_coefficients_combined,
      input$coef_retained_only,
      input$coef_max_terms,
      "Combined Coefficient HDIs"
    )
  })

  output$sublevel_network_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view sublevel CIRN."))
    plot_static_network(
      filter_result_mode(res$edges_combined, "sublevel"),
      "Sublevel CIRN Network",
      "Retained state, first-derivative, and second-derivative sublevel edges"
    )
  })

  output$all_predictors_network_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view all-predictors CIRN."))
    plot_static_network(
      filter_result_mode(res$edges_combined, "all_predictors"),
      "All-Predictors CIRN Network",
      "Retained edges from the joint state + derivative predictor model"
    )
  })

  output$pairwise_network_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view pairwise CIRN."))
    plot_static_network(
      filter_result_mode(res$edges_combined, "pairwise"),
      "Pairwise CIRN Network",
      "Retained one target-regulator fit at a time; used as a robustness/support view"
    )
  })

  consistency_summary <- reactive({
    res <- cirn_results()
    if (is.null(res)) {
      return(tibble())
    }
    summarize_mode_consistency(res$edges_combined)
  })

  edge_consistency_summary <- reactive({
    res <- cirn_results()
    if (is.null(res)) {
      return(tibble())
    }
    edge_consistency_from_result(res)
  })

  output$consistent_network_plot <- renderPlot({
    shiny::validate(shiny::need(!is.null(cirn_results()), "Run CIRN Algorithm to view consistency figures."))
    plot_static_network(
      consistency_edges(consistency_summary(), min_modes = 2),
      "Mode-Consistent CIRN Network",
      "Edges retained with the same sign in at least two of: sublevel, all-predictors, pairwise"
    )
  })

  output$three_mode_network_plot <- renderPlot({
    shiny::validate(shiny::need(!is.null(cirn_results()), "Run CIRN Algorithm to view three-mode consistency."))
    plot_static_network(
      consistency_edges(consistency_summary(), min_modes = 3),
      "Three-Mode Consistent CIRN Network",
      "Edges retained with the same sign in sublevel, all-predictors, and pairwise CIRN"
    )
  })

  output$sublevel_coef_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view sublevel coefficient HDIs."))
    plot_hdi_coefficients(
      filter_result_mode(res$all_coefficients_combined, "sublevel"),
      input$coef_retained_only,
      input$coef_max_terms,
      "Sublevel Coefficient HDIs"
    )
  })

  output$all_predictors_coef_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view all-predictors coefficient HDIs."))
    plot_hdi_coefficients(
      filter_result_mode(res$all_coefficients_combined, "all_predictors"),
      input$coef_retained_only,
      input$coef_max_terms,
      "All-Predictors Coefficient HDIs"
    )
  })

  output$pairwise_coef_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view pairwise coefficient HDIs."))
    plot_hdi_coefficients(
      filter_result_mode(res$all_coefficients_combined, "pairwise"),
      input$coef_retained_only,
      input$coef_max_terms,
      "Pairwise Coefficient HDIs"
    )
  })

  output$consistency_heatmap <- renderPlot({
    shiny::validate(shiny::need(!is.null(cirn_results()), "Run CIRN Algorithm to view mode consistency."))
    plot_mode_consistency_grid(edge_consistency_summary())
  })

  output$consistency_table <- renderDT({
    shiny::validate(shiny::need(!is.null(cirn_results()), "Run CIRN Algorithm to view mode consistency."))
    safe_dt(edge_consistency_summary(), page_length = 8)
  })

  output$edge_table <- renderDT({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view retained edges."))
    safe_dt(res$edges_combined, page_length = 12)
  })

  output$coef_table <- renderDT({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view coefficients."))
    safe_dt(res$all_coefficients_combined, page_length = 12)
  })

  output$interpretation_note <- renderUI({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to generate interpretation notes."))
    n_edges <- nrow(res$edges_combined)
    n_jitter <- if (!is.null(res$diagnostics) && nrow(res$diagnostics) > 0) sum(res$diagnostics$jitter_used, na.rm = TRUE) else 0
    div(
      class = if (n_jitter > 0) "warning-box" else "note-box",
      tags$p(strong("Retained signed edges: "), n_edges),
      tags$p(strong("Adaptive jitter fits: "), n_jitter),
      tags$p("Read retained edges together with HDIs, class balance, VIF, jitter diagnostics, and sensitivity stability.")
    )
  })

  output$diagnostics_table <- renderDT({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view diagnostics."))
    safe_dt(res$diagnostics, page_length = 25, scroll_x = TRUE, scroll_y = "520px")
  })

  output$effective_sample_size_table <- renderDT({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view data-sufficiency diagnostics."))
    safe_dt(effective_sample_size_from_result(res), page_length = 25, scroll_x = TRUE, scroll_y = "520px")
  })

  output$pairwise_diagnostics_table <- renderDT({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view pairwise diagnostics."))
    pair_diag <- if (!is.null(res$pairwise)) res$pairwise$diagnostics else tibble()
    safe_dt(pair_diag, page_length = 25, scroll_x = TRUE, scroll_y = "520px")
  })

  output$vif_table <- renderDT({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view VIF diagnostics."))
    safe_dt(res$vif_group, page_length = 25, scroll_x = TRUE, scroll_y = "520px")
  })

  output$vif_pairs_table <- renderDT({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view VIF pair diagnostics."))
    safe_dt(res$vif_pairs %||% tibble(), page_length = 25, scroll_x = TRUE, scroll_y = "520px")
  })

  output$pairwise_vif_pairs_table <- renderDT({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view pairwise VIF pair diagnostics."))
    pair_vif_pairs <- if (!is.null(res$pairwise)) res$pairwise$vif_pairs %||% tibble() else tibble()
    safe_dt(pair_vif_pairs, page_length = 25, scroll_x = TRUE, scroll_y = "520px")
  })

  output$jitter_magnitude_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view jitter diagnostics."))
    plot_jitter_magnitude(res$diagnostics)
  })

  observeEvent(cirn_results(), {
    res <- cirn_results()
    targets <- character()
    if (!is.null(res$diagnostics) && nrow(res$diagnostics) > 0) {
      targets <- unique(res$diagnostics$target[res$diagnostics$jitter_used])
    }
    targets <- sub("^d", "", targets)
    updateSelectInput(
      session,
      "jitter_target",
      choices = targets,
      selected = first_or(targets, character(0))
    )
    updateSelectInput(
      session,
      "allplots_jitter_target",
      choices = targets,
      selected = first_or(targets, character(0))
    )

    bayes_targets <- bayes_target_choices(res)
    updateSelectInput(
      session,
      "allplots_bayes_target",
      choices = bayes_targets,
      selected = first_or(bayes_targets, character(0))
    )
  })

  observeEvent(list(cirn_results(), input$allplots_bayes_target), {
    res <- cirn_results()
    choices <- bayes_representation_choices(res, input$allplots_bayes_target)
    updateSelectInput(
      session,
      "allplots_bayes_representation",
      choices = choices,
      selected = first_or(unname(choices), character(0))
    )
  }, ignoreInit = FALSE)

  output$jitter_detail_plot <- renderPlot({
    req(input$jitter_target)
    spar_value <- if (isTRUE(input$auto_spar)) NULL else input$spar
    plot_adaptive_jitter_diagnostic(
      df = selected_data(),
      target = input$jitter_target,
      time_col = input$time_col,
      points_per_interval = input$points_per_interval,
      spar = spar_value,
      outlier_method = "MAD",
      outlier_thresh = input$outlier_thresh,
      outlier_action = input$outlier_action,
      response_eps = response_eps(),
      jitter_min_class_count = input$jitter_min_class_count,
      jitter_scale_grid = jitter_grid(),
      jitter_scale_basis = input$jitter_scale_basis,
      seed = input$model_seed
    )
  })

  output$rf_table <- renderDT({
    shiny::validate(shiny::need(!is.null(rf_results()), "Run RF support check to view RF diagnostics."))
    safe_dt(rf_results(), page_length = 25, scroll_x = TRUE, scroll_y = "520px")
  })

  output$rf_supported_edges <- renderDT({
    res <- cirn_results()
    rf <- rf_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm first."))
    shiny::validate(shiny::need(!is.null(rf), "Run RF support check to view supported Bayesian edges."))
    tab <- res$edges_combined %>%
      left_join(rf %>% select(target, term, rf_supported = supported, rf_support_rate = support_rate), by = c("target", "term")) %>%
      filter(coalesce(.data$rf_supported, FALSE))
    safe_dt(tab, page_length = 8)
  })

  output$latent_table <- renderDT({
    shiny::validate(shiny::need(!is.null(latent_results()), "Run Latent-Z screening to view diagnostics."))
    safe_dt(latent_results(), page_length = 25, scroll_x = TRUE, scroll_y = "520px")
  })

  parse_numeric_vector <- function(text) {
    x <- suppressWarnings(as.numeric(trimws(unlist(strsplit(text, ",")))))
    x[is.finite(x)]
  }

  sensitivity_scope_config <- reactive({
    resolve_sensitivity_scope(
      input$sensitivity_scope,
      input$representation_mode,
      isTRUE(input$run_pairwise),
      input$pairwise_representation_mode
    )
  })

  output$sensitivity_scope_note <- renderUI({
    cfg <- sensitivity_scope_config()
    box_class <- if (cfg$key %in% c("pairwise_only", "everything")) "warning-box" else "note-box"
    div(
      class = box_class,
      tags$p(strong(cfg$title), ": ", cfg$body),
      tags$p(class = "subtle", cfg$detail)
    )
  })

  sensitivity_plan <- eventReactive(input$preview_sensitivity, {
    make_cirn_sensitivity_plan(
      noise_sd_fractions = parse_numeric_vector(input$sens_noise),
      lag_units = as.integer(parse_numeric_vector(input$sens_lags)),
      downsample_intervals = as.integer(parse_numeric_vector(input$sens_downsample)),
      target_sample_sizes = as.integer(parse_numeric_vector(input$sens_sample_sizes)),
      missing_fractions = parse_numeric_vector(input$sens_missing),
      replicates = input$sens_replicates
    )
  }, ignoreInit = FALSE)

  output$sensitivity_plan_table <- renderDT({
    safe_dt(sensitivity_plan(), page_length = 15, scroll_y = "420px")
  })

  output$sensitivity_progress_status <- renderUI({
    log <- sensitivity_progress_log()
    if (nrow(log) == 0) {
      return(div(class = "diagnostic-status diagnostic-status-ready",
                 tags$div(class = "diagnostic-status-title", "Sensitivity progress"),
                 tags$div(class = "diagnostic-status-message", "Ready. Click Run Sensitivity to execute the scenarios in the previewed plan.")))
    }
    last <- log[nrow(log), , drop = FALSE]
    total <- suppressWarnings(as.integer(last$total))
    completed <- suppressWarnings(as.integer(last$completed))
    percent_done <- suppressWarnings(as.numeric(last$percent_done))
    completed <- ifelse(is.finite(completed), completed, 0L)
    total <- ifelse(is.finite(total) && total > 0, total, nrow(sensitivity_plan()))
    percent_done <- ifelse(is.finite(percent_done), percent_done, 100 * completed / max(total, 1L))
    failed <- sum(grepl("^failed", log$status %||% character()), na.rm = TRUE)
    state <- if (completed >= total) {
      if (failed > 0) "failed" else "completed"
    } else {
      "running"
    }
    div(
      class = paste("diagnostic-status", paste0("diagnostic-status-", state)),
      tags$div(class = "diagnostic-status-title", "Sensitivity progress"),
      tags$div(class = "diagnostic-status-message", completed, " of ", total, " scenario runs completed."),
      tags$div(class = "diagnostic-status-meta", "Overall progress: ", round(percent_done, 1), "%."),
      tags$div(class = "diagnostic-status-meta", "Current/last scenario: ", last$condition, " | status: ", last$status),
      if (completed >= total) tags$div(class = "diagnostic-status-meta", "Completed runs table and edge-stability summaries are now available on this tab.")
    )
  })

  output$sensitivity_progress_table <- renderDT({
    log <- sensitivity_progress_log()
    if (nrow(log) == 0) {
      return(safe_dt(tibble(message = "No sensitivity run has started yet."), page_length = 5))
    }
    safe_dt(log, page_length = 25, scroll_y = "520px")
  })

  sensitivity_results <- eventReactive(input$run_sensitivity, {
    req(selected_data(), input$time_col, input$targets, input$predictors)
    spar_value <- if (isTRUE(input$auto_spar)) NULL else input$spar
    scope_cfg <- sensitivity_scope_config()
    base_config <- list(
      targets = input$targets,
      predictors = input$predictors,
      representation_mode = input$representation_mode,
      lag_units = input$lag_units,
      run_pairwise = isTRUE(input$run_pairwise),
      pairwise_representation_mode = input$pairwise_representation_mode,
      points_per_interval = input$points_per_interval,
      spar = spar_value,
      outlier_method = "MAD",
      outlier_thresh = input$outlier_thresh,
      outlier_action = input$outlier_action,
      response_eps = response_eps(),
      adaptive_jitter = isTRUE(input$adaptive_jitter),
      jitter_predictors = isTRUE(input$jitter_predictors),
      jitter_min_class_count = input$jitter_min_class_count,
      jitter_scale_grid = jitter_grid(),
      jitter_scale_basis = input$jitter_scale_basis,
      model_iter = input$model_iter,
      model_warmup = input$model_warmup,
      model_chains = input$model_chains,
      model_cores = input$model_cores,
      prior_mean = input$prior_mean,
      prior_sd = input$prior_sd,
      adapt_delta = input$adapt_delta,
      compute_loo = FALSE,
      seed = input$model_seed,
      sensitivity_inference_scope = input$sensitivity_scope,
      sensitivity_replicates = input$sens_replicates,
      sensitivity_noise_sd_fractions = parse_numeric_vector(input$sens_noise),
      sensitivity_lag_units = as.integer(parse_numeric_vector(input$sens_lags)),
      sensitivity_downsample_intervals = as.integer(parse_numeric_vector(input$sens_downsample)),
      sensitivity_target_sample_sizes = as.integer(parse_numeric_vector(input$sens_sample_sizes)),
      sensitivity_missing_fractions = parse_numeric_vector(input$sens_missing)
    )
    plan_tbl <- sensitivity_plan()
    sensitivity_progress_log(tibble())
    result <- withProgress(message = "Running sensitivity analysis", value = 0, {
      progress_callback <- function(row_index,
                                    total,
                                    completed,
                                    percent_done,
                                    sensitivity_id,
                                    condition,
                                    scenario,
                                    scenario_value,
                                    replicate,
                                    status,
                                    message = NA_character_,
                                    scenario_n = NA_integer_,
                                    runtime_seconds = NA_real_) {
        progress_value <- max(0, min(1, percent_done / 100))
        setProgress(
          progress_value,
          detail = paste0(
            "Scenario ", row_index, " of ", total,
            ": ", condition,
            " | ", status
          )
        )
        sensitivity_progress_log(bind_rows(
          sensitivity_progress_log(),
          tibble(
            time = format(Sys.time(), "%H:%M:%S"),
            sensitivity_id = sensitivity_id,
            row_index = row_index,
            total = total,
            completed = completed,
            percent_done = round(percent_done, 1),
            condition = condition,
            scenario = scenario,
            scenario_value = scenario_value,
            replicate = replicate,
            status = status,
            message = message,
            scenario_n = scenario_n,
            runtime_seconds = runtime_seconds
          )
        ))
      }
      out <- run_cirn_sensitivity_analysis(
        df = selected_data(),
        time_col = input$time_col,
        base_config = base_config,
        sensitivity_plan = plan_tbl,
        sensitivity_inference_scope = input$sensitivity_scope,
        true_adj = truth_matrix(),
        output_dir = if (isTRUE(input$sens_save_outputs)) tempdir() else NULL,
        save_outputs = isTRUE(input$sens_save_outputs),
        show_progress = FALSE,
        progress_bar = FALSE,
        progress_callback = progress_callback
      )
      setProgress(1, detail = "Sensitivity analysis completed")
      out
    })
    result$sensitivity_scope <- result$sensitivity_scope %||% scope_cfg
    result
  }, ignoreInit = TRUE)

  output$sensitivity_runs_table <- renderDT({
    shiny::validate(shiny::need(!is.null(sensitivity_results()), "Run sensitivity analysis to view results."))
    safe_dt(sensitivity_results()$runs, page_length = 25, scroll_y = "520px")
  })

  feature_sensitivity_stability <- reactive({
    sr <- sensitivity_results()
    scope_cfg <- sr$sensitivity_scope %||% sensitivity_scope_config()
    feature_tbl <- as_result_tbl(sr$feature_edge_stability)
    if (nrow(feature_tbl) > 0) {
      return(feature_tbl)
    }
    summarize_sensitivity_feature_edge_stability(
      sr,
      edge_focus = scope_cfg$edge_focus
    )
  })

  output$sensitivity_stability_table <- renderDT({
    shiny::validate(shiny::need(!is.null(sensitivity_results()), "Run sensitivity analysis to view edge stability."))
    safe_dt(feature_sensitivity_stability(), page_length = 25, scroll_y = "520px")
  })

  output$sensitivity_state_stability_table <- renderDT({
    shiny::validate(shiny::need(!is.null(sensitivity_results()), "Run sensitivity analysis to view state-level edge stability."))
    safe_dt(sensitivity_results()$edge_stability %||% tibble(), page_length = 25, scroll_y = "520px")
  })

  output$sensitivity_edges_table <- renderDT({
    shiny::validate(shiny::need(!is.null(sensitivity_results()), "Run sensitivity analysis to view retained sensitivity edges."))
    safe_dt(sensitivity_results()$edges %||% tibble(), page_length = 25, scroll_y = "520px")
  })

  output$sensitivity_diagnostics_table <- renderDT({
    shiny::validate(shiny::need(!is.null(sensitivity_results()), "Run sensitivity analysis to view sensitivity diagnostics."))
    safe_dt(sensitivity_results()$diagnostics %||% tibble(), page_length = 25, scroll_y = "520px")
  })

  output$sensitivity_sample_size_table <- renderDT({
    shiny::validate(shiny::need(!is.null(sensitivity_results()), "Run sensitivity analysis to view sensitivity data-sufficiency diagnostics."))
    safe_dt(sensitivity_results()$effective_sample_size %||% tibble(), page_length = 25, scroll_y = "520px")
  })

  output$sensitivity_truth_metrics_table <- renderDT({
    shiny::validate(shiny::need(!is.null(sensitivity_results()), "Run sensitivity analysis to view sensitivity ground-truth metrics."))
    safe_dt(sensitivity_results()$ground_truth_metrics %||% tibble(), page_length = 25, scroll_y = "520px")
  })

  output$sensitivity_heatmap <- renderPlot({
    shiny::validate(shiny::need(!is.null(sensitivity_results()), "Run sensitivity analysis to view edge stability."))
    cirn_plot_typography(plot_sensitivity_feature_heatmap(feature_sensitivity_stability()))
  })

  benchmark_results <- eventReactive(input$run_benchmark, {
    req(cirn_results())
    mat <- truth_matrix()
    shiny::validate(shiny::need(!is.null(mat), "Upload a ground-truth matrix or use a built-in simulation with true adjacency."))
    metrics <- evaluate_representation_agnostic(mat, prepare_edges_for_benchmark(cirn_results()$edges_combined))
    list(metrics = metrics, metrics_table = metrics_to_long_table(metrics), truth = mat)
  }, ignoreInit = TRUE)

  inferred_state_adj <- reactive({
    res <- cirn_results()
    mat <- truth_matrix()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm first."))
    if (is.null(mat)) {
      vars <- unique(c(input$targets, input$predictors))
      mat <- matrix(0, length(vars), length(vars), dimnames = list(vars, vars))
    }
    pred <- matrix(0, nrow(mat), ncol(mat), dimnames = dimnames(mat))
    edges <- prepare_edges_for_benchmark(res$edges_combined)
    if (!is.null(edges) && nrow(edges) > 0) {
      for (i in seq_len(nrow(edges))) {
        source <- base_term(edges$term[[i]])
        target <- sub("^d", "", edges$target[[i]])
        if (source %in% rownames(pred) && target %in% colnames(pred)) {
          pred[source, target] <- sign(edges$omega[[i]])
        }
      }
    }
    pred
  })

  plot_adj_heatmap <- function(mat, title) {
    as.data.frame(as.table(mat)) %>%
      setNames(c("source", "target", "value")) %>%
      ggplot(aes(source, target, fill = value)) +
      geom_tile(color = "white") +
      geom_text(aes(label = value), size = 4) +
      scale_fill_gradient2(low = CIRN_INHIBITION_COLOR, mid = "#F7F7F7", high = CIRN_ACTIVATION_COLOR, limits = c(-1, 1)) +
      labs(title = title, x = "Source", y = "Target", fill = "Sign") +
      theme_minimal(base_size = CIRN_PLOT_BASE_SIZE) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  }

  output$allplots_bayes_posterior_density_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view Bayesian posterior plots."))
    shiny::validate(shiny::need(nzchar(input$allplots_bayes_target %||% ""), "Choose a Bayesian target."))
    shiny::validate(shiny::need(nzchar(input$allplots_bayes_representation %||% ""), "Choose a Bayesian representation."))
    model <- selected_bayes_model(res, input$allplots_bayes_target, input$allplots_bayes_representation)
    shiny::validate(shiny::need(!is.null(model), "No fitted Bayesian model is available for this target/representation."))
    draw_plot_object(plot_selected_model_posteriors(model, input$allplots_bayes_max_terms))
  })

  output$allplots_bayes_cross_posterior_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view Bayesian posterior plots."))
    shiny::validate(shiny::need(nzchar(input$allplots_bayes_target %||% ""), "Choose a Bayesian target."))
    draw_plot_object(plot_target_representation_posteriors(res, input$allplots_bayes_target))
  })

  output$allplots_bayes_hist_trace_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view MCMC diagnostics."))
    shiny::validate(shiny::need(nzchar(input$allplots_bayes_target %||% ""), "Choose a Bayesian target."))
    shiny::validate(shiny::need(nzchar(input$allplots_bayes_representation %||% ""), "Choose a Bayesian representation."))
    model <- selected_bayes_model(res, input$allplots_bayes_target, input$allplots_bayes_representation)
    shiny::validate(shiny::need(!is.null(model), "No fitted Bayesian model is available for this target/representation."))
    draw_plot_object(plot_selected_model_hist_trace(model, input$allplots_bayes_max_terms))
  })

  output$allplots_bayes_trace_burnin_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view MCMC traces."))
    shiny::validate(shiny::need(nzchar(input$allplots_bayes_target %||% ""), "Choose a Bayesian target."))
    draw_plot_object(plot_target_trace_with_burnin(res, input$allplots_bayes_target))
  })

  output$allplots_bayes_diagnostics_summary_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view Bayesian diagnostics."))
    cirn_plot_typography(plot_diagnostics_summary(res$diagnostics))
  })

  output$allplots_missing_plot <- renderPlot({
    cirn_plot_typography(plot_missingness(raw_data()))
  })

  output$allplots_simulation_plot <- renderPlot({
    df <- simulation_data()
    cirn_plot_typography(plot_time_series(df, "t", setdiff(names(df), "t")))
  })

  output$allplots_simulation_truth_network <- renderVisNetwork({
    mat <- truth_for_system(input$sim_system)
    if (is.null(mat)) {
      return(visNetwork(data.frame(id = "Custom", label = "Custom truth not specified"), data.frame()))
    }
    edge_tbl <- as.data.frame(as.table(mat)) %>%
      setNames(c("term", "target", "omega")) %>%
      filter(.data$omega != 0) %>%
      mutate(
        hdi_lower95 = .data$omega,
        hdi_upper95 = .data$omega,
        jitter_used = FALSE
    )
    make_network_widget(edge_tbl, height = "430px", show_legend = FALSE)
  })

  output$allplots_eda_raw_small_multiples_plot <- renderPlot({
    req(raw_data(), input$time_col)
    cirn_plot_typography(plot_time_series_small_multiples(raw_data(), input$time_col, input$eda_vars))
  })

  output$allplots_eda_time_plot <- renderPlot({
    req(processed_data(), input$time_col, input$eda_vars)
    cirn_plot_typography(plot_time_series(processed_data(), input$time_col, input$eda_vars))
  })

  output$allplots_eda_normalized_overlay_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_normalized_overlay(processed_data(), input$time_col, input$eda_vars))
  })

  output$allplots_eda_distribution_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_variable_distributions(processed_data(), input$time_col, input$eda_vars))
  })

  output$allplots_eda_outlier_timeline_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_outlier_timeline(processed_data(), input$time_col, input$eda_vars, input$eda_outlier_thresh))
  })

  output$allplots_eda_time_gap_plot <- renderPlot({
    req(raw_data(), input$time_col)
    cirn_plot_typography(plot_time_gap_profile(raw_data(), input$time_col))
  })

  output$allplots_phase_plot <- renderPlot({
    cirn_plot_typography(plot_phase_plane(processed_data(), input$time_col, input$phase_x, input$phase_y))
  })

  output$allplots_eda_phase_pairs_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_phase_pairs(processed_data(), input$time_col, input$eda_vars))
  })

  output$allplots_corr_plot <- renderPlot({
    req(processed_data(), input$eda_vars)
    cirn_plot_typography(plot_correlation_heatmap(processed_data(), input$eda_vars))
  })

  output$allplots_eda_pairwise_scatter_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_pairwise_scatter_matrix(processed_data(), input$time_col, input$eda_vars))
  })

  output$allplots_eda_lagged_corr_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_lagged_correlation_heatmap(processed_data(), input$time_col, input$eda_vars, input$eda_lag_max, "spearman"))
  })

  output$allplots_eda_acf_pacf_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_acf_pacf(processed_data(), input$time_col, input$eda_vars, input$eda_lag_max))
  })

  output$allplots_eda_pca_plot <- renderPlot({
    req(processed_data(), input$time_col)
    cirn_plot_typography(plot_pca_trajectory(processed_data(), input$time_col, input$eda_vars))
  })

  output$allplots_eda_missing_plot <- renderPlot({
    cirn_plot_typography(plot_missingness(processed_data()))
  })

  output$allplots_eda_missing_summary_plot <- renderPlot({
    req(raw_data())
    cirn_plot_typography(plot_missingness_summary(raw_data()))
  })

  output$allplots_derivative_plot <- renderPlot({
    req(derivative_data(), input$targets)
    cirn_plot_typography(plot_derivative_orders(derivative_data(), input$time_col, input$targets))
  })

  output$allplots_class_balance_plot <- renderPlot({
    cirn_plot_typography(plot_class_balance(class_preview()))
  })

  output$allplots_eda_derivative_state_plot <- renderPlot({
    req(derivative_data(), input$time_col, input$eda_vars, input$targets)
    cirn_plot_typography(plot_derivative_state_relationships(derivative_data(), input$time_col, input$eda_vars, input$targets))
  })

  output$allplots_eda_derivative_distribution_plot <- renderPlot({
    req(derivative_data(), input$time_col, input$eda_vars)
    cirn_plot_typography(plot_derivative_distributions(derivative_data(), input$time_col, input$eda_vars))
  })

  output$allplots_eda_epsilon_sensitivity_plot <- renderPlot({
    req(derivative_data(), input$targets)
    cirn_plot_typography(plot_epsilon_sensitivity(derivative_data(), input$targets, input$response_eps_log))
  })

  output$allplots_cirn_network <- renderVisNetwork({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view the network."))
    make_network_widget(res$edges_combined, show_legend = FALSE)
  })

  output$allplots_sublevel_network_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view sublevel CIRN."))
    plot_static_network(
      filter_result_mode(res$edges_combined, "sublevel"),
      "Sublevel CIRN Network",
      "Retained state, first-derivative, and second-derivative sublevel edges"
    )
  })

  output$allplots_all_predictors_network_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view all-predictors CIRN."))
    plot_static_network(
      filter_result_mode(res$edges_combined, "all_predictors"),
      "All-Predictors CIRN Network",
      "Retained edges from the joint state + derivative predictor model"
    )
  })

  output$allplots_pairwise_network_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view pairwise CIRN."))
    plot_static_network(
      filter_result_mode(res$edges_combined, "pairwise"),
      "Pairwise CIRN Network",
      "Retained one target-regulator fit at a time; used as a robustness/support view"
    )
  })

  output$allplots_consistent_network_plot <- renderPlot({
    shiny::validate(shiny::need(!is.null(cirn_results()), "Run CIRN Algorithm to view consistency figures."))
    plot_static_network(
      consistency_edges(consistency_summary(), min_modes = 2),
      "Mode-Consistent CIRN Network",
      "Edges retained with the same sign in at least two of: sublevel, all-predictors, pairwise"
    )
  })

  output$allplots_three_mode_network_plot <- renderPlot({
    shiny::validate(shiny::need(!is.null(cirn_results()), "Run CIRN Algorithm to view three-mode consistency."))
    plot_static_network(
      consistency_edges(consistency_summary(), min_modes = 3),
      "Three-Mode Consistent CIRN Network",
      "Edges retained with the same sign in sublevel, all-predictors, and pairwise CIRN"
    )
  })

  output$allplots_coef_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view coefficient HDIs."))
    plot_hdi_coefficients(
      res$all_coefficients_combined,
      input$allplots_coef_retained_only,
      input$allplots_coef_max_terms,
      "Combined Coefficient HDIs"
    )
  })

  output$allplots_sublevel_coef_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view sublevel coefficient HDIs."))
    plot_hdi_coefficients(
      filter_result_mode(res$all_coefficients_combined, "sublevel"),
      input$allplots_coef_retained_only,
      input$allplots_coef_max_terms,
      "Sublevel Coefficient HDIs"
    )
  })

  output$allplots_all_predictors_coef_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view all-predictors coefficient HDIs."))
    plot_hdi_coefficients(
      filter_result_mode(res$all_coefficients_combined, "all_predictors"),
      input$allplots_coef_retained_only,
      input$allplots_coef_max_terms,
      "All-Predictors Coefficient HDIs"
    )
  })

  output$allplots_pairwise_coef_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view pairwise coefficient HDIs."))
    plot_hdi_coefficients(
      filter_result_mode(res$all_coefficients_combined, "pairwise"),
      input$allplots_coef_retained_only,
      input$allplots_coef_max_terms,
      "Pairwise Coefficient HDIs"
    )
  })

  output$allplots_consistency_heatmap <- renderPlot({
    shiny::validate(shiny::need(!is.null(cirn_results()), "Run CIRN Algorithm to view mode consistency."))
    plot_mode_consistency_grid(edge_consistency_summary())
  })

  output$allplots_jitter_magnitude_plot <- renderPlot({
    res <- cirn_results()
    shiny::validate(shiny::need(!is.null(res), "Run CIRN Algorithm to view jitter diagnostics."))
    plot_jitter_magnitude(res$diagnostics)
  })

  output$allplots_jitter_detail_plot <- renderPlot({
    req(input$allplots_jitter_target)
    spar_value <- if (isTRUE(input$auto_spar)) NULL else input$spar
    plot_adaptive_jitter_diagnostic(
      df = selected_data(),
      target = input$allplots_jitter_target,
      time_col = input$time_col,
      points_per_interval = input$points_per_interval,
      spar = spar_value,
      outlier_method = "MAD",
      outlier_thresh = input$outlier_thresh,
      outlier_action = input$outlier_action,
      response_eps = response_eps(),
      jitter_min_class_count = input$jitter_min_class_count,
      jitter_scale_grid = jitter_grid(),
      jitter_scale_basis = input$jitter_scale_basis,
      seed = input$model_seed
    )
  })

  output$allplots_sensitivity_heatmap <- renderPlot({
    shiny::validate(shiny::need(!is.null(sensitivity_results()), "Run sensitivity analysis to view edge stability."))
    plot_sensitivity_feature_heatmap(feature_sensitivity_stability())
  })

  output$allplots_truth_heatmap <- renderPlot({
    mat <- truth_matrix()
    shiny::validate(shiny::need(!is.null(mat), "No ground truth available."))
    plot_adj_heatmap(mat, "Ground truth")
  })

  output$allplots_inferred_heatmap <- renderPlot({
    plot_adj_heatmap(inferred_state_adj(), "Inferred")
  })

  output$benchmark_metrics <- renderDT({
    shiny::validate(shiny::need(!is.null(benchmark_results()), "Run benchmark to view metrics."))
    safe_dt(benchmark_results()$metrics_table, page_length = 20, scroll_x = FALSE)
  })

  output$truth_heatmap <- renderPlot({
    mat <- truth_matrix()
    shiny::validate(shiny::need(!is.null(mat), "No ground truth available."))
    plot_adj_heatmap(mat, "Ground truth")
  })

  output$inferred_heatmap <- renderPlot({
    plot_adj_heatmap(inferred_state_adj(), "Inferred")
  })

  output$benchmark_note <- renderUI({
    div(
      class = "method-box",
      "Benchmarking is valid only for simulations or datasets with a known signed ground-truth network. ",
      "Wrong-sign recoveries count as both false positives and false negatives in the signed evaluation."
    )
  })

  current_settings <- reactive({
    make_settings_list(input, response_eps(), jitter_grid())
  })

  feedback_record <- reactive({
    dat <- maybe_value(selected_data())
    deriv <- maybe_value(derivative_data())
    res <- maybe_value(cirn_results())
    include_context <- isTRUE(input$feedback_include_context)

    context <- list()
    if (include_context) {
      context <- list(
        app = "CIRN Studio",
        current_tab = input$main_nav %||% "",
        data_mode = input$data_mode %||% "",
        time_column = input$time_col %||% "",
        data_rows = if (!is.null(dat)) nrow(dat) else NA_integer_,
        data_columns = if (!is.null(dat)) names(dat) else character(),
        derivative_rows = if (!is.null(deriv)) nrow(deriv) else NA_integer_,
        targets = input$targets %||% character(),
        predictors = input$predictors %||% character(),
        normalization = input$normalization %||% "",
        points_per_interval = input$points_per_interval %||% NA_integer_,
        response_eps = maybe_value(response_eps()) %||% NA_real_,
        representation_mode = input$representation_mode %||% "",
        run_pairwise = isTRUE(input$run_pairwise),
        pairwise_representation_mode = input$pairwise_representation_mode %||% "",
        model_iter = input$model_iter %||% NA_integer_,
        model_warmup = input$model_warmup %||% NA_integer_,
        model_chains = input$model_chains %||% NA_integer_,
        model_cores = input$model_cores %||% NA_integer_,
        adapt_delta = input$adapt_delta %||% NA_real_,
        cirn_fitted = !is.null(res),
        retained_edges = if (!is.null(res) && !is.null(res$edges_combined)) nrow(res$edges_combined) else NA_integer_,
        r_version = as.character(getRversion()),
        shiny_version = as.character(utils::packageVersion("shiny")),
        platform = R.version$platform,
        host = session$clientData$url_hostname %||% "",
        browser_width = session$clientData$output_cirn_network_width %||% NA_integer_
      )
    }

    list(
      timestamp = as.character(Sys.time()),
      type = input$feedback_type %||% "",
      severity = input$feedback_severity %||% "",
      area = input$feedback_area %||% "",
      summary = input$feedback_summary %||% "",
      details = input$feedback_details %||% "",
      steps = input$feedback_steps %||% "",
      expected = input$feedback_expected %||% "",
      actual = input$feedback_actual %||% "",
      name = input$feedback_name %||% "",
      email = input$feedback_email %||% "",
      affiliation = input$feedback_affiliation %||% "",
      follow_up = isTRUE(input$feedback_followup),
      context = context
    )
  })

  output$feedback_contact_note <- renderUI({
    if (nzchar(feedback_contact_email)) {
      div(
        class = "note-box",
        tags$p(
          strong("This app does not send email automatically. "),
          "After completing the one-sentence summary, click ",
          strong("Open Email Draft"),
          " to prepare a message in your default email application. Review the message and click Send yourself."
        ),
        tags$p(
          strong("Feedback recipients: "),
          gsub(",", ", ", feedback_contact_email, fixed = TRUE)
        )
      )
    } else {
      div(
        class = "warning-box",
        strong("Feedback email is not configured. "),
        "You may still download the feedback report or CSV record."
      )
    }
  })

  output$feedback_preview <- renderText({
    format_feedback_report(feedback_record())
  })

  output$feedback_email_link <- renderUI({
    record <- feedback_record()
    summary_ready <- nzchar(trimws(record$summary %||% ""))

    if (!summary_ready) {
      return(div(class = "warning-box", "Write a one-sentence summary to prepare an email draft."))
    }

    if (!nzchar(feedback_contact_email)) {
      return(div(class = "warning-box", "Email draft is disabled until CIRN_FEEDBACK_EMAIL is configured."))
    }

    tags$a(
      href = feedback_mailto_uri(record, feedback_contact_email),
      target = "_blank",
      class = "btn btn-primary studio-action-button feedback-mail-button",
      button_label(icon("envelope"), "Open Email Draft")
    )
  })

  feedback_saved_path <- reactiveVal("")

  observeEvent(input$save_feedback_local, {
    record <- feedback_record()
    if (!nzchar(trimws(record$summary %||% ""))) {
      showNotification("Please write a one-sentence summary before saving feedback.", type = "warning")
      return(NULL)
    }

    path <- tryCatch({
      dir.create(feedback_default_dir, recursive = TRUE, showWarnings = FALSE)
      if (!dir.exists(feedback_default_dir)) {
        stop("Feedback directory could not be created.")
      }
      log_path <- file.path(feedback_default_dir, "CIRN_feedback_log.csv")
      row <- feedback_record_row(record)
      if (file.exists(log_path)) {
        readr::write_csv(row, log_path, append = TRUE)
      } else {
        readr::write_csv(row, log_path)
      }
      log_path
    }, error = function(e) {
      fallback_dir <- file.path(tempdir(), "cirn_feedback")
      dir.create(fallback_dir, recursive = TRUE, showWarnings = FALSE)
      fallback_path <- file.path(fallback_dir, "CIRN_feedback_log.csv")
      row <- feedback_record_row(record)
      if (file.exists(fallback_path)) {
        readr::write_csv(row, fallback_path, append = TRUE)
      } else {
        readr::write_csv(row, fallback_path)
      }
      fallback_path
    })

    feedback_saved_path(path)
    showNotification("Feedback record saved on the Shiny server.", type = "message")
  })

  output$feedback_log_status <- renderUI({
    saved <- feedback_saved_path()
    if (!nzchar(saved)) {
      return(NULL)
    }
    div(
      class = "note-box",
      strong("Latest saved copy: "),
      saved
    )
  })

  output$download_feedback_txt <- downloadHandler(
    filename = function() paste0("CIRN_feedback_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"),
    content = function(file) {
      writeLines(format_feedback_report(feedback_record()), file)
    }
  )

  output$download_feedback_csv <- downloadHandler(
    filename = function() paste0("CIRN_feedback_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"),
    content = function(file) {
      readr::write_csv(feedback_record_row(feedback_record()), file)
    }
  )

  output$settings_text <- renderPrint({
    str(current_settings(), max.level = 2)
  })

  output$export_manifest <- renderDT({
    res <- maybe_value(cirn_results())
    rf <- maybe_value(rf_results())
    latent <- maybe_value(latent_results())
    sens <- maybe_value(sensitivity_results())
    bench <- maybe_value(benchmark_results())
    rows <- tibble(
      item = c(
        "Selected data", "Derivative preview", "Class balance", "CIRN edges",
        "Edge consistency", "All coefficients", "Diagnostics", "Data sufficiency",
        "VIF", "VIF pair diagnostics", "RF support", "Latent-Z", "Sensitivity plan",
        "Sensitivity runs", "Sensitivity retained edges", "Sensitivity diagnostics",
        "Sensitivity state stability", "Sensitivity feature stability", "Sensitivity scope",
        "Sensitivity benchmark metrics", "Benchmark", "Figure pack", "Settings",
        "Session info", "App runtime algorithm"
      ),
      available = c(
        !is.null(selected_data()),
        !is.null(derivative_data()),
        !is.null(class_preview()),
        !is.null(res),
        !is.null(res),
        !is.null(res),
        !is.null(res),
        !is.null(res),
        !is.null(res),
        !is.null(res),
        !is.null(rf),
        !is.null(latent),
        !is.null(sens),
        !is.null(sens),
        !is.null(sens),
        !is.null(sens),
        !is.null(sens),
        !is.null(sens),
        !is.null(sens),
        !is.null(sens) && !is.null(sens$ground_truth_metrics),
        !is.null(bench),
        !is.null(res),
        TRUE,
        TRUE,
        file.exists(algorithm_path)
      )
    )
    safe_dt(rows, page_length = 20, scroll_x = FALSE)
  })

  session_info_text <- function() {
    paste(capture.output(sessionInfo()), collapse = "\n")
  }

  write_plot_png <- function(path, plot_fun, width = 1600, height = 1100, res = 180) {
    grDevices::png(path, width = width, height = height, res = res)
    on.exit(grDevices::dev.off(), add = TRUE)
    plot_obj <- tryCatch(
      plot_fun(),
      error = function(e) {
        plot.new()
        text(0.5, 0.5, paste("Plot unavailable:", conditionMessage(e)), cex = 0.85)
        NULL
      }
    )
    if (inherits(plot_obj, c("gtable", "grob", "gTree"))) {
      grid::grid.draw(plot_obj)
    } else if (!is.null(plot_obj)) {
      print(plot_obj)
    }
    invisible(path)
  }

  default_bayes_selection <- function(res) {
    targets <- bayes_target_choices(res)
    if (length(targets) == 0) {
      return(list(target = character(0), representation = character(0), model = NULL))
    }
    target <- first_or(input$allplots_bayes_target, targets[[1]])
    if (!target %in% targets) target <- targets[[1]]
    reps <- bayes_representation_choices(res, target)
    rep_value <- first_or(input$allplots_bayes_representation, first_or(unname(reps), character(0)))
    if (!rep_value %in% unname(reps)) rep_value <- first_or(unname(reps), character(0))
    list(target = target, representation = rep_value, model = selected_bayes_model(res, target, rep_value))
  }

  write_figure_pack <- function(file) {
    tmp <- tempfile("cirn_figures_")
    dir.create(tmp, recursive = TRUE, showWarnings = FALSE)
    paths <- character()

    add_fig <- function(name, plot_fun, width = 1600, height = 1100, res = 180) {
      path <- file.path(tmp, paste0(name, ".png"))
      write_plot_png(path, plot_fun, width = width, height = height, res = res)
      paths <<- c(paths, path)
    }

    res <- maybe_value(cirn_results())
    if (!is.null(res)) {
      add_fig(
        "01_sublevel_network",
        function() plot_static_network(
          filter_result_mode(res$edges_combined, "sublevel"),
          "Sublevel CIRN Network",
          "Retained state, first-derivative, and second-derivative sublevel edges"
        ),
        width = 1800, height = 1300
      )
      add_fig(
        "02_all_predictors_network",
        function() plot_static_network(
          filter_result_mode(res$edges_combined, "all_predictors"),
          "All-Predictors CIRN Network",
          "Retained edges from the joint state + derivative predictor model"
        ),
        width = 1800, height = 1300
      )
      add_fig(
        "03_pairwise_network",
        function() plot_static_network(
          filter_result_mode(res$edges_combined, "pairwise"),
          "Pairwise CIRN Network",
          "Retained one target-regulator fit at a time; used as a robustness/support view"
        ),
        width = 1800, height = 1300
      )
      add_fig(
        "04_mode_consistent_ge2_network",
        function() plot_static_network(
          consistency_edges(summarize_mode_consistency(res$edges_combined), min_modes = 2),
          "Mode-Consistent CIRN Network",
          "Edges retained with the same sign in at least two modes"
        ),
        width = 1800, height = 1300
      )
      add_fig(
        "05_three_mode_consistent_network",
        function() plot_static_network(
          consistency_edges(summarize_mode_consistency(res$edges_combined), min_modes = 3),
          "Three-Mode Consistent CIRN Network",
          "Edges retained with the same sign in sublevel, all-predictors, and pairwise CIRN"
        ),
        width = 1800, height = 1300
      )
      add_fig("06_edge_consistency_grid", function() plot_mode_consistency_grid(edge_consistency_from_result(res)), width = 1800, height = 1300)
      add_fig("07_combined_coefficient_hdi", function() plot_hdi_coefficients(res$all_coefficients_combined, FALSE, input$allplots_coef_max_terms %||% 80, "Combined Coefficient HDIs"), width = 1800, height = 1300)
      add_fig("08_sublevel_coefficient_hdi", function() plot_hdi_coefficients(filter_result_mode(res$all_coefficients_combined, "sublevel"), FALSE, input$allplots_coef_max_terms %||% 80, "Sublevel Coefficient HDIs"), width = 1800, height = 1300)
      add_fig("09_all_predictors_coefficient_hdi", function() plot_hdi_coefficients(filter_result_mode(res$all_coefficients_combined, "all_predictors"), FALSE, input$allplots_coef_max_terms %||% 80, "All-Predictors Coefficient HDIs"), width = 1800, height = 1300)
      add_fig("10_pairwise_coefficient_hdi", function() plot_hdi_coefficients(filter_result_mode(res$all_coefficients_combined, "pairwise"), FALSE, input$allplots_coef_max_terms %||% 80, "Pairwise Coefficient HDIs"), width = 1800, height = 1300)
      add_fig("11_bayesian_diagnostics_summary", function() plot_diagnostics_summary(res$diagnostics), width = 1700, height = 1000)
      add_fig("12_adaptive_jitter_magnitude", function() plot_jitter_magnitude(res$diagnostics), width = 1500, height = 950)

      bayes_sel <- default_bayes_selection(res)
      if (!is.null(bayes_sel$model)) {
        add_fig("13_bayesian_posterior_density", function() plot_selected_model_posteriors(bayes_sel$model, input$allplots_bayes_max_terms %||% 12), width = 1800, height = 1300)
        add_fig("14_bayesian_hist_trace", function() plot_selected_model_hist_trace(bayes_sel$model, input$allplots_bayes_max_terms %||% 8), width = 1800, height = 1300)
        add_fig("15_bayesian_cross_representation_posterior", function() plot_target_representation_posteriors(res, bayes_sel$target), width = 1800, height = 1300)
        add_fig("16_bayesian_target_trace_burnin", function() plot_target_trace_with_burnin(res, bayes_sel$target), width = 1800, height = 1300)
      }
    }

    sens <- maybe_value(sensitivity_results())
    if (!is.null(sens)) {
      add_fig("17_sensitivity_feature_stability_heatmap", function() plot_sensitivity_feature_heatmap(feature_sensitivity_stability()), width = 1800, height = 1300)
    }

    mat <- maybe_value(truth_matrix())
    if (!is.null(mat)) {
      add_fig("18_ground_truth_signed_adjacency", function() plot_adj_heatmap(mat, "Ground truth"), width = 1300, height = 1000)
    }
    if (!is.null(res)) {
      add_fig("19_inferred_signed_adjacency", function() plot_adj_heatmap(inferred_state_adj(), "Inferred"), width = 1300, height = 1000)
    }

    if (length(paths) == 0) {
      readme <- file.path(tmp, "README.txt")
      writeLines("No CIRN figures are available yet. Run CIRN Algorithm first, then export the figure pack.", readme)
      paths <- readme
    }

    oldwd <- getwd()
    on.exit(setwd(oldwd), add = TRUE)
    setwd(tmp)
    utils::zip(zipfile = file, files = basename(paths))
  }

  write_workbook <- function(path) {
    wb <- openxlsx::createWorkbook()
    add_sheet <- function(name, data) {
      openxlsx::addWorksheet(wb, substr(name, 1, 31))
      openxlsx::writeData(wb, substr(name, 1, 31), data)
    }
    add_sheet("selected_data", selected_data())
    add_sheet("derivatives", derivative_data())
    add_sheet("class_balance", class_preview())
    res <- maybe_value(cirn_results())
    if (!is.null(res)) {
      add_sheet("edges", res$edges_combined)
      add_sheet("sublevel_edges", filter_result_mode(res$edges_combined, "sublevel"))
      add_sheet("all_predictor_edges", filter_result_mode(res$edges_combined, "all_predictors"))
      add_sheet("pairwise_edges", filter_result_mode(res$edges_combined, "pairwise"))
      add_sheet("mode_consistency", summarize_mode_consistency(res$edges_combined))
      add_sheet("edge_consistency", edge_consistency_from_result(res))
      add_sheet("all_coefficients", res$all_coefficients_combined)
      add_sheet("diagnostics", res$diagnostics)
      add_sheet("effective_sample_size", effective_sample_size_from_result(res))
      if (!is.null(res$pairwise)) {
        add_sheet("pairwise_diagnostics", res$pairwise$diagnostics %||% tibble())
        add_sheet("pairwise_vif", res$pairwise$vif_group %||% tibble())
        add_sheet("pairwise_vif_pairs", res$pairwise$vif_pairs %||% tibble())
      }
      add_sheet("vif", res$vif_group)
      add_sheet("vif_pairs", res$vif_pairs)
    }
    rf <- maybe_value(rf_results())
    latent <- maybe_value(latent_results())
    sens <- maybe_value(sensitivity_results())
    bench <- maybe_value(benchmark_results())
    if (!is.null(rf)) add_sheet("rf_support", rf)
    if (!is.null(latent)) add_sheet("latent_z", latent)
    if (!is.null(sens)) {
      add_sheet("sensitivity_plan", sens$plan %||% sensitivity_plan())
      add_sheet("sensitivity_runs", sens$runs)
      add_sheet("sensitivity_edges", sens$edges %||% tibble())
      add_sheet("sensitivity_diagnostics", sens$diagnostics %||% tibble())
      add_sheet("sensitivity_effective_n", sens$effective_sample_size %||% tibble())
      add_sheet("feature_edge_stability", feature_sensitivity_stability())
      add_sheet("state_edge_stability", sens$edge_stability)
      add_sheet("sensitivity_truth_metrics", sens$ground_truth_metrics %||% tibble())
      scope <- sens$sensitivity_scope %||% list()
      add_sheet(
        "sensitivity_scope",
        tibble(setting = names(scope), value = vapply(scope, function(x) paste(x, collapse = ", "), character(1)))
      )
    }
    if (!is.null(bench)) add_sheet("benchmark", bench$metrics_table)
    add_sheet("settings", tibble(setting = names(current_settings()), value = vapply(current_settings(), function(x) paste(x, collapse = ", "), character(1))))
    add_sheet("session_info", tibble(line = strsplit(session_info_text(), "\n", fixed = TRUE)[[1]]))
    openxlsx::saveWorkbook(wb, path, overwrite = TRUE)
  }

  output$download_edges_csv <- downloadHandler(
    filename = function() "CIRN_edges.csv",
    content = function(file) {
      req(cirn_results())
      readr::write_csv(cirn_results()$edges_combined, file)
    }
  )

  output$download_coefficients_csv <- downloadHandler(
    filename = function() "CIRN_all_coefficients.csv",
    content = function(file) {
      req(cirn_results())
      readr::write_csv(cirn_results()$all_coefficients_combined, file)
    }
  )

  output$download_diagnostics_csv <- downloadHandler(
    filename = function() "CIRN_diagnostics.csv",
    content = function(file) {
      req(cirn_results())
      readr::write_csv(cirn_results()$diagnostics, file)
    }
  )

  output$download_effective_sample_size_csv <- downloadHandler(
    filename = function() "CIRN_effective_sample_size.csv",
    content = function(file) {
      req(cirn_results())
      readr::write_csv(effective_sample_size_from_result(cirn_results()), file)
    }
  )

  output$download_vif_pairs_csv <- downloadHandler(
    filename = function() "CIRN_vif_pair_diagnostics.csv",
    content = function(file) {
      req(cirn_results())
      res <- maybe_value(cirn_results())
      main_vif <- as_result_tbl(res$vif_pairs %||% tibble())
      if (nrow(main_vif) > 0 && !"analysis_mode" %in% names(main_vif)) {
        main_vif <- mutate(main_vif, analysis_mode = "multivariable")
      }
      pairwise_vif <- if (!is.null(res$pairwise)) as_result_tbl(res$pairwise$vif_pairs %||% tibble()) else tibble()
      if (nrow(pairwise_vif) > 0 && !"analysis_mode" %in% names(pairwise_vif)) {
        pairwise_vif <- mutate(pairwise_vif, analysis_mode = "pairwise")
      }
      readr::write_csv(bind_rows(main_vif, pairwise_vif), file)
    }
  )

  output$download_workbook <- downloadHandler(
    filename = function() "CIRN_Studio_Workbook.xlsx",
    content = function(file) {
      write_workbook(file)
    }
  )

  output$download_settings_json <- downloadHandler(
    filename = function() "CIRN_Studio_Settings.json",
    content = function(file) {
      writeLines(jsonlite::toJSON(current_settings(), pretty = TRUE, auto_unbox = TRUE), file)
    }
  )

  output$download_html_report <- downloadHandler(
    filename = function() "CIRN_Studio_Report.html",
    content = function(file) {
      res <- maybe_value(cirn_results())
      edge_count <- if (!is.null(res)) nrow(res$edges_combined) else 0
      html <- paste0(
        "<!doctype html><html><head><meta charset='utf-8'><title>CIRN Studio Report</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:960px;margin:32px auto;line-height:1.5;color:#102033} table{border-collapse:collapse;width:100%;margin:16px 0} td,th{border:1px solid #d7dde5;padding:6px;text-align:left} h1,h2{color:#0f766e}</style>",
        "</head><body><h1>CIRN Studio Report</h1>",
        "<p>Generated: ", htmltools::htmlEscape(as.character(Sys.time())), "</p>",
        "<h2>Summary</h2><p>Rows: ", nrow(selected_data()), "; targets: ", length(input$targets), "; predictors: ", length(input$predictors), "; retained edges: ", edge_count, ".</p>",
        "<h2>Interpretation Reminder</h2><p>CIRN edges are exploratory signed regulatory hypotheses based on Bayesian classification of target derivative direction. They are not automatic causal proof.</p>",
        "</body></html>"
      )
      writeLines(html, file)
    }
  )

  output$download_session_info <- downloadHandler(
    filename = function() "CIRN_Studio_sessionInfo.txt",
    content = function(file) {
      writeLines(session_info_text(), file)
    }
  )

  output$download_figure_pack <- downloadHandler(
    filename = function() "CIRN_Studio_Figure_Pack.zip",
    content = function(file) {
      write_figure_pack(file)
    }
  )

  output$download_zip <- downloadHandler(
    filename = function() "CIRN_Studio_Bundle.zip",
    content = function(file) {
      tmp <- tempfile("cirn_studio_bundle_")
      dir.create(tmp, recursive = TRUE, showWarnings = FALSE)
      paths <- character()
      res <- maybe_value(cirn_results())
      rf <- maybe_value(rf_results())
      latent <- maybe_value(latent_results())
      sens <- maybe_value(sensitivity_results())
      bench <- maybe_value(benchmark_results())
      write_csv(selected_data(), file.path(tmp, "selected_data.csv"))
      paths <- c(paths, file.path(tmp, "selected_data.csv"))
      write_csv(derivative_data(), file.path(tmp, "derivatives.csv"))
      paths <- c(paths, file.path(tmp, "derivatives.csv"))
      write_csv(class_preview(), file.path(tmp, "class_balance.csv"))
      paths <- c(paths, file.path(tmp, "class_balance.csv"))
      if (!is.null(res)) {
        write_csv(res$edges_combined, file.path(tmp, "edges.csv"))
        write_csv(filter_result_mode(res$edges_combined, "sublevel"), file.path(tmp, "sublevel_edges.csv"))
        write_csv(filter_result_mode(res$edges_combined, "all_predictors"), file.path(tmp, "all_predictor_edges.csv"))
        write_csv(filter_result_mode(res$edges_combined, "pairwise"), file.path(tmp, "pairwise_edges.csv"))
        write_csv(summarize_mode_consistency(res$edges_combined), file.path(tmp, "mode_consistency.csv"))
        write_csv(edge_consistency_from_result(res), file.path(tmp, "edge_consistency.csv"))
        write_csv(res$all_coefficients_combined, file.path(tmp, "all_coefficients.csv"))
        write_csv(res$diagnostics, file.path(tmp, "diagnostics.csv"))
        write_csv(effective_sample_size_from_result(res), file.path(tmp, "effective_sample_size.csv"))
        pairwise_paths <- character()
        if (!is.null(res$pairwise)) {
          write_csv(res$pairwise$diagnostics %||% tibble(), file.path(tmp, "pairwise_diagnostics.csv"))
          write_csv(res$pairwise$vif_group %||% tibble(), file.path(tmp, "pairwise_vif.csv"))
          write_csv(res$pairwise$vif_pairs %||% tibble(), file.path(tmp, "pairwise_vif_pairs.csv"))
          pairwise_paths <- file.path(tmp, c("pairwise_diagnostics.csv", "pairwise_vif.csv", "pairwise_vif_pairs.csv"))
        }
        write_csv(res$vif_group, file.path(tmp, "vif.csv"))
        write_csv(res$vif_pairs, file.path(tmp, "vif_pairs.csv"))
        paths <- c(
          paths,
          file.path(tmp, c(
            "edges.csv", "sublevel_edges.csv", "all_predictor_edges.csv",
            "pairwise_edges.csv", "mode_consistency.csv", "edge_consistency.csv", "all_coefficients.csv",
            "diagnostics.csv", "effective_sample_size.csv", "vif.csv", "vif_pairs.csv"
          )),
          pairwise_paths
        )
      }
      if (!is.null(rf)) {
        write_csv(rf, file.path(tmp, "rf_support.csv"))
        paths <- c(paths, file.path(tmp, "rf_support.csv"))
      }
      if (!is.null(latent)) {
        write_csv(latent, file.path(tmp, "latent_z.csv"))
        paths <- c(paths, file.path(tmp, "latent_z.csv"))
      }
      if (!is.null(sens)) {
        write_csv(sens$plan %||% sensitivity_plan(), file.path(tmp, "sensitivity_plan.csv"))
        write_csv(sens$runs, file.path(tmp, "sensitivity_runs.csv"))
        write_csv(sens$edges %||% tibble(), file.path(tmp, "sensitivity_edges.csv"))
        write_csv(sens$diagnostics %||% tibble(), file.path(tmp, "sensitivity_diagnostics.csv"))
        write_csv(sens$effective_sample_size %||% tibble(), file.path(tmp, "sensitivity_effective_sample_size.csv"))
        write_csv(feature_sensitivity_stability(), file.path(tmp, "feature_edge_stability.csv"))
        write_csv(sens$edge_stability, file.path(tmp, "state_edge_stability.csv"))
        write_csv(sens$ground_truth_metrics %||% tibble(), file.path(tmp, "sensitivity_ground_truth_metrics.csv"))
        scope <- sens$sensitivity_scope %||% list()
        write_csv(
          tibble(setting = names(scope), value = vapply(scope, function(x) paste(x, collapse = ", "), character(1))),
          file.path(tmp, "sensitivity_scope.csv")
        )
        paths <- c(
          paths,
          file.path(tmp, c(
            "sensitivity_plan.csv", "sensitivity_runs.csv", "sensitivity_edges.csv",
            "sensitivity_diagnostics.csv", "sensitivity_effective_sample_size.csv",
            "feature_edge_stability.csv", "state_edge_stability.csv",
            "sensitivity_ground_truth_metrics.csv", "sensitivity_scope.csv"
          ))
        )
      }
      if (!is.null(bench)) {
        write_csv(bench$metrics_table, file.path(tmp, "benchmark_metrics.csv"))
        paths <- c(paths, file.path(tmp, "benchmark_metrics.csv"))
      }
      writeLines(jsonlite::toJSON(current_settings(), pretty = TRUE, auto_unbox = TRUE), file.path(tmp, "settings.json"))
      paths <- c(paths, file.path(tmp, "settings.json"))
      writeLines(session_info_text(), file.path(tmp, "sessionInfo_CIRN_Studio.txt"))
      paths <- c(paths, file.path(tmp, "sessionInfo_CIRN_Studio.txt"))
      wb_path <- file.path(tmp, "CIRN_Studio_Workbook.xlsx")
      write_workbook(wb_path)
      paths <- c(paths, wb_path)
      figure_zip <- file.path(tmp, "CIRN_Studio_Figure_Pack.zip")
      write_figure_pack(figure_zip)
      paths <- c(paths, figure_zip)
      if (file.exists(algorithm_path)) {
        algorithm_copy <- file.path(tmp, "CIRN_Algorithm_app_runtime.R")
        file.copy(algorithm_path, algorithm_copy, overwrite = TRUE)
        paths <- c(paths, algorithm_copy)
      }
      oldwd <- getwd()
      on.exit(setwd(oldwd), add = TRUE)
      setwd(tmp)
      utils::zip(zipfile = file, files = basename(paths))
    }
  )
}

shinyApp(ui, server)
