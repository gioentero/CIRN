################################################################################
####################### GRANGER CAUSALITY NETWORK INFERENCE ####################
################################################################################

# ============================================================
# Working Directory
# ============================================================

setwd('/Users/gio/Desktop/My Dissertation/Dissertation Focus/Main Version/0_Predator_Prey_Simulation/Classical_Lotka_Volterra/Stochastic_Predator_Prey')

# ============================================================
# Required Libraries
# ============================================================

library(dplyr)
library(lmtest)
library(tseries)
library(ggplot2)
library(tidyr)

################################################################################
################################################################################

# ============================================================
# ADF Stationarity Check
# ============================================================

check_stationarity <- function(df, time_col){
  
  vars <- setdiff(names(df), time_col)
  df_out <- df
  
  non_stationary <- FALSE
  
  for(v in vars){
    
    ### skip constant / near-constant variables
    if (sd(df[[v]], na.rm = TRUE) < 1e-8) {
      warning(paste("⚠ Variable", v, "is constant → skipping ADF test"))
      next
    }
    
    test <- adf.test(df[[v]])
    
    cat("\nADF test for", v, "\n")
    print(test)
    
    ### handle NA p-values safely
    if (is.na(test$p.value)) {
      warning(paste("⚠ Variable", v, "returned NA p-value → skipping"))
      next
    }
    
    if(test$p.value > 0.05){
      non_stationary <- TRUE
    }
    
  }
  
  if(non_stationary){
    
    cat("\n⚠ At least one variable is non-stationary → applying FIRST DIFFERENCE TO ALL VARIABLES\n")
    
    for(v in vars){
      
      ### avoid differencing constant variables
      if (sd(df[[v]], na.rm = TRUE) < 1e-8) {
        df_out[[v]] <- df[[v]]   # keep as-is
      } else {
        df_out[[v]] <- c(NA, diff(df[[v]]))
      }
      
    }
    
  }
  
  df_out <- na.omit(df_out)
  
  return(df_out)
}

################################################################################
################################################################################

# ============================================================
# Pairwise Granger Causality Inference
# ============================================================

infer_granger_network <- function(df, time_col,
                                  max_lag = 1,
                                  p_threshold = 0.05,
                                  collinearity_threshold = 0.999){
  
  vars <- setdiff(names(df), time_col)
  data_mat <- df[, vars]
  
  start_time <- Sys.time()
  
  edges <- data.frame()
  
  p_matrix <- matrix(
    NA,
    nrow = length(vars),
    ncol = length(vars),
    dimnames = list(vars, vars)
  )
  
  for(cause in vars){
    for(target in vars){
      
      if(cause == target) next
      
      # ------------------------------------------------------
      # 🔥 NEW: Skip near-perfect collinearity
      # ------------------------------------------------------
      corr_val <- cor(data_mat[[cause]], data_mat[[target]])
      
      if(!is.na(corr_val) && abs(corr_val) > collinearity_threshold){
        message("⚠ Skipping ", cause, " → ", target, " (collinearity: ", round(corr_val,4), ")")
        next
      }
      
      # ------------------------------------------------------
      # Run Granger test safely
      # ------------------------------------------------------
      test <- tryCatch(
        grangertest(
          data_mat[[target]] ~ data_mat[[cause]],
          order = max_lag
        ),
        error = function(e){
          message("⚠ Error for ", cause, " → ", target, ": ", e$message)
          return(NULL)
        }
      )
      
      if(is.null(test)) next
      
      p_value <- test$`Pr(>F)`[2]
      
      p_matrix[cause, target] <- p_value
      
      if(!is.na(p_value) && p_value < p_threshold){
        
        edges <- rbind(
          edges,
          data.frame(
            Cause = cause,
            Effect = target,
            p_value = p_value,
            Strength = -log10(max(p_value,1e-16))
          )
        )
        
      }
      
    }
  }
  
  end_time <- Sys.time()
  
  runtime_seconds <- as.numeric(
    difftime(end_time,start_time,units="secs")
  )
  
  message(
    "⏱ Granger causality runtime: ",
    round(runtime_seconds,4),
    " seconds"
  )
  
  return(list(
    edges = edges,
    runtime = runtime_seconds,
    p_matrix = p_matrix
  ))
}

################################################################################
################################################################################

# ============================================================
# Granger Heatmap
# ============================================================

plot_granger_heatmap <- function(p_matrix){
  
  heatmap_df <- as.data.frame(p_matrix) %>%
    tibble::rownames_to_column("Cause") %>%
    pivot_longer(
      -Cause,
      names_to = "Target",
      values_to = "p_value"
    ) %>%
    filter(!is.na(p_value))
  
  ggplot(
    heatmap_df,
    aes(Cause, Target, fill = -log10(p_value))
  ) +
    geom_tile(color="white") +
    geom_text(
      aes(label=sprintf("%.2e",p_value)),
      size=4
    ) +
    scale_fill_gradient(
      low = "#e8f8f5",
      high = "#27ae60",
      name = "-log10(p)"
    ) +
    labs(
      title="Granger Causality Significance",
      x="Cause",
      y="Target"
    ) +
    theme_minimal(base_size=13)
}

################################################################################
################################################################################

# ============================================================
# Load Dataset
# ============================================================

sim_df <- read.csv("Stochastic_Lotka_Volterra.csv") %>%
  na.omit()

time_col <- names(sim_df)[1]

################################################################################
################################################################################

# ============================================================
# Stationarity Check
# ============================================================

sim_df_stationary <- check_stationarity(sim_df, time_col)

### remove zero / near-zero variance variables
vars <- setdiff(names(sim_df_stationary), time_col)

valid_vars <- vars[
  sapply(sim_df_stationary[, vars], sd) > 1e-8
]

sim_df_stationary <- sim_df_stationary[, c(time_col, valid_vars)]

message("Variables used for Granger: ", paste(valid_vars, collapse = ", "))

################################################################################
################################################################################

# ============================================================
# Run Granger Causality
# ============================================================

granger_res <- infer_granger_network(
  df = sim_df_stationary,
  time_col = time_col,
  max_lag = 9
)

granger_edges <- granger_res$edges

print(granger_edges)

################################################################################
################################################################################

# ============================================================
# Plot Heatmap
# ============================================================

plot_granger_heatmap(granger_res$p_matrix)

################################################################################
############################### ~ END ~ ########################################
################################################################################

