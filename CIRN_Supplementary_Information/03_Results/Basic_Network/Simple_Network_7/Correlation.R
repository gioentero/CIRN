################################################################################
################ CORRELATION-BASED NETWORK INFERENCE ###########################
################################################################################

# ============================================================
# Working Directory
# ============================================================

setwd('/Users/gio/Desktop/My Dissertation/Dissertation Focus/Main Version/1_Basic_Net_Raw/Simple_Network_7')

# ============================================================
# Required Libraries
# ============================================================

library(dplyr)
library(tidyr)
library(corrplot)

################################################################################
################################################################################

# ============================================================
# CORRELATION-BASED NETWORK INFERENCE
# Pearson and Spearman Correlation
# ============================================================
#
# Purpose:
#   • Provide classical baseline methods for comparison with CIRN
#   • Infer signed but UNDIRECTED associations
#   • Convert associations into directed edges for evaluation
#
# NOTE:
#   Correlation does NOT infer causality or direction.
#   Edges are duplicated in both directions for comparability.
# ============================================================

infer_network_correlation <- function(df, time_col, method = "pearson",
                                      cor_threshold = 0.3,
                                      p_threshold = 0.05) {
  
  # ------------------------------------------------------------
  # 1. Remove time column
  # ------------------------------------------------------------
  
  vars <- setdiff(names(df), time_col)
  data_mat <- df[, vars]
  
  # ------------------------------------------------------------
  # 2. Start runtime measurement
  # ------------------------------------------------------------
  
  start_time <- Sys.time()
  
  # ------------------------------------------------------------
  # 3. Compute correlation matrix
  # ------------------------------------------------------------
  
  cor_matrix <- cor(
    data_mat,
    method = method,
    use = "pairwise.complete.obs"
  )
  
  # ------------------------------------------------------------
  # 4. Compute p-values
  # ------------------------------------------------------------
  
  p_matrix <- matrix(
    NA,
    nrow = length(vars),
    ncol = length(vars),
    dimnames = list(vars, vars)
  )
  
  for (i in seq_along(vars)) {
    for (j in seq_along(vars)) {
      
      if (i == j) next
      
      test <- suppressWarnings(
        cor.test(
          data_mat[[vars[i]]],
          data_mat[[vars[j]]],
          method = method
        )
      )
      
      p_matrix[i, j] <- test$p.value
    }
  }
  
  # ------------------------------------------------------------
  # 5. Edge extraction
  # ------------------------------------------------------------
  
  edges <- data.frame()
  
  for (i in seq_along(vars)) {
    for (j in seq_along(vars)) {
      
      if (i == j) next
      
      r <- cor_matrix[i, j]
      p <- p_matrix[i, j]
      
      if (!is.na(r) &&
          abs(r) >= cor_threshold &&
          p < p_threshold) {
        
        edges <- rbind(
          edges,
          data.frame(
            predictor_set = paste0(method, "_correlation"),
            term = vars[i],
            target = vars[j],
            omega = r,
            odds_ratio = NA,
            sign = sign(r),
            regulation_type = ifelse(
              r > 0,
              "activation",
              "inhibition"
            ),
            rel_strength = abs(r),
            hdi_lower95 = NA,
            hdi_upper95 = NA,
            eti_lower95 = NA,
            eti_upper95 = NA,
            method = paste(method, "correlation")
          )
        )
      }
    }
  }
  
  # ------------------------------------------------------------
  # 6. Stop runtime measurement
  # ------------------------------------------------------------
  
  end_time <- Sys.time()
  
  runtime_seconds <- as.numeric(
    difftime(end_time, start_time, units = "secs")
  )
  
  message(
    "⏱ ", method, " correlation runtime: ",
    round(runtime_seconds, 4),
    " seconds"
  )
  
  return(list(
    edges = edges,
    runtime = runtime_seconds,
    cor_matrix = cor_matrix,
    p_matrix = p_matrix
  ))
}

################################################################################
################################################################################

# ============================================================
# Bubble Correlation Heatmap (like your example)
# ============================================================

plot_correlation_bubble <- function(cor_matrix, method_name = "Correlation") {
  
  corrplot(
    cor_matrix,
    
    method = "circle",           # circles like your figure
    type = "upper",              # upper triangle
    
    addCoef.col = "black",       # numbers inside circles
    number.cex = 0.8,
    
    col = colorRampPalette(
      c("#FFD92F", "#FF8C00", "#FF0000")
    )(200),                      # yellow → orange → red
    
    tl.col = "black",
    tl.srt = 45,
    
    diag = FALSE,
    
    title = paste(method_name, "Pairwise Correlation"),
    mar = c(0,0,2,0)
  )
}

################################################################################
################################################################################

# ============================================================
# Load Dataset
# ============================================================

sim_df <- read.csv("Simple_Network_7.csv") %>%
  na.omit()

time_col <- names(sim_df)[1]

################################################################################
################################################################################

# ============================================================
# Pearson Correlation Network
# ============================================================

pearson_res <- infer_network_correlation(
  df = sim_df,
  time_col = time_col,
  method = "pearson"
)

pearson_edges <- pearson_res$edges
pearson_edges

pearson_cor_mat <- pearson_res$cor_matrix
pearson_cor_mat

pearson_p_mat <- pearson_res$p_matrix
pearson_p_mat

# Pearson bubble heatmap

plot_correlation_bubble(
  pearson_res$cor_matrix,
  "Pearson"
)

################################################################################
################################################################################

# ============================================================
# Spearman Correlation Network
# ============================================================

spearman_res <- infer_network_correlation(
  df = sim_df,
  time_col = time_col,
  method = "spearman"
)

spearman_edges <- spearman_res$edges
spearman_edges

spearman_cor_mat <- spearman_res$cor_matrix
spearman_cor_mat

spearman_p_mat <- spearman_res$p_matrix
spearman_p_mat

# Spearman bubble heatmap

plot_correlation_bubble(
  spearman_res$cor_matrix,
  "Spearman"
)

################################################################################
s################################################################################

# ============================================================
# Optional: Save figures for thesis / manuscript
# ============================================================

#pdf("Pearson_Correlation_Bubble.pdf", width = 6, height = 6)

#plot_correlation_bubble(
#  pearson_res$cor_matrix,
#  "Pearson"
#)

#dev.off()


#pdf("Spearman_Correlation_Bubble.pdf", width = 6, height = 6)

#plot_correlation_bubble(
#  spearman_res$cor_matrix,
#  "Spearman"
#)

#dev.off()

################################################################################
############################### ~ END ~ ########################################
################################################################################