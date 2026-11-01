################################################################################
################### CONVERGENT CROSS MAPPING (CCM) #############################
################################################################################

setwd('/Users/gio/Desktop/My Dissertation/Dissertation Focus/Main Version/1_Basic_Net_Raw/Simple_Network_4')

library(rEDM)
library(dplyr)
library(ggplot2)
library(tidyr)

################################################################################
# Load data
################################################################################

sim_df <- read.csv("Simple_Network_4.csv") %>% na.omit()

time_col <- names(sim_df)[1]
vars <- setdiff(names(sim_df), time_col)

################################################################################
# Select embedding dimension
################################################################################

select_embedding_dimension <- function(df, variable, max_E = 10){
  
  embed_out <- EmbedDimension(
    dataFrame = df,
    lib = c(1, nrow(df)),
    pred = c(1, nrow(df)),
    columns = variable,
    target = variable,
    maxE = max_E
  )
  
  best_E <- embed_out$E[which.max(embed_out$rho)]
  
  cat("Best embedding dimension for", variable, ":", best_E, "\n")
  
  return(best_E)
}

E_values <- sapply(vars, function(v){
  select_embedding_dimension(sim_df, v)
})

E <- round(mean(E_values))

cat("Using embedding dimension:", E, "\n")

################################################################################
# CCM inference
################################################################################

infer_network_ccm <- function(df, time_col, E, lib_sizes = NULL, num_samples = 100){
  
  # ------------------------------------------------------------
  # Identify variables
  # ------------------------------------------------------------
  
  vars <- setdiff(names(df), time_col)
  
  # ------------------------------------------------------------
  # Determine safe library sizes automatically
  # ------------------------------------------------------------
  # CCM requires: libSize ≤ N − (E − 1)
  
  N <- nrow(df)
  max_lib <- N - (E - 1)
  
  # If user did not specify library sizes, construct safe sequence
  if(is.null(lib_sizes)){
    
    start_lib <- max(5, floor(max_lib * 0.2))
    
    lib_sizes <- paste(start_lib, max_lib, 5)
  }
  
  # ------------------------------------------------------------
  # Start runtime measurement
  # ------------------------------------------------------------
  
  start_time <- Sys.time()
  
  edges <- data.frame()
  
  rho_matrix <- matrix(
    NA,
    length(vars),
    length(vars),
    dimnames = list(vars, vars)
  )
  
  # ------------------------------------------------------------
  # Pairwise CCM
  # ------------------------------------------------------------
  
  for(i in 1:(length(vars)-1)){
    for(j in (i+1):length(vars)){
      
      if(i == j) next
      
      x <- vars[i]
      y <- vars[j]
      
      ccm_out <- CCM(
        dataFrame = df,
        E = E,
        columns = x,
        target = y,
        libSizes = lib_sizes,
        sample = num_samples
      )
      
      # --------------------------------------------------------
      # Extract converged cross-map skill
      # --------------------------------------------------------
      # Last row corresponds to largest library size
      
      rho_xy <- tail(ccm_out[[2]], 1)
      rho_yx <- tail(ccm_out[[3]], 1)
      
      rho_matrix[x, y] <- rho_xy
      rho_matrix[y, x] <- rho_yx
      
      edges <- rbind(edges,
                     data.frame(
                       Cause = x,
                       Effect = y,
                       CCM_rho = rho_xy
                     ))
      
      edges <- rbind(edges,
                     data.frame(
                       Cause = y,
                       Effect = x,
                       CCM_rho = rho_yx
                     ))
    }
  }
  
  # ------------------------------------------------------------
  # Runtime
  # ------------------------------------------------------------
  
  runtime <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  
  cat("CCM runtime:", round(runtime,4), "seconds\n")
  
  # ------------------------------------------------------------
  # Return results
  # ------------------------------------------------------------
  
  return(list(
    edges = edges,
    rho_matrix = rho_matrix,
    runtime = runtime
  ))
}


################################################################################
# Run CCM
################################################################################

ccm_res <- infer_network_ccm(
  df = sim_df,
  time_col = time_col,
  E = E
)

ccm_edges <- ccm_res$edges
print(ccm_edges)

################################################################################
# Heatmap
################################################################################

plot_ccm_heatmap <- function(rho_matrix){
  
  df <- as.data.frame(rho_matrix) %>%
    tibble::rownames_to_column("Cause") %>%
    pivot_longer(-Cause, names_to="Effect", values_to="rho") %>%
    filter(!is.na(rho))
  
  ggplot(df, aes(Cause, Effect, fill = rho)) +
    geom_tile(color="white") +
    geom_text(aes(label = round(rho,2)), size = 5) +
    scale_fill_gradient(low="#e8f8f5", high="#27ae60", name="ρ") +
    theme_minimal(base_size = 13) +
    labs(title="Convergent Cross Mapping", x="Cause", y="Effect")
}

plot_ccm_heatmap(ccm_res$rho_matrix)

################################################################################
################################ END ###########################################
################################################################################