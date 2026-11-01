################################################################################
######################## PARTIAL CROSS MAPPING (PCM) ###########################
################################################################################

# ============================================================
# Working Directory
# ============================================================

setwd('/Users/gio/Desktop/My Dissertation/Dissertation Focus/Main Version/3_GeneNetWeaver_Ecoli_Networks/Size_5_Network/Stochastic_0_10')

# ============================================================
# Install / Load Libraries
# ============================================================

if(!require(tEDM)) install.packages("tEDM")
library(rEDM)
library(tEDM)
library(dplyr)
library(ggplot2)
library(tidyr)

################################################################################
# Load dataset
################################################################################

sim_df <- read.csv("Stochastic_0_10.csv") %>% na.omit()

time_col <- names(sim_df)[1]
vars <- setdiff(names(sim_df), time_col)

################################################################################
# PCM Inference Function
################################################################################

infer_network_pcm <- function(df,
                              time_col,
                              E = E,
                              tau = 1,
                              libsizes = seq(10,100,10)){
  
  vars <- setdiff(names(df), time_col)
  
  start_time <- Sys.time()
  
  edges <- data.frame()
  
  strength_matrix <- matrix(
    NA,
    length(vars),
    length(vars),
    dimnames = list(vars, vars)
  )
  
  # PCM requires conditioning variables
  if(length(vars) < 3){
    message("PCM requires at least 3 variables. No PCM computed.")
    
    return(list(
      edges = edges,
      strength_matrix = strength_matrix,
      runtime = 0
    ))
  }
  
  for(cause in vars){
    for(effect in vars){
      
      if(cause == effect) next
      
      conds <- setdiff(vars, c(cause,effect))
      
      pcm_res <- pcm(
        data = df,
        cause = cause,
        effect = effect,
        conds = conds,
        libsizes = libsizes,
        E = E,
        tau = tau,
        threads = 1
      )
      
      # Extract partial cross mapping result
      pxmap <- pcm_res$pxmap
      
      # final convergence value
      strength <- tail(pxmap[[2]],1)
      
      strength_matrix[cause,effect] <- strength
      
      edges <- rbind(
        edges,
        data.frame(
          Cause = cause,
          Effect = effect,
          PCM_strength = strength
        )
      )
      
    }
  }
  
  runtime <- as.numeric(difftime(Sys.time(),start_time,units="secs"))
  
  cat("PCM runtime:",round(runtime,4),"seconds\n")
  
  return(list(
    edges = edges,
    strength_matrix = strength_matrix,
    runtime = runtime
  ))
}

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
  
  ### skip weak variables
  if (sd(sim_df[[v]]) < 1e-8) {
    warning(paste("⚠ Skipping", v, "for embedding (low variability)"))
    return(NA)
  }
  
  select_embedding_dimension(sim_df, v)
})

E_values <- na.omit(E_values)
E <- round(mean(E_values))

cat("Using embedding dimension:", E, "\n")

################################################################################
# Run PCM
################################################################################

pcm_res <- infer_network_pcm(
  df = sim_df,
  time_col = time_col,
  E = E
)

pcm_edges <- pcm_res$edges
print(pcm_edges)

################################################################################
# PCM Heatmap
################################################################################

plot_pcm_heatmap <- function(strength_matrix){
  
  df <- as.data.frame(strength_matrix) %>%
    tibble::rownames_to_column("Cause") %>%
    pivot_longer(-Cause,names_to="Effect",values_to="strength") %>%
    filter(!is.na(strength))
  
  ggplot(df,aes(Cause,Effect,fill=strength))+
    geom_tile(color="white")+
    geom_text(aes(label=round(strength,2)),size=5)+
    scale_fill_gradient(low="white",high="darkblue",name="PCM")+
    theme_minimal(base_size=13)+
    labs(title="Partial Cross Mapping",x="Cause",y="Effect")
}

plot_pcm_heatmap(pcm_res$strength_matrix)

################################################################################
############################### END ############################################
################################################################################