################################################################################
######################## PARTIAL CROSS MAPPING (PCM) ###########################
################################################################################

# ============================================================
# Working Directory
# ============================================================

setwd('/Users/gio/Desktop/My Dissertation/Dissertation Focus/Main Version/0_Predator_Prey_Simulation')

# ============================================================
# Install / Load Libraries
# ============================================================

if(!require(tEDM)) install.packages("tEDM")

library(tEDM)
library(dplyr)
library(ggplot2)
library(tidyr)

################################################################################
# Load dataset
################################################################################

sim_df <- read.csv("0_Predator_Prey_Simulation.csv") %>% na.omit()

time_col <- names(sim_df)[1]
vars <- setdiff(names(sim_df), time_col)

################################################################################
# PCM Inference Function
################################################################################

infer_network_pcm <- function(df,
                              time_col,
                              E = 3,
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
# Run PCM
################################################################################

pcm_res <- infer_network_pcm(
  df = sim_df,
  time_col = time_col,
  E = 3
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