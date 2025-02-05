#install.packages("sf")
# Load required packages
install_if_missing <- function(package) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, dependencies = TRUE)
    library(package, character.only = TRUE)
  }
}

required_packages <- c("sf","spatstat", "dplyr", "future.apply", "ggplot2", "assertthat", "grid", "VGAM", "fastDummies", "forecast", "pryr", "plyr", "reshape2")
invisible(sapply(required_packages, install_if_missing))


library(spatstat)
library(dplyr)
library(future.apply)

#use this in my M3 macbook Air
setwd("/Users/pramitd/Library/Mobile Documents/com~apple~CloudDocs/Desktop/diff-crime-reporting")

#use this in my iMac
#setwd("~/Desktop/diff-crime-reporting") #making this so that I can run the code from my desktop in any computer be it the iMac or the M1 laptop
source('utils/model_utils.R')
source('utils/real_bogota_utils.R')


#this function generates Hawkes stream of ALL CRIMES in bogota given parameters
generate_data_all_crimes <- function(params, time_steps, version = 2024) {
  
  mu_bar = params[1]
  theta = params[2] / params[3]
  omega = params[3]
  sigma_x = params[4]
  sigma_y = params[4]
  
  meta_path = 'metadata/bogota_victimization.csv'
  meta = read.csv(meta_path)
  
  meta$population_scaled = meta$Population / 40
  meta$n_crimes = round(meta$population_scaled * meta$Victimization)
  meta$thinning_rate = 1 - meta$Percent_reported
  
  centers = get_centers('realistic_bogota')
  mu_partial = mu_bar / length(centers)
  
  for (ii in 1:length(centers)) {
    local({
      idx = ii
      assign(paste0('background_intensity_part', idx), get_unimodal_background_intensity_function(x, y, centers[[idx]], mu_partial), pos = .GlobalEnv)
    })
  }
  
  data = data.frame(matrix(ncol = 5, nrow = 0))
  xrange = c(-13.15241, 13.15241)
  yrange = c(-31.35409, 31.35409)
  
  for (t in 1:time_steps) {
    for (ii in 1:length(centers)) {
      pp = rpoispp(lambda = get(paste0('background_intensity_part', ii)), 1, win = owin(xrange, yrange))
      data = rbind(data, cbind(pp$x, pp$y, rep(t, pp$n), rep(0, pp$n), rep(ii, pp$n)))
    }
  }
  
  colnames(data) = c('x', 'y', 't', 'parent_index', 'group')
  data$group = as.factor(data$group)
  rownames(data) = NULL
  data$id = 1:nrow(data)
  
  data = add_districts(data, cell_length = 1)
  data = data[, c('id', 'x', 'y', 't', 'parent_index', 'district')]
  background_data = data
  
  l = 0
  n_background = c(nrow(data))
  m = theta * 2 * pi * sigma_x * sigma_y
  
  while (nrow(background_data) > 0) {
    
    l = l + 1
    background_data$n_offspring = rpois(nrow(background_data), m)
    sum_offspring = sum(background_data$n_offspring)
    
    x = rnorm(sum_offspring, mean = 0, sd = sigma_x)
    y = rnorm(sum_offspring, mean = 0, sd = sigma_y)
    t = ceiling(rexp(sum_offspring, rate = omega))
    
    offspring_data = data.frame(matrix(ncol = 5, nrow = 0))
    colnames(offspring_data) = c('id', 'x', 'y', 't', 'parent_index')
    
    YY = future_lapply(1:nrow(background_data), function(ii) {
      if (background_data[ii, 'n_offspring'] > 0) {
        x_center = background_data[ii, 'x']
        y_center = background_data[ii, 'y']
        t_center = background_data[ii, 't']
        parent_id = background_data[ii, 'id']
        n_offspring = background_data[ii, 'n_offspring']
        
        start_idx = cumsum(background_data$n_offspring)[ii] - n_offspring
        data_offspring_partial = data.frame(matrix(ncol = 4, nrow = 0))
        
        for (j in 1:n_offspring) {
          data_offspring_partial = rbind(data_offspring_partial, c(x_center + x[start_idx + j], y_center + y[start_idx + j], t_center + t[start_idx + j], parent_id))
        }
        
        colnames(data_offspring_partial) = c('x', 'y', 't', 'parent_index')
        data_offspring_partial
        
      } else {
        data_offspring_partial = data.frame(matrix(ncol = 4, nrow = 0))
        colnames(data_offspring_partial) = c('x', 'y', 't', 'parent_index')
        data_offspring_partial
      }
    })
    
    offspring_data = bind_rows(YY, .id = NULL)
    if (nrow(offspring_data) > 0) {
      offspring_data$id = (max(data$id) + 1):(max(data$id) + nrow(offspring_data))
    } else {
      offspring_data = data.frame(matrix(ncol = 5, nrow = 0))
      colnames(offspring_data) = c('x', 'y', 't', 'parent_index', 'id')
    }
    
    offspring_data = add_districts(offspring_data, cell_length = 1)
    offspring_data = offspring_data[offspring_data$district != 0,]
    data = rbind(data, offspring_data)
    n_background[length(n_background) + 1] = nrow(data)
    
    background_data = offspring_data
    background_data = background_data[background_data$t <= time_steps,]
  }
  
  data = data[data$t <= time_steps,]
  
  
  rownames(data)= NULL
  return(data[data$district!=0,])
}


#generate MC_REPS many Monte Carlo samples of Crime streams over which I shall calculate the average 
generate_multiple_streams <- function(params, time_steps, mc_reps) {
  all_data <- list()
  for (rep in 1:mc_reps) {
    data <- generate_data_all_crimes(params, time_steps, rep)
    all_data[[rep]] <- data
  }
  return(all_data)
}





#given grid and list of data, compute expected crimes - this is absiaclly aggregation and drawing heatmap from teh MC_reps
# Updated function to compute expected crimes and generate a heatmap using Bogotá grid
compute_expected_crimes <- function(data_list, bogota_shapefile, grid_size, filename_prefix) {
  # Generate grid based on Bogotá shapefile
  
  bogota_bounds <- st_bbox(st_read(bogota_shapefile))
  x_breaks <- seq(bogota_bounds["xmin"], bogota_bounds["xmax"], length.out = grid_size[1]+1)
  y_breaks <- seq(bogota_bounds["ymin"], bogota_bounds["ymax"], length.out = grid_size[2]+1)
  
  combined_data <- do.call(rbind, data_list)
  combined_data <- combined_data[combined_data$x >= min(x_breaks) & combined_data$x <= max(x_breaks) &
                                   combined_data$y >= min(y_breaks) & combined_data$y <= max(y_breaks), ]
  
  combined_data$x_bin <- cut(combined_data$x, breaks = x_breaks, include.lowest = TRUE, labels = FALSE)
  combined_data$y_bin <- cut(combined_data$y, breaks = y_breaks, include.lowest = TRUE, labels = FALSE)
  
  combined_data_grouped <- aggregate(id ~ x_bin + y_bin + t, data = combined_data, FUN = length)
  names(combined_data_grouped)[names(combined_data_grouped) == "id"] <- "crime_count"
  
  # Calculate the average daily crimes
  combined_data_grouped$day <- ceiling(combined_data_grouped$t / (max(combined_data$t) / 7))
  avg_daily_crimes <- aggregate(crime_count ~ x_bin + y_bin, data = combined_data_grouped, FUN = mean)
  names(avg_daily_crimes)[names(avg_daily_crimes) == "crime_count"] <- "avg_daily_crimes"
  
  # Ensure all grid cells are represented
  all_cells <- expand.grid(
    x_bin = seq_along(x_breaks[-1]),
    y_bin = seq_along(y_breaks[-1])
  )
  avg_daily_crimes <- merge(all_cells, avg_daily_crimes, by = c("x_bin", "y_bin"), all.x = TRUE)
  avg_daily_crimes$avg_daily_crimes[is.na(avg_daily_crimes$avg_daily_crimes)] <- 0
  
  # Generate the heatmap
  heatmap_data <- reshape2::acast(avg_daily_crimes, y_bin ~ x_bin, value.var = "avg_daily_crimes", fill = 0)
  heatmap_data <- heatmap_data[rev(rownames(heatmap_data)), ]
  
  heatmap_plot <- ggplot2::ggplot(reshape2::melt(heatmap_data), aes(Var2, Var1, fill = value)) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_gradient(low = "white", high = "blue") +
    ggplot2::labs(title = paste0("Expected Crime Heatmap for ", filename_prefix), x = "X Bin", y = "Y Bin") +
    ggplot2::theme_minimal() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
  
  ggplot2::ggsave(paste0(filename_prefix, "_heatmap.png"), plot = heatmap_plot)
  
  return(avg_daily_crimes)
}

# Function to compare true and estimated hotspots and save results
# Updated function to compare true and estimated hotspots and calculate MAE
compare_hotspots <- function(expected_crimes_true, expected_crimes_est, true_params, estimated_params) {
  hotspots_true <- expected_crimes_true[order(-expected_crimes_true$avg_daily_crimes), ][1:10, ]
  hotspots_est <- expected_crimes_est[order(-expected_crimes_est$avg_daily_crimes), ][1:10, ]
  
  common_hotspots <- merge(hotspots_true, hotspots_est, by = c("x_bin", "y_bin"))
  
  accuracy <- nrow(common_hotspots) / 10
  
  # Calculate MAE
  merged_data <- merge(expected_crimes_true, expected_crimes_est, by = c("x_bin", "y_bin"), all = TRUE)
  merged_data$avg_daily_crimes.x[is.na(merged_data$avg_daily_crimes.x)] <- 0
  merged_data$avg_daily_crimes.y[is.na(merged_data$avg_daily_crimes.y)] <- 0
  mae <- mean(abs(merged_data$avg_daily_crimes.x - merged_data$avg_daily_crimes.y) / (merged_data$avg_daily_crimes.x + 1e-8))
  
  # Save top 10 cells to a CSV file
  true_filename <- paste0('top_10_hotspots_true_mu_', true_params[1], '_alpha_', true_params[2], '_beta_', true_params[3], '_sigma_', true_params[4], '.csv')
  est_filename <- paste0('top_10_hotspots_est_mu_', estimated_params[1], '_alpha_', estimated_params[2], '_beta_', estimated_params[3], '_sigma_', estimated_params[4], '.csv')
  write.csv(hotspots_true, true_filename, row.names = FALSE)
  write.csv(hotspots_est, est_filename, row.names = FALSE)
  
  return(list(accuracy = accuracy, mae = mae, true_filename = true_filename, est_filename = est_filename))
}


# Define a function to calculate top-10 accuracy
calculate_top_10_accuracy_and_relative_MAE <- function(true_params, estimated_params) {
  time_steps <- 7
  mc_reps <- 100
  bogota_shape_file <- "~/metadata/bogota.shp"
  grid_size <- c(7, 16)
  
  start_time = Sys.time()
  
  # Generate data
  #this data generates MC_reps many replications 
  data_true <- generate_multiple_streams(true_params, time_steps, mc_reps)
  data_est <- generate_multiple_streams(estimated_params, time_steps, mc_reps)
  
  # Compute expected crimes
  expected_crimes_true <- compute_expected_crimes(data_true, grid_size, paste0("true_", true_params[1], "_", true_params[2], "_", true_params[3], "_", true_params[4]))
  expected_crimes_est <- compute_expected_crimes(data_est, grid_size, paste0("est_", estimated_params[1], "_", estimated_params[2], "_", estimated_params[3], "_", estimated_params[4]))
  
  # Compare hotspots
  result <- compare_hotspots(expected_crimes_true, expected_crimes_est, true_params, estimated_params)
  
  end_time <- Sys.time()
  execution_time <- end_time - start_time
  cat("Execution time for this job:", execution_time, "seconds\n")
  
  cat("Top-10 hotspot prediction accuracy for true params (mu =", true_params[1], 
      ", alpha =", true_params[2], ", beta =", true_params[3], ", sigma =", true_params[4], 
      "):", result$accuracy * 100, "%", "\n")
  
  # Return the results
  return(list(true_params = true_params, estimated_params = estimated_params, accuracy = result$accuracy, count_relative_MAE= result$mae, true_filename = result$true_filename, est_filename = result$est_filename))
}

#Input the values of true parameters and estimated parameters 
#to get the top_10_accuracy and relative MAE between expected 
#calculate_top_10_accuracy_and_relative_MAE (true_params, estimated_params)
