options(repos = c(CRAN = "https://cloud.r-project.org"))
library(ggplot2)
library(RobustGaSP)
install.packages("gridExtra")  # Add this package for plotting
library(gridExtra)  # Load the package after installation


data <- read.csv("droplet_data.csv", header = TRUE)  # Use header=TRUE because your first row has column names
clean_data <- na.omit(data)

x <- clean_data[, 1:3]
y <- clean_data[, 12:13]

x_scaled <- scale(x)

# Number of folds
k <- 5
set.seed(123)  # reproducible splits

n <- nrow(x_scaled)
folds <- sample(rep(1:k, length.out = n))  # assign fold numbers randomly

# Initialize vectors to store metrics
rmse_values <- matrix(NA, nrow = k, ncol = ncol(y))
r2_values <- matrix(NA, nrow = k, ncol = ncol(y))

# Helper functions for metrics
rmse <- function(true, pred) sqrt(mean((true - pred)^2))
r_squared <- function(true, pred) 1 - sum((true - pred)^2) / sum((true - mean(true))^2)

for (fold in 1:k) {
  cat("Fold", fold, "\n")
  
  # Split train/test based on fold
  train_indices <- which(folds != fold)
  test_indices <- which(folds == fold)
  
  x_train <- x_scaled[train_indices, , drop=FALSE]
  y_train <- y[train_indices, , drop=FALSE]
  x_test <- x_scaled[test_indices, , drop=FALSE]
  y_test <- y[test_indices, , drop=FALSE]
  
  # Fit model on train
  model <- ppgasp(
    design = x_train,
    response = y_train,
    nugget.est = TRUE,
    num_initial_values = 3
  )
  
  model.predict<-predict(model, x_test)
  
  # Calculate metrics for each output dimension
  for (i in 1:ncol(y)) {
    rmse_values[fold, i] <- rmse(y_test[, i], model.predict$mean[, i])
    r2_values[fold, i] <- r_squared(y_test[, i], model.predict$mean[, i])
  }
}

# Print average RMSE and R2 across folds
cat(paste0("Output variable volume average metrics:\n"))
cat("  Mean RMSE: ", mean(rmse_values[, 1]), "\n")
cat("  Mean R-squared: ", mean(r2_values[, 1]), "\n\n")

cat(paste0("Output variable time average metrics:\n"))
cat("  Mean RMSE: ", mean(rmse_values[, 2]), "\n")
cat("  Mean R-squared: ", mean(r2_values[, 2]), "\n\n")


model <- ppgasp(
  design = x_scaled,
  response = y,
  nugget.est=TRUE, 
  num_initial_values=3
)

pdf_given <- function(G, Temp) {
    Nsims <- 10000

    # Fixed G and Temp arrays
    G_array <- rep(G, Nsims)
    Temp_array <- rep(Temp, Nsims)

    # Generate wave uniformly
    wave <- runif(Nsims, min = 1/300, max = 1/100)

    # Combine into a data frame (ensemble)
    ensembles <- data.frame(G = G_array, wave = wave, Temp = Temp_array)

    # Normalization constants (same as training)
    G_mean <- 19.99292310090832
    G_std <- 5.768017604154681

    wave_mean <- mean(wave)
    wave_std <- sd(wave)

    Temp_mean <- 168.00554590178015
    Temp_std <- 58.87173432703981

    # Scale the ensembles
    ensembles_scaled <- data.frame(
        G = (ensembles$G - G_mean) / G_std,
        wave = (ensembles$wave - wave_mean) / wave_std,
        Temp = (ensembles$Temp - Temp_mean) / Temp_std
    )

    model.pred <- predict(model, ensembles_scaled)

    volume <- model.pred$mean[,1]  # first output column
    time <- model.pred$mean[,2]

    # Calculate standard deviations from prediction
    vol_sd <- sqrt(model.pred$sd[,1]^2)
    time_sd <- sqrt(model.pred$sd[,2]^2)

    # Create separate plots for volume and time
    p1 <- ggplot(data.frame(value = volume), aes(x = value)) +
        geom_density(fill = "skyblue", alpha = 0.5) +
        labs(
            title = "Volume PDF",
            x = "Predicted Volume",
            y = "Density"
        ) +
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5))

    p2 <- ggplot(data.frame(value = time), aes(x = value)) +
        geom_density(fill = "lightcoral", alpha = 0.5) +
        labs(
            title = "Time PDF",
            x = "Predicted Time",
            y = "Density"
        ) +
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5))

    # Save the combined plot to a PDF file
    pdf("pdf_predictions.pdf", width = 12, height = 6)
    grid.arrange(p1, p2, ncol = 2)
    dev.off()

    # Also display the plot in the R graphics window
    grid.arrange(p1, p2, ncol = 2)

    # Return summary statistics
    return(list(
        volume = list(
            mean = mean(volume),
            sd = vol_sd,
            quantiles = quantile(volume, probs = c(0.025, 0.5, 0.975))
        ),
        time = list(
            mean = mean(time),
            sd = time_sd,
            quantiles = quantile(time, probs = c(0.025, 0.5, 0.975))
        )
    ))
}

# Test the function
pdf_given(30, 200)

find_params_given_mfr <- function(target_mfr, n_simulations = 1000) {
    # Define parameter ranges
    G_range <- seq(10, 30, length.out = 20)  # 20 evenly spaced G values
    Temp_range <- seq(66, 270, length.out = 20)  # 20 evenly spaced Temp values
    
    # Initialize storage for results
    results <- data.frame(
        G = numeric(),
        Temp = numeric(),
        predicted_mfr = numeric(),
        error = numeric()
    )
    
    # Progress bar
    pb <- txtProgressBar(min = 0, max = length(G_range) * length(Temp_range), style = 3)
    counter <- 0
    
    # Run simulations for each parameter combination
    for (G in G_range) {
        for (Temp in Temp_range) {
            # Get predictions for this parameter combination
            pred <- pdf_given(G, Temp)
            
            # Calculate mass flow rate (using mean predictions)
            # Assuming mass flow rate is related to volume/time
            predicted_mfr <- pred$volume$mean / pred$time$mean
            
            # Calculate error (absolute difference from target)
            error <- abs(predicted_mfr - target_mfr)
            
            # Store results
            results <- rbind(results, data.frame(
                G = G,
                Temp = Temp,
                predicted_mfr = predicted_mfr,
                error = error
            ))
            
            # Update progress bar
            counter <- counter + 1
            setTxtProgressBar(pb, counter)
        }
    }
    close(pb)
    
    # Find the best parameter combination
    best_idx <- which.min(results$error)
    best_params <- results[best_idx, ]
    
    # Create a contour plot of the error surface
    error_matrix <- matrix(results$error, 
                          nrow = length(G_range), 
                          ncol = length(Temp_range))
    
    # Create contour plot
    filename <- sprintf("parameter_search_mfr_%.5f.pdf", target_mfr)
    pdf(filename, width = 10, height = 8)
    filled.contour(G_range, Temp_range, error_matrix,
                  xlab = "G (kg/m²s)",
                  ylab = "Temperature (K)",
                  main = "Error Surface for Parameter Search",
                  color.palette = colorRampPalette(c("blue", "white", "red")),
                  key.title = title(main = "Error", cex.main = 1),
                  key.axes = axis(4, cex.axis = 0.7))
    dev.off()
    
    # Print results
    cat("\nBest parameters found:\n")
    cat("G =", best_params$G, "kg/m²s\n")
    cat("Temperature =", best_params$Temp, "K\n")
    cat("Predicted MFR =", best_params$predicted_mfr, "\n")
    cat("Error =", best_params$error, "\n")
    
    # Return the best parameters and full results
    return(list(
        best_params = best_params,
        all_results = results,
        error_surface = error_matrix
    ))
}

result <- find_params_given_mfr(target_mfr = 0.0001)
    