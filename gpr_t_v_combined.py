import numpy as np
import pandas as pd
import gpflow
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Load data
filename = 'droplet_data.csv'
df = pd.read_csv(filename).dropna()
data = df.to_numpy()

# Extract inputs and outputs
x = data[:, 0:3]
y_time = data[:, 12]
y_vol = data[:, 11]

# Standardize input
x_scaled = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

# Set up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# To store the performance scores
mse_time = []
mse_vol = []

# Perform K-Fold Cross-Validation
for train_index, test_index in kf.split(x_scaled):
    # Split the data
    X_train, X_test = x_scaled[train_index], x_scaled[test_index]
    y_time_train, y_time_test = y_time[train_index], y_time[test_index]
    y_vol_train, y_vol_test = y_vol[train_index], y_vol[test_index]
    
    # Prepare augmented data for training
    num_train = X_train.shape[0]
    X_train_aug = np.vstack([
        np.hstack([X_train, np.full((num_train, 1), 0)]),  # output index 0 = time
        np.hstack([X_train, np.full((num_train, 1), 1)])   # output index 1 = vol
    ])
    Y_train_aug = np.hstack([y_time_train, y_vol_train])[:, None]
    
    # Prepare augmented data for testing
    num_test = X_test.shape[0]
    X_test_aug = np.vstack([
        np.hstack([X_test, np.full((num_test, 1), 0)]),  # output index 0 = time
        np.hstack([X_test, np.full((num_test, 1), 1)])   # output index 1 = vol
    ])
    
    # Convert to TensorFlow tensors
    X_train_tf = tf.convert_to_tensor(X_train_aug, dtype=tf.float64)
    Y_train_tf = tf.convert_to_tensor(Y_train_aug, dtype=tf.float64)
    X_test_tf = tf.convert_to_tensor(X_test_aug, dtype=tf.float64)
    
    # Define kernel with coregionalization
    kern = gpflow.kernels.Matern32(active_dims=[0, 1, 2]) * gpflow.kernels.Coregion(
        output_dim=2, rank=1, active_dims=[3]
    )
    
    # Define and train VGP model
    model = gpflow.models.VGP(
        data=(X_train_tf, Y_train_tf),
        kernel=kern,
        likelihood=gpflow.likelihoods.Gaussian()
    )
    
    # Optimize model
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    
    # Make predictions
    mean, var = model.predict_f(X_test_tf)
    mean = mean.numpy()
    var = var.numpy()
    
    # Split predictions back into time and volume
    time_pred = mean[:num_test].flatten()
    vol_pred = mean[num_test:].flatten()
    
    # Compute metrics
    mse_t = mean_squared_error(y_time_test, time_pred)
    mse_v = mean_squared_error(y_vol_test, vol_pred)
    mse_time.append(mse_t)
    mse_vol.append(mse_v)
    
    r2_t = r2_score(y_time_test, time_pred)
    r2_v = r2_score(y_vol_test, vol_pred)
    print("R² score(cross validation time):", r2_t)
    print("R² score(cross validation volume):", r2_v)

# Print average MSE
avg_mse_t = np.mean(mse_time)
avg_mse_v = np.mean(mse_vol)
print(f"Average Mean Squared Error from Cross-Validation on time model: {avg_mse_t}")
print(f"Average Mean Squared Error from Cross-Validation on vol model: {avg_mse_v}")

print("MSE time:", mse_time)
print("MSE vol:", mse_vol)

# Train final model on all data
num_data = x_scaled.shape[0]
X_aug = np.vstack([
    np.hstack([x_scaled, np.full((num_data, 1), 0)]),  # output index 0 = time
    np.hstack([x_scaled, np.full((num_data, 1), 1)])   # output index 1 = vol
])
Y_aug = np.hstack([y_time, y_vol])[:, None]

# Convert to TensorFlow tensors
X_aug_tf = tf.convert_to_tensor(X_aug, dtype=tf.float64)
Y_aug_tf = tf.convert_to_tensor(Y_aug, dtype=tf.float64)

# Define and train final model
kern = gpflow.kernels.Matern32(active_dims=[0, 1, 2]) * gpflow.kernels.Coregion(
    output_dim=2, rank=1, active_dims=[3]
)

model = gpflow.models.VGP(
    data=(X_aug_tf, Y_aug_tf),
    kernel=kern,
    likelihood=gpflow.likelihoods.Gaussian()
)

opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)

# Make predictions
mean, var = model.predict_f(X_aug_tf)
mean = mean.numpy()
var = var.numpy()

# Split predictions
y_hat_time = mean[:num_data].flatten()
y_hat_vol = mean[num_data:].flatten()
y_sigma_time = np.sqrt(var[:num_data].flatten())
y_sigma_vol = np.sqrt(var[num_data:].flatten())

# Compute R² scores
r2_t = r2_score(y_time, y_hat_time)
r2_v = r2_score(y_vol, y_hat_vol)
print("Final R² score (time):", r2_t)
print("Final R² score (volume):", r2_v)

# Compute confidence intervals
lower_time = y_hat_time - 1.96 * y_sigma_time
upper_time = y_hat_time + 1.96 * y_sigma_time
lower_vol = y_hat_vol - 1.96 * y_sigma_vol
upper_vol = y_hat_vol + 1.96 * y_sigma_vol

# Find outliers
yt_outliers = []
xt_outliers = []
yv_outliers = []
xv_outliers = []

for i in range(len(x_scaled)):
    if not lower_time[i] < y_time[i] < upper_time[i]:
        yt_outliers.append(y_time[i])
        xt_outliers.append(x_scaled[i])

    if not lower_vol[i] < y_vol[i] < upper_vol[i]:
        yv_outliers.append(y_vol[i])
        xv_outliers.append(x_scaled[i])
    
xv_outliers = np.array(xv_outliers)
yv_outliers = np.array(yv_outliers)
xt_outliers = np.array(xt_outliers)
yt_outliers = np.array(yt_outliers)

# Plot predictions and confidence intervals
fig, axs = plt.subplots(2, 3, figsize=(12, 6))

# Volume vs inputs
axs[0, 0].scatter(x[:, 0], y_vol, c='red', s=5, label='Simulation data')
axs[0, 0].errorbar(x[:, 0], y_hat_vol, yerr=1.96 * y_sigma_vol, fmt='x', 
                  color='blue', markersize=2, label='GaSP Mean ± 95% CI', alpha=0.3)
axs[0, 0].set_ylabel(r'$V_{po} \ (m^3)$')

axs[0, 1].scatter(x[:, 1], y_vol, c='red', s=5)
axs[0, 1].errorbar(x[:, 1], y_hat_vol, yerr=1.96 * y_sigma_vol, fmt='x', 
                  color='blue', markersize=2, alpha=0.1)

axs[0, 2].scatter(10 * x[:, 2], y_vol, c='red', s=5)
axs[0, 2].errorbar(10 * x[:, 2], y_hat_vol, yerr=1.96 * y_sigma_vol, fmt='x', 
                  color='blue', markersize=2, alpha=0.3)

# Time vs inputs
axs[1, 0].scatter(x[:, 0], y_time, c='red', s=5)
axs[1, 0].errorbar(x[:, 0], y_hat_time, yerr=1.96 * y_sigma_time, fmt='x', 
                  color='blue', markersize=2, alpha=0.3)
axs[1, 0].set_xlabel(r'$T^l$ (K)', fontsize=11)
axs[1, 0].set_ylabel(r'$t_{po} \ (s)$')
axs[1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

axs[1, 1].scatter(x[:, 1], y_time, c='red', s=5)
axs[1, 1].errorbar(x[:, 1], y_hat_time, yerr=1.96 * y_sigma_time, fmt='x', 
                  color='blue', markersize=2, alpha=0.3)
axs[1, 1].set_xlabel(r'$\lambda^l$ (m)', fontsize=11)
axs[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

axs[1, 2].scatter(10 * x[:, 2], y_time, c='red', s=5)
axs[1, 2].errorbar(10 * x[:, 2], y_hat_time, yerr=1.96 * y_sigma_time, fmt='x', 
                  color='blue', markersize=2, alpha=0.3)
axs[1, 2].set_xlabel(r'$G \left(\frac{kg}{m^2s}\right)$', fontsize=11)
axs[1, 2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# Layout and legend
fig.tight_layout()
fig.subplots_adjust(bottom=0.15, wspace=0.5)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), 
          fancybox=False, shadow=False, ncol=2)

# Save figure
plt.savefig('GPflow_train_accuracy.png', bbox_inches='tight', dpi=800)
plt.show()

# Create a new figure for outliers
fig_outliers, axs_outliers = plt.subplots(1, 3, figsize=(12, 4))

# Time vs inputs - only plot time outliers since that's where they exist
if len(xt_outliers) > 0:  # Only plot if there are outliers
    axs_outliers[0].scatter(xt_outliers[:, 0], yt_outliers, c='red', s=5, label='Outliers')
    axs_outliers[0].set_xlabel(r'$T^l$ (K)', fontsize=11)
    axs_outliers[0].set_ylabel(r'$t_{po} \ (s)$')
    axs_outliers[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    axs_outliers[1].scatter(xt_outliers[:, 1], yt_outliers, c='red', s=5)
    axs_outliers[1].set_xlabel(r'$\lambda^l$ (m)', fontsize=11)
    axs_outliers[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    axs_outliers[2].scatter(10 * xt_outliers[:, 2], yt_outliers, c='red', s=5)
    axs_outliers[2].set_xlabel(r'$G \left(\frac{kg}{m^2s}\right)$', fontsize=11)
    axs_outliers[2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# Layout and legend
fig_outliers.tight_layout()
fig_outliers.subplots_adjust(bottom=0.2, wspace=0.3)
handles, labels = axs_outliers[0].get_legend_handles_labels()
fig_outliers.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
                   fancybox=False, shadow=False, ncol=1)

# Save figure
plt.savefig('GPflow_outliers.png', bbox_inches='tight', dpi=800)
plt.show()


