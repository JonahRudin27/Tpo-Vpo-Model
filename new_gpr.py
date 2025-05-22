import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Load data
filename = 'droplet_data.csv'
df = pd.read_csv(filename).dropna()
data = df.to_numpy()

# Input and output extraction
x_train = data[:, 0:3]
y_train_time = data[:, 12]
y_train_vol = data[:, 11]



x_train_scaled = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)

kernel = 1.0 * Matern(nu=0.5)

# best_score = -np.inf
# best_nu = None

# #find best nu
# for nu in [0.5, 1.5, 2.5]:
#     kernel = 1.0 * Matern(nu=nu)
#     gpr_time = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
#     gpr_vol = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
#     score = np.mean(cross_val_score(gpr_time, x_train_scaled, y_train_time, scoring='neg_mean_squared_error', cv=5))
#     + np.mean(cross_val_score(gpr_vol, x_train_scaled, y_train_vol, scoring='neg_mean_squared_error', cv=5))
#     if score > best_score:
#         best_score = score
#         best_nu = nu

# print(f"nu: {nu}")
# kernel = 1.0 * Matern(nu=best_nu)

# Train separate GPR models for time and volume

gpr_time = GaussianProcessRegressor(kernel=kernel, random_state=0, alpha=0.1, normalize_y=True)

gpr_vol = GaussianProcessRegressor(kernel=kernel, random_state=0, alpha=0.1, normalize_y=True)

# Set up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# To store the performance scores (MSE in this case)
mse_time = []
mse_vol = []

# Perform K-Fold Cross-Validation
for train_index, test_index in kf.split(x_train_scaled):
    # Split the data into train and test sets
    X_train, X_test = x_train_scaled[train_index], x_train_scaled[test_index]
    y_time_train, y_time_test = y_train_time[train_index], y_train_time[test_index]
    y_vol_train, y_vol_test = y_train_vol[train_index], y_train_vol[test_index]
    
    gpr_time.fit(X_train, y_time_train)
    gpr_vol.fit(X_train, y_vol_train)
    
    time_pred = gpr_time.predict(X_test)
    
    # Compute the mean squared error (MSE) for the fold
    mse_t = mean_squared_error(y_time_test, time_pred)
    mse_time.append(mse_t)

    vol_pred = gpr_vol.predict(X_test)
    
    mse_v = mean_squared_error(y_vol_test, vol_pred)
    mse_vol.append(mse_v)

    r2_t = r2_score(y_time_test, time_pred)
    print("R² score(cross validation time):", r2_t)
    r2_v = r2_score(y_vol_test, vol_pred)
    print("R² score(cross validation volume):", r2_v)


avg_mse = np.mean(mse_t)
print(f"Average Mean Squared Error from Cross-Validation on time model: {avg_mse}")
avg_mse = np.mean(mse_v)
print(f"Average Mean Squared Error from Cross-Validation on vol model: {avg_mse}")

print(mse_time)
print(mse_vol)

gpr_time = GaussianProcessRegressor(kernel=kernel, random_state=0, alpha=0.1, normalize_y=True)
gpr_time.fit(x_train_scaled, y_train_time)
y_hat_time, y_sigma_time = gpr_time.predict(x_train_scaled, return_std=True)

gpr_vol = GaussianProcessRegressor(kernel=kernel, random_state=0, alpha=0.1, normalize_y=True)
gpr_vol.fit(x_train_scaled, y_train_vol)
y_hat_vol, y_sigma_vol = gpr_vol.predict(x_train_scaled, return_std=True)


r2_t = r2_score(y_train_time, y_hat_time)
print("R² score:", r2_t)
r2_v = r2_score(y_train_vol, y_hat_vol)
print("R² score:", r2_v)

# Compute confidence intervals
lower_time = y_hat_time - 1.96 * y_sigma_time
upper_time = y_hat_time + 1.96 * y_sigma_time

lower_vol = y_hat_vol - 1.96 * y_sigma_vol
upper_vol = y_hat_vol + 1.96 * y_sigma_vol

yt_outliers = []
xt_outliers = []
yv_outliers = []
xv_outliers = []

for i in range(len(x_train_scaled)):
    if not lower_time[i] < y_train_time[i] < upper_time[i]:
        yt_outliers.append(y_train_time[i])
        xt_outliers.append(x_train_scaled[i])

    if not lower_vol[i] < y_train_vol[i] < upper_vol[i]:
        yv_outliers.append(y_train_vol[i])
        xv_outliers.append(x_train_scaled[i])
    
xv_outliers = np.array(xv_outliers)
yv_outliers = np.array(yv_outliers)
xt_outliers = np.array(xt_outliers)
yt_outliers = np.array(yt_outliers)
print(xt_outliers.shape)

# Plot predictions on top of data with confidence intervals
fig, axs = plt.subplots(2, 3, figsize=(12, 6))

# Volume vs inputs
axs[0, 0].scatter(x_train[:, 0], y_train_vol, c='red', s=5, label='Simulation data')
axs[0, 0].errorbar(x_train[:, 0], y_hat_vol, yerr=1.96 * y_sigma_vol, fmt='x', color='blue', markersize=2, label='GaSP Mean ± 95% CI', alpha = .3)
axs[0, 0].set_ylabel(r'$V_{po} \ (m^3)$')

axs[0, 1].scatter(x_train[:, 1], y_train_vol, c='red', s=5)
axs[0, 1].errorbar(x_train[:, 1], y_hat_vol, yerr=1.96 * y_sigma_vol, fmt='x', color='blue', markersize=2, alpha = .1)

axs[0, 2].scatter(10 * x_train[:, 2], y_train_vol, c='red', s=5)
axs[0, 2].errorbar(10 * x_train[:, 2], y_hat_vol, yerr=1.96 * y_sigma_vol, fmt='x', color='blue', markersize=2, alpha = .3)

# Time vs inputs
axs[1, 0].scatter(x_train[:, 0], y_train_time, c='red', s=5)
axs[1, 0].errorbar(x_train[:, 0], y_hat_time, yerr=1.96 * y_sigma_time, fmt='x', color='blue', markersize=2, alpha = .3)
axs[1, 0].set_xlabel(r'$T^l$ (K)', fontsize=11)
axs[1, 0].set_ylabel(r'$t_{po} \ (s)$')
axs[1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

axs[1, 1].scatter(x_train[:, 1], y_train_time, c='red', s=5)
axs[1, 1].errorbar(x_train[:, 1], y_hat_time, yerr=1.96 * y_sigma_time, fmt='x', color='blue', markersize=2, alpha = .3)
axs[1, 1].set_xlabel(r'$\lambda^l$ (m)', fontsize=11)
axs[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

axs[1, 2].scatter(10 * x_train[:, 2], y_train_time, c='red', s=5)
axs[1, 2].errorbar(10 * x_train[:, 2], y_hat_time, yerr=1.96 * y_sigma_time, fmt='x', color='blue', markersize=2, alpha = .3)
axs[1, 2].set_xlabel(r'$G \left(\frac{kg}{m^2s}\right)$', fontsize=11)
axs[1, 2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# Layout and legend
fig.tight_layout()
fig.subplots_adjust(bottom=0.15, wspace=0.5)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=False, shadow=False, ncol=2)

# Save figure
plt.savefig('GPtrain_accuracy_gravity_2D.png', bbox_inches='tight', dpi=800)
plt.show()


if len(xv_outliers) > 0:  # Only plot if there are outliers
    axs[0, 0].scatter(xv_outliers[:, 0], yv_outliers, c='red', s=5, label='Outliers')
    axs[0, 0].set_ylabel(r'$V_{po} \ (m^3)$')

    axs[0, 1].scatter(xv_outliers[:, 1], yv_outliers, c='red', s=5)

    axs[0, 2].scatter(10 * xv_outliers[:, 2], yv_outliers, c='red', s=5)

# Time vs inputs
if len(xt_outliers) > 0:  # Only plot if there are outliers
    axs[1, 0].scatter(xt_outliers[:, 0], yt_outliers, c='red', s=5)
    axs[1, 0].set_xlabel(r'$T^l$ (K)', fontsize=11)
    axs[1, 0].set_ylabel(r'$t_{po} \ (s)$')
    axs[1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    axs[1, 1].scatter(xt_outliers[:, 1], yt_outliers, c='red', s=5)
    axs[1, 1].set_xlabel(r'$\lambda^l$ (m)', fontsize=11)
    axs[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    axs[1, 2].scatter(10 * xt_outliers[:, 2], yt_outliers, c='red', s=5)
    axs[1, 2].set_xlabel(r'$G \left(\frac{kg}{m^2s}\right)$', fontsize=11)
    axs[1, 2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# Layout and legend
fig.tight_layout()
fig.subplots_adjust(bottom=0.15, wspace=0.5)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), 
          fancybox=False, shadow=False, ncol=2)

# Save figure
plt.savefig('GPtrain_outliers.png', bbox_inches='tight', dpi=800)
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
plt.savefig('GPtrain_outliers.png', bbox_inches='tight', dpi=800)
plt.show()
