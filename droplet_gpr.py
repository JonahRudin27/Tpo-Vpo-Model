# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 00:03:51 2021

@author: George
"""
import numpy as np
from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
import pandas as pd
import time

#%% No Gravity
filename = 'droplet_data.csv'
df = pd.read_csv(filename)
df = df.dropna()

dfgpr = df.to_numpy()

#k = np.array(range(np.size(dfgpr,0)))+1


# #Simplots
# fig, axs = plt.subplots(1, 2)
# axs[0].scatter(k,dfgpr[:,11], marker = 'x', color='k')
# axs[0].set_xlabel('N')
# axs[0].set_ylabel('Pinch-off volume (m^3)')
# axs[1].scatter(k, dfgpr[:,12], marker = 'x', color='r')
# axs[1].set_xlabel('N')
# axs[1].set_ylabel('Pinch-off time (s)')
# plt.subplots_adjust(left=0.1,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.9, 
#                     wspace=0.4, 
#                     hspace=0.4)
# plt.show()

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, DotProduct, RBF
import matplotlib.pyplot as plt
import numpy as np


# Create kernel and define GPR
from sklearn.gaussian_process.kernels import ConstantKernel
kernel = 1.0 * RBF(1.0)
random_state = 0
gpr = GaussianProcessRegressor(kernel=kernel, random_state=random_state, alpha = 0.1, normalize_y=True)


x_train = dfgpr[:,0:3]
y_train_time = dfgpr[:, 12]
y_train_vol = dfgpr[:, 11]

x_train_scaled = np.zeros((np.size(x_train, 0) , np.size(x_train, 1)))

x_train_scaled[:,0] = (x_train[:,0]-np.mean(x_train[:,0]))/np.std(x_train[:,0])
x_train_scaled[:,1] = (x_train[:,1]-np.mean(x_train[:,1]))/np.std(x_train[:,1])
x_train_scaled[:,2] = (x_train[:,2]-np.mean(x_train[:,2]))/np.std(x_train[:,2])


from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
#3d scatter of train data
#from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(25,25), dpi=600) 
ax = fig.add_subplot(1, 1, 1, projection='3d')  
ax.scatter(x_train[:,0], x_train[:,1], 10*x_train[:,2], marker='o', s = 200, c = 'red')
ax.set_xlabel(r'Temperature $T^l$ (K)', fontsize = 22, labelpad=15)
ax.set_ylabel(r'Wavelength $\lambda^l (m)$', fontsize = 22, labelpad=18)
ax.set_zlabel(r'Oxidizer flux G ($\frac{kg}{m^2s}$)', fontsize = 22, labelpad=15)
ax.tick_params(axis='both', labelsize=22)
ax.set_facecolor('white')

plt.show()

#plt.savefig('Traindata_nogravity.png', bbox_inches='tight')

#2D subplots of train data
fig, axs = plt.subplots(2, 3)
axs[0, 0].scatter(x_train[:,0], y_train_vol, marker = 'o', c ='red', s = 5)
#axs[0, 0].set_xlabel(r'$T^l$ (K)', fontsize = 11)
axs[0,0].set_ylabel(r'$V_{po} (m^3)$')
axs[0, 1].scatter(x_train[:,1], y_train_vol, marker = 'o', c ='red', s = 5)
#axs[0, 1].set_xlabel(r'$\lambda^l (m)$', fontsize = 11)
axs[0, 2].scatter(10*x_train[:,2], y_train_vol, marker = 'o', c ='red', s = 5)
#axs[0, 2].set_xlabel(r'G ($\frac{kg m^2}{s}$)', fontsize = 11)
axs[1, 0].scatter(x_train[:,0], y_train_time, marker = 'o', c ='red', s = 5)
axs[1, 0].set_xlabel(r' $T^l$ (K)', fontsize = 11)
axs[1,0].set_ylabel(r'$t_{po} (s)$')
axs[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axs[1, 1].scatter(x_train[:,1], y_train_time, marker = 'o', c ='red', s = 5)
axs[1, 1].set_xlabel(r' $\lambda^l (m)$', fontsize = 11)
axs[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axs[1, 2].scatter(10*x_train[:,2], y_train_time, marker = 'o', c ='red', s = 5)
axs[1, 2].set_xlabel(r' G($\frac{kg}{m^2s}$)', fontsize = 11)
axs[1,2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.tight_layout()
#plt.savefig('Traindata_nogravity_2D.png', bbox_inches='tight', dpi = 800)

plt.show()
# Create test data

# import random

# x_test_T = np.linspace(66,270,100)
# x_test_lambda = np.linspace(1/300,1/100,100)
# x_test_G = np.linspace(10,30,100)

# x_test = np.array(np.meshgrid(x_test_T, x_test_lambda, x_test_G)).T.reshape(-1,3)

# x_test_scaled = np.zeros((np.size(x_test, 0) , np.size(x_test, 1)))


# x_test_scaled[:,0] = (x_test[:,0]-np.mean(x_train[:,0]))/np.std(x_train[:,0])
# x_test_scaled[:,1] = (x_test[:,1]-np.mean(x_train[:,1]))/np.std(x_train[:,1])
# x_test_scaled[:,2] = (x_test[:,2]-np.mean(x_train[:,2]))/np.std(x_train[:,2])



# # Predict mean
# y_hat_time, y_sigma_time = timegpr.predict(x_test_scaled, return_std=True)

# y_hat_time = y_hat_time[...,None]
# y_sigma_time = y_sigma_time[...,None]

# df_time_pred  = np.concatenate((x_test, y_hat_time,y_sigma_time), axis=1)

# mean_yhat_time_T = np.zeros(100)
# mean_ysigma_time_T = np.zeros(100)

# for i in range(0,np.size(x_test_T,0)):
#   mean_yhat_time_T[i] = np.mean(df_time_pred[df_time_pred[:,0] == x_test_T[i], 3])
#   mean_ysigma_time_T[i] = np.mean(df_time_pred[df_time_pred[:,0] == x_test_T[i], 4])
  
# #T ~ time plot

# plt.figure()
# plt.plot(x_train[:,0], y_train_time, 'r.', markersize=10, label='Observations')
# plt.plot(x_test_T, mean_yhat_time_T, 'b-', label='Mean Prediction')
# plt.fill(np.concatenate([x_test_T, x_test_T[::-1]]),
#           np.concatenate([mean_yhat_time_T - 1.9600 * mean_ysigma_time_T,
#                         (mean_yhat_time_T + 1.9600 * mean_ysigma_time_T)[::-1]]),
#           alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.xlabel('Temperature $(^{o}C)$')
# plt.ylabel('Pinch-off time $(s)$')
# plt.ylim(0, 0.05)
# plt.legend(loc='upper right')




# #lambda ~ time plot
# mean_yhat_time_lambda = np.zeros(100)
# mean_ysigma_time_lambda = np.zeros(100)

# for i in range(0,np.size(x_test_lambda,0)):
#   mean_yhat_time_lambda[i] = np.mean(df_time_pred[df_time_pred[:,1] == x_test_lambda[i], 3])
#   mean_ysigma_time_lambda[i] = np.mean(df_time_pred[df_time_pred[:,1] == x_test_lambda[i], 4])

# plt.figure()
# plt.plot(x_train[:,1], y_train_time, 'r.', markersize=10, label='Observations')
# plt.plot(x_test_lambda, mean_yhat_time_lambda, 'b-', label='Mean Prediction')
# plt.fill(np.concatenate([x_test_lambda, x_test_lambda[::-1]]),
#           np.concatenate([mean_yhat_time_lambda - 1.9600 * mean_ysigma_time_lambda,
#                         (mean_yhat_time_lambda + 1.9600 * mean_ysigma_time_lambda)[::-1]]),
#           alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.xlabel('Wavelength')
# plt.ylabel('Pinch-off time $(s)$')
# plt.ylim(0, 0.5)
# plt.legend(loc='upper right')


# #G ~ time plot

# mean_yhat_time_G = np.zeros(100)
# mean_ysigma_time_G = np.zeros(100)

# for i in range(0,np.size(x_test_G,0)):
#   mean_yhat_time_G[i] = np.mean(df_time_pred[df_time_pred[:,2] == x_test_G[i], 3])
#   mean_ysigma_time_G[i] = np.mean(df_time_pred[df_time_pred[:,2] == x_test_G[i], 4])

# plt.figure()
# plt.plot(x_train[:,2], y_train_time, 'r.', markersize=10, label='Observations')
# plt.plot(x_test_G, mean_yhat_time_G, 'b-', label='Mean Prediction')
# plt.fill(np.concatenate([x_test_G, x_test_G[::-1]]),
#           np.concatenate([mean_yhat_time_G - 1.9600 * mean_ysigma_time_G,
#                         (mean_yhat_time_G + 1.9600 * mean_ysigma_time_G)[::-1]]),
#           alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.xlabel('Oxidizer flux G $(kg/m^{2}s)$')
# plt.ylabel('Pinch-off time $(s)$')
# plt.ylim(0, 0.5)
# plt.legend(loc='upper right')





#Build GPs
start = time.time()
timegpr = gpr.fit(x_train_scaled[9:387], y_train_time[9:387])
end = time.time()
print(end-start)
y_hat_time, y_sigma_time = timegpr.predict(x_train_scaled[0:10], return_std=True)

volgpr = gpr.fit(x_train_scaled[9:387], y_train_vol[9:387])
y_hat_vol, y_sigma_vol = volgpr.predict(x_train_scaled[0:10], return_std=True)

#Plots with the GP at training spots
fig, axs = plt.subplots(2, 3)
axs[0, 0].scatter(x_train[:,0], y_train_vol, marker = 'o', c ='red', s = 5,label = 'Simulation data')
#axs[0, 0].set_xlabel(r'$T^l$ (K)', fontsize = 11)
axs[0,0].set_ylabel(r'$V_{po} (m^3)$')
axs[0,0].scatter(x_train[:,0], y_hat_vol, c='blue', marker = 'x', label='GaSP Mean Prediction', s = 2)
axs[0, 1].scatter(x_train[:,1], y_train_vol, marker = 'o', c ='red', s = 5, label = 'Simulation')
axs[0,1].scatter(x_train[:,1], y_hat_vol, c='blue', marker = 'x', label='Mean Prediction', s = 2)
#axs[0, 1].set_xlabel(r'$\lambda^l (m)$', fontsize = 11)
axs[0, 2].scatter(10*x_train[:,2], y_train_vol, marker = 'o', c ='red', s = 5, label = 'Simulation')
axs[0,2].scatter(10*x_train[:,2], y_hat_vol, c='blue', marker = 'x', label='Mean Prediction', s = 2)
#axs[0, 2].set_xlabel(r'G ($\frac{kg m^2}{s}$)', fontsize = 11)
axs[1, 0].scatter(x_train[:,0], y_train_time, marker = 'o', c ='red', s = 5, label = 'Simulation')
axs[1, 0].set_xlabel(r' $T^l$ (K)', fontsize = 11)
axs[1,0].set_ylabel(r'$t_{po} (s)$')
axs[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
axs[1,0].scatter(x_train[:,0], y_hat_time, c='blue', marker = 'x', label='Mean Prediction', s = 2)
axs[1, 1].scatter(x_train[:,1], y_train_time, marker = 'o', c ='red', s = 5, label = 'Simulation')
axs[1, 1].set_xlabel(r' $\lambda^l (m)$', fontsize = 11)
axs[1,1].scatter(x_train[:,1], y_hat_time, c='blue', marker = 'x', label='Mean Prediction', s = 2)
axs[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
axs[1, 2].scatter(10*x_train[:,2], y_train_time, marker = 'o', c ='red', s = 5, label = 'Simulation')
axs[1, 2].set_xlabel(r' G($\frac{kg}{m^2s}$)', fontsize = 11)
axs[1,2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
axs[1,2].scatter(10*x_train[:,2], y_hat_time, c='blue', marker = 'x', label='Mean Prediction', s = 2)
fig.tight_layout()
fig.subplots_adjust(bottom=0.15, wspace=0.5)
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5, -0.01),fancybox=False, shadow=False, ncol=2)
plt.savefig('GPtrain_accuracy_nogravity_2D.png', bbox_inches='tight', dpi = 800)


train_accuracy_df = np.column_stack((x_train, y_hat_vol, y_sigma_vol, y_train_vol, y_hat_time, y_sigma_time, y_train_time, np.abs(100*(y_hat_vol-y_train_vol)/y_train_vol),np.abs(100*(y_hat_time-y_train_time)/y_train_time) ))

#Plot of errors as a function of the inputs
fig, axs = plt.subplots(2, 3)
axs[0, 0].scatter(x_train[:,0], train_accuracy_df[:,9], marker = 'x', c ='blue', s = 2)
axs[0,0].set_ylabel(r'$\frac{\|\hat{V}_{po} - V_{po}\|}{V_{po}} (\%) $', fontsize=14)
axs[0, 1].scatter(x_train[:,1], train_accuracy_df[:,9], marker = 'x', c ='blue', s = 2)
axs[0, 2].scatter(10*x_train[:,2], train_accuracy_df[:,9], marker = 'x', c ='blue', s = 2)
axs[1, 0].set_xlabel(r' $T^l$ (K)', fontsize = 11)
axs[1,0].set_ylabel(r'$\frac{\|\hat{t}_{po} - t_{po}\|}{t_{po}} (\%)$', fontsize=14)
axs[1,0].scatter(x_train[:,0], train_accuracy_df[:,10], c='blue', marker = 'x', label='Mean Prediction', s = 2)
axs[1, 1].set_xlabel(r' $\lambda^l (m)$', fontsize = 11)
axs[1,1].scatter(x_train[:,1], train_accuracy_df[:,10], c='blue', marker = 'x', label='Mean Prediction', s = 2)
axs[1, 2].set_xlabel(r' G($\frac{kg}{m^2s}$)', fontsize = 11)
axs[1,2].scatter(10*x_train[:,2], train_accuracy_df[:,10], c='blue', marker = 'x', label='Mean Prediction', s = 2)
fig.tight_layout()
fig.subplots_adjust(bottom=0.15, wspace=0.5)
plt.savefig('errors_nogravity.png', bbox_inches='tight', dpi = 800)


#np.savetxt("train_accuracy_df.csv", train_accuracy_df, delimiter=",")

#Volume pdf MCM

#Assume uniform inputs G, T, lambda
Nsims = 1000000

G = np.random.uniform(10, 30, Nsims)
wave = np.random.uniform(1/300,1/100, Nsims)
Temp = np.random.uniform(66,270, Nsims)

ensembles = np.column_stack([G, wave, Temp])

ensembles_scaled = np.zeros((np.size(ensembles, 0), np.size(ensembles, 1)))

ensembles_scaled[:,0] = (ensembles[:,0]-np.mean(ensembles[:,0]))/np.std(ensembles[:,0])
ensembles_scaled[:,1] = (ensembles[:,1]-np.mean(ensembles[:,1]))/np.std(ensembles[:,1])
ensembles_scaled[:,2] = (ensembles[:,2]-np.mean(ensembles[:,2]))/np.std(ensembles[:,2])

y_hat_vol, y_sigma_vol = volgpr.predict(ensembles_scaled, return_std=True)


import seaborn as sns

figure = plt.figure()
sns.kdeplot(y_hat_vol, color = '#0504aa', shade = True)
plt.xlabel(r'$V_{po} (m^3)$', fontsize = 18 )
plt.ylabel('Probability Density', fontsize = 18 )
plt.savefig('vpo_caseb.png', bbox_inches='tight', dpi = 800)


#Time pdf MCM

#Assume uniform inputs G, T, lambda
Nsims = 1000000

G = np.random.uniform(10, 30, Nsims)
wave = np.random.uniform(1/300,1/100, Nsims)
Temp = np.random.uniform(66,270, Nsims)

ensembles = np.column_stack([G, wave, Temp])

ensembles_scaled = np.zeros((np.size(ensembles, 0), np.size(ensembles, 1)))

ensembles_scaled[:,0] = (ensembles[:,0]-np.mean(ensembles[:,0]))/np.std(ensembles[:,0])
ensembles_scaled[:,1] = (ensembles[:,1]-np.mean(ensembles[:,1]))/np.std(ensembles[:,1])
ensembles_scaled[:,2] = (ensembles[:,2]-np.mean(ensembles[:,2]))/np.std(ensembles[:,2])

timegpr = gpr.fit(x_train_scaled, y_train_time)
y_hat_time, y_sigma_time = timegpr.predict(ensembles_scaled, return_std=True)


import seaborn as sns

figure = plt.figure()
sns.kdeplot(y_hat_time, color = '#0504aa', shade = True)
plt.xlabel(r'$t_{po} (s)$', fontsize = 18 )
plt.ylabel('Probability Density', fontsize = 18 )
plt.savefig('tpo_caseb.png', bbox_inches='tight', dpi = 800)

#%% With Gravity
import sys
sys.modules[__name__].__dict__.clear()
import numpy as np
from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
import pandas as pd

filename = 'droplet_data_withg.csv'
df = pd.read_csv(filename)
df = df.dropna()

dfgpr = df.to_numpy()

# k = np.array(range(np.size(dfgpr,0)))+1


# #Simplots
# fig, axs = plt.subplots(1, 2)
# axs[0].scatter(k,dfgpr[:,12], marker = 'x', color='k')
# axs[0].set_xlabel('N')
# axs[0].set_ylabel('Pinch-off volume (m^3)')
# axs[1].scatter(k, dfgpr[:,13], marker = 'x', color='r')
# axs[1].set_xlabel('N')
# axs[1].set_ylabel('Pinch-off time (s)')
# plt.subplots_adjust(left=0.1,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.9, 
#                     wspace=0.4, 
#                     hspace=0.4)
# plt.show()

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, DotProduct, RBF
import matplotlib.pyplot as plt
import numpy as np


# Create kernel and define GPR
from sklearn.gaussian_process.kernels import ConstantKernel
kernel = 1.0 * RBF(1.0)
random_state = 0
gpr = GaussianProcessRegressor(kernel=kernel, random_state=random_state, alpha = 0.1, normalize_y=True)


x_train = dfgpr[:,1:4]
y_train_time = dfgpr[:, 13]
y_train_vol = dfgpr[:, 12]

x_train_scaled = np.zeros((np.size(x_train, 0) , np.size(x_train, 1)))

x_train_scaled[:,0] = (x_train[:,0]-np.mean(x_train[:,0]))/np.std(x_train[:,0])
x_train_scaled[:,1] = (x_train[:,1]-np.mean(x_train[:,1]))/np.std(x_train[:,1])
x_train_scaled[:,2] = (x_train[:,2]-np.mean(x_train[:,2]))/np.std(x_train[:,2])


from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

#Build GPs
timegpr = gpr.fit(x_train_scaled, y_train_time)
y_hat_time, y_sigma_time = timegpr.predict(x_train_scaled, return_std=True)

volgpr = gpr.fit(x_train_scaled, y_train_vol)
y_hat_vol, y_sigma_vol = volgpr.predict(x_train_scaled, return_std=True)

#Plots with the GP at training spots
fig, axs = plt.subplots(2, 3)
axs[0, 0].scatter(x_train[:,0], y_train_vol, marker = 'o', c ='red', s = 5,label = 'Simulation data')
axs[0,0].set_ylabel(r'$V_{po} (m^3)$')
axs[0,0].scatter(x_train[:,0], y_hat_vol, c='blue', marker = 'x', label='GaSP Mean Prediction', s = 2)
axs[0, 1].scatter(x_train[:,1], y_train_vol, marker = 'o', c ='red', s = 5, label = 'Simulation')
axs[0,1].scatter(x_train[:,1], y_hat_vol, c='blue', marker = 'x', label='Mean Prediction', s = 2)
axs[0, 2].scatter(10*x_train[:,2], y_train_vol, marker = 'o', c ='red', s = 5, label = 'Simulation')
axs[0,2].scatter(10*x_train[:,2], y_hat_vol, c='blue', marker = 'x', label='Mean Prediction', s = 2)
axs[1, 0].scatter(x_train[:,0], y_train_time, marker = 'o', c ='red', s = 5, label = 'Simulation')
axs[1, 0].set_xlabel(r' $T^l$ (K)', fontsize = 11)
axs[1,0].set_ylabel(r'$t_{po} (s)$')
axs[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
axs[1,0].scatter(x_train[:,0], y_hat_time, c='blue', marker = 'x', label='Mean Prediction', s = 2)
axs[1, 1].scatter(x_train[:,1], y_train_time, marker = 'o', c ='red', s = 5, label = 'Simulation')
axs[1, 1].set_xlabel(r' $\lambda^l (m)$', fontsize = 11)
axs[1,1].scatter(x_train[:,1], y_hat_time, c='blue', marker = 'x', label='Mean Prediction', s = 2)
axs[1,1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
axs[1, 2].scatter(10*x_train[:,2], y_train_time, marker = 'o', c ='red', s = 5, label = 'Simulation')
axs[1, 2].set_xlabel(r' G($\frac{kg}{m^2s}$)', fontsize = 11)
axs[1,2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
axs[1,2].scatter(10*x_train[:,2], y_hat_time, c='blue', marker = 'x', label='Mean Prediction', s = 2)
fig.tight_layout()
fig.subplots_adjust(bottom=0.15, wspace=0.5)
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5, -0.01),fancybox=False, shadow=False, ncol=2)
plt.savefig('GPtrain_accuracy_gravity_2D.png', bbox_inches='tight', dpi = 800)


train_accuracy_df = np.column_stack((x_train, y_hat_vol, y_sigma_vol, y_train_vol, y_hat_time, y_sigma_time, y_train_time, np.abs(100*(y_hat_vol-y_train_vol)/y_train_vol),np.abs(100*(y_hat_time-y_train_time)/y_train_time) ))

#Plot of errors as a function of the inputs
fig, axs = plt.subplots(2, 3)
axs[0, 0].scatter(x_train[:,0], train_accuracy_df[:,9], marker = 'x', c ='blue', s = 2)
axs[0,0].set_ylabel(r'$\frac{\|\hat{V}_{po} - V_{po}\|}{V_{po}} (\%) $', fontsize=14)
axs[0, 1].scatter(x_train[:,1], train_accuracy_df[:,9], marker = 'x', c ='blue', s = 2)
axs[0, 2].scatter(10*x_train[:,2], train_accuracy_df[:,9], marker = 'x', c ='blue', s = 2)
axs[1, 0].set_xlabel(r' $T^l$ (K)', fontsize = 11)
axs[1,0].set_ylabel(r'$\frac{\|\hat{t}_{po} - t_{po}\|}{t_{po}} (\%)$', fontsize=14)
axs[1,0].scatter(x_train[:,0], train_accuracy_df[:,10], c='blue', marker = 'x', label='Mean Prediction', s = 2)
axs[1, 1].set_xlabel(r' $\lambda^l (m)$', fontsize = 11)
axs[1,1].scatter(x_train[:,1], train_accuracy_df[:,10], c='blue', marker = 'x', label='Mean Prediction', s = 2)
axs[1, 2].set_xlabel(r' G($\frac{kg}{m^2s}$)', fontsize = 11)
axs[1,2].scatter(10*x_train[:,2], train_accuracy_df[:,10], c='blue', marker = 'x', label='Mean Prediction', s = 2)
fig.tight_layout()
fig.subplots_adjust(bottom=0.15, wspace=0.5)
plt.savefig('errors_gravity.png', bbox_inches='tight', dpi = 800)


np.savetxt("train_accuracy_df_gravity.csv", train_accuracy_df, delimiter=",") 


#Volume pdf MCM

#Assume uniform inputs G, T, lambda
Nsims = 1000000

G = np.random.uniform(10, 30, Nsims)
wave = np.random.uniform(1/300,1/100, Nsims)
Temp = np.random.uniform(66,270, Nsims)

ensembles = np.column_stack([G, wave, Temp])

ensembles_scaled = np.zeros((np.size(ensembles, 0), np.size(ensembles, 1)))

ensembles_scaled[:,0] = (ensembles[:,0]-np.mean(ensembles[:,0]))/np.std(ensembles[:,0])
ensembles_scaled[:,1] = (ensembles[:,1]-np.mean(ensembles[:,1]))/np.std(ensembles[:,1])
ensembles_scaled[:,2] = (ensembles[:,2]-np.mean(ensembles[:,2]))/np.std(ensembles[:,2])


volgpr = gpr.fit(x_train_scaled, y_train_vol)

y_hat_vol, y_sigma_vol = volgpr.predict(ensembles_scaled, return_std=True)

#plt.scatter(ensembles[:,1], y_hat_vol)

#test = np.column_stack([sorted(ensembles[:,1]), sorted(y_hat_vol)])


import seaborn as sns

figure = plt.figure()
sns.kdeplot(y_hat_vol, color = '#0504aa', shade = True)
plt.xlabel(r'$V_{po} (m^3)$', fontsize = 18 )
plt.ylabel('Probability Density', fontsize = 18 )
#len(np.where(y_hat_vol<1.5*10**(-8))[0])
#len(np.where(ensembles[:,1]<0.0066666)[0])
#len(np.where(y_hat_vol>1.5*10**(-8))[0])
#len(np.where(ensembles[:,1]>0.0066666)[0])
plt.savefig('vpo_casea.png', bbox_inches='tight', dpi = 800)

# #Special plot with both (need to edit above to create)
# import seaborn as sns

# figure = plt.figure()
# sns.kdeplot(y_hat_vol_a, color = '#0504aa', shade = True)
# sns.kdeplot(y_hat_vol_b, color = 'grey', shade = True)
# plt.xlabel(r'$V_{po} (m^3)$', fontsize = 18 )
# plt.ylabel('Probability Density', fontsize = 18 )
# plt.legend(ncol = 1, loc='upper right', labels=[r'Case A: a = -g',r'Case B: a = 0'], fontsize = 12)
# plt.savefig('vpo_a_vs_b.png', bbox_inches='tight', dpi = 800)


#Time pdf MCM

#Assume uniform inputs G, T, lambda
Nsims = 1000000

G = np.random.uniform(10, 30, Nsims)
wave = np.random.uniform(1/300,1/100, Nsims)
Temp = np.random.uniform(66,270, Nsims)

ensembles = np.column_stack([G, wave, Temp])

ensembles_scaled = np.zeros((np.size(ensembles, 0), np.size(ensembles, 1)))

ensembles_scaled[:,0] = (ensembles[:,0]-np.mean(ensembles[:,0]))/np.std(ensembles[:,0])
ensembles_scaled[:,1] = (ensembles[:,1]-np.mean(ensembles[:,1]))/np.std(ensembles[:,1])
ensembles_scaled[:,2] = (ensembles[:,2]-np.mean(ensembles[:,2]))/np.std(ensembles[:,2])

timegpr = gpr.fit(x_train_scaled, y_train_time)
y_hat_time, y_sigma_time = timegpr.predict(ensembles_scaled, return_std=True)


import seaborn as sns

figure = plt.figure()
sns.kdeplot(y_hat_time, color = '#0504aa', shade = True)
plt.xlabel(r'$t_{po} (s)$', fontsize = 18 )
plt.ylabel('Probability Density', fontsize = 18 )
plt.savefig('tpo_casea.png', bbox_inches='tight', dpi = 800)

# figure = plt.figure()
# sns.kdeplot(y_hat_time_a, color = '#0504aa', shade = True)
# sns.kdeplot(y_hat_time_b, color = 'grey', shade = True)
# plt.xlabel(r'$t_{po} (s)$', fontsize = 18 )
# plt.ylabel('Probability Density', fontsize = 18 )
# plt.legend(ncol = 1, loc='upper right', labels=[r'Case A: a = -g',r'Case B: a = 0'], fontsize = 12)
# plt.savefig('tpo_a_vs_b.png', bbox_inches='tight', dpi = 800)








