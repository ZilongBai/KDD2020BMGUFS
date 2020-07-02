# This is file of methods for paper "Block Model Guided Unsupervised Feature Selection" accepted at SIGKDD 2020
# Authors: Zilong Bai, Hoa Nguyen, Ian Davidson
# Code implemented by: Zilong Bai @ UC Davis
# This script demonstrates how to use Algorithm 1 BMGUFS from our paper on the BlogCatalog data for feature selection.

import numpy as np
import torch
import block_model_feature_selection as bmfs 

datastr = 'dataset/'
resultstr = 'results/'

print('Loading raw graph...')
X = np.genfromtxt((datastr+'BlogCatalog_Network.csv'),delimiter=',')
E = X + X.transpose()
E[np.where(X>0)] = 1

print('Loading nodal attributes...')
Y = np.genfromtxt((datastr+'BlogCatalog_Attributes.csv'),delimiter=',')

kmin = 6; kmax = 6;
repmax = 10;

gammas = [1, 2, 3] # Grid-search
betas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

MAXITER = 200 # 200 iterations. 
step_size = 1e-2
delta = 1e-6

repselected = 3 # Lowest RRE.

for k in range(kmin,kmax+1): # Iterate from k = kmin to k = kmax
	kstr = str(k);

# At each k, we first find which block model achieves the lowest reconstruction loss
	rep = repselected

	repstr = str(rep) # Already aligned rep to start at 1. No need to add 1 to translate to matlab indexing.
		
	print('Loading block allocation ...')
	F = np.genfromtxt((datastr+'blogcatalog_nmtf_'+kstr+'_'+repstr+'_F.csv'),delimiter=',');

	print('Loading mixing/image matrix ...')
	Ms = np.genfromtxt((datastr+'blogcatalog_nmtf_'+kstr+'_'+repstr+'_M.csv'),delimiter=',');

	for beta in betas:
		for gamma in gammas:
			r = bmfs.solver_delta_enhanced_sparse(Y,F,Ms,beta,gamma,delta,MAXITER,step_size)
			print(r.cpu().numpy())
			betastr = '0'+str(beta*10)
			np.savetxt(resultstr+'ours_blogcatalog_k_'+kstr+'_beta_'+betastr+'_gamma_'+str(gamma)+'_epoch_'+str(MAXITER)+'_r.csv',(r.cpu().numpy()),delimiter=',')
