# This is file of methods for paper "Block Model Guided Unsupervised Feature Selection" accepted at SIGKDD 2020
# Authors: Zilong Bai, Hoa Nguyen, Ian Davidson
# Code implemented by: Zilong Bai @ UC Davis
# This module implements Algorithm 1 for BMGUFS from our paper.

import numpy as np
from numpy import linalg as LA
import torch

def solver_delta_enhanced_sparse(Y,F,Ms,beta,gamma,delta,MAXITER,step_size):
# Y is n x f feature matrix. Nonnegative.
# F is n x k block allocation matrix. Binary.
# Ms is k x k image/mixing matrix of the raw network topology.
# beta is the ratio of KL-divergence term.
# delta is minor postive scalar to avoid trivial solution.
# MAXITER is a large integer for the maximum amount of iterations.
# step_size is the step size for gradient descent
	[n,f] = Y.shape
	k = F.shape[1]
	D = np.identity(k)
	for j in range(k):
		D[j,j] = LA.norm(F[:,j])**2

#	cuda = torch.device('cuda')
	cuda = torch.device('cuda:1')
	F = torch.from_numpy(F).float().to(cuda)
	Y = torch.from_numpy(Y).float().to(cuda)
	D = torch.from_numpy(D).float().to(cuda)

	Ybar = torch.mm(torch.mm(F,D.inverse()),torch.mm(F.t(),Y))
	Dbar = torch.mm(D.inverse(),torch.mm(F.t(),Y))

	YY = torch.mm(Y.t(),Y)
	YbYb = torch.mm(Ybar.t(),Ybar)
	YbY = torch.mm(Ybar.t(),Y)

	Ms = Ms + np.ones((k,k))*delta
	Ms = torch.from_numpy(Ms).float().to(cuda)

	#delta = torch.from_numpy(delta).to(cuda)

	P = torch.zeros(k,k).float().to(cuda)
	for i in range(k):
		P[i,:] = Ms[i,:]/torch.sum(Ms[i,:])

#       P = torch.from_numpy(P).float().to(cuda)

	Ls_dr = (torch.ones(f)/torch.norm(torch.ones(f))).float().to(cuda) # For sparsity regularization. Weighted by float scalar gamma.

	r = torch.ones(f).to(cuda) # Make sure it is a vector, not a matrix. Otherwise torch.diag(r) wouldn't work for R.

	for iteration in range(MAXITER):
		print('inside iteraction:'+str(iteration+1))
		R = torch.diag(r)
		print(r.shape)
		print(R.shape)
		print(Dbar.shape)
		M = torch.mm(torch.mm(Dbar,R),Dbar.t())
		M = M + torch.ones(k,k).to(cuda)*delta
		A = torch.mm(torch.mm(Y,R),Y.t())
		X = torch.mm(torch.mm(Ybar,R),Ybar.t())
		YAY = torch.mm(torch.mm(Y.t(),A),Y)
		Lb = (torch.norm((A - X))**2)/(torch.norm(A)**2)
		Lb_dr = torch.diag(YAY + torch.mm(torch.mm(Ybar.t(),X-(A*2)),Ybar))*2/(torch.norm(A)**2) - torch.diag(YAY)*2*Lb/(torch.norm(A)**2)
#               Lb_dr = torch.diag(torch.mm(torch.mm(Y.t(),A),Y) + torch.mm(torch.mm(Ybar.t(),X),Ybar) - torch.mm(torch.mm(Ybar.t(),A),Ybar)*2)*2/(torch.norm(A)**2) - torch.diag(torch.mm(torch.mm(Y.t(),A),Y))*2*Lb/(torch.norm(A)**2)

		Lm_dr = torch.zeros(f).float().to(cuda)
		Q = torch.zeros(k,k).float().to(cuda)
		for i in range(k):
			Q[i,:] = M[i,:]/torch.sum(M[i,:])

		for l in range(f):
			Q_rl = torch.mm(torch.diag(torch.div(torch.ones(k).to(cuda),torch.sum(M,1))) , ( (  torch.ger(Dbar[:,l],Dbar[:,l])  )  -  torch.mm(torch.diag(Dbar[:,l]), Q)*torch.sum(Dbar[:,l]) ))
			Lm_dr[l] = torch.trace( torch.mm( (torch.log( torch.div(Q,P) )  + P ).t() , Q_rl ) )

		L_dr = ( Lb_dr/torch.norm(Lb_dr) )*(1.0-beta) + ( Lm_dr/torch.norm(Lm_dr) )*beta + Ls_dr*gamma

		g = L_dr*step_size
		r = r - g
		for l in range(f):
			r[l] = max(r[l],0)

		normr = torch.norm(r)
		r = r/normr

	return r

