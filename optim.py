import torch
import numpy as np

import time

from scipy.sparse.linalg import eigs

torch.set_default_dtype(torch.float64)

def accelerated_gd(U, W, Kmm, Kmn, lbda=1e-3, tau=1e-4, niter=1000,  reg='trace',
					acceleration=True, factored=True, verbose=True, report_interval=1000):

	"""
	Fit a sos density using FISTA (Beck and Teboulle 2009)

	Parameters
	----------
	U : ndarray
		m x m x m tensor of u_pq vectors (1 <= p, q <= m) as per the paper.
	W : ndarray
		Kernel covariance matrix of the reference measure.
	Kmm : ndarray
		Matrix of pairwise kernel evaluations of Xm.
	Kmn : ndarray
		Matrix of pairwise kernel evaluations of Xm and Xn.
	lbda : float, default=1e-3
		Regularization strength.
	tau : float, default=1e-4
		Stopping criterion (loss decrease).
	niter : int, default=10000
		Maximum total number of iterations.
	reg : str, default = 'trace'
		Regularization type.
		Valid values: 'trace' or 'frobenius'.
	acceleration : bool, default=True
		If true, use FISTA. Otherwise, use projected gradient descent.
	factored : bool, default=True
		If true, use a factor of Kmm for computations (more stable).
	verbose : bool, default=True
		If true, report loss value periodically.
	report_interval : int, default=1000
		If verbose, period at which the loss is reported.

	Returns
	-------
	B: ndarray
		Optimization parameter.
	losses: list
		Values of objective throughout optimization.
	"""

	assert reg in ['trace', 'frobenius']
	
	start_time = time.time()

	m, n = Kmn.shape

	Ri = torch.inverse(torch.cholesky(W).T)
	K_inv = torch.inverse(Kmm)

	if factored:
		LKi = torch.inverse(torch.cholesky(Kmm))
	else:
		LKi = K_inv

	### Compute smoothness constant. 
	L = smoothness_constant(U, Kmm, Ri, lbda=lbda, reg=reg)

	lr = 1. / L
	t = 1.

	eta = torch.eye(m) / m
	C = eta.clone()

	eta.requires_grad = True

	losses = []

	for i in range(niter):

		### Compute objective
		loss = reg_mmd(Ri @ eta @ Ri.T, U, Kmm, Kmn, LKi, lbda=lbda, reg=reg, factored=factored)   

		## Gradient update
		loss.backward()
		with torch.no_grad():
			C_ = project_C(eta - lr * eta.grad)

			## FISTA
			if acceleration:
				t_ = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
				eta = C_ + (t - 1) / t_ * (C_ - C)
				eta = .5 * (eta + eta.T)
				t = t_
			else:
				eta.data = C_
				
		if eta.grad is not None:
			eta.grad *= 0
		else:
			eta.requires_grad=True
		
		losses.append(loss.item())

		### Check if we should stop
		if i > 0 and np.abs(losses[-2] - losses[-1]) < tau:
			if verbose:
				print(f"iter {i}:\tloss: {losses[i]:4e}")
				print(f"Precision {tau:.2e} reached in {time.time() - start_time:.2e} seconds")
			return Ri @ C_ @ Ri.T, losses

		C = C_.clone()

		if verbose:
			if i % report_interval == 0:
				print(f"iter {i + 1}:\tloss: {losses[i]:.2e}")

	if verbose:
		print(f"iter {i}:\tloss: {losses[i]:.4e}")
		print(f"Precision {tau:.2e} not reached in {time.time() - start_time:.2e} seconds")
	return Ri @ C @ Ri.T, losses


def euclidean_proj_simplex(v, s=1, presorted=False):
	""" 
	Compute the Euclidean projection on a positive simplex.

	Adapted from https://gist.github.com/daien/1272551 
	Adrien Gaidon - INRIA - 2011

	Parameters
	----------
	v : ndarray
		Vector to project.
	s : float, default=1.
		Target sum.
	presorted : bool, default=False
		Set to true if v is already sorted in decreasing order.
	Returns
	-------
	ndarray
		Euclidean projection of v on the s-sum positive simplex.

	"""
	
	assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
	n, = v.shape  # will raise ValueError if v is not 1-D
	# check if we are already on the simplex
	if v.sum() == s and np.alltrue(v >= 0):
		# best projection: itself!
		return v
	# get the array of cumulative sums of a sorted (decreasing) copy of v
	if not presorted:
		u = np.sort(v)[::-1]
	else:
		u = v[::-1]
	cssv = np.cumsum(u)
	# get the number of > 0 components of the optimal solution
	rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
	# compute the Lagrange multiplier associated to the simplex constraint
	theta = (cssv[rho] - s) / (rho + 1.0)
	# compute the projection by thresholding v using theta
	w = (v - theta).clip(min=0)
	return w

def project_C(matrix):
	""" 
	Project the matrix onto {C PSD with Trace(C)=1}

	Parameters
	----------
	matrix : ndarray
		Matrix to project.

	Returns
	-------
	ndarray
		Euclidean projection onto {C PSD with Trace(C)=1}.

	"""
	matrix = 0.5 * (matrix + matrix.T)
	vals, vecs = torch.symeig(matrix, True)
	proj_vals = euclidean_proj_simplex(vals)
	return (vecs * proj_vals) @ vecs.T


def smoothness_constant(U, K, Ri, lbda=1e-2, reg='trace'):
	""" 
	Compute the smoothness constant of the objective function.

	Parameters
	----------
	U : ndarray
		m x m x m tensor of u_pq vectors (1 <= p, q <= m) as per the paper.
	K : ndarray
		Matrix of pairwise kernel evaluations of density support points.
	Ri : ndarray
		Inverse of the Cholesky factor of W. 
	lbda : float, default = 1e-2
		Matrix to project
	reg : str, default = 'trace'
		Regularization type.
		Valid values: 'trace' or 'frobenius'.

	Returns
	-------
	float
		Smoothness constant of the objective function (largest eigenvalue of the Hessian).

	"""

	assert reg in ['trace', 'frobenius']

	m = len(U)
	nsym = int((m * (m + 1)) / 2)

	start_time = time.time()

	K_inv = torch.inverse(K)

	H = 2. * torch.tensordot(U, K_inv @  U, dims = ([0], [1]))

	Hw = torch.kron(Ri.T, Ri.T) @ H.reshape(m**2, m**2) @ torch.kron(Ri, Ri)

	if reg == 'frobenius':
		Kw = Ri.T @ K @ Ri 
		Kw = .5 * (Kw + Kw.T)
		Hw += 2 * lbda * torch.kron(Kw, Kw)

	Hw = (Hw + Hw.T) / 2
	
	la_max = np.real(eigs(Hw.numpy(), k=1, which='LR', return_eigenvectors=False)[0])
	print(f'Largest eigenvalue = {la_max:.4e} obtained in {time.time() - start_time:.2e} seconds')
	return la_max

def reg_mmd(B, U, Kmm, Kmn, K_inv, lbda=1e-2, factored=False, reg='trace'):
	""" 
	MMD distance between model and empirical distribution, plus optional regularization.
	This is the objective function of the optimization problem.

	Parameters
	----------
	B : ndarray
		Density model parameter (PSD matrix).
	U : ndarray
		m x m x m tensor of u_pq vectors (1 <= p, q <= m) as per the paper.
	Kmm : ndarray
		Matrix of pairwise kernel evaluations of Xm.
	Kmn : ndarray
		Matrix of pairwise kernel evaluations of Xm and Xn.
	K_inv : ndarray
		If factored = True, inverse of the Cholesky factor of Kmm.
		Else, inverse of Kmm.
	lbda : float, default = 1e-2
		Regularization strength.
	factored : bool, default= False
		If true, use a factor of Kmm for computations (more stable).
	reg : str, default = 'trace'
		Valid values: 'trace' or 'frobenius'.

	Returns
	-------
	ndarray
		Objective function (MMD + lambda * regularization).

	"""

	assert reg in ['trace', 'frobenius']

	UB = torch.tensordot(U, B)
	if not factored:
		c = K_inv @ Kmn.mean(1)
		loss =  UB.T @ K_inv @ UB - 2 * c @ UB + c.T @ Kmm @ c
	else:
		## In that case, K_inv is actually a Cholesky factor
		c = K_inv.T @ K_inv @ Kmn.mean(1)
		loss = ((K_inv @ UB)**2).sum() - 2 * c @ UB + c.T @ Kmm @ c

	if reg == 'trace':
		return loss + lbda * (B * Kmm).sum() 
	elif reg == 'frobenius':
		return loss + lbda * (B * (Kmm @ B @ Kmm)).sum()

