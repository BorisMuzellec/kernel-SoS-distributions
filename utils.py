import numpy as np
import torch


def make_kernels(Xm, Xn, sigma, eps=0):
	""" 
	Build matrices required for optimization.

	Parameters
	----------
	Xm : ndarray
		Support points for the density model.
	Xn : ndarray
		Support points of the empirical measure that we approximate.
	sigma : float
		Gaussian kernel bandwidth.
	eps : float, default = 0
		Diagonal regularization strength for W.
		
	Returns
	-------
	Kmm : ndarray
		Matrix of pairwise kernel evaluations of Xm.
	Kmn : ndarray
		Matrix of pairwise kernel evaluations of Xm and Xn.
	W : ndarray
		Kernel covariance matrix of the reference measure.
		p_B is a density function iff Tr(BW) = 1.
	U : ndarray
		Tensor of u_pq vectors (1 <= p, q <= m) as per the paper.
	"""
	
	m = len(Xm)
	n = len(Xn)


	W = make_W(Xm, sigma) + eps * torch.eye(m)
	U = make_U(Xm, sigma)

	Kmm = torch.exp((-(Xm[:, None] - Xm)**2).sum(-1) / sigma**2)
	Kmn = torch.exp((-(Xm[:, None] - Xn)**2).sum(-1) / sigma**2)

	return Kmm, Kmn, W, U


def make_W(X, sigma):
	""" 
	Build the covariance matrix of the reference measure.
	p_B is a density function iff Tr(BW) = 1.

	Parameters
	----------
	X : ndarray
		Support points for the density model.
	sigma : float
		Gaussian kernel bandwidth.

	Returns
	-------
	W : ndarray
		Kernel covariance matrix.

	"""

	n, d = X.shape
	C = (sigma * np.sqrt(np.pi / 2)) ** d
	return C * torch.exp(- ((X[:, None] - X)**2).sum(2) / (2 * sigma**2))


def make_U(X, sigma):
	""" 
	Build the tensor of u_pq vectors (1 <= p, q <= m) as per the paper.

	Parameters
	----------
	X : ndarray
		Support points for the density model.
	sigma : float
		Gaussian kernel bandwidth.

	Returns
	-------
	U : ndarray
		m x m x m tensor.

	"""
	m, d = X.shape
	
	norms = (X**2).sum(1)
	
	arg = norms[:, None, None] + norms[None, :, None] + norms[None, None, :]  \
		- ((X[:, None, None] + X[None, :, None] + X[None, None, :])**2).sum(-1) / 3.
	return ((sigma *  np.sqrt(np.pi / 3) ) ** d) *  torch.exp(-arg/sigma**2)



def density(x, X, B, sigma=1.):
	""" 
	Evaluate the density model.

	Parameters
	----------
	x : float or ndarray
		Points at wich to evaluate the density.
	X : ndarray
		Support points for the density model.
	B : ndarray
		Model parameter (PSD matrix).
	sigma : float, default = 1.
		Gaussian kernel bandwidth.

	Returns
	-------
	float or ndarray
		Value of the density model at x.
	"""
	
	if len(x.shape) == 1:
		q = np.exp((-(x - X)**2).sum(-1) / sigma**2)
		return q @ B @ q
	else:
		q = np.exp((-(x[:, None] - X)**2).sum(-1) / sigma**2)
		return (q @ B * q).sum(-1)
