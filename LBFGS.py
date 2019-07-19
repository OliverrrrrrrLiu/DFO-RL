import numpy as np
from numpy.linalg import multi_dot
from collections import deque

class LBFGS(object):
	"""L-BFGS to compute search direction d_k = -H_k * grad_k
	@param x: starting point
	@param m: number of vector pairs (s_i, y_i) stored
	@param s: matrix with rows  s_k = x_k+1 - x_k 
	@param y: matrix with rows y_k = grad_k+1 - grad_k
	@param rho: 1/y_kT*s_k rho = np.sum(s*y, axis = 1)
	"""
	def __init__(self, m):
		self.m = m
		self.s = []
		self.y = []
		self.rho = []
		self.iter = 0

	def update_history(self, s_new, y_new):
		self.s.append(s_new)
		self.y.append(y_new)
		self.rho.append(np.inner(s_new, y_new))
		if len(self.s) <= self.m:
			pass
		else:
			self.s = self.s[1:]
			self.y = self.y[1:]
			self.rho = self.rho[1:]

	def calculate_direction(self, grad, H, k):
		self.iter += 1
		if k == 0:
			return -grad
		elif k < self.m:
			tmp = np.eye(len(grad)) - self.rho[-1] * np.outer(self.s[-1], self.y[-1])
			H_new = multidot([tmp, H, tmp]) + self.rho[-1] * np.outer(self.s[-1], self.s[-1])
			return -np.matmul(H_new, grad)
		q = grad
		alpha = np.zeros(self.m)
		for i in range(self.m):
			alpha[i] = self.rho[i] * np.inner(self.s[i], q)
			q -= alpha[i] * self.y[i]
		gamma_k = np.inner(self.s[-1],self.y[-1]) / np.inner(self.y[-1],self.y[-1])
		r = gamma_k * q
		for i in range(self.m-1,-1,-1):
			beta = self.rho[i] * np.inner(self.y[i],r)
			r += self.s[i] * (alpha[i] - beta)
		self.iter += 1
		return -r

if __name__ == "__main__":
	grad = np.ones(3)
	m = 3
	s = np.array([[1,2,3],[2,1,4],[0,2,1]])
	y = np.array([[0,1,1],[2,1,3],[0,0,3]])
	rho = np.sum(s*y, axis = 1)
	lbfgs = LBFGS(m)
	lbfgs.s, lbfgs.y, lbfgs.rho = s, y, rho
	print(lbfgs.calculate_direction(grad, None, 3))
    





