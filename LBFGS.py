import numpy as np
from numpy.linalg import multi_dot
from collections import deque
from numpy.linalg import norm

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
		"""
		update (s,y) correction pairs so that only the most recent self.m pairs are stored
		@param s_new: new s vector x_k+1 - x_k
		@param y_new: new y vector grad_k+1 - grad_k

		@return: update the storage (s,y)
		"""
		self.s.append(s_new)
		self.y.append(y_new)
		self.rho.append(1/np.inner(s_new, y_new))
		if len(self.s) <= self.m: #if less then 
			pass
		else: #the oldest (s,y,rho) is repalced by the newest (s,y,rho) pair
			self.s = self.s[1:]
			self.y = self.y[1:]
			self.rho = self.rho[1:]

	def calculate_direction(self, grad):
		"""
		calculate the LBFGS direction - H * grad

		@param: current gradient

		@return: LBFGS search direction 
		"""
		self.iter += 1 #initialize iteration counter
		k = len(self.rho) #check number of (s,y) pairs stored
		if k == 0: # if no (s,y) pair stored use gradient descent direction
			return -grad
		# elif k < self.m: #if less than m pairs of (s,y) stored, apply bfgs direction
		# 	tmp = np.eye(len(grad)) - self.rho[-1] * np.outer(self.s[-1], self.y[-1])
		# 	self.H = multi_dot([tmp, self.H, tmp]) + self.rho[-1] * np.outer(self.s[-1], self.s[-1])
		# 	return -np.matmul(self.H, grad)
		q = grad
		alpha = np.zeros(k)
		for i in range(k-1,-1,-1):
			alpha[i] = self.rho[i] * np.inner(self.s[i],q)
			q -= alpha[i] * self.y[i]
		gamma_k = np.inner(self.s[-1],self.y[-1]) / np.inner(self.y[-1], self.y[-1])
		r = gamma_k * q
		for i in range(k):
			beta = self.rho[i] * np.inner(self.y[i], r)
			r += self.s[i] * (alpha[i] - beta)
		self.iter += 1
		return -r


		# #LBFGS two-loop recursion: recusively compute direction = -(H*grad)
		# q = grad #create copy of current gradient
		# alpha = np.zeros(self.m) 
		# for i in range(self.m):
		# 	alpha[i] = self.rho[i] * np.inner(self.s[i], q)
		# 	q -= alpha[i] * self.y[i]
		# gamma_k = np.inner(self.s[-1],self.y[-1]) / np.inner(self.y[-1],self.y[-1])
		# r = gamma_k * q
		# for i in range(self.m-1,-1,-1):
		# 	beta = self.rho[i] * np.inner(self.y[i],r)
		# 	r += self.s[i] * (alpha[i] - beta)
		# self.iter += 1 #increase iteration counter
		# return -r

"""
test problem 
"""

if __name__ == "__main__":
	grad = np.ones(3)
	m = 3
	s = np.array([[1,2,3],[2,1,4],[0,2,1]])
	y = np.array([[0,1,1],[2,1,3],[0,0,3]])
	rho = np.sum(s*y, axis = 1)
	lbfgs = LBFGS(m)
	lbfgs.s, lbfgs.y, lbfgs.rho = s, y, rho
	print(s,y,rho)
	print(lbfgs.calculate_direction(grad))
    





