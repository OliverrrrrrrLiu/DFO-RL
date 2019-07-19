import numpy as np
from ECNoise import ECNoise
from np.linalg import norm
from linesearch import LineSearch

#TODO: separate grad estimation from h calculation
#TODO: function counter

class recovery(object):
	def __init__(self, f, gamma_1, gamma_2):
		self.f = f
		self.gamma_1 = gamma_1
		self.gamma_2 = gamma_2
		self.t_rec = 0

	def recover(self, f, x_k, f_k, grad_hk, h, d_k, x_s, f_s):
		"""
		@param x_k: current iterate
		@param f_k: current function value
		@param grad_hk: current gradient estimate
		@param h: current finite difference interval
		@param gamma1: 
		@param gamma2: finite difference interval acceptance/rejection parameter
		@param d_k: search direction
		@param (x_s,f_s): best point on the stencil
		"""
		noise = ECNoise.estimate(self, f, x_k, direction = d_k) #reestimate noise
		grad, h_new = fd_gradient(f, x_k, noise)
		if h_new < self.gamma_1 * h or h_new > self.gamma_2 * h:
			x, fval, h = x_k, f_k, h_new
		else:
			step = h/norm(d_k)
			x_h = x_k + step * d_k
			f_h = f(x_h)
			self.t_rec += 1
			if LineSearch.is_armijo((f_k, grad_hk), f_h, step, d_k):
				x, fval = x_h, f_h
			elif f_h <= f_s and f_h <= f_k:
				x, fval = x_h, f_h
			elif f_k > f_s and f_h > f_s:
				x, fval = x_s, f_s
			else:
				x, fval = x_k, f_k
				noise_new = ECNoise.estimate(self, f, x_k)
				grad, h = fd_gradient(f, x_k, noise_new)
		return x, fval, h, t_rec