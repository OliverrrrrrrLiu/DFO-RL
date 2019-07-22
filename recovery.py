import numpy as np
from ECNoise import ECNoise
from FD import fd_gradient
from numpy.linalg import norm
from linesearch import LineSearch

#TODO: separate grad estimation from h calculation
#TODO: function counter

class Recovery(object):
	def __init__(self, f, gamma_1, gamma_2, noise_f, ls):
		self.f = f
		self.gamma_1 = gamma_1
		self.gamma_2 = gamma_2
		self.t_rec = 0
		self.noise_f = noise_f
		self.ls = ls

	def recover(self, orig_pt, h, d_k, stencil_pt):
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
		x_k, f_k, grad_hk = orig_pt
		x_s, f_s = stencil_pt
		noise = self.noise_f.estimate(x_k, direction = d_k) #reestimate noise
		print("noise", noise, d_k)
		grad, h_new = fd_gradient(self.f, x_k, noise)
		if h_new < self.gamma_1 * h or h_new > self.gamma_2 * h:
			x, fval, h = x_k, f_k, h_new
		else:
			step = h / norm(d_k)
			x_h = x_k + step * d_k
			f_h = self.f(x_h)
			self.t_rec += 1
			if self.ls.is_armijo((f_k, grad_hk), f_h, step, d_k):
				x, fval = x_h, f_h
			elif f_h <= f_s and f_h <= f_k:
				x, fval = x_h, f_h
			elif f_k > f_s and f_h > f_s:
				x, fval = x_s, f_s
			else:
				x, fval = x_k, f_k
				noise_new = self.noise.estimate(x_k)
				grad, h = fd_gradient(f, x_k, noise_new)
		return (x, fval), h, noise_new