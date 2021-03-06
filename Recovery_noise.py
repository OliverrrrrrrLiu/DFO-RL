import numpy as np
from ECNoise import ECNoise
from FD import fd_gradient
from numpy.linalg import norm
from linesearch import LineSearch

###This code assumes that the noise level of the function is known!!

class Recovery(object):
	"""Recovery scheme to determine the cause of the failure of linesearch and to take corrective action"""
	def __init__(self, f, gamma_1, gamma_2, noise, ls):
		"""initilization
		@param f: noisy function f
		@param gamma_i: finite difference interval acceptance/rejection parameters

		"""
		self.f = f
		self.gamma_1 = gamma_1
		self.gamma_2 = gamma_2
		self.t_rec = 0
		self.noise = noise
		self.ls = ls

	def recover(self, orig_pt, h, d_k, stencil_pt):
		"""
		Corrective algorithm to find the next iterate when linesearch fails 

		@param orig_pt: tuple (fx, grad_fx) current iterate
		@param h: current finite difference interval
		@param d_k: current search direction
		@param stencil_pt: (x_s,f_s) best point on the stencil

		@return (x, fval), h, noise: next iterate, new finite difference interval, new noise estimate
		"""
		rec_counter = 0 
		print("RECOVER")
		x_k, f_k, grad_hk = orig_pt #extract current iterate information
		x_s, f_s = stencil_pt #extract current stencil point information
		#print("noise", noise, d_k)
		grad, h_new = fd_gradient(self.f, x_k, self.noise) #recalculate finite difference interval and new finite difference gradient estimator
		dim = len(x_k)
		rec_counter += dim
		#print("h_new:", h_new, h)
		if h_new < self.gamma_1 * h or h_new > self.gamma_2 * h: #if the current differencing interval differs significantly from new estimate h
			x, fval, h = x_k, f_k, h_new #return new differencing interval without changing the current iterate
		else: 
			step = h / norm(d_k) #normalize search direction
			x_h = x_k + step * d_k #compute small perturbation of x_k, of size h
			f_h = self.f(x_h) #evaluate function value at x_h
			self.t_rec += 1 #increase iteration counter
			if self.ls.is_armijo((x_k, f_k, grad_hk), f_h, step, d_k): #if the perturbed point satisfies the armijo condition
				x, fval = x_h, f_h #return the perturbed points and keep old finite difference interval
			#if armijo condition not satisfied
			elif f_h <= f_s and f_h <= f_k: #if perturbed point is better than current point and stencil point
				x, fval = x_h, f_h #accept the perturbed point
			elif f_k > f_s and f_h > f_s: # if the stencil point is smaller than perturbed and current point
				x, fval = x_s, f_s #accept stencil point
			else:
				x, fval = x_k, f_k
				grad, h = fd_gradient(self.f, x_k, self.noise) #reestimate finite difference interval
				rec_counter += dim 
		return (x, fval), h, self.noise, rec_counter #return new interval and noise level without changing curren iterate