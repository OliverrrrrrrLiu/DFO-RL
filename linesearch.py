import math
import numpy as np
from numpy.linalg import norm
from FD import fd_gradient
from ECNoise import ECNoise

# Current TODO's:
#			number of function evaluations
#			modularity

class LineSearch(object):
	def __init__(self, f, c1, c2, max_iter):
		"""
		@param f:  noisy function f to be evaluated
		@param c1: Armijo coefficient
		@param c2: Wolfe coefficient
		@param max_iter: maximum number of trials to perform line search
		"""
		self.f = f
		self.c1 = c1
		self.c2 = c2
		self.max_iter = max_iter

	def is_armijo(self, orig_pt, fx_new, step, d, noise = 0.0):
		"""verify if the (relaxed) Armijo condition is satisfied 
		@param orig_pt: tuple (fx, grad_fx): starting point at which we examine the Armijo condition
		@param fx_new:  f(x + step * d): trial point function value
		@param step:    step size
		@param d:       search direction
		"""
		x, fx_orig, grad_fx_orig = orig_pt #extract current point information
		#print("f_new", fx_new, "f_old", fx_orig)
		return fx_new <= fx_orig + self.c1 * step * np.inner(grad_fx_orig, d) + 2 * noise #relaxed armijo-condition

	def is_wolfe(self, orig_pt, x_new, d, noise = 0.0, mode = "cd"):
		"""
		Verify if the wolfe condition is satisfied
		@param orig_pt: tuple (fx, grad_fx): starting point at which we examine the Wolfe condition
		@param x_new: trial point
		@param d: search direction
		"""
		x, fx_orig, grad_fx_orig = orig_pt #extract current point information
		#grad_fx_new, _ = fd_gradient(self.f, x_new, noise, mode=mode) #evaluate gradient at trial point
		#print("gTP_old",np.inner(grad_fx_orig, d), "gTP_new", np.inner(grad_fx_new, d))
		#return np.inner(grad_fx_new, d) >= self.c2 * np.inner(grad_fx_orig, d)
		dirdev_new, _ = fd_gradient(self.f, x_new, noise, mode=mode, direction = d) #direcctional derivative
		#grad_new, _ = fd_gradient(self.f, x_new, noise, mode=mode)
		return dirdev_new >= self.c2 * np.inner(grad_fx_orig, d) #directional derivative
		#return np.inner(grad_new, d) >= self.c2 * np.inner(grad_fx_orig, d)
	# def search(self, orig_pt, d, noise = 0.0):
	# 	"""
	# 	Search for stepsize that satisfies the armijo-wolfe conditions
	# 	"""
	# 	ls_counter = 0 #initialize iteration counter
	# 	step = 1 #initialize stepsize
	# 	xk, f_xk, grad_f_xk = orig_pt #extract current point information
	# 	while ls_counter < self.max_iter: #if max number of iterations has not been exceeded
	# 		relax = ls_counter >= 1 #armijo condition is relaxed if initial stepsize does not satisfy the armijo-wolfe conditions
	# 		x_trial = xk + step * d #trial point
	# 		f_trial = self.f(x_trial) #trial point function value
	# 		if ls_counter != self.max_iter - 1: #if not at max number of iterations, check both armijo and wolfe condition
	# 			condition_satisfied = self.is_armijo(orig_pt, f_trial, step, d, relax * noise) and \
	# 								  self.is_wolfe(orig_pt, x_trial, d, noise)
	# 		else: #only check the relaxed armijo wolfe condition
	# 			condition_satisfied = self.is_armijo(orig_pt, f_trial, step, d, relax * noise)
	# 		if condition_satisfied: #if a stepsize is found, return the corresponding trial point, value and stepsize
	# 			return (x_trial, f_trial), step, True
	# 		ls_counter += 1 #increase counter number
	# 		step /= 2 #half stepsize
	# 	return (x_trial, f_trial), step, False #linesearch failue: no stepsize satisfied the A-W conditions


	# def search(self, orig_pt, d, noise = 0.0, mode = "cd"):
	# 	l = 0
	# 	u = np.inf
	# 	alpha = 1
	# 	ls_counter = 0
	# 	xk, f_xk, grad_xk = orig_pt
	# 	phid = np.inner(grad_xk, d)
	# 	while ls_counter < self.max_iter:
	# 		relax = ls_counter >= 1
	# 		x_trial = xk + alpha * d
	# 		f_trial = self.f(x_trial)
	# 		grad_trial,_ = fd_gradient(self.f, x_trial, noise, mode = mode)
	# 		if f_trial > f_xk + self.c1 * alpha * phid + 2 * noise * relax:
	# 			u = alpha
	# 		elif np.inner(grad_trial,d) < self.c2 * phid:
	# 			l = alpha
	# 		else:
	# 			return (x_trial, f_trial), alpha, True 
	# 		if u < np.inf:
	# 			alpha = (l+u)/2
	# 		else:
	# 			alpha = 2 * l
	# 		ls_counter += 1
	# 	return (x_trial, f_trial), alpha, False

###TODO: check ls_counter

	def search(self, orig_pt, d, noise = 0.0, mode = "fd"):
		l = 0 #lower bound
		u = np.inf #upper boundí
		alpha = 1 #stepsize 
		ls_counter = 0 #line search counter
		xk, f_xk, grad_xk = orig_pt
		dim = len(xk)
		eval_counter = 0
		while ls_counter < self.max_iter:
			relax = ls_counter >= 1
			x_trial = xk + alpha * d
			f_trial = self.f(x_trial)
			eval_counter += 1
			armijo_satisfied = self.is_armijo(orig_pt, f_trial, step = alpha, d = d, noise = noise)
			wolfe_satisfied = self.is_wolfe(orig_pt, x_trial, d, noise= 0, mode=mode)
			if mode == "fd":
				eval_counter += 1#*dim
			else:
				eval_counter += 2 #* dim
			#if ls_counter != self.max_iter - 1:	
			if not armijo_satisfied:
				u = alpha
			elif not wolfe_satisfied:
				l = alpha
			else:
				#print("ls", ls_counter)
				return (x_trial, f_trial), alpha, True, eval_counter
			if u < np.inf:
				alpha = (l + u)/2
			else:
				alpha = 2 * l 
			# else:
			# 	if armijo_satisfied:
			# 		return(x_trial, f_trial), alpha, True
			ls_counter+= 1
		return (x_trial, f_trial), alpha, False, eval_counter

