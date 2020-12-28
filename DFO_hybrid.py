import numpy as np

class DFO_hybrid(object):
	"""
	the class of solving non-linear least square problems with a hybrid approach that combines interpolation and finite-difference with a trust-region framework.
	"""
	def __init__(self, noise, r,m, delta_max, delta_init, eta_1, eta_2, eta_3, ecn_params,output_level,his_len = 5, mode = "fd",  tol = 1e-6):








	def run(self, x):
		n = len(x) #dimension
		residual = self.r(x,self.m) #calculate the residual
		self.eval_counter += 1 #update evaluation counter
		print("Running from starting point:", x)
		f_val = self.f(x) #evaluate the function at current iterate x
		J = np.empty([self.m,n]) #Jacobian matrix
		for i in range(self.m):
			r_i = lambda x: self.r(x,self.m)[i] #i-th component of residual
			grad_i, h_i = fd_gradient(r_i, x, self.noise, mode = self.mode) #estimate gradient and difference interval
			#if self.mode == "fd":
			#	self.eval_counter += n
			#else:
			#	self.eval_counter += 2 * n 
			J[i] = grad_i
		if self.mode == "fd": #update function evaluation counter
			self.eval_counter += n
		else:
			self.eval_counter += 2 * n
		k, fval_history = 0, [f_val] #update function value history
		eval_history = [self.eval_counter]
		alpha = 0
		grad = 2*np.dot(J.T, residual) #update gradient of f
		H = 2*np.dot(J.T,J) #update hessian of f
		condition_number = np.linalg.cond(H) #compute condition number of hessian
		norm_grad_k = np.max(np.abs(grad)) #compute norm of gradient of f
		

		##Build interpolation model

				