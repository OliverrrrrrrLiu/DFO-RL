import numpy as np
from ECNoise import ECNoise
from FD import fd_gradient
from LBFGS import LBFGS
from linesearch import LineSearch
from numpy.linalg import norm
from Recovery_noise import Recovery
from numpy.core.umath_tests import inner1d
from LSfunc_def import *


class DFOGN(object):
	"""
	The class of finite difference method to solve least-square problem with predetermined noise level
	"""
	def __init__(self, noise, r,m, ecn_params, ls_params, rec_params, regularization, output_level, his_len = 5, mode = "fd", tol = 1e-6):
		"""
		@param r: residual function(vector function)
		@param f: ||r(x)||^2
		@param noise: noise level 
		"""
		self.r = r #residual function 
		#self.f = lambda r: norm(r)**2
		self.m = m #dimension of residual 
		self.f = lambda x: norm(r(x,self.m))**2 #objective function
		self.eval_counter = 0 #function evaluation counter
		self.ls_counter = 0 #linesearch function evaluation counter
		self.rec_counter = 0 #recover function evaluation counter
		self.noise_f = ECNoise(self.f, *ecn_params) #ECNoise - noise level estimation
		self.ls = LineSearch(self.f, *ls_params) #line search 
		self.rec = Recovery(self.f, *rec_params, noise, self.ls) #recovery 
		self.output_level = output_level #print level 
		self.mode = mode # finite difference mode: FD/CD
		self.tol = tol #convergence tolerance
		self.his_len = his_len #convergence moving average window length
		self.noise = noise #noise level
		self.reg = regularization #regularization parameter for hessian

	def get_stencil_pt(self, x, h):
		"""
		Compute and store the best point on the stencil

		@param x: current point
		@param h: current finite-difference interval 

		@return: (x_s, f_s): best point on the stencil S = {x_i: x_i = x + h * e_i, i = 1,...,n} and its function value
		"""
		stencil_pts = []
		dim = len(x)
		for i in range(dim): #evaluate the stencil points
			basis = np.zeros(len(x))
			basis[i] = h
			stencil = x + basis
			val = self.f(stencil)
			stencil_pts.append((stencil, val))
		return min(stencil_pts, key=lambda t:t[1])

	def is_convergent(self, f_val, fval_history, norm_grad, tol, history_len = 5):
		"""
		check convergence of the algorithm
		"""
		if len(fval_history) <= 1:
			return norm_grad <= tol
		fval_ma = np.mean(fval_history)
		return abs(f_val - fval_ma) <= tol * max(1, abs(fval_ma)) or norm_grad <= tol #or abs(f_val) <= self.noise


	def run(self, x):
		"""
		finite difference Jacobian + Least square
		@param: x starting iterate
		"""
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
		alpha = 0
		grad = 2*np.dot(J.T, residual) #update gradient of f
		H = 2*np.dot(J.T,J) + self.reg * np.eye(n) #update hessian of f
		condition_number = np.linalg.cond(H) #compute condition number of hessian
		norm_grad_k = np.max(np.abs(grad)) #compute norm of gradient of f

		#print(J)
		#print(r_jacobian(x))

		if self.output_level >= 2:
			output_header = '%6s %23s %9s %6s %9s %9s' % \
			('iter', 'f',  'alpha', '#func', '||grad_f||','cond')
			print(output_header)
			print('%6i %23.16e  %9.2e %6i %9.2e %9.2e' %
			  (k, f_val, alpha, self.eval_counter, norm_grad_k, condition_number))
		

		while not self.is_convergent(f_val, fval_history, norm_grad_k, self.tol) and k <=10000:
			d = -np.dot(np.linalg.pinv(H), grad) #compute search direction
			new_pt, step, flag, ls_counter = self.ls.search((x, f_val, grad), d, noise=self.noise, mode=self.mode) #conduct linesearch to find the next iterate
			self.eval_counter += ls_counter #update function evaluation counter
			#x, f_val = new_pt #update iterate
			x_trial, f_trial = new_pt
			#ared = f_val - f_trial
			#pred = f_val - norm(residual + np.dot(J,x_trial - x))
			#ratio = ared/pred
			#if ratio > 0.75:
			#	self.reg = self.reg * 10
			#	print('here')
			#else:
			#	self.reg = self.reg/10
			#	print('red')
			x = x_trial
			f_val = f_trial 
			fval_history.append(f_val) #update function value history
			if len(fval_history) > self.his_len:
				fval_history = fval_history[1:]
			#if k % 2 == 0 or f_val <= 1e-3:
			for i in range(self.m):
				r_i = lambda x: self.r(x,self.m)[i] #i-th component of residual
				grad_i, _ = fd_gradient(r_i, x, self.noise, mode = self.mode) #compute gradient of r_i
				#if self.mode == "fd":
				#	self.eval_counter += n
				#else:
				#	self.eval_counter += 2 * n 
				J[i] = grad_i
			if self.mode == "fd": # update function evaluation counter
				self.eval_counter += n
			else:
				self.eval_counter += 2 * n
			#print(J)
			#print(r_jacobian(x))
			residual = self.r(x,self.m) #compute updated residual #TODO: avoid double computation
			grad = 2*np.dot(J.T, residual) #compute gradient of f
			H = 2*np.dot(J.T,J) + self.reg * np.eye(n) # compute hessian of f
			condition_number = np.linalg.cond(H) #compute condition number of H
			norm_grad_k = np.max(np.abs(grad)) # compute norm of gradient of f
			k += 1
			if self.output_level >= 2:
				if k % 10 == 0:
					print(output_header)
				print('%6i %23.16e  %9.2e %6i %9.2e %9.2e' %
			  (k, f_val, step, self.eval_counter, norm_grad_k, condition_number))

		stats = {}
		stats['num_iter'] = k
		stats['norm_grad'] = norm_grad_k
		stats['num_func_it'] = self.eval_counter

		#Final output message
		if self.output_level >= 1:
			print('')
			print('Final objective.................: %g' % f_val)
			print('||grad|| at final point.........: %g' % norm_grad_k)
			print('Number of iterations............: %d' % k)
			print('Number of function evaluations..: %d' % self.eval_counter)
			print('')


		# Return output arguments
		print("Returned point:", x)
		return x, f_val, stats




def r(x):
	#return np.array([10.0 * (x[1] - x[0] ** 2), (1.0 - x[0])])
	return np.array([10.0 * (x[1] - x[0] ** 2) +1e-3 * np.random.normal(), (1.0 - x[0])+1e-3 * np.random.normal()])
	#return np.array([10.0 * (x[1] - x[0] ** 2) *(1+1e-3 * np.random.normal()), (1.0 - x[0])*(1+1e-3 * np.random.normal())])
	#return np.array([10.0 * (x[1] - x[0] ** 2) *(1+ 1e-3 * np.random.chisquare(5)), (1.0 - x[0]) *(1+ 1e-3 * np.random.chisquare(5))])


	#return np.array([(x[0]+1)**2 + 1e-3 * np.random.rand(), x[1]**2 +1e-3 * np.random.rand()    ])

def r_jacobian(x):
    return np.array([[-20.0*x[0], 10.0], [-1.0, 0.0]])



def r_noise(x,m):
	return fun21(x,m) + 1e-3 *np.random.normal()



if __name__ == "__main__":
	np.random.seed(123)
	n = 100
	x = np.ones(n).astype(float)
	x[::2] = -1.2
	#x = np.array([1.3,0.65,0.65,0.7,0.6,3,5,7,2,4.5,5.5])
	m = n
	dfogn = DFOGN(1e-3,r_noise,m, (1e-8, 7, 100), (1e-4, 0.9, 20), (0.9, 1.1), 0, 3)

	dfogn.run(x)














