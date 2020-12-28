import numpy as np
from ECNoise import ECNoise
from FD import fd_gradient
from numpy.linalg import norm
from numpy.core.umath_tests import inner1d
from LSfunc_def import *
from CG import steihaug_cg
from linesearch import LineSearch

#Gauss-Newton + Coordinate descent 

class DFO_GNCD(object):
	"""
	The class of finite difference method to solve least-square problem with predetermined noise level
	"""
	def __init__(self, noise, r,m, delta_max, delta_init, eta_1, eta_2, eta_3, ecn_params, output_level, rule,  his_len = 5, mode = "fd",  tol = 1e-6):
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
		self.noise_f = ECNoise(self.f, *ecn_params) #ECNoise - noise level estimation
		self.output_level = output_level #print level 
		self.mode = mode # finite difference mode: FD/CD
		self.tol = tol #convergence tolerance
		self.his_len = his_len #convergence moving average window length
		self.noise = noise #noise level
		self.delta_max = delta_max
		self.delta = delta_init
		self.eta_1 = eta_1
		self.eta_2 = eta_2
		self.eta_3 = eta_3
		self.rule = rule


	def is_convergent(self, f_val, fval_history, grad, tol, history_len):
		"""
		check convergence of the algorithm
		"""
		if len(fval_history) <= 1:
			#return np.abs(grad) <= tol
			return f_val < tol
		fval_ma = np.mean(fval_history[-history_len:])
		return  f_val < tol or abs(f_val - fval_ma) <= tol * max(1, abs(fval_ma))

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
		idx = np.random.choice(n) #randomly pick one coordinate direction
		J = np.empty(self.m) #Evaluate Jacobian at idx-column
		for i in range(self.m):
			r_i = lambda x: self.r(x,self.m)[i] #i-th component of residual
			grad_i, h_i = fd_gradient(r_i, x, self.noise, mode = self.mode) #estimate gradient and difference interval
			#if self.mode == "fd":
			#	self.eval_counter += n
			#else:
			#	self.eval_counter += 2 * n 
			J[i] = grad_i[idx]
		if self.mode == "fd": #update function evaluation counter
			self.eval_counter += 1
		else:
			self.eval_counter += 2 
		k, fval_history = 0, [f_val] #update function value history
		eval_history = [self.eval_counter]
		alpha = 0
		grad = 2*np.dot(J.T, residual) #update gradient of f along coodinate direction, g_i
		H = 2*np.dot(J.T,J) #update hessian of f along coordinate direction, H_ii
		#condition_number = np.linalg.cond(H) #compute condition number of hessian
		#norm_grad_k = np.max(np.abs(grad)) #compute norm of gradient of f
		#print(J)
		#print(r_jacobian(x))

		if self.output_level >= 2:
			output_header = '%6s %23s %6s %9s %9s ' % \
			('iter', 'f',  '#func', 'grad_i','H_ii')
			print(output_header)
			print('%6i %23.16e %6i %9.2e %9.2e ' %
			  (k, f_val,  self.eval_counter, grad, H))

		# initialize iterator, sample without replacement
		idx_list = list(np.random.choice(n, n, replace=False))
		while not self.is_convergent(f_val, fval_history, grad, self.tol,self.his_len) and k <=200:
			#cg_options = init_options_cg()
			#p_k,cg_status,_ = steihaug_cg(H, grad, self.delta, cg_options)
			step = - grad/H
			p_k = np.zeros(n)
			p_k[idx] = step 
			#grad_ls = np.zeros(n)
			#grad_ls[idx] = grad
			#new_pt, step_k, flag = self.search((x, f_val, grad_ls), p_k, noise=self.noise, mode=self.mode)
			#self.eval_counter += ls_counter
			x = x +  p_k
			f_val = self.f(x)
			self.eval_counter += 1
			#x, f_val = new_pt


			if self.rule == "random":
				idx_list = list(np.random.choice(n, n, replace=False)) if len(idx_list) == 0 else idx_list
				idx = idx_list.pop()
				#idx = np.random.choice(n)
				#idx = k % n
				for i in range(self.m):
					r_i = lambda x: self.r(x,self.m)[i] #i-th component of residual
					grad_i, _ = fd_gradient(r_i, x, self.noise, mode = self.mode)
					J[i] = grad_i[idx]
				if self.mode == "fd":
					self.eval_counter += 1
				else:
					self.eval_counter += 2 
			fval_history.append(f_val) #update function value history
			eval_history.append(self.eval_counter)	
			residual = self.r(x,self.m) #compute updated residual #TODO: avoid double computation
			grad = 2*np.dot(J.T, residual) #compute gradient of f along coordinate direction
			H = 2*np.dot(J.T,J) # compute hessian of f along coordinate direction
			
			k += 1
			if self.output_level >= 2:
				if k % 10 == 0:
					print(output_header)
				print('%6i %23.16e   %6i %9.2e %9.2e' %
			  (k, f_val, self.eval_counter, grad, H))

		stats = {}
		stats['num_iter'] = k
		stats['grad'] = grad
		stats['num_func_it'] = self.eval_counter

		#Final output message
		if self.output_level >= 1:
			print('')
			print('Final objective.................: %g' % f_val)
			print('grad_i at final point.........: %g' % grad)
			print('Number of iterations............: %d' % k)
			print('Number of function evaluations..: %d' % self.eval_counter)
			print('')


		# Return output arguments
		print("Returned point:", x)
		return x, f_val, stats, fval_history, eval_history




def r(x):
	#return np.array([10.0 * (x[1] - x[0] ** 2), (1.0 - x[0])])
	return np.array([10.0 * (x[1] - x[0] ** 2) +1e-3 * np.random.normal(), (1.0 - x[0])+1e-3 * np.random.normal()])
	#return np.array([10.0 * (x[1] - x[0] ** 2) *(1+1e-3 * np.random.normal()), (1.0 - x[0])*(1+1e-3 * np.random.normal())])
	#return np.array([10.0 * (x[1] - x[0] ** 2) *(1+ 1e-3 * np.random.chisquare(5)), (1.0 - x[0]) *(1+ 1e-3 * np.random.chisquare(5))])


	#return np.array([(x[0]+1)**2 + 1e-3 * np.random.rand(), x[1]**2 +1e-3 * np.random.rand()    ])

def r_jacobian(x):
    return np.array([[-20.0*x[0], 10.0], [-1.0, 0.0]])






def r_noise(x,m):
	return fun15(x,m) + 1e-3 *np.random.normal()


if __name__ == "__main__":
	np.random.seed(123)
	n = 6
	#x =  np.ones(n).astype(float)
	#x[::2] = -1.2
	#x = np.array([0,10,20])
	#x = np.array([0.5,1.5,-1,0.01,0.02])
	#x = np.array([1.3,0.65,0.65,0.7,0.6,3,5,7,2,4.5,5.5])
	m = n
	x = initial_points(n,15)
	dfogncd = DFO_GNCD(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),3, rule = "random")

	dfogncd.run(x)














