import numpy as np
from ECNoise import ECNoise
from FD import fd_gradient
from numpy.linalg import norm
from numpy.core.umath_tests import inner1d
from LSfunc_def import *
from CG import steihaug_cg


class DFO_LSTR(object):
	"""
	The class of finite difference method to solve least-square problem with predetermined noise level
	"""
	def __init__(self, noise, r,m, delta_max, delta_init, eta_1, eta_2, eta_3, ecn_params,k_accept,k_reject, output_level, rule, j_len, his_len = 5, mode = "fd",  tol = 1e-6):
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
		self.j_len = j_len
		self.k_accept = k_accept
		self.k_reject = k_reject

	def is_convergent(self, f_val, fval_history, norm_grad, tol, history_len):
		"""
		check convergence of the algorithm
		"""
		if len(fval_history) <= 1:
			return norm_grad <= tol
		fval_ma = np.mean(fval_history[-history_len:])
		return  norm_grad <= tol or self.delta <= 1e-5 or abs(f_val - fval_ma) <= tol * max(1, abs(fval_ma))


	def run(self, x):
		"""
		finite difference Jacobian + Least square
		@param: x starting iterate
		"""
		n = len(x) #dimension
		update_len = np.ceil(self.m * self.j_len / 100).astype(int)
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
		#print(J)
		#print(r_jacobian(x))

		if self.output_level >= 2:
			output_header = '%6s %23s %6s %9s %9s %9s' % \
			('iter', 'f',  '#func', '||grad_f||','cond','delta')
			print(output_header)
			print('%6i %23.16e %6i %9.2e %9.2e %9.2e' %
			  (k, f_val,  self.eval_counter, norm_grad_k, condition_number, self.delta))
		

		while not self.is_convergent(f_val, fval_history, norm_grad_k, self.tol,self.his_len) and k <=10000:
			cg_options = init_options_cg()
			p_k,cg_status,_ = steihaug_cg(H, grad, self.delta, cg_options)
			x_trial = x + p_k
			f_trial = self.f(x_trial)
			self.eval_counter += 1
			#compute rho_k
			pred_k = - np.dot(grad, p_k) - (1/2) * np.linalg.multi_dot([p_k, H, p_k])
			ared_k = f_val - f_trial
			rho_k = ared_k/pred_k

			#update radius
			norm_pk = norm(p_k)
			if rho_k < self.eta_2:
				self.delta = 0.5 * norm_pk #self.delta = 0.25 * self.delta
			elif rho_k > self.eta_3 and cg_status == 2:
				self.delta = min(2 * self.delta, self.delta_max)

			#update iterate
			if rho_k > self.eta_1:
				x = x_trial
				f_val = f_trial
				fval_history.append(f_val) #update function value history
				eval_history.append(self.eval_counter)
				#if len(fval_history) > self.his_len:
				#	fval_history = fval_history[1:]
				#Randomly update Jacobian at arbitrary rows
				if k % self.k_accept == 0: #or f_val <= 1e-3:
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
				else:
					if self.rule == "random":
						#batch = min(np.ceil(2.2**k).astype('int'), m)
						#batch = min(k+35, m)
						#idx = np.random.choice(m, batch, replace = False)
						idx = np.random.choice(self.m,update_len, replace = False)
						for i in idx:
							r_i = lambda x: self.r(x,self.m)[i]
							grad_i, _ = fd_gradient(r_i, x, self.noise, mode = self.mode)
							J[i] = grad_i
							if self.mode == "fd":
								self.eval_counter += n/self.m
							else:
								self.eval_counter += 2*n/self.m
					elif self.rule == "cyclic":
						idx_start = k % self.m 
						idx = np.array(range(update_len))
						idx = (idx + idx_start * update_len) % self.m
						for i in idx:
							r_i = lambda x: self.r(x,self.m)[i]
							grad_i, _ = fd_gradient(r_i, x, self.noise, mode = self.mode)
							J[i] = grad_i
							if self.mode == "fd":
								self.eval_counter += n/self.m
							else:
								self.eval_counter += 2*n/self.m					
				"""

				if k % 5 == 0 or f_val <= 1e-3:
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
				"""
				residual = self.r(x,self.m) #compute updated residual #TODO: avoid double computation
				grad = 2*np.dot(J.T, residual) #compute gradient of f
				H = 2*np.dot(J.T,J) # compute hessian of f
				condition_number = np.linalg.cond(H) #compute condition number of H
				norm_grad_k = np.max(np.abs(grad)) # compute norm of gradient of f
			
			else:
				if self.k_reject != 0 and k % self.k_reject == 0: #or f_val <= 1e-3:
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
					residual = self.r(x,self.m) #compute updated residual #TODO: avoid double computation
					grad = 2*np.dot(J.T, residual) #compute gradient of f
					H = 2*np.dot(J.T,J) # compute hessian of f
					condition_number = np.linalg.cond(H) #compute condition number of H
					norm_grad_k = np.max(np.abs(grad))
			
			k += 1
			if self.output_level >= 2:
				if k % 10 == 0:
					print(output_header)
				print('%6i %23.16e   %6i %9.2e %9.2e %9.2e' %
			  (k, f_val, self.eval_counter, norm_grad_k, condition_number, self.delta))

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
		return x, f_val, stats, fval_history, eval_history




def r(x):
	#return np.array([10.0 * (x[1] - x[0] ** 2), (1.0 - x[0])])
	return np.array([10.0 * (x[1] - x[0] ** 2) +1e-3 * np.random.normal(), (1.0 - x[0])+1e-3 * np.random.normal()])
	#return np.array([10.0 * (x[1] - x[0] ** 2) *(1+1e-3 * np.random.normal()), (1.0 - x[0])*(1+1e-3 * np.random.normal())])
	#return np.array([10.0 * (x[1] - x[0] ** 2) *(1+ 1e-3 * np.random.chisquare(5)), (1.0 - x[0]) *(1+ 1e-3 * np.random.chisquare(5))])


	#return np.array([(x[0]+1)**2 + 1e-3 * np.random.rand(), x[1]**2 +1e-3 * np.random.rand()    ])

def r_jacobian(x):
    return np.array([[-20.0*x[0], 10.0], [-1.0, 0.0]])







def init_options_cg():
    '''
   Option initialization

   Initialize algorithm options for CG with default values

   Output values:
   ==============

   options:
       This is a structure with field that correspond to algorithmic
       options of our method.  In particular:

       cg_max_iter:   
           Maximum number of iterations
       cg_tol:        
           Convergence tolerance for residual
       step_type:
           Different ways to calculate the search direction:
           'Steepest Descent'
           'Newton'
       cg_output_level:
           Amount of output printed
           0: No output
           1: Only summary information
           2: One line per iteration 
    '''
    
    options = {}
    options['cg_max_iter'] = 1e5
    options['cg_tol'] = 1e-6
    options['cg_output_level'] = 0
    
    return options


def r_noise(x,m):
	return fun16(x,m) + 1e-3 *np.random.normal()


if __name__ == "__main__":
	np.random.seed(123)
	n = 100
	x = 0.5 * np.ones(n).astype(float)
	#x[::2] = -1.2
	#x = np.array([0,10,20])
	#x = np.array([0.5,1.5,-1,0.01,0.02])
	#x = np.array([1.3,0.65,0.65,0.7,0.6,3,5,7,2,4.5,5.5])
	m = n
	dfogn = DFO_LSTR(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,10, 3, rule = "random", j_len = 50)

	dfogn.run(x)














