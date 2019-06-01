def linesearch(f, x_k, f_k, grad_hk, d_k, a_max):
# The linesearch function aims to find a steplength alpha_k that satisfies the Armijo-Wolfe conditions
"""
input arguments:
=================================================================================
	f: callable function 'f(x)'
	x_k: current iterate
	f_k: function value of f at point x_k
	grad_hk: current finite difference gradient estimation based on interval grad_h
	d_k: LBFGS search direction
	a_max: maximum number of linesearch iterations

output values:
==================================================================================
	x_sol: new iterates
	f_sol: new function value
	alpha_k: steplength
	t_ls = number of function evaluations
	LS_flag: return code for indicating reason for terminzation
		0: a stepleangth that satisfies the Armijo-Wolfe conditions is found
		1: length search failure; recovery function is needed to determine the cause
"""

def phi(alpha):
	t_ls += 1
	return f(x_k + alpha*d_k)
def phid(alpha):
	t_ls += 1
