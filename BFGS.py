import numpy as np 


def BFGS(grad, H, s, y):
	rho = np.inner(s,y)
	tmp = np.eye(len(grad)) - rho* np.outer(s, y)
	H = multi_dot([tmp, H, tmp]) + rho * np.outer(s, s)
	return -np.matmul(H, grad)