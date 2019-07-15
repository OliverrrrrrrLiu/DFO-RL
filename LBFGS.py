import numpy as np

class LBFGS(object):
	"""L-BFGS to compute search direction d_k = -H_k * grad_k
	@param x: starting point
	@param m: number of vector pairs (s_i, y_i) stored
	@param s: matrix with rows  s_k = x_k+1 - x_k 
	@param y: matrix with rows y_k = grad_k+1 - grad_k
	@param rho: 1/y_kT*x_k rho = np.sum(s*y, axis = 1)
	"""
	def __init__(self, grad_k, m, s, y, rho):
		self.grad_k = grad_k
		self.m = m
		self.s = s
		self.y = y
		self.rho = rho

	def calculate_direction(self):
		q = self.grad_k
		alpha = np.zeros(m)
		for i in range(self.m):
			alpha[i] = self.rho[i]*np.inner(self.s[i],q)
			q = q - alpha * self.y[i]
		"""	
		alpha = rho * np.inner(s, grad_k)
		q = grad_k - np.sum(alpha*y)
		"""
		gamma_k = np.inner(self.s[m-1],self.y[m-1])/np.inner(self.y[m-1],self.y[m-1])
		r = gamma_k * q
		for i in range(self.m-1,-1,-1):
			beta = self.rho[i]*np.inner(self.y[i],r)
			r = r + self.s[i]*(alpha[i] - beta)
		return -r

if __name__ == "__main__":
	grad_k = np.array([1,1,1])
	m = 3
	s = np.array([[1,2,3],[2,1,4],[0,2,1]])
	y = np.array([[0,1,1],[2,1,3],[0,0,3]])
	rho = np.sum(s*y,axis = 1)
d_k = LBFGS(grad_k,m,s,y,rho)
print(d_k.calculate_direction())
    





