import numpy as np
def recovery(x_k, f_k, grad_hk, h, gamma_1, gamma_2, d_k, x_s, f_s):
	t_rec = 0 #functions evaluation counter
	#TODO: Compute new noise estimate and new h
	#Update function evaluation counter

	if h_new < gamma_1 * h or h_new > gamma_2*h:
		h = h_new
		x_sol = x_k
		f_sol = f_k
	else:
		x_h = x_k + h* d_k/np.linalg.norm(d_k)
		f_h = f(x_h)
		t_rec = t_rec + 1
		if f(x_h) <= f_k + c1 * h * np.inner(grad_hk,d_k):
			x_sol = x_h
			f_sol = f_h
		else:
			if f_h <= f_s and f_h <= f_k:
				x_sol = x_h
				f_sol = f_h
			elif f_k > f_s and f_h > f_s:
				x_sol = x_s
				f_sol = f_s
			else:
				x_sol = x_k
				f_sol = f_k
				#TODO: compute new noise estimate along a random direction and new h
				h = h_new
				t_rec = t_rec + f_ecn
	return x_sol,f_sol,h,t_rec