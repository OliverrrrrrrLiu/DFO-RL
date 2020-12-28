import numpy as np

"""
This file defines the functions of 22 nonlinear least squares problems
Input:
	x: input array of length n
	m: dimension of residual n<= m
Output: 
	r: vector of residuals

"""

def fun1(x,m):
	# linear function - full rank
	n = np.size(x)
	temp = 2 * np.sum(x)/m + 1
	res = -temp * np.ones(m)
	res[:n] += x
	return res 

def fun2(x,m):
	# linear function - rank 1
	n = np.size(x)
	temp = np.sum((np.arange(n) + 1)*x)
	res = (np.arange(m) + 1) * temp - 1
	return res

def fun3(x,m):
	#linear function - rank 1 with zero columns and rows
	n = np.size(x)
	temp = np.arange(n)
	temp = np.sum(temp[2:] * x[1:-1])
	res = np.zeros(m)
	res[:m-1] = np.arange(m-1) * temp - 1
	res[-1] = -1
	return res

def fun4(x,m = 2):
	#rosenbrock function
	# m = 2, n = 2
	if m != 2:
		raise ValueError('This function only allows m = 2')
	return np.array([10 * (x[1]- x[0]**2), 1 - x[0]])


def fun5(x,m = 3):
	#Helical valley function
	# m = 3, n = 3
	if m != 3:
		raise ValueError('This function only allows m = 3')
	if x[0] > 0:
		theta = np.arctan(x[1]/x[0]) / (2 * np.pi )
	elif x[0] < 0:
		theta = np.arctan(x[1]/x[0]) / (2 * np.pi ) + 0.5
	else:
		theta = 0.25
	r = np.sqrt(x[0]**2 + x[1] ** 2)
	return np.array([10 * (x[2] - 10 * theta), 10 * (r-1), x[2]])

def fun6(x, m = 4):
	#Powell singular function 
	# m = 4, n = 4
	if m != 4:
		raise ValueError('This function only allows m = 4')
	return np.array([x[0] + 10 * x[1], np.sqrt(5) * (x[2] - x[3]), (x[1]- 2*x[2])**2, np.sqrt(10) *(x[0]-x[3])**2 ])

def fun7(x, m = 2):
	#Freudenstein and Roth function
	# m = 2, n = 2
	if m != 2:
		raise ValueError('This function only allows m = 2')
	return np.array([-13.0 + x[0] + ((5 - x[1])*x[1] - 2.0) * x[1], -29.0 + x[0] + ((1 + x[1]) * x[1] - 14.0) * x[1]])
		

def fun8(x,m = 15):
	#Bard function
	# m = 15, n = 3
	if m != 15:
		raise ValueError('This function only allows m = 15')
	y = np.array([0.14,0.18,0.22,0.25,0.29,0.32,0.35,0.39,0.37,0.58,0.73,0.96,1.34,2.10,4.39])
	u = np.arange(m) + 1
	v = m - np.arange(m)
	w = np.minimum(u,v)
	res = y - (x[0] + u/(x[1] * v + x[2] * w))
	return res

def fun9(x,m = 11):
	#Kowalik and Osborne function
	# m = 11, n = 4
	if m != 11:
		raise ValueError('This function only allows m = 11')
	y = np.array([0.1957,0.1947,0.1735,0.1600,0.0844,0.0627,0.0456,0.0342,0.0323,0.0235,0.0246])
	u = np.array([4.0,2.0,1.0,0.5,0.25,0.1670,0.1250,0.1,0.0833,0.0714,0.0625])
	res = y - (x[0] * (u**2 + u * x[1])) / (u**2 + u * x[2] + x[3])
	return res

def fun10(x, m = 16):
	#Meyer function
	# m = 16, n = 3
	if m!= 16:
		raise ValueError('This function only allows m = 16')
	y = np.array([34780,28610,23650,19630,16370,13720,11540,9744,8261,7030,6005,5147,4427,3820,3307,2872])
	t = 45 + (np.arange(m)+1) * 5
	res = x[0] * np.exp(x[1]/(t + x[2])) - y
	return res


def fun11(x, m = 31):
	#Watson function
	# m = 31, 2 <= n <= 31
	n = len(x)
	assert 2 <= n <= 31
	assert m == 31

	ts = (np.arange(29) + 1) / 29
	Js = np.arange(n)

	res = [np.sum(np.arange(n)[1:] * x[1:] * (ts[i] ** Js[:-1])) - (np.sum(x * (ts[i] ** Js))) ** 2 - 1 for i in range(29)] 
	res += [x[0], x[1] - x[0] ** 2 - 1]
	return np.array(res)

def fun12(x, m):
	#box 3-dimensional function
	#m >= n, n = 3
	n = len(x)
	assert n == 3
	t = (np.arange(m) + 1) * 0.1
	res = np.exp(- t * x[0]) - np.exp(- t * x[1]) - x[2] * (np.exp(-t) - np.exp(-10 * t))
	return res

def fun13(x,m):
	#Jennrich and Sampson function
	#m >=n, n = 2
	n = len(x)
	assert n == 2

	temp = np.arange(m) + 1
	res = 2 + 2 * temp - (np.exp(temp * x[0]) + np.exp(temp * x[1]))
	return res

def fun14(x,m):
	#Brown and Dennis function
	# m>= n, n = 4
	n = len(x)
	assert n == 4
	t = (np.arange(m) + 1)/5
	res = (x[0] + t * x[1] - np.exp(t))**2 + (x[2] + x[3] * np.sin(t) - np.cos(t))**2
	return res

def fun15(x,m):
	#Chebyquad function
	#m,n
	n = len(x)
	res = np.zeros(m)
	for j in range(n):
		u = 1
		v = 2 * x[j] - 1
		t = 2 * v
		for i in range(m):
			res[i] = res[i] + v
			th = t * v - u
			u = v
			v = th
	iev = -1
	for i in range(m):
		res[i] = res[i]/n
		if (iev > 0):
			res[i] = res[i] + 1/((i+1)**2 - 1)
		iev = - iev
	return res

def fun16(x,m):
	#Brown almost-linear function
	# m = n
	#solution: x = (1,1,....,1)
	n = len(x)
	assert m == n
	
	s = sum(x)
	res = x + s - (n+1)
	res[-1] = np.prod(x) - 1
	return res

def fun17(x,m):
	#Osborne 1 function
	# m = 33, n = 5
	n = len(x)
	assert m == 33
	assert n == 5

	y = np.array([0.844,0.908,0.932,0.936,0.925,0.908,0.881,0.850,0.818,0.784,0.751,\
		0.718,0.685,0.658,0.628,0.603,0.580,0.558,0.538,0.522,0.506,0.490,\
		0.478,0.467,0.457,0.448,0.438,0.431,0.424,0.420,0.414,0.411,0.406])
	t = 10 * np.arange(m)
	res = y - (x[0] + x[1] * np.exp(- t * x[3]) + x[2] * np.exp(-t * x[4]))
	return res

def fun18(x,m):
	#Osborne 2 function
	#m = 65, n = 11
	n = len(x)
	assert m == 65
	assert n == 11
	y = np.array([1.366,1.191,1.112,1.013,0.991,0.885,0.831,0.847,0.786,0.725,0.746,\
		0.679,0.608,0.655,0.616,0.606,0.602,0.626,0.651,0.724,0.649,0.649,\
		0.694,0.644,0.624,0.661,0.612,0.558,0.533,0.495,0.500,0.423,0.395,\
		0.375,0.372,0.391,0.396,0.405,0.428,0.429,0.523,0.562,0.607,0.653,\
		0.672,0.708,0.633,0.668,0.645,0.632,0.591,0.559,0.597,0.625,0.739,\
		0.710,0.729,0.720,0.636,0.581,0.428,0.292,0.162,0.098,0.054])
	t = np.arange(m) * 0.1
	res = y - (x[0] * np.exp(-t * x[4]) + x[1] * np.exp(- (t - x[8])**2 * x[5]) + x[2]*np.exp(- (t-x[9])**2 * x[6]) + x[3] * np.exp(- (t - x[10])**2 *x[7]))
	return res

def fun19(x,m):
	#Bdqrtic "banded quartic model"
	# n >= 5, m = (n-4) * 2
	n = len(x)
	assert n >= 5
	assert m == (n-4)* 2
	#m = (n-4) * 2
	res1 = list(- 4 * x[:n-4] + 3)
	res2 = list(x[:n-4] **2 + 2 * x[1:n-3] **2 + 3 * x[2:n-2] **2 + 4 * x[3:n-1] **2 + 5 * x[-1]**2)
	res = res1 + res2
	return np.array(res)


def fun20(x,m):
	#Cube
	#n, m= n
	n = len(x)
	assert m == n

	res = np.zeros(m)
	res[0] = x[0] - 1
	res[1:] = 10 * (x[1:] - x[0:n-1]**3)
	return res


def fun21(x,m):
	#Mancino
	#n >=2; m = n
	n = len(x)
	assert n >= 2
	assert m == n
	res = np.zeros(m)
	for i in range(n):
		ss = 0
		for j in range(n):
			v2 = np.sqrt(x[i]**2 + (i+1)/(j+1))
			ss += v2 * ((np.sin(np.log(v2)))**5 + (np.cos(np.log(v2)))**5)
		res[i] = 1400 * x[i] + (i - 49)**3 + ss
	return res

def fun22(x,m):
	#Heart8ls
	#m = n  =8
	n = len(x)
	assert n == 8
	assert m == n
	res = np.zeros(m)
	res[0] = x[0] + x[1] + 0.69
	res[1] = x[2] + x[3] + 0.044 
	res[2] = x[4]*x[0] + x[5]*x[1] - x[6]* x[2] - x[7]*x[3] + 1.57
	res[3] = x[6] * x[0] + x[7] * x[1] + x[4] * x[2] + x[5]*x[3] + 1.31
	res[4] = x[0] * (x[4]**2 - x[6]**2) - 2.0 * x[2] * x[4] * x[6] + x[1] * (x[5]**2 - x[7]**2) - 2.0 * x[3] * x[5]*x[7] + 2.65
	res[5] = x[2] * (x[4]**2 - x[6]**2) + 2.0*x[0]*x[4]*x[6]+x[3]*(x[5]**2-x[7]**2)+2.0*x[1]*x[5]*x[7] - 2.0
	res[6] = x[0]*x[4]*(x[4]**2-3.0*x[6]**2) + x[2]*x[6]*(x[6]**2-3.0*x[4]**2)+x[1]*x[5]*(x[5]**2 - 3.0*x[7]**2)+x[3]*x[7]*(x[7]**2 - 3.0*x[5]**2) + 12.6
	res[7] = x[2]*x[4]*(x[4]**2 - 3.0*x[6]**2)-x[0]*x[6]*(x[6]**2-3.0*x[4]**2) + x[3]*x[5]*(x[5]**2-3.0*x[7]**2)-x[1]*x[7]*(x[7]**2 - 3.0*x[5]**2) - 9.48
	return res

def fun23(x,m):
	#Extended Rosenbrock function
	#n, m = n
	n = len(x)
	assert n%2 == 0
	#assert m == n
	x_odd = x[::2] #odd rows of x
	x_even = x[1::2] #even rows of x
	res = np.zeros(m)
	res[::2] = 10 * (x_even - x_odd **2)
	res[1::2] = 1 - x_odd
	return res







#initial points


def initial_points(n,prob):
	if prob == 1: #linear - full rank
		x = np.ones(n)
	elif prob == 2:#linear - rank 1
		x = np.ones(n)
	elif prob == 3:#linear - rank 1 with zero columns and rows
		x = np.ones(n)
	elif prob == 4:#rosenbrock
		x = np.array([-1.2,1])
	elif prob == 5:#helical valley
		x = np.zeros(n)
		x[0] = -1
	elif prob == 6:#powell singular 
		x = np.array([3,-1,0,1])
	elif prob == 7:#freudenstein and roth
		x = np.array([0.5,-2])
	elif prob == 8:#bard
		x = np.ones(n)
	elif prob == 9:#kowalik and osborne
		x = np.array([0.25,0.39,0.415,0.39])
	elif prob == 10:#meyer
		x = np.array([0.02,4000,250])
	elif prob == 11:#watson
		x = 0.5 * np.ones(n)
	elif prob == 12:#box 3-dimensional 
		x = np.array([0,10,20])
	elif prob == 13:#jeenrich and sampson
		x = np.array([0.3,0.4])
	elif prob == 14:#brown and dennis
		x = np.array([25,5,-5,-1])
	elif prob == 15:#chebyquad
		x = (np.arange(n) + 1) / (n+1)
	elif prob == 16:#brown almost-linear
		x = 0.5 * np.ones(n)
	elif prob == 17:#osborne 1
		x = np.array([0.5,1.5,1,0.01,0.02])
	elif prob == 18:#osborne 2 
		x = np.array([1.3,0.65,0.65,0.7,0.6,3,5,7,2,4.5,5.5])
	elif prob == 19: #bdqrtic
		x = np.ones(n)
	elif prob == 20: #cube
		x = 0.5 * np.ones(n)
	elif prob == 21:
		x = np.zeros(n)
		for i in range(n):
			ss = 0
			for j in range(n):
				ss += np.sqrt((i+1)/(j+1)) * ((np.sin(np.log(np.sqrt((i+1)/(j+1)))))**5 + (np.cos(np.log(np.sqrt((i+1)/(j+1)))))**5)
			x[i] = - 8.710996e-4 * ((i-49)**3 + ss)
	elif prob == 22:
		x = np.array([-0.3,-0.39,0.3,-0.344,-1.2,2.69,1.59,-1.5])
	elif prob == 23: #Extended Rosenbrock
		x = np.ones(n).astype(float)
		x[::2] = -1.2
	return x























