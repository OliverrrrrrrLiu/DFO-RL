import numpy as np 
import matplotlib.pyplot as plt
from DFO_randomJacobian import DFO_LSTR
from DFO_Jacobian_col import DFO_LSTR_COL

from LSfunc_def import *

from FDLM_Noise import *

from DFO_GNCD2 import *

from DFO_linear import *

np.random.seed(123)
#n = 100
#x = np.ones(n).astype(float)
#x[::2] = -1.2
#x = np.array([0,10,20])
#x = np.array([0.5,1.5,-1,0.01,0.02])
#x = np.array([1.3,0.65,0.65,0.7,0.6,3,5,7,2,4.5,5.5])
#m = n


#initialization
n = 10 #dimension of x
m = n
prob = 23 #problem number
x = initial_points(n,prob) #starting point
def r_noise(x,m):
	return fun23(x,m) + 1e-3 *np.random.normal()

"""
dfogncd = DFO_GNCD(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),3, rule = "random")

dfogncd.run(x)
"""




def f(x):
	return np.linalg.norm(r_noise(x,m)) **2
	#return np.sum((r_noise(x,23))**2)

fdlm = FDLM(1e-3,f, (1e-8, 7, 100), (1e-4, 0.9, 20), (0.9, 1.1), 0, 10,3)
fdlm.run(x)


#col_0 = DFO_LSTR_COL(0,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),1,0, 3, rule = "random", j_len = 0)
#col_2 = DFO_LSTR_COL(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 0)
#_, _, _, fval_history_0, eval_history_0 = col_0.run(x)
#_, _, _, fval_history_1, eval_history_1 = col_2.run(x)

##============Compare different k_accept and k_reject====================================================
"""																#k_accept, k_reject, j_len = #rows to be updated
col_0 = DFO_LSTR_COL(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),1,0, 3, rule = "random", j_len = 0)
col_1 = DFO_LSTR_COL(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),2,0, 3, rule = "random", j_len = 0)
col_2 = DFO_LSTR_COL(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 0)
col_3 = DFO_LSTR_COL(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,5, 3, rule = "random", j_len = 0)
col_4 = DFO_LSTR_COL(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,10, 3, rule = "random", j_len = 0)
col_5 = DFO_LSTR_COL(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),10,10, 3, rule = "random", j_len = 0)
col_6 = DFO_LSTR_COL(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),10,0, 3, rule = "random", j_len = 0)
"""
"""
row_0 = DFO_LSTR(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),1,0, 3, rule = "random", j_len = 3)
row_1 = DFO_LSTR(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),2,0, 3, rule = "random", j_len = 3)
row_2 = DFO_LSTR(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 3)
row_3 = DFO_LSTR(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,5, 3, rule = "random", j_len = 3)
row_4 = DFO_LSTR(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,10, 3, rule = "random", j_len = 3)
row_5 = DFO_LSTR(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),10,10, 3, rule = "random", j_len = 3)
row_6 = DFO_LSTR(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),10,20, 3, rule = "random", j_len = 3)
"""


##======== with fixed frequency, compare the amount of rows/cols updated in intermediate iterations==========
"""
col_0 = DFO_LSTR_COL(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 0)
col_1 = DFO_LSTR_COL(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 1)
col_2 = DFO_LSTR_COL(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 30)
col_3 = DFO_LSTR_COL(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 50)
col_4 = DFO_LSTR_COL(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 100)
col_5 = DFO_L(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 100)
"""

"""
row_0 = DFO_LSTR(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 0)
row_1 = DFO_LSTR(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 1)
row_2 = DFO_LSTR(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 30)
row_3 = DFO_LSTR(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 50)
row_4 = DFO_LSTR(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 100)
"""


#==========================================================================================================
#        RUN
#==========================================================================================================
"""
_, _, _, fval_history_0, eval_history_0 = col_0.run(x)
_, _, _, fval_history_1, eval_history_1 = col_1.run(x)
_, _, _, fval_history_2, eval_history_2 = col_2.run(x)
_, _, _, fval_history_3, eval_history_3 = col_3.run(x)
_, _, _, fval_history_4, eval_history_4 = col_4.run(x)
_, _, _, fval_history_5, eval_history_5 = col_5.run(x)
#_, _, _, fval_history_6, eval_history_6 = col_6.run(x)
"""

"""
_, _, _, fval_history_0, eval_history_0 = row_0.run(x)
_, _, _, fval_history_1, eval_history_1 = row_1.run(x)
_, _, _, fval_history_2, eval_history_2 = row_2.run(x)
_, _, _, fval_history_3, eval_history_3 = row_3.run(x)
_, _, _, fval_history_4, eval_history_4 = row_4.run(x)
#_, _, _, fval_history_5, eval_history_5 = row_5.run(x)
#_, _, _, fval_history_6, eval_history_6 = row_6.run(x)
"""

#Plot
"""
plt.plot(eval_history_0, np.log(fval_history_0), label = "1,0")
plt.plot(eval_history_1, np.log(fval_history_1), label = "2,0")
plt.plot(eval_history_2, np.log(fval_history_2), label = "5,0")
plt.plot(eval_history_3, np.log(fval_history_3), label = "5,5")
plt.plot(eval_history_4, np.log(fval_history_4), label = "5,10")
plt.plot(eval_history_5, np.log(fval_history_5), label = "10,10")
plt.plot(eval_history_6, np.log(fval_history_6), label = "10,20")
plt.xlabel("Number of function evaluations")
plt.ylabel("Log(Objective Value)")
plt.title("Rosenbrock: Update Jacobian every Few iterations")
plt.legend()
plt.show()
"""

"""
plt.plot(eval_history_0, np.log(fval_history_0), label = "NONE")
plt.plot(eval_history_1, np.log(fval_history_1), label = "1 row")
plt.plot(eval_history_2, np.log(fval_history_2), label = "30% rows")
plt.plot(eval_history_3, np.log(fval_history_3), label = "50% rows")
plt.plot(eval_history_4, np.log(fval_history_4), label = "ALL")
plt.xlabel("Number of function evaluations")
plt.ylabel("Log(Objective Value)")
plt.title("Heart8ls (8,8): update every 5 iterations")
plt.legend()
plt.savefig("/Users/melody/Desktop/Experiments/Heart8ls_row.png")

"""

"""
plt.plot(eval_history_0,np.log(fval_history_0), label = "NONE")
plt.plot(eval_history_1,np.log(fval_history_1), label = "1 col")
plt.plot(eval_history_2,np.log(fval_history_2), label = "30% cols")
plt.plot(eval_history_3,np.log(fval_history_3), label = "50% cols")
plt.plot(eval_history_4,np.log(fval_history_4), label = "ALL")
plt.xlabel("Number of function evaluations")
plt.ylabel("Log(Objective Value)")
plt.title("Heart8ls (8,8): update every 5 iterations")
plt.legend()
plt.savefig("/Users/melody/Desktop/Experiments/Heart8ls_col.png")

"""

"""
plt.plot(eval_history_0,np.log(fval_history_0), label = "NONE,update frequency 5")
#plt.plot(eval_history_1,np.log(fval_history_1), label = "1 col")
#plt.plot(eval_history_2,np.log(fval_history_2), label = "30% cols")
#plt.plot(eval_history_3,np.log(fval_history_3), label = "50% cols")
plt.plot(eval_history_5,np.log(fval_history_5), label = "linear criterion")
plt.plot(eval_history_4,np.log(fval_history_4), label = "ALL")
plt.xlabel("Number of function evaluations")
plt.ylabel("Log(Objective Value)")
plt.title("Chebyquad (10,10)")
plt.legend()
plt.show()
#plt.savefig("/Users/melody/Desktop/Experiments/Heart8ls_col.png")
"""




