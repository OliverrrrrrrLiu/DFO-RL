import numpy as np 
import matplotlib.pyplot as plt
from DFO_randomJacobian import DFO_LSTR
from DFO_Jacobian_col import DFO_LSTR_COL
from LSfunc_def import *  #functions
from FDLM_Noise import *  #QuasiNewton-Linesearch
from DFO_GNCD2 import *  #Coordinate Gauss-Newton
from DFO_linear import * #DFO with Linear 
import pandas as pd


np.random.seed(123)

#initialization
n = 9 #dimension of x
m = 45
prob = 1 #problem number
x = initial_points(n,prob) #starting point
def r_noise(x,m):
	return fun1(x,m) + 1e-3 *np.random.normal()


#========================================================================================================================================
#
#========================================================================================================================================

dfogncd = DFO_GNCD(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),3, rule = "random") #coordiante GN


col_0 = DFO_LSTR_COL(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),1,0, 3, rule = "random", j_len = 0) #everyiter
col_1 = DFO_LSTR_COL(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 0) #every5iter
col_2 = DFO_L(1e-3,r_noise,m,100,1,0.1,1/4,3/4, (1e-8, 7, 100),5,0, 3, rule = "random", j_len = 100)
_, _, _, fval_history_0, eval_history_0 = col_0.run(x)
_, _, _, fval_history_1, eval_history_1 = col_1.run(x)
_, _, _, fval_history_2, eval_history_2 = col_2.run(x)
_, _, _, fval_history_3, eval_history_3 = dfogncd.run(x)

#lindon roberts DFO-LS
DFOLS_lr = pd.read_csv("/Users/melody/Desktop/DFO_Experiments/DFOLS_Roberts/Linear_full_rank.csv", usecols = [1,13])
f_vals = DFOLS_lr.values

#PLOT 

plt.plot(eval_history_1,np.log(fval_history_1), label = "NONE,update frequency 5")
plt.plot(eval_history_2,np.log(fval_history_2), label = "linear criterion")
plt.plot(eval_history_0,np.log(fval_history_0), label = "ALL")
plt.plot(eval_history_3,np.log(fval_history_3), label = "coordinate GN")
plt.plot(f_vals[:,1], np.log(f_vals[:,0]), label = "DFO-LS")
plt.xlabel("Number of function evaluations")
plt.ylabel("Log(Objective Value)")
plt.title("Linear Full Rank (9,45)")
plt.legend()
#plt.show()
plt.savefig("/Users/melody/Desktop/DFO_Experiments/DFOLS_Roberts/LinearFullRank.png")



