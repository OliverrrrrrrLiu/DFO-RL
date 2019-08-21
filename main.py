import os, time, argparse
import numpy as np
from FDLM import *


def rosenbrock(x):
	return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2 #* (1 + 1e-6 * np.random.rand())

def f1(x):
	return (np.inner(x,x)) #* (1 + 1e-6*np.random.rand())

def f2(x):
	return (x[0] - 1)**2 + (np.exp(x[1]) - 1)/(np.exp(x[1]) + 1) + 0.1 * np.exp(-x[1])

def f3(x):
	return (x[0] - 1)**4 + (np.exp(x[1]) - 1)/(np.exp(x[1]) + 1) + 0.1 * np.exp(-x[1])

mode_dict = {"fd": "forward difference", "cd": "central difference"}
function_dict = {"rosenbrock": rosenbrock, "f1": f1, "f2": f2, "f3": f3}

parser = argparse.ArgumentParser(description="Argument parser for FDLM driver script")
parser.add_argument("--function", default="rosenbrock", type=str, help="Function to be optimized")
parser.add_argument("--ecn_parameter", default=(1e-8, 7, 100), type=tuple, help="ECNoise parameters in the order of (spacing, breadth, max_iter)")
parser.add_argument("--ls_parameter", default=(1e-4, 0.5, 20), type=tuple, help="Line search parameters in the order of (c_1, c_2, max_iter)")
parser.add_argument("--rec_parameter", default=(0.9, 1.1), type=tuple, help="Recovery parameter in the order of (gamma_1, gamma_2)")
parser.add_argument("--zeta", default=0.1, type=float, help="Curvature condition parameter")
parser.add_argument("--m", default=10, type=int, help="L-BFGS memory length")
parser.add_argument("--output_level", default=2, type=int, help="Output level")
parser.add_argument("--mode", default="cd", type=str, help="Finite-difference estimation mode")
parser.add_argument("--tol", default=1e-8, type=float, help="Tolerance level for convergence")
#.add_argument("--method", default="lbfgs", help="Optimization method")
parser.add_argument("--x0", default=[2, 2], help="Starting point")



def main(args):
	assert len(args.ecn_parameter) == 3
	assert len(args.ls_parameter) == 3
	assert len(args.rec_parameter) == 2
	assert args.output_level in [0, 1, 2]
	assert args.mode in ["fd", "cd"]
	assert args.function in function_dict.keys()



	print("Evaluating {} function".format(args.function))
	print("ECNoise parameter: {}".format(args.ecn_parameter))
	print("LineSearch parameter: {}".format(args.ls_parameter))
	print("Recovery parameter: {}".format(args.rec_parameter))
	print("Cuvature parameter: {}".format(args.zeta))
	print("LBFGS memory length: {}".format(args.m))
	print("Gradient evaluation mode: {}".format(mode_dict[args.mode]))
	print("convergence tolerance level: {}".format(args.tol))

	f = function_dict[args.function]
	fdlm = FDLM(f, args.ecn_parameter, args.ls_parameter, args.rec_parameter, args.zeta, args.m, args.output_level, args.mode, args.tol)
	time.sleep(1)
	start = time.time()
	fdlm.run(np.array(args.x0))
	print("Optimization completed! Total time elapsed: {:.3e}s".format(time.time()-start))

if __name__ == "__main__":
	args = parser.parse_args()
	XX = np.asarray(args.x0)
	print(XX.shape)
	main(args)

#python main.py --function f2  --mode cd
