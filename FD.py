import numpy as np
from numpy.core.umath_tests import inner1d
from numpy.linalg import norm

# TODO: implement second-order finite difference table when mu_2 is not found
# TODO: estimate maxHessian only once
############TODO: the f_incr of CD is wrong

class MaxHessian(object):
    """Estimate the maximum absolute Hessian entry for function f"""
    def __init__(self, f, eps, tau_1 = 100, tau_2 = 0.1):
        """
        @param f:         noisy function to be evaluated
        @param eps:       estimated noise level
        @param tau_i:     constant used for Hessian estimation, see More and Wild "Estimating Derivatives of Noisy Simulations", 2012
        """
        self.f = f
        self.eps = eps
        self.tau_1 = tau_1
        self.tau_2 = tau_2

    def calculate_delta(self, x, h, direction = None):
        """
        Calculate the second-order difference |f(x-h) - 2*f(x) + f(x+h)|
        
        @param x: current point
        @param h: spacing
        @param direction: direction at which the finite difference is conducted
        """
        fx = self.f(x) #evaluate f(x)
        if isinstance(x, (list, np.ndarray)): #check type
            x = np.array(x)
            d = direction if direction is not None else np.random.randn(len(x)) #random direction is no direction is given
            d /= norm(d) #normalize direction
            h *= d #stepsize h in direction h
            """
            d = direction if direction is not None else np.random.choice(len(x)) #random direction e_i if no direction is given
            tmp = np.zeros(len(x))
            tmp[d] = h
            h = tmp
            """
        fx_plus_h = self.f(x + h)  #evaluate f(x+h)
        fx_minus_h = self.f(x - h) #evaluate f(x-h)
        delta = abs(fx_plus_h + fx_minus_h - 2 * fx) 
        return delta, fx, fx_plus_h, fx_minus_h

    def exit(self, h, dh, fx, fx_plus_h, fx_minus_h):
        """Verify if the exit condition has been satisfied: if the choise h is acceptable
        @param h:          step size
        @param dh:         delta_h: the second-difference calculated by calculate_delta(x, h)
        @param fx:         function value evaluated at x, i.e., f(x)
        @param fx_plus_h:  function value evaluated at f(x+h)
        @param fx_minus_h: function value evaluated at f(x-h)
        """
        c1 = dh / self.eps >= self.tau_1 #check whether h is sufficiently large
        c2 = abs(fx_plus_h - fx)  <= self.tau_2 * max(fx, fx_plus_h) #check whether h is not too large
        c3 = abs(fx_minus_h - fx) <= self.tau_2 * max(fx, fx_minus_h) # check whether h is not too large
        return c1 and c2 and c3

    def calculate_mu(self, x, mu = 1, direction = None):
        """Calculate the Hessian estimate
        @param mu: previous estimate of Hessian, to be used as a scalar (default: 1)
        """
        h = (self.eps / mu) ** 0.25
        dh, fx, fx_plus_h, fx_minus_h = self.calculate_delta(x, h, direction) #calculate second-order difference
        mu = dh / h ** 2
        return h, dh, mu, fx, fx_plus_h, fx_minus_h

    def estimate(self, x, direction = None):
        """Estimate the max absolute Hessian
        @param x:         number or list type. Input value
        @param direction: nonnegative integer or None. Direction with respect to
                          which the finite-difference Hessian estimation will be
                          conducted (default: None, random direction)
        """
        h_a, dh_a, mu_a, fx, fx_plus_h_a, fx_minus_h_a = self.calculate_mu(x, direction = direction)
        if self.exit(h_a, dh_a, fx, fx_plus_h_a, fx_minus_h_a):
            return mu_a #return mu_a as the abs hessian estimate
        h_b, dh_b, mu_b, fx, fx_plus_h_b, fx_minus_h_b = self.calculate_mu(x, mu = mu_a, direction = direction)
        if self.exit(h_b, dh_b, fx, fx_plus_h_b, fx_minus_h_b) or abs(mu_a - mu_b) <= mu_b / 2: #if exit condition is satisfied or if mu_a and mu_b are similar
            return mu_b #exit with mu_b as the abs hessian estimate
        else:
            return 2.0 #backup when neither exit conditions were satisfied
"""
def f(x):
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    n, d = x.shape
    return inner1d(x, x) + np.random.uniform(1e-3,1e-3)
"""
def fd_gradient(f, x, eps,h = None, mode = "cd", tau_1 = 100, tau_2 = 0.1, direction = None):
    """
    estimate the finite difference interval and finite difference gradient given noise 
    @param f: function f
    @param x: current point
    @param eps: noise level of function f
    @param h: finite difference interval (default = None)
    @param mode: "fd": forward difference "cd": central difference
    @param tau_1: acceptance parameter for second order difference interval (>>1)
    @param tau_2: acceptance parameter for second order difference interval (0,1)

    @return: finite difference gradient, finite difference interval 
    """
    dim = len(x)
    #return np.array([-400 * x[0] * (x[1]-x[0]**2) - 2 * (1-x[0]), 200 * (x[1]-x[0]**2)]), 0.1
    #return np.array([400 * x[0]**3 - 400 * x[0] * x[1] + 2 * x[0] - 2, 200 * (x[1]-x[0]**2)]),0.1
    #return np.array([200*(x[1] - x[0]**2)*(-2*x[0]) + 2*(x[0]-1), 200*(x[1] - x[0]**2)]),0.1
    if direction is None:     
        if eps <= 1e-20:#np.finfo(float).eps: #If no noise is observed
            if mode == "fd":
                h = np.sqrt(np.finfo(float).eps) 
            else:
                h = (np.finfo(float).eps)**(1/3) 
            fx = f(x)
            f_incr = []
            h_d = np.ones(dim) * h
            #calculate d-th component of the forward difference approximation of the gradient of f at x
            for d in range(dim): 
                tmp = np.zeros(dim)
                h_d[d] = h * max(1, np.abs(x[d]))
                tmp[d] = h_d[d]
                f_incr.append(f(x + tmp))
            f_incr = np.array(f_incr)
            if mode == "fd":
                f_decr = fx
            else:
                #central difference
                f_decr = []#fx
                for d in range(dim):
                    tmp = np.zeros(dim)
                    h_d[d]=h * max(1, np.abs(x[d]))
                    tmp[d] = h_d[d]
                    f_decr.append(f(x - tmp))
                h_d = 2*h_d
            return (f_incr - f_decr) / h_d, h_d

        mu = MaxHessian(f, eps, tau_1, tau_2).estimate(x, direction = None)
        if mu is None: return None #if hessian estimate was not found
        """
        f_incr = []
        f_decr = fx if mode == "fd" else []
        for d in range(dim):
            tmp = np.zeros(dim)
            tmp[d] = h
            f_incr.append(f(x + tmp))
            if mode == "fd":
                h = 8 ** 0.25 * (eps / mu) ** 0.5 if h is None else h
                tmp[d] = h
            else:
                h = 3 ** (1/3) * (eps / mu) ** (1/3) if h is None else h
                tmp[d] = h
                h *= 2
                f_decr.append(f(x - tmp))
        if isinstance(f_decr, list): f_decr = np.array(f_decr)
        f_incr = np.array(f_incr)
        return (f_incr - f_decr) / h, h
        """
        fx = f(x) #evaluate f(x)
        if h is None: #if finite difference interval not given
            h = 8 ** 0.25 * (eps / mu) ** 0.5
        #h_mat = np.diag(np.ones(dim) * h) TODO vectorize

        
        if mode == "fd": #forward difference mode
            f_decr = fx
            f_incr = []
        #calculate d-th component of the forward difference approximation of the gradient of f at x
            for d in range(dim): 
                tmp = np.zeros(dim)
                tmp[d] = h
                f_incr.append(f(x + tmp))
            f_incr = np.array(f_incr)
        else:
            f_incr = []
            f_decr = []
            #calculate d-th component of the central difference approximation of the gradient of f at x
            if h is None:
                h = 3 ** (1/3) * (eps / mu) ** (1/3)
        #calculate d-th component of the forward difference approximation of the gradient of f at x
            for d in range(dim): 
                tmp = np.zeros(dim)
                tmp[d] = h
                f_incr.append(f(x + tmp))
                f_decr.append(f(x - tmp))
            f_incr = np.array(f_incr)
            f_decr = np.array(f_decr)
            h = 2*h
        #print("mu, h", mu, h)
        return (f_incr - f_decr) / h, h
    else:
        if eps <= 1e-20:
            h = np.sqrt(np.finfo(float).eps) #* max(1, np.max(np.abs(x)))
            f_incr = f(x + h * direction)
            if mode == "fd":
                f_decr = f(x)
            else:
                f_decr = f(x - h * direction)
            return (f_incr - f_decr)/h, h
        mu = MaxHessian(f, eps, tau_1, tau_2).estimate(x, direction = direction)
        if mu is None: return None #if hessian estimate was not found
        fx = f(x)
        if h is None:
            h = 8 ** 0.25 * (eps / mu) ** 0.5
        if mode == "fd":
            f_decr = fx
            f_incr = f(x + h * direction)
        else:
            if h is None:
                h = 3 ** (1/3) * (eps / mu) ** (1/3)
            f_decr = f(x - h * direction)
            f_incr = f(x + h * direction)
            h = 2 * h
        return (f_incr - f_decr) / h, h        

"""
test
"""
def f(x):
    #return (100*(x[1]-x[0]**2)**2 + (1-x[0])**2 )*(1 + 1e-6*np.random.normal(0,1))
    return np.inner(x,x) *(1+ 1e-2*np.random.randn())

if __name__ == "__main__":
    print(fd_gradient(f, np.array([500.0,500.0]),1830.9999))


# if __name__ == "__main__":
#     lst = []
#     for _ in range(1000):
#         tmp = fd_gradient(f, np.array([5, 1]), 0.01)
#         if tmp is not None: lst.append(tmp[0])
#     lst = np.array(lst)
#     print(np.mean(lst, axis=0))
