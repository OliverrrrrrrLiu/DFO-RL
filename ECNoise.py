import numpy as np

def function(nf,fval):
    # Determines the noise of a function from the function values

    # The user must provide the function value at nf equally-spaced points.
    # For example, if nf = 7, the user could provide

    #    f(x-3h), f(x-2h), f(x-h), f(x), f(x+h), f(x+2h), f(x+3h)

    # in the array fval. Although nf >= 4 is allowed, the use of at least
    # nf = 7 function evaluations is recommended.

    # Noise will not be detected by this code if the function values differ
    # in the first digit.

    # If noise is not detected, the user should increase or decrease the 
    # spacing h according to the ouput value of inform.  In most cases, 
    # the subroutine detects noise with the initial value of h.

    # On exit:
    #   fnoise is set to an estimate of the function noise; 
    #      fnoise is set to zero if noise is not detected.

    #   level is set to estimates for the noise. The k-th entry is an
    #     estimate from the k-th difference.

    #   inform is set as follows:
    #     inform = 1  Noise has been detected.
    #     inform = 2  Noise has not been detected; h is too small.
    #                 Try 100*h for the next value of h.
    #     inform = 3  Noise has not been detected; h is too large.
    #                 Try h/100 for the next value of h.

    #    Argonne National Laboratory
    #    Jorge More' and Stefan Wild. November 2009

    level = np.zeros((nf-1, 1))
    dsgn = np.zeros((nf-1,1))
    fnoise = 0.0
    gamma = 1.0

    #Compute the range of function values
    fmin = np.min(fval)
    fmax = np.max(fval)
    if (fmax - fmin)/np.max(np.abs(fmax), np.abs(fmin)) >.1:
        inform = 3
        return fnoise, level, inform

    #Construct the difference table
    for j in range(nf-1):
        for i in range(nf-j-1):
            fval[i] = fval[i+1] - fval[i]
        #h is too small only when half the function values are equal
        if j==0 and np.sum(fval[0:nf-1] == 0) >= nf/2:
            inform = 2
            return fnoise,level,inform

        gamma = 0.5 * ((j+1)/(2*(j+1) - 1)) * gamma

        #Compute the estimates for the noise level
        level[j] = np.sqrt(gamma*np.mean(fval[0:nf-j-1]**2))

        #Determine differences in sign
        emin = np.min(fval[0:nf-j-1])
        emax = np.max(fval[0:nf-j-1])
        if emin*emax < 0.0:
            dsgn[j] = 1

    #Determine the noise level
    for k in range(nf-3):
        emin = np.min(level[k:k+3])
        emax = np.max(level[k:k+3])
        if emax <= 4*emin and dsgn[k]:
            fnoise = level[k]
            inform = 1
            return fnoise, level, inform
    #If noise not detected then h is too large
    inform = 3
    return fnoise, level, inform










