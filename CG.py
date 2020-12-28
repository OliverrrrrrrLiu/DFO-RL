import numpy as np

def init_options_cg():
    '''
   Option initialization

   Initialize algorithm options for CG with default values

   Output values:
   ==============

   options:
       This is a structure with field that correspond to algorithmic
       options of our method.  In particular:

       cg_max_iter:   
           Maximum number of iterations
       cg_tol:        
           Convergence tolerance for residual
       step_type:
           Different ways to calculate the search direction:
           'Steepest Descent'
           'Newton'
       cg_output_level:
           Amount of output printed
           0: No output
           1: Only summary information
           2: One line per iteration 
    '''
    
    options = {}
    options['cg_max_iter'] = 1e5
    options['cg_tol'] = 1e-6
    options['cg_output_level'] = 0
    
    return options


def steihaug_cg(Bk, grad_fk, Delta, options):
    '''
    Optimization method for unconstrainted optimization

    Author: Andreas Waechter
    Date:   2019-02-08

    This is an implementation of Algorihm 7.2 in Nocedal/Wright text book.
    It solves the optimization problem

    min  1/2 p' Bk * p + grad_fk' * p
    s.t. ||p||_2 <= Delta

    Input arguments:
    ================

    Bk:     
        Hessian of quadratic objective function (does not have to be
        positive definite).
    grad_fk:
        gradient of quadratic objective function
    Delta:  
        Trust region radius
    options:
        This is a structure with options for the algorithm.
        For details see the init_options function.

    Output values:
    ==============

    p_sol:
        (Approximate) minimizer
    status:
        Return code indicating reason for termination:
           -1: Number of iterations exceeded
            0: CG converged to tolerance
            1: CG stopped because of negative curvature
            2: CG stopped because of trust region
    stats:
        Structure with statistics for the run.  Its fields are
        num_iter    Number of iterations taken
        resid_grad  2-norm of residual rk
    '''

    # iteration counter
    iter = 0

    # return flag
    status = -99 # set to nonsensical value to make sure it will be explicitly set

    # Determine size of the problem
    n = len(grad_fk)

    ### Initialization
    zj = np.zeros(n)
    rj = grad_fk
    rj_norm = np.linalg.norm(rj)
    zj_norm = 0.
    alpha = 0.
    beta = 0.
    dj = -rj

    # Dummy solution if exit occurs before any assignment
    p_sol = np.zeros((n, 1))

    if options['cg_output_level'] >= 2:
        # Prepare header for output
        output_header = '\t%6s %15s %15s %15s %15s' % ('iter', '||rk||', '||zk||', 'alpha', 'beta')

    ###########################
    # Main Loop
    ###########################    
    while True:
        
        ######################################################
        # Print iteration output
        ######################################################
        if options['cg_output_level'] >= 2:
            # Print the output header every 10 iterations
            if iter%10 == 0:
                print(output_header)
        
            print('\t%6i %15.8e %15.8e %15.8e %15.8e' % (iter, rj_norm, zj_norm, alpha, beta))

        ######################################################
        # Check termination
        ######################################################
        if rj_norm < options['cg_tol']:
            # Restore solution
            p_sol = zj
            # Set flag to indicate convergence to tolerance
            status = 0
            # exit from the while loop
            break;

        if iter > options['cg_max_iter']:
            # Set flag to indicate the maximum number of iterations have been
            # exceeded
            status = -1
            # exit from the while loop
            break
        
        ######################################################
        # Perform steps of the algorithm
        ######################################################
        
        ### Check if we have non-positive curvature here
        Bdj = Bk.dot(dj)
        dBdj = np.dot(Bdj, dj)
        if dBdj <= 0.:
            # Find tau such that pk = zj + tau*dj minimizes model and satisfies
            # ||pk||=Delta
            dTdj = np.dot(dj, dj)
            dTzj = np.dot(dj, zj)
            sqrroot = np.sqrt(dTzj**2 - dTdj*(zj_norm**2-Delta**2))
            tau1 = (-dTzj + sqrroot)/dTdj
            tau2 = (-dTzj - sqrroot)/dTdj
            pk1 = zj + tau1*dj
            pk2 = zj + tau2*dj
            # Pick the point with smaller model value
            mk1 = np.dot(pk1, grad_fk + 0.5*Bk.dot(pk1))
            mk2 = np.dot(pk2, grad_fk + 0.5*Bk.dot(pk2))
            if mk1 <= mk2:
                p_sol = pk1
            else:
                p_sol = pk2
            status = 1
            break

        ### Check if iterate is exceeding size of trust region
        alpha = (rj_norm**2)/dBdj
        zj_last = zj
        zj_norm_last = zj_norm
        zj = zj + alpha*dj
        zj_norm = np.linalg.norm(zj)
        if zj_norm >= Delta:
            # Find tau >= 0 such that pk = zj + tau*dj satisfies ||pk||=Delta
            zj = zj_last
            zj_norm = zj_norm_last
            dTdj = np.dot(dj, dj)
            dTzj = np.dot(dj, zj)
            sqrroot = np.sqrt(dTzj**2 - dTdj*(zj_norm**2-Delta**2))
            tau = (-dTzj + sqrroot)/dTdj
            p_sol = zj + tau*dj
            status = 2
            break

        ### Perform final updates in this iteration
        rj_norm_last = rj_norm
        rj = rj + alpha*Bdj
        rj_norm = np.linalg.norm(rj)
        beta = (rj_norm/rj_norm_last)**2
        dj = -rj + beta*dj
            
        # Increment iteration counter
        iter +=1 
    
                
    ######################################################
    # Finalize results
    ######################################################

    # Set the statistics
    stats = {}
    stats['num_iter'] = iter
    stats['resid_norm'] = rj_norm

    # Final output message
    if options['cg_output_level'] >= 1:
        if status == 0:
            print('\tConvergence tolerance satisfied.')
        elif status == -1:
            print('\tMaximum number of iterations (%d) excdeed.' % iter)
        elif status == 1:
            print('\tNegative curvature direction encountered.')
        elif status == 2:
            print('\tTrust-region is active.')
        else:
            print('\tUnexpected error.')
            
        # and more of your stuff
        print('  ||residual|| at final point is %e' % rj_norm)
        print('  ||p_sol|| is %e' % np.linalg.norm(p_sol))
        
    # Return output arguments
    return p_sol, status, stats    
                