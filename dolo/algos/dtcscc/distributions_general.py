import numpy as np
import scipy.sparse as spa
from dolo.algos.dtcscc.time_iteration import time_iteration, time_iteration_direct
from dolo.numeric.misc import mlinspace
from dolo.numeric.discretization.discretization import rouwenhorst


# # Check whether inverse transition is in the model.
# ('transition_inv' in model.functions)

def stat_dist(model, dr, Nf, Nq=7, itmaxL=5000, tolL=1e-8, verbose=False):
    '''
    Compute a histogram of the stationary distribution for some fixed set of
    aggregate variables.

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    dr : Decision rule
        Decision rule associated with solution to the model
    Nf : array
        Number of fine grid points in each dimension
    Nq : int
        Number of quadrature nodes over the iid shock
    itmaxL : int
        Maximum number of iterations over the distribution evolution equation
    tolL : int
        Tolerance on the distance between subsequent distributions

    Returns
    -------
    L : array
        The density across states for the model. Note, the implied
        grid that L lays on follows the convention that for [N1, N2, N3, ...], earlier states vary slower than later states. Therefore, if reshaping L, make sure to use Fortran ordering: L.reshape([Nf[0], Nf[1]],order='F')
    QT : array
        The distribution transition matrix, i.e. L' = QT*L
    '''
    # TODO: once we know how many endogenous vs. exogenous states there are, it's easy to create additional transition matrices Q by simply looping over each of the respective states, and using the single_state_transition_matrix function.

    # NOTE: as it stands, variables are ordered as: [endogenous, exogenous]

    # HACK: get number of exogenous states from the number of shocks in the model.
    # We are assuming that each shock is associated with an exogenous state.
    # That is, no IID shocks enter the model on their own.
    Nexo = len(model.calibration['shocks'])
    Nend = len(model.calibration['states']) - Nexo

    # Total number of continuous states
    Ntot = np.prod(Nf)

    # Create fine grid for the histogram
    sgridf = fine_grid(model, Nf)
    parms = model.calibration['parameters']

    # Get the quadrature nodes for the iid normal shocks
    distrib = model.get_distribution()
    nodes, weights = distrib.discretize(orders=Nq)

    # Find the state tomorrow on the fine grid
    sprimef = dr_to_sprime(model, dr, Nf)


    # Compute exogenous state transition matrices by discretizing the processes
    # First state:

    [nodes, Qm] = rouwenhorst(rho, sigma, N)

    Qm = spa.csr_matrix((Ntot, Nf[Nend]))   # Start from first exogenous state
    mgrid = np.unique(sgridf[:,Nend])
    for i in range(Nq):
        mprimef = gtilde(model, sgridf[:,Nend], nodes[i])
        Qm += weights[i]*single_state_transition_matrix(mgrid, mprimef, Nf, Nf[Nend])
    Qm = Qm.toarray()

    # TODO: Finish extra dimensions. Tricky because of extra dimensions in Nq...
    # Second (and further) state transitions created via repeated tensor products
    # for i_m in range(Nend, Nend+Nexo+1):
    #     Qtmp = spa.csr_matrix((Ntot, Nf[i_m]))   # Start from first exogenous state



    # Compute endogenous state transition matrices
    # First state:
    sgrid = np.unique(sgridf[:,0])
    Qs = single_state_transition_matrix(sgrid, sprimef[:,0], Nf, Nf[0]).toarray()
    # Second (and further) state transitions created via repeated tensor products
    for i_s in range(1,Nend):
        sgrid = np.unique(sgridf[:,i_s])
        Qtmp = single_state_transition_matrix(sgrid, sprimef[:,i_s], Nf, Nf[i_s]).toarray()
        N = Qtmp.shape[1]*Qs.shape[1]
        Qs = Qs[:, :, None]*Qtmp[:, None, :]
        Qs = Qs.reshape([N, -1])

    # Construct all-state transitions via endogenous-exogenous tensor product
    Q = Qs[:, :, None]*Qm[:, None, :]
    Q = Q.reshape([Ntot, -1])
    QT = spa.csr_matrix(Q).T
    # TODO: Need to keep the row kronecker product in sparse matrix format

    # Find stationary distribution, starting from uniform distribution
    L = np.ones(Ntot)
    L = L/sum(L)
    for itL in range(itmaxL):
        Lnew = QT*L      # Sparse matrices can be multipled
        dL = np.linalg.norm(Lnew-L)/np.linalg.norm(L)
        if (dL < tolL):
            break

        L = np.copy(Lnew)

        if verbose is True:
            if np.mod(itL, 100) == 0:
                print('Iteration = %i, dist = %f \n' % (itL, dL))

    return L, QT


def single_state_transition_matrix(grid, vals, Nf, Nstate):
    '''
    Compute the transition matrix for an individual state variable. Transitions are
    from states defined on a fine grid with the dimensions in Nf, to the unique values
    that lie on the individual state grid (with dimension Nstate).

    Parameters
    ----------
    grid : Array
        The approximated model's state space defined on a grid. Must be unique
        values, sorted in ascending order.
    vals : Array
        Actual values of state variable next period (computed using a transition or decision rule)
    Nf : array
        Number of fine grid points in each dimension
    Nstate : int
        Number of fine grid points for the state variable in question.

    Returns
    -------
    Qstate : sparse matrix
        An [NtotxNstate] Transition probability matrix for the state variable in quesiton.

    '''
    Ntot = np.prod(Nf)

    idxlist = np.arange(0,Ntot)
    # Find upper and lower bracketing indices for the state values on the grid
    idL, idU = lookup(grid, vals)

    # Assign probability weight to the bracketing points on the grid
    weighttoupper = ( (vals - grid[idL])/(grid[idU] - grid[idL]) ).flatten()
    weighttolower = ( (grid[idU] - vals)/(grid[idU] - grid[idL]) ).flatten()

    # Construct sparse transition matrices. Note: we convert to CSR for better sparse matrix arithmetic performance.
    QstateL = spa.coo_matrix((weighttolower, (idxlist, idL.flatten())), shape=(Ntot, Nstate))
    QstateU = spa.coo_matrix((weighttoupper, (idxlist, idU.flatten())), shape=(Ntot, Nstate))
    Qstate =(QstateL + QstateU).tocsr()

    return Qstate



def solve_eqm(model, Nf, Kinit=50, Nq=7, itermaxKeq=100, tolKeq=1e-4, verbose=False):
    '''
    Solve for the equilibrium value of the aggregate capital stock in the model.
    Do this via a damping algorithm over the capital stock. Iterate until
    aggregate capital yields an interest rate that induces a distribution of
    capital accumulation across agents that is consistent with that capital stock.

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    Kinit : float
        Initial guess for the aggregate capital stock
    Nf : array
        Number of fine grid points in each dimension
    Nq : int
        Number of quadrature nodes over the iid shock
    itermaxKeq : int
        Maximum number of iterations over the capital stock
    tolKeq : int
        Maximum distance between succesive iterations over capital

    Returns
    -------
    K : float
        Equilibrium aggregate capital
    '''

    # TODO: Need to allow for multiple aggregate variables to be computed. E.g. a model with aggregate capital and labor.

    K = Kinit
    model.set_calibration(kagg=K)
    if ('direct_response' in model.symbolic.equations):
        dr = time_iteration_direct(model, with_complementarities=True, verbose=False)
    else:
        dr = time_iteration(model, with_complementarities=True, verbose=False)

    sgridf = fine_grid(model, Nf)

    damp = 0.999
    for iteq in range(itermaxKeq):
        # Solve for decision rule given current guess for K
        if ('direct_response' in model.symbolic.equations):
            dr = time_iteration_direct(model, with_complementarities=True, verbose=False)
        else:
            dr = time_iteration(model, with_complementarities=True, verbose=False)

        # Solve for stationary distribution given decision rule
        L, QT = stat_dist(model, dr, Nf, Nq=Nq, verbose=False)
        Kagg = np.dot(L, sgridf[:,0])

        dK = np.linalg.norm(Kagg-K)/K
        if (dK < tolKeq):
            break

        if verbose is True:
            print('Iteration = \t%i: K=\t%1.4f  Kagg=\t%1.4f\n' % (iteq, K, Kagg) )

        # Update guess for aggregate capital using damping
        K = damp*K + (1-damp)*Kagg

        # Update calibration and reduce damping paramter
        model.set_calibration(kagg=K)
        damp = 0.995*damp

    return K


def supply_demand(model, varname, pricename, Nf, Nq=7, numpoints=20, lower=40, upper=75, verbose=True):
    '''
    Solve the model at a range of aggregate capital values to generate supply and demand curves a given aggregate variable.
    Note, the aggregate demand for the variable must be (at least implicitly) defined in the model file, .e.g., K = f(r) or
    L = g(w).

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    varname : string
        The string name of the aggregate variable in question. e.g.
        varname = 'kagg' picks out aggregate capital.
    price : string
        The string name of the price of the aggregate variable, e.g. price = 'r' picks out the interest rate
    Nf : array
        Number of fine grid points in each dimension
    Nq : int
        Number of quadrature nodes over the iid shock
    numpoints : int
        Number of points at which to evaluate the curves
    lower : float
        Lower bound on aggregate variable (demand)
    upper : float
        Upper bound on aggregate variable (demand)

    Returns
    -------
    Ad : array
        Set of aggreate demands
    As : array
        Set of aggregate supplies
    p : array
        Set of prices at each point on the demand-supply curves
    '''

    sgridf = fine_grid(model, Nf)

    Ad = np.linspace(lower,upper,numpoints)
    As = np.zeros([numpoints,1])
    p = np.zeros([numpoints,1])

    for i in range(numpoints):
        # Set new aggregate variable value and solve
        model.set_calibration(varname,Ad[i])
        if ('direct_response' in model.symbolic.equations):
            dr = time_iteration_direct(model, with_complementarities=True, verbose=False)
        else:
            dr = time_iteration(model, with_complementarities=True, verbose=False)

        # Compute aggregate supply using computed distribution
        L, QT = stat_dist(model, dr, Nf, Nq=Nq, verbose=False)
        As[i] = np.dot(L, sgridf[:,0])

        # Get price of the aggregate variable
        p[i] = model.calibration_dict[pricename]    # p[i] = model.calibration_dict['r']
        if verbose is True:
            print('Iteration = %i\n' % i)

    return Ad, As, p



# TODO: allow for multiple exogenous processes
# NOTE: To do this, would need to know which variables are exogenous...
# NOTE: Alternative would be to use the tanh trick for all exogenous
# variable transitions. But then need that to be in any/every
# heterogeneous agents model...
def gtilde(model, e, eps):
    '''
    Transition rule for an exogeneous variable that ensures it remains on the grid.

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    e : float or array
        Current state of the exogenous variable
    eps : float
        Value of shock to the exogenous variable

    Returns
    -------
    gtilde : array
        Next period's exogenous variable, bounded to remain on the grid space
    '''
    grid = model.get_grid()
    emin = grid.a[1]
    emax = grid.b[1]
    rho_e = model.calibration_dict['rho_e']
    # gtilde = e**rho_e*np.exp(eps)
    gtilde = rho_e*e + eps
    gtilde = np.minimum(gtilde, emax)
    gtilde = np.maximum(gtilde, emin)

    return gtilde


def dr_to_sprime(model, dr, Nf):
    '''
    Solve the decision rule on the fine grid, and compute the next
    period's state variable.

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    dr : Decision rule
        Decision rule associated with solution to the model
    Nf : array
        Number of fine grid points in each dimension

    Returns
    -------
    kprimef : array
        Next period's state variable, given the decision rule mdr
    '''

    sgridf = fine_grid(model, Nf)
    trans = model.functions['transition']      # trans(s, x, e, p, out)
    parms = model.calibration['parameters']
    grid = model.get_grid()
    a = grid.a
    b = grid.b

    drc = dr(sgridf)

    # NOTE: sprimef has second variable moving fastest.
    sprimef = trans(sgridf, drc, np.zeros([np.prod(Nf),1]), parms)

    # Keep state variables on their respective grids
    for i_s in range(len(a)):
        sprimef[:,i_s] = np.maximum(sprimef[:,i_s], a[i_s])
        sprimef[:,i_s] = np.minimum(sprimef[:,i_s], b[i_s])

    return sprimef


# TODO: GENERALIZE!
# TODO: Will need to take into account potentially multiple state variables.
def lookup(grid, x):
    '''
    Finds indices of points in the grid that bracket the values in x.
    Grid must be sorted in ascending order. Find the first index, i, in grid such that grid[i] <= x.
    This is the index of the upper bound for x, unless x is equal to the lowest value on the grid,
    in which case set the upper bound index equal to 1. The lower bound index is simply one less
    than the upper bound index.
    '''
    N = grid.shape[0]-1   # N = last index in grid
    m = grid.min()
    M = grid.max()
    x = np.maximum(x, m)
    x = np.minimum(x, M)
    idU = np.searchsorted(grid, x)   # Get insertion index = index of the upper bound
    idU = np.maximum(idU, 1)      # Make sure upper bound is always the second index, 1, or higher.
    idL = idU -1                  # lower bound index = upper bound index - 1

    return idL, idU



def fine_grid(model, Nf):
    '''
    Construct evenly spaced fine grids for endogenous state variables, using the
    upper and lower bounds of the state space as specified in the yaml file.

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    Nf : array
        Number of points on a fine grid for each endogeneous state variable, for
        use in computing the stationary distribution.

    Returns
    -------
    sgridf : array
        Fine grid for each endogenous state variable.

    '''
    # HACK: trick to get number of exogenous states
    Nexo = len(model.calibration['shocks'])
    Nend = len(model.calibration['states']) - Nexo

    rho_e = model.calibration_dict['rho_e']
    sig_e = model.calibration_dict['sig_e']
    mgrid, Qm = rouwenhorst(rho_e, sig_e, Nf[1])

    grid = model.get_grid()
    a = grid.a
    b = grid.b
    sgridf = mlinspace(a,b,Nf)

    return sgridf
