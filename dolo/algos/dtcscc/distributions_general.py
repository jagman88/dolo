import numpy as np
import scipy.sparse as spa
from dolo.algos.dtcscc.time_iteration import time_iteration, time_iteration_direct
from dolo.numeric.misc import mlinspace
from dolo.numeric.discretization.discretization import rouwenhorst


# # Check whether inverse transition is in the model.
# ('transition_inv' in model.functions)

def stat_dist(model, dr, Nf, itmaxL=5000, tolL=1e-8, verbose=False):
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
    itmaxL : int
        Maximum number of iterations over the distribution evolution equation
    tolL : int
        Tolerance on the distance between subsequent distributions

    Returns
    -------
    L : array
        The density across states for the model. Note, the implied
        grid that L lays on follows the convention that for [N1, N2, N3, ...],
        earlier states vary slower than later states. Therefore, if reshaping L,
        make sure to use Fortran ordering: L.reshape([Nf[0], Nf[1]],order='F')
    QT : array
        The distribution transition matrix, i.e. L' = QT*L
    '''
    # NOTE: as it stands, variables are ordered as: [endogenous, exogenous]

    # HACK: get number of exogenous states from the number of shocks in
    # the model. We are assuming that each shock is associated with an
    # exogenous state. That is, no IID shocks enter the model on their own.
    Nexo = len(model.calibration['shocks'])
    Nend = len(model.calibration['states']) - Nexo

    # Total number of continuous states
    Ntot = np.prod(Nf)

    # Create fine grid for the histogram
    sgridf = fine_grid(model, Nf)
    parms = model.calibration['parameters']

    # Find the state tomorrow on the fine grid
    sprimef = dr_to_sprime(model, dr, Nf)

    # Compute exogenous state transition matrices
    mgrid, Qm = exog_grid_trans(model, Nf)

    # Compute endogenous state transition matrices
    # First state:
    sgrid = np.unique(sgridf[:,0])
    Qs = single_state_transition_matrix(sgrid, sprimef[:,0], Nf, Nf[0]).toarray()
    # Subsequent state transitions created via repeated tensor products
    for i_s in range(1,Nend):
        sgrid = np.unique(sgridf[:,i_s])
        Qtmp = single_state_transition_matrix(sgrid, sprimef[:,i_s], Nf, Nf[i_s]).toarray()
        N = Qs.shape[1]*Qtmp.shape[1]
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
    Compute the transition matrix for an individual state variable.
    Transitions are from states defined on a fine grid with the dimensions
    in Nf, to the unique values that lie on the individual state grid
    (with dimension Nstate).

    Parameters
    ----------
    grid : Array
        The approximated model's state space defined on a grid. Must be unique
        values, sorted in ascending order.
    vals : Array
        Actual values of state variable next period (computed using a
        transition or decision rule)
    Nf : array
        Number of fine grid points in each dimension
    Nstate : int
        Number of fine grid points for the state variable in question.

    Returns
    -------
    Qstate : sparse matrix
        An [NtotxNstate] Transition probability matrix for the state
        variable in quesiton.

    '''
    Ntot = np.prod(Nf)

    idxlist = np.arange(0,Ntot)
    # Find upper and lower bracketing indices for the state values on the grid
    idL, idU = lookup(grid, vals)

    # Assign probability weight to the bracketing points on the grid
    weighttoupper = ( (vals - grid[idL])/(grid[idU] - grid[idL]) ).flatten()
    weighttolower = ( (grid[idU] - vals)/(grid[idU] - grid[idL]) ).flatten()

    # Construct sparse transition matrices.
    # Note: we convert to CSR for better sparse matrix arithmetic performance.
    QstateL = spa.coo_matrix((weighttolower, (idxlist, idL.flatten())), shape=(Ntot, Nstate))
    QstateU = spa.coo_matrix((weighttoupper, (idxlist, idU.flatten())), shape=(Ntot, Nstate))
    Qstate =(QstateL + QstateU).tocsr()

    return Qstate


# TODO: Need to allow for multiple aggregate variables to be computed.
# E.g. a model with aggregate capital and labor.
def solve_eqm(model, Nf, Kinit=50, itermaxKeq=100, tolKeq=1e-4, verbose=False):
    '''
    Solve for the equilibrium value of the aggregate capital stock in the
    model. Do this via a damping algorithm over the capital stock. Iterate
    until aggregate capital yields an interest rate that induces a
    distribution of capital accumulation across agents that is consistent
    with that capital stock.

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    Kinit : float
        Initial guess for the aggregate capital stock
    Nf : array
        Number of fine grid points in each dimension
    itermaxKeq : int
        Maximum number of iterations over the capital stock
    tolKeq : int
        Maximum distance between succesive iterations over capital

    Returns
    -------
    K : float
        Equilibrium aggregate capital
    '''

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
        L, QT = stat_dist(model, dr, Nf, verbose=False)
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


def supply_demand(model, varname, pricename, Nf, numpoints=20, lower=40,
                  upper=75, verbose=True):
    '''
    Solve the model at a range of aggregate capital values to generate
    supply and demand curves a given aggregate variable. Note, the aggregate
    demand for the variable must be (at least implicitly) defined in the
    model file, .e.g., K = f(r) or L = g(w).

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    varname : string
        The string name of the aggregate variable in question. e.g.
        varname = 'kagg' picks out aggregate capital.
    price : string
        The string name of the price of the aggregate variable,
        e.g. price = 'r' picks out the interest rate
    Nf : array
        Number of fine grid points in each dimension
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
        L, QT = stat_dist(model, dr, Nf, verbose=False)
        As[i] = np.dot(L, sgridf[:,0])

        # Get price of the aggregate variable
        p[i] = model.calibration_dict[pricename]
        if verbose is True:
            print('Iteration = %i\n' % i)

    return Ad, As, p


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

    gridf = fine_grid(model, Nf)
    trans = model.functions['transition']      # trans(s, x, e, p, out)
    parms = model.calibration['parameters']
    grid = model.get_grid()
    a = grid.a
    b = grid.b

    drc = dr(gridf)

    # NOTE: sprimef has second variable moving fastest.
    sprimef = trans(gridf, drc, np.zeros([np.prod(Nf),1]), parms)

    # Keep state variables on their respective grids
    for i_s in range(len(a)):
        sprimef[:,i_s] = np.maximum(sprimef[:,i_s], a[i_s])
        sprimef[:,i_s] = np.minimum(sprimef[:,i_s], b[i_s])

    return sprimef


def lookup(grid, x):
    '''
    Finds indices of points in the grid that bracket the values in x.
    Grid must be sorted in ascending order. Find the first index, i, in
    grid such that grid[i] <= x. This is the index of the upper bound for x,
    unless x is equal to the lowest value on the grid, in which case set the
    upper bound index equal to 1. The lower bound index is simply one less
    than the upper bound index.
    '''
    N = grid.shape[0]-1   # N = last index in grid
    m = grid.min()
    M = grid.max()
    x = np.maximum(x, m)
    x = np.minimum(x, M)
    idU = np.searchsorted(grid, x)   # Index of the upper bound
    idU = np.maximum(idU, 1)      # Upper bound is always greater than 1
    idL = idU -1                  # lower bound index = upper bound index - 1

    return idL, idU



def fine_grid(model, Nf):
    '''
    Construct evenly spaced fine grids for state variables. For endogenous
    variables use a uniform grid with the upper and lower bounds as
    specified in the yaml file. For exogenous variables, use grids from
    the discretization of the AR(1) process via Rouwenhorst.

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    Nf : array
        Number of points on a fine grid for each endogeneous state
        variable, for use in computing the stationary distribution.

    Returns
    -------
    grid : array
        Fine grid for state variables. Note, endogenous ordered first,
        then exogenous. Later variables are "fastest" moving, earlier
        variables are "slowest" moving.
    '''

    # HACK: trick to get number of exogenous states
    Nexo = len(model.calibration['shocks'])
    Nend = len(model.calibration['states']) - Nexo

    # ENDOGENOUS VARIABLES
    grid = model.get_grid()
    a = grid.a
    b = grid.b
    sgrid = mlinspace(a[:Nend],b[:Nend],Nf[:Nend])

    mgrid, Qm = exog_grid_trans(model, Nf)

    # Put endogenous and exogenous grids together
    gridf = np.hstack([np.repeat(sgrid, mgrid.shape[0],axis=0), np.tile(mgrid, (sgrid.shape[0],1))])

    # grid = model.get_grid()
    # a = grid.a
    # b = grid.b
    # sgridf = mlinspace(a,b,Nf)

    return gridf


def exog_grid_trans(model, Nf):
    '''
    Construct the grid and transition matrix for exogenous variables. Both
    elements are drawn from the Rouwenhorst descritization of the exogenous
    AR(1) process. The grids and transition matrices are compounded if
    there are multiple exogenous variables, and are constructed such that
    late variables are "fastest" moving, earlier variables are "slowest" moving.

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    Nf : array
        Number of points on a fine grid for each endogeneous state variable,
        for use in computing the stationary distribution.

    Returns
    -------
    mgrid : array
        Fine grid for exogenous state variables.
    Qm : array
        Transition matrix (dimension: N x Nm)
    '''

    # HACK: trick to get number of exogenous states
    Nexo = len(model.calibration['shocks'])
    Nend = len(model.calibration['states']) - Nexo
    Nendtot = np.prod(Nf[:Nend])

    # Get AR(1) persistence parameters via the derivative of the
    # transition rule at the steady state
    trans = model.functions['transition']    #  tmp(s, x, e, p, out)
    sss, xss, ess, pss = model.calibration['states', 'controls', 'shocks', 'parameters']
    diff = trans(sss, xss, ess, pss, diff=True)
    diff_s = diff[0]  # derivative wrt state variables

    # Get AR(1) std dev from the variance covariance matrix
    distr = model.distribution

    # EXOGENOUS VARIABLES
    # Get first grid
    rho = diff_s[Nend]
    sig = np.sqrt(distr.sigma[0,0])
    mgrid, Qm = rouwenhorst(rho, sig, Nf[Nend])
    mgrid = mgrid[:,None]
    # Get subsequent grids
    for i_m in range(1,Nexo):
        rho = diff_s[Nend+i_m]
        sig = np.sqrt(distr.sigma[i_m,i_m])
        tmpgrid, Qtmp = rouwenhorst(rho, sig, Nf[i_m])
        # Compound the grids
        tmpgrid = np.tile(tmpgrid, mgrid.shape[0])
        mgrid = np.repeat(mgrid, Nf[i_m],axis=0)
        mgrid = np.hstack([mgrid, tmpgrid[:,None]])
        # Compound the transition matrices
        Qm = np.kron(Qm, Qtmp)

    # Repeat to match full sized state space.
    Qm = np.kron(np.ones([Nendtot,1]), Qm)

    return mgrid, Qm