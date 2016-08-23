import numpy as np
import scipy.sparse as spa
from dolo.algos.dtcscc.time_iteration import time_iteration
from dolo.numeric.misc import mlinspace


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
        The density across states for the model
    QT : array
        The distribution transition matrix, i.e. L' = QT*L
    '''

    # Exogenous/Markov state variable
    Nkf = Nf[0]
    Nef = Nf[1]
    Nq = 7
    kgridf, egridf = fine_grid(model, Nf)
    # NOTE: this seems backwards, but we reverse the columns so we're back to [k, e] ordering
    sf = mlinspace(np.array([min(egridf), min(kgridf)]), np.array([max(egridf), max(kgridf)]), np.array([Nef, Nkf]))
    sf[:,[0, 1]] = sf[:,[1, 0]]
    parms = model.calibration['parameters']

    # Get the quadrature nodes for the iid normal shocks
    distrib = model.get_distribution()
    nodes, weights = distrib.discretize(orders=Nq)
    # TODO: get the quadrature appropriate to the shock (i.e. normal, beta, etc..)

    Qe = spa.csr_matrix((Nkf*Nef, Nef))
    idxlist = np.arange(0,Nkf*Nef)

    for i in range(Nq):
        eprimef = gtilde(model, sf[:,1], nodes[i])
        idL, idU = lookup(egridf, eprimef)  # Upper and lower bracketing indices
        weighttoupper = ( (eprimef - egridf[idL])/(egridf[idU] - egridf[idL]) ).flatten()
        weighttolower = ( (egridf[idU] - eprimef)/(egridf[idU] - egridf[idL]) ).flatten()
        QeL = spa.coo_matrix((weighttolower, (idxlist, idL.flatten())), shape=(Nkf*Nef, Nef))
        QeU = spa.coo_matrix((weighttoupper, (idxlist, idU.flatten())), shape=(Nkf*Nef, Nef))
        Qe += weights[i]*(QeL + QeU).tocsr()

    # Find the state tomorrow on the fine grid
    kprimef = dr_to_sprime(model, dr, Nf)

    idxlist = np.arange(0,Nkf*Nef)
    idL, idU = lookup(kgridf, kprimef)  # Upper and lower bracketing indices

    weighttoupper = ( (kprimef - kgridf[idL])/(kgridf[idU] - kgridf[idL]) ).flatten()
    weighttolower = ( (kgridf[idU] - kprimef)/(kgridf[idU] - kgridf[idL]) ).flatten()

    QkL = spa.coo_matrix((weighttolower, (idxlist, idL.flatten())), shape=(Nkf*Nef, Nkf))
    QkU = spa.coo_matrix((weighttoupper, (idxlist, idU.flatten())), shape=(Nkf*Nef, Nkf))
    Qk =(QkL + QkU).tocsr()    # convert to CSR for better sparse matrix arithmetic performance.

    # TODO: Need to keep the row kronecker product in sparse matrix format
    Qk = Qk.toarray()
    Qe = Qe.toarray()
    rowkron = Qe[:, :, None]*Qk[:, None, :]
    rowkron = rowkron.reshape([Nkf*Nef, -1])
    QT = spa.csr_matrix(rowkron).T

    # Find stationary distribution, starting from uniform distribution
    L = np.ones(Nkf*Nef)
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

    # TODO: need option that selects which algorithm will be used to solve for
    # the decision rule

    Nkf = Nf[0]
    Nef = Nf[1]
    K = Kinit
    model.set_calibration(kagg=K)
    dr = time_iteration(model, with_complementarities=True, verbose=False)
    kprimef = dr_to_sprime(model, dr, Nf)
    kgridf, egridf = fine_grid(model, Nf)

    damp = 0.999
    for iteq in range(itermaxKeq):
        # Solve for decision rule given current guess for K
        dr = time_iteration(model, with_complementarities=True, verbose=False)

        # Solve for stationary distribution given decision rule
        L, QT = stat_dist(model, dr, Nf, Nq=Nq, verbose=False)
        Kagg = np.dot(L, np.tile(kgridf,Nef))

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


def supply_demand(model, Nf, numpoints=20, lower=40, upper=75, verbose=True):
    '''
    Solve the model at a range of aggregate capital values to generate supply
    and demand curves for the aggregate capital stock.

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    Nf : array
        Number of fine grid points in each dimension
    numpoints : int
        Number of points at which to evaluate the curves
    lower : float
        Lower bound on aggregate capital stock (demand)
    Upper : float
        Upper bound on aggregate capital stock (demand)

    Returns
    -------
    Kd : array
        Set of aggreate capital demands
    Ks : array
        Set of aggregate capital supplies
    r : array
        Set of interest rates at each point on the demand-supply curves
    '''

    Nkf = Nf[0]
    Nef = Nf[1]
    Nq = 7
    kgridf, egridf = fine_grid(model, Nf)

    Kd = np.linspace(lower,upper,numpoints)
    Ks = np.zeros([numpoints,1])
    r = np.zeros([numpoints,1])

    for i in range(numpoints):
        model.set_calibration(kagg=Kd[i])
        dr = time_iteration(model, with_complementarities=True, verbose=False)
        kprimef = dr_to_sprime(model, dr, Nf)
        L, QT = stat_dist(model, dr, Nf, Nq=Nq, verbose=False)
        Ks[i] = np.dot(L, np.tile(kgridf,Nef))
        r[i] = model.calibration_dict['r']
        if verbose is True:
            print('Iteration = %i\n' % i)

    return Kd, Ks, r



def gtilde(model, e, eps):
    '''
    Transition rule for the exogeneous variable that ensures it remains on the grid.

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
    gtilde = np.maximum(np.minimum(e**rho_e*np.exp(eps), emax), emin)
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

    Nkf = Nf[0]
    Nef = Nf[1]
    kgridf, egridf = fine_grid(model, Nf)
    trans = model.functions['transition']      # trans(s, x, e, p, out)
    parms = model.calibration['parameters']

    # NOTE: This looks backwards, but we reverse columns so we're back to [k, e] ordering
    sf = mlinspace(np.array([min(egridf), min(kgridf)]), np.array([max(egridf), max(kgridf)]), np.array([Nef, Nkf]))
    sf[:,[0, 1]] = sf[:,[1, 0]]

    drc = dr(sf)

    # NOTE: sprimef has second variable moving fastest.
    sprimef = trans(sf, drc, np.zeros([Nkf*Nef,1]), parms)    # Goes off the grid bounds for kprimef
    kprimef = sprimef[:,0]
    kprimef = np.maximum(kprimef, min(kgridf))
    kprimef = np.minimum(kprimef, max(kgridf))

    return kprimef



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
    Construct fine grid for state variables.
    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    Nf : array
        Number of fine grid points in each dimension

    Returns
    -------
    kgridf : array
        Fine gird of points using dimensions in Nf
    egridf : array
        Fine gird of points using dimensions in Nf

    '''
    Nkf = Nf[0]
    Nef = Nf[1]
    grid = model.get_grid()
    a = grid.a
    b = grid.b
    kgridf = np.linspace(a[0], b[0], num=Nkf)
    egridf = np.linspace(a[1], b[1], num=Nef)

    return kgridf, egridf
