import numpy as np
import scipy.sparse as spa
from dolo.algos.dtmscc.time_iteration import time_iteration

# # Check whether inverse transition is in the model.
# ('transition_inv' in model.functions)

def stat_dist(model, mdr, Nkf=1000, itmaxL=5000, tolL=1e-8, verbose=False):
    '''
    Compute a histogram of the stationary distribution for some fixed set of aggregate variables.

    Parameters
    ----------
    model : NumericModel
        "dtmscc" model to be solved
    mdr : Markov decision rule
        Markov decision rule associated with solution to the model
    Nkf : int
        Number of grid points used in the range of the stationary distribution
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
    Ne = model.markov_chain[0].shape[0]
    P = model.markov_chain[1]              # Markov transition matrix
    Qe = np.kron(P, np.ones([Nkf,1]))      # Exogenous state transitions

    # Find fine grid and the state tomorrow on the fine grid
    kgridf = fine_grid(model, Nkf)
    kprimef = mdr_to_sprime(model, mdr, Nkf)

    # Compute endogenous transitions
    # NOTE: we want to assign the distance away from the *lower* bound to the *upper* bound (and conversely, assign distance awway from *lower* bound to the *upper* bound. E.g., if k' is 3/4 of the way between kj and kj+1, then want to assign 3/4 weight to kj+1.

    idxlist = np.arange(0,Nkf*Ne)
    idL, idU = lookup(kgridf, kprimef)  # Upper and lower bracketing indices

    weighttoupper = ( (kprimef - kgridf[idL])/(kgridf[idU] - kgridf[idL]) ).flatten()
    weighttolower = ( (kgridf[idU] - kprimef)/(kgridf[idU] - kgridf[idL]) ).flatten()

    QkL = spa.coo_matrix((weighttolower, (idxlist, idL.flatten())), shape=(Nkf*Ne, Nkf))
    QkU = spa.coo_matrix((weighttoupper, (idxlist, idU.flatten())), shape=(Nkf*Ne, Nkf))
    Qk =(QkL + QkU).tocsr()    # convert to CSR for better sparse matrix arithmetic performance.

    # TODO: Need to keep the row kronecker product in sparse matrix format
    Qk = Qk.toarray()
    rowkron = Qe[:, :, None]*Qk[:, None, :]
    rowkron = rowkron.reshape([Nkf*Ne, -1])
    QT = spa.csr_matrix(rowkron).T
    # NOTE: CSR format (Compressed Sparse Row) has efficient arithmetic operations.
    # Useful here since we are using QT*L to compute L'.

    # Find stationary distribution, starting from uniform distribution
    L = np.ones(Nkf*Ne)
    L = L/sum(L)
    for itL in range(itmaxL):
        Lnew = QT*L      # Sparse matrices can be multipled
        dL = np.linalg.norm(Lnew-L)/np.linalg.norm(L)
        if (dL < tolL):
            break

        if verbose is True:
            if np.mod(itL, 100) == 0:
                print('Iteration = %i, dist = %f \n' % (itL, dL))

        L = np.copy(Lnew)

    # L = L.reshape(Ne, Nkf).T

    return L, QT


def solve_eqm(model, Kinit=38, Nkf=1000, itermaxKeq=100, tolKeq=1e-4, verbose=False):
    '''
    Solve for the equilibrium value of the aggregate capital stock in the model.
    Do this via a damping algorithm over the capital stock. Iterate until
    aggregate capital yields an interest rate that induces a distribution of
    capital accumulation across agents that is consistent with that capital stock.

    Parameters
    ----------
    model : NumericModel
        "dtmscc" model to be solved
    Kinit : float
        Initial guess for the aggregate capital stock
    Nkf : int
        Number of points in the fine grid for the distribution
    itermaxKeq : int
        Maximum number of iterations over the capital stock
    tolKeq : int
        Maximum distance between succesive iterations over capital

    Returns
    -------
    K : float
        Equilibrium aggregate capital
    '''

    # TODO: need option for which algorithm will be used to solve for the decision rule

    K = Kinit
    model.set_calibration(kagg=K)
    mdr = time_iteration(model, with_complementarities=True, verbose=False, output_type='dr')
    kgridf = fine_grid(model, Nkf)
    kprimef = mdr_to_sprime(model, mdr, Nkf)

    damp = 0.999
    for iteq in range(itermaxKeq):
        # Solve for decision rule given current guess for K
        mdr = time_iteration(model, with_complementarities=True, verbose=False, output_type='dr')

        # Solve for stationary distribution given decision rule
        L, QT = stat_dist(model, mdr, Nkf=Nkf, verbose=False)
        Kagg = np.dot(L, np.hstack([kgridf, kgridf]))

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


def supply_demand(model, Nkf=1000, numpoints=20, lower=37, upper=45, verbose=True):
    '''
    Solve the model at a range of aggregate capital values to generate supply
    and demand curves for the aggregate capital stock.

    Parameters
    ----------
    model : NumericModel
        "dtmscc" model to be solved
    Nkf : int
        Number of points in the fine grid for the distribution
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
    Ne = model.markov_chain[0].shape[0]

    Kd = np.linspace(lower,upper,numpoints)
    Ks = np.zeros([numpoints,1])
    r = np.zeros([numpoints,1])

    for i in range(numpoints):
        model.set_calibration(kagg=Kd[i])
        mdr = time_iteration(model, with_complementarities=True, verbose=False, output_type='dr')
        kgridf = fine_grid(model, Nkf)
        kprimef = mdr_to_sprime(model, mdr, Nkf)
        L, QT = stat_dist(model, mdr, Nkf=Nkf, verbose=False)
        Ks[i] = np.dot(L, np.tile(kgridf,Ne))    # Ks[i] = np.dot(L, np.hstack([kgridf, kgridf]))
        r[i] = model.calibration_dict['r']
        print('Iteration = %i\n' % i)

    return Kd, Ks, r



def mdr_to_sprime(model, mdr, Nkf):
    '''
    Solve the Markov decision rule on the fine grid, and compute the next
    period's state variable.

    Parameters
    ----------
    model : NumericModel
        "dtmscc" model to be solved
    mdr : Markov decision rule
        Markov decision rule associated with solution to the model
    Nkf : int
        Number of points in the fine grid for the distribution

    Returns
    -------
    kprimef : array
        Next period's state variable, given the decision rule mdr
    '''

    Ne = model.markov_chain[0].shape[0]
    egrid = model.markov_chain[0]
    kgridf = fine_grid(model, Nkf)

    trans = model.functions['transition']      # trans(m, s, x, M, p, out)
    parms = model.calibration['parameters']

    # Construct next period's state variable using Markov decision rule
    mdrc = np.zeros([Nkf, Ne])
    kprimef = np.zeros([Nkf, Ne])
    for i_m in range(Ne):
        mdrc[:, i_m] = mdr(i_m, kgridf.reshape(-1,1)).flatten()
        kprimef[:, i_m] = trans(egrid[i_m], kgridf.reshape(-1,1), mdrc[:,i_m].reshape(-1,1), egrid[i_m], parms).flatten()

    # NOTE: Need Fortran ordering here, because k dimension changes fastest
    kprimef = np.reshape(kprimef, (Nkf*Ne, 1), order='F')

    # Force kprimef onto the grid for use in computing the stationary distribution.
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



def fine_grid(model, Nkf):
    '''
    Construct fine grid for endogenous state variable.
    '''
    grid = model.get_grid()
    a = grid.a
    b = grid.b
    kgridf = np.linspace(a, b, num=Nkf)

    return kgridf
