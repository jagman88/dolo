import numpy as np
import scipy.sparse as spa

# # Check whether inverse transition is in the model.
# ('transition_inv' in model.functions)

def stat_dist(model, mdr, Nkf=500, itmaxL=5000, tolL=1e-11, verbose=False):
    '''
    Idea is:
    First, create a histogram of the stationary distribution for some fixed set of aggregate variables.
    Second, allow the computation of the equilibrium interest rate


    Parameters
    ----------
    model : NumericModel
        "dtmscc" model to be solved
    mdr : Markov decision rule
        Markov decision rule associated with solution to the model

    Returns
    -------
    L : array
        The density across states for the model
    QT : array
        The distribution transition matrix, i.e. L' = QT*L
    kgridf : array
        Fine grid of state variables used for the histogram.
    '''

    #TODO: Need to generalize the code. At the moment only takes one dimensional models.

    a = mdr.a
    b = mdr.b
    orders = mdr.orders
    Ne = model.markov_chain[0].shape[0]
    Nk = model.get_grid().orders[0]
    egrid = model.markov_chain[0]
    kgridf = np.linspace(a, b, num=Nkf)
    trans = model.functions['transition']      # trans(m, s, x, M, p, out)
    parms = model.calibration['parameters']

    P = model.markov_chain[1]              # Markov transition matrix
    Qe = np.kron(P, np.ones([Nkf,1]))      # Exogenous state transitions

    # Construct next period's state variable using Markov decision rule
    mdrc = np.zeros([Nkf, Ne])
    kprimef = np.zeros([Nkf, Ne])
    for i_m in range(Ne):
        mdrc[:, i_m] = mdr(i_m, kgridf.reshape(-1,1)).flatten()
        kprimef[:, i_m] = trans(egrid[i_m], kgridf.reshape(-1,1), mdrc[:,i_m].reshape(-1,1), egrid[i_m], parms).flatten()

    # NOTE: Need Fortran ordering here, because k dimension changes fastest
    kprimef = np.reshape(kprimef, (Nkf*Ne, 1), order='F')

    # Compute endogenous transitions
    idxlist = np.arange(0,Nkf*Ne)
    idL, idU = lookup(kgridf, kprimef)

    # NOTE: we want to assign the distance away from the *lower* bound to the *upper* bound (and conversely, assign distance awway from *lower* bound to the *upper* bound. E.g., if k' is 3/4 of the way between kj and kj+1, then want to assign 3/4 weight to kj+1.

    weighttoupper = ( (kprimef - kgridf[idL])/(kgridf[idU] - kgridf[idL]) ).flatten()
    weighttolower = ( (kgridf[idU] - kprimef)/(kgridf[idU] - kgridf[idL]) ).flatten()

    QkL = spa.coo_matrix((weighttolower, (idxlist, idL.flatten())), shape=(Nkf*Ne, Nkf))
    QkU = spa.coo_matrix((weighttoupper, (idxlist, idU.flatten())), shape=(Nkf*Ne, Nkf))
    Qk =(QkL + QkU).tocsr()    # convert to CSR for better sparse matrix arithmetic performance.


    # # TODO: Need to keep the row kronecker product in sparse matrix format
    Qk = Qk.toarray()
    rowkron = Qe[:, :, None]*Qk[:, None, :]
    rowkron = rowkron.reshape([Nkf*Ne, -1])
    QT = spa.csr_matrix(rowkron).T
    # NOTE: CSR format (Compressed Sparse Row) has efficient arithmetic operations.
    # Useful here since we are using QT*L to compute L'.

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

    L = L.reshape(Ne, Nkf).T


    return L, QT, kgridf



# Lookup function
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
