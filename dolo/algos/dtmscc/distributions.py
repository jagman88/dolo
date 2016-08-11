import numpy

# # Check whether inverse transition is in the model.
# ('transition_inv' in model.functions)

def stat_dist(model, dr, Nkf):
    '''
    Idea is:
    First, create a histogram of the stationary distribution for some fixed set of aggregate variables.
    Second, allow the computation of the equilibrium interest rate


    Parameters
    ----------
    model : NumericModel
        "dtmscc" model to be solved
    dr : decision rule
        Decision rule from the solved model

    Returns
    -------
    L : array
        The density across states for the model
    QT : array
        The distribution transition matrix, i.e. L' = QT*L
    '''

    #TODO: Need to make code take a decision rule object. Do we take the object before
    # or after it has been solved on the fine grid used for the histogram?
    #TODO: Need to generalize the code. At the moment only takes one dimensional models.

    a = model.options['grid'].a
    b = model.options['grid'].b
    kgridf = np.linspace(a, b, num=Nkf)

    P = model.markov_chain[1]
    Qe = np.kron(P, np.ones([Nkf,1]))

    #
    # Qk = np.zeros([Nkf*Ne, Nkf])

    idxlist = np.arange(0,Nkf*Ne).reshape(-1,1)
    idL, idU = lookup(kgridf, kprimef)

    # IMPORTANT: we want to assign the distance away from the *lower* bound to the *upper* bound (and conversely, assign
    # distance awway from *lower* bound to the *upper* bound. That is, if k' is 3/4 of the way between kj and kj+1,
    # then want to assign 3/4 weight to kj+1.

    toupper = (kprimef - kgridf[idL])/(kgridf[idU] - kgridf[idL])
    tolower = (kgridf[idU] - kprimef)/(kgridf[idU] - kgridf[idL])

    # #TODO: Is there a way to make the Qk matrix in one line?
    QkL = spa.coo_matrix((tolower.flatten(), (idxlist.flatten(), idL.flatten())), shape=(Nkf*Ne, Nkf))
    QkU = spa.coo_matrix((toupper.flatten(), (idxlist.flatten(), idU.flatten())), shape=(Nkf*Ne, Nkf))
    Qk =(QkL + QkU).tocsr()    # convert to CSR for better sparse matrix arithmatic performance.


    # # TODO: NEED TO KEEP THIS AS A SPARSE MATRIX!!
    Qk = Qk.toarray()
    rowkron = Qe[:, :, None]*Qk[:, None, :]
    rowkron = rowkron.reshape([Nkf*Ne, -1])

    QT = spa.csr_matrix(rowkron).T
    # # NOTE: CSR (Compressed Sparse Row) has efficient arithmetic operations, but inefficient column slicing.
    # # Useful here since we are just using QT for transitions.

    L = np.ones(Nkf*Ne)
    L = L/sum(L)
    itmaxL = 5000
    tolL = 1e-11
    verbose = False

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


    return L, QT



# Lookup function
def lookup(grid, x):
    '''
    Finds the indices of the points in grid that brack the values in x: grid[idL] <=
    Grid must be sorted in ascending order. Find the first index, i, in grid such that grid[i] <= x.
    At the end of table, if x = grid[N], then set i = N-1. This is so that the upper index, i+1, would, be N-1.
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
