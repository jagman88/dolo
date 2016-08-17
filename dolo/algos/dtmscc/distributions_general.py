import numpy as np
import scipy.sparse as spa
from dolo.algos.dtmscc.time_iteration import time_iteration
from dolo.numeric.misc import mlinspace

# # Check whether inverse transition is in the model.
# ('transition_inv' in model.functions)


# MOSTlY DONE
def stat_dist(model, mdr, Nf=1000, itmaxL=5000, tolL=1e-8, verbose=False):
    '''
    Compute a histogram of the stationary distribution for some fixed set of aggregate variables.

    Parameters
    ----------
    model : NumericModel
        "dtmscc" model to be solved
    mdr : Markov decision rule
        Markov decision rule associated with solution to the model
    Nf : int
        Number of endogenous state grid points used in the range of the
        stationary distribution
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
    # TODO: Don't yet know how to deal with multiple Markov states or multiple
    # endogenous states.
    # TODO: Seems like will need to loop over state variable in order to
    # construct the Q matrices separately for each separately.


    # Number of Markov state variables + transition matrix
    Nm = model.markov_chain[0].shape[0]
    P = model.markov_chain[1]              # Markov transition matrix

    stotdim = np.prod(Nf)    # Total number of endogenous state dimensions
    totdim = stotdim*Nm             # Total number of all state dimensions
    Qm = np.kron(P, np.ones([stotdim,1]))      # Exogenous state transitions

    # Find fine grid and the state tomorrow on the fine grid
    sgridf = fine_grid(model, Nf).flatten()
    sprimef = mdr_to_sprime(model, mdr, Nf).flatten()

    # Compute endogenous transitions
    # NOTE: we want to assign the distance away from the *lower* bound to the *upper* bound
    # (and conversely, assign distance awway from *lower* bound to the *upper* bound.
    # E.g., if k' is 3/4 of the way between kj and kj+1, then want to assign 3/4 weight to kj+1.

    idxlist = np.arange(0,totdim)
    idL, idU = lookup(sgridf, sprimef)  # Upper and lower bracketing indices

    weighttoupper = ( (sprimef - sgridf[idL])/(sgridf[idU] - sgridf[idL]) ).flatten()
    weighttolower = ( (sgridf[idU] - sprimef)/(sgridf[idU] - sgridf[idL]) ).flatten()

    QsL = spa.coo_matrix((weighttolower, (idxlist, idL.flatten())), shape=(totdim, Nf))
    QsU = spa.coo_matrix((weighttoupper, (idxlist, idU.flatten())), shape=(totdim, Nf))
    Qs =(QsL + QsU).tocsr()    # convert to CSR for better sparse matrix arithmetic performance.

    # TODO: Need to keep the row kronecker product in sparse matrix format
    Qs = Qs.toarray()
    rowkron = Qm[:, :, None]*Qs[:, None, :]
    rowkron = rowkron.reshape([totdim, -1])
    QT = spa.csr_matrix(rowkron).T
    # NOTE: CSR format (Compressed Sparse Row) has efficient arithmetic operations.
    # Useful here since we are using QT*L to compute L'.

    # Find stationary distribution, starting from uniform distribution
    L = np.ones(totdim)
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

    return L, QT


# TODO: GENERALIZE. Might be kind of hard...
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


# MOSTlY DONE
def mdr_to_sprime(model, mdr, Nf):
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
    # TODO: Although this is set up generally, I don't think it will work with
    # multiple endogenous state variables or multiple Markov states.
    # TODO: Need to deal with cases with multiple transition rules.


    # NOTE: I'm assuming that with multiple markov chains, the chains are taken
    # as a product, so that Nm is the total number of states, and mgrid is a
    # matrix of state combinations
    Nm = model.markov_chain[0].shape[0]
    mgrid = model.markov_chain[0]
    sgridf = fine_grid(model, Nf)

    # NOTE: Does transition take multi-valued markov states? Multi-valued endogeneous states?
    trans = model.functions['transition']      # trans(m, s, x, M, p, out)
    parms = model.calibration['parameters']

    # Construct next period's state variable using Markov decision rule
    totdimlist = Nf.tolist().copy()
    totdimlist.append(Nm)             # A list containing all state dimensions
    mdrc = np.zeros(totdimlist)
    sprimef = np.zeros(totdimlist)
    # NOTE: Does the markov decision rule take multi-valued state variables?
    # NOTE: What happens if there is more than one control variable?
    for i_m in range(Nm):
        mdrc[:, i_m] = mdr(i_m, sgridf.reshape(-1,1)).flatten()
        sprimef[:, i_m] = trans(mgrid[i_m], sgridf.reshape(-1,1),mdrc[:,i_m].reshape(-1,1), mgrid[i_m], parms).flatten()

    # NOTE: Need Fortran ordering here, because endogenous state dimensions change fastest
    totdim = np.prod(Nf)*Nm
    sprimef = np.reshape(sprimef, (totdim, 1), order='F')

    # Force sprimef onto the grid for use in computing the stationary distribution.
    sprimef = np.maximum(sprimef, min(sgridf))
    sprimef = np.minimum(sprimef, max(sgridf))

    return sprimef


# DONE
def fine_grid(model, Nf):
    '''
    Construct evenly spaced fine grids for endogenous state variables, using the
    upper and lower bounds of the state space as specified in the yaml file.

    Parameters
    ----------
    model : NumericModel
        "dtmscc" model to be solved
    Nf : array
        Number of points on a fine grid for each endogeneous state variable, for
        use in computing the stationary distribution.
    Returns
    -------
    gridf : list
        Fine grid for each endogenous state variable.
    '''

    grid = model.get_grid()
    a = grid.a
    b = grid.b
    sgridf = mlinspace(a,b,Nf)

    return sgridf
