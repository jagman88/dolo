import numpy as np

def solve_policy(model, grid={}):

    assert(model.model_type == 'dtmscc')
    # assert(set(['g','r']).issubset(set(model.model_spec)))

    discount = model.calibration['beta']

    felicity = model.functions['felicity']

    [P, Q] = model.markov_chain

    n_ms = P.shape[0]   # number of markov states
    n_mv = P.shape[1] # number of markov variables

    x0 = model.calibration['controls']
    p = model.calibration['parameters']
    m0 = model.calibration['markov_states']

    n_x = len(x0)

    n_s = len(model.symbols['states'])

    approx = model.get_grid(**grid)
    a = approx.a
    b = approx.b
    orders = approx.orders

    from dolo.numeric.decision_rules_markov import MarkovDecisionRule
    mdrv = MarkovDecisionRule(n_ms, a, b, orders) # values

    grid = mdrv.grid
    N = grid.shape[0]

    s0 = model.calibration['states']
    r0 = felicity(m0,s0,x0,p)
    from dolo.misc.dprint import dprint
    dprint(r0)

    controls_0 = np.zeros((n_ms, N, n_x))
    controls_0[:,:,:] = model.calibration['controls'][None,None,:]
    #
    values_0 = np.zeros((n_ms, N, 1))
    values_0[:,:,:] = r0/(1-discount)

    transition = model.functions['transition']
    felicity = model.functions['felicity']
    controls_lb = model.functions['controls_lb']
    controls_ub = model.functions['controls_ub']

    mdrv = MarkovDecisionRule(n_ms, a, b, orders) # values

    import scipy.optimize

    maxit = 500
    it = 0
    tol_v = 1e-6
    err_v = 100
    err_x = 100

    while it<maxit and err_v>tol_v:

        it+=1

        mdrv.set_values(values_0)

        values = values_0.copy()
        controls = controls_0.copy()

        for i_m in range(n_ms):
            for n in range(N):
                m = P[i_m,:]
                s = grid[n,:]
                x = controls[i_m,n,:]
                values[i_m,n,0] = choice_value(transition,felicity, i_m, s, x, mdrv, P, Q, p, discount)

        err_x = abs(controls - controls_0).max()
        err_v = abs(values - values_0).max()

        values_0 = values

        print((it,err_v))

    maxit = 1000
    it = 0
    tol_v = 1e-6
    err_v = 100
    err_x = 100

    while it<maxit and err_v>tol_v:

        it+=1

        mdrv.set_values(values_0)

        values = values_0.copy()
        controls = controls_0.copy()

        for i_m in range(n_ms):
            for n in range(N):

                m = P[i_m,:]
                s = grid[n,:]
                x = controls[i_m,n,:]
                lb = controls_lb(m,s,p)
                ub = controls_ub(m,s,p)
                bounds = [e for e in zip(lb,ub)]

                fun = lambda t: -choice_value(transition,felicity, i_m, s, t, mdrv, P, Q, p, discount)[0]
                res = scipy.optimize.minimize(fun, x, bounds=bounds)

                controls[i_m,n,:] = res.x
                values[i_m,n,0] = -fun(res.x)

        err_x = abs(controls - controls_0).max()
        err_v = abs(values - values_0).max()

        values_0 = values
        controls_0 = controls

        print((it,err_x,err_v))

    mdr = MarkovDecisionRule(n_ms, a, b, orders) # values
    mdr.set_values(controls)

    return mdr, mdrv

def choice_value(transition,felicity, i_ms, s, x, drv, P, Q, parms, beta):

    n_ms = P.shape[0]   # number of markov states
    m = P[i_ms,:]
    cont_v = 0.0
    for I_ms in range(n_ms):
        M = P[I_ms,:]
        prob = Q[i_ms, I_ms]
        S = transition(m,s,x,M,parms)
        cont_v += prob*drv(I_ms, S)[0]
    return felicity(m,s,x,parms) + beta*cont_v



def evaluate_policy(model, mdr, tol=1e-8,  maxit=2000, grid={}, verbose=True, initial_guess=None, hook=None, integration_orders=None):

    """Compute value function corresponding to policy ``dr``

    Parameters:
    -----------

    model:
        "dtcscc" model. Must contain a 'value' function.

    mdr:
        decision rule to evaluate

    Returns:
    --------

    decision rule:
        value function (a function of the space similar to a decision rule
        object)

    """

    assert(model.model_type == 'dtmscc')

    [P, Q] = model.markov_chain

    n_ms = P.shape[0]   # number of markov states
    n_mv = P.shape[1]   # number of markov variables

    x0 = model.calibration['controls']
    v0 = model.calibration['values']
    parms = model.calibration['parameters']
    n_x = len(x0)
    n_v = len(v0)
    n_s = len(model.symbols['states'])

    approx = model.get_grid(**grid)
    a = approx.a
    b = approx.b
    orders = approx.orders

    from dolo.numeric.decision_rules_markov import MarkovDecisionRule
    mdrv = MarkovDecisionRule(n_ms, a, b, orders) # values

    grid = mdrv.grid
    N = grid.shape[0]

    controls = np.zeros((n_ms, N, n_x))
    for i_m in range(n_ms):
        controls[i_m,:,:] = mdr(i_m,grid) #x0[None,:]

    values_0 = np.zeros((n_ms, N, n_v))
    if initial_guess is None:
        for i_m in range(n_ms):
            values_0[i_m,:,:] = v0[None,:]
    else:
        for i_m in range(n_ms):
            values_0[i_m,:,:] = initial_guess(i_m, grid)


    ff = model.functions['arbitrage']
    gg = model.functions['transition']
    aa = model.functions['auxiliary']
    vaval = model.functions['value']


    f = lambda m,s,x,M,S,X,p: ff(m,s,x,M,S,X,p)
    g = lambda m,s,x,M,p: gg(m,s,x,M,p)
    def val(m,s,x,v,M,S,X,V,p):
        return vaval(m,s,x,v,M,S,X,V,p)
    # val = lambda m,s,x,v,M,S,X,V,p: vaval(m,s,x,aa(m,s,x,p),v,M,S,X,aa(M,S,X,p),V,p)


    sh_v = values_0.shape

    err = 10
    inner_maxit = 50
    it = 0


    if verbose:
        headline = '|{0:^4} | {1:10} | {2:8} | {3:8} |'.format( 'N',' Error', 'Gain','Time')
        stars = '-'*len(headline)
        print(stars)
        print(headline)
        print(stars)

    import time
    t1 = time.time()

    err_0 = np.nan

    verbit = (verbose == 'full')

    while err>tol and it<maxit:

        it += 1

        t_start = time.time()

        mdrv.set_values(values_0.reshape(sh_v))

        values = update_value(val, g, grid, controls, values_0, mdr, mdrv, P, Q, parms).reshape((-1,n_v))

        err = abs(values.reshape(sh_v)-values_0).max()

        err_SA = err/err_0
        err_0 = err

        values_0 = values.reshape(sh_v)

        t_finish = time.time()
        elapsed = t_finish - t_start

        if verbose:
            print('|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} |'.format( it, err, err_SA, elapsed  ))

    # values_0 = values.reshape(sh_v)

    t2 = time.time()

    if verbose:
        print(stars)
        print("Elapsed: {} seconds.".format(t2-t1))
        print(stars)

    return mdrv


def update_value(val, g, s, x, v, dr, drv, P, Q, parms):

    N = s.shape[0]
    n_s = s.shape[1]

    n_ms = P.shape[0]   # number of markov states

    res = np.zeros_like(v)

    for i_ms in range(n_ms):

        m = P[i_ms,:][None,:].repeat(N,axis=0)

        xm = x[i_ms,:,:]
        vm = v[i_ms,:,:]

        for I_ms in range(n_ms):

            # M = P[I_ms,:][None,:]
            M = P[I_ms,:][None,:].repeat(N,axis=0)
            prob = Q[i_ms, I_ms]

            S = g(m, s, xm, M, parms)
            XM = dr(I_ms, S)
            VM = drv(I_ms, S)

            rr = val(m,s,xm,vm,M,S,XM,VM,parms)

            res[i_ms,:,:] += prob*rr

    return res
