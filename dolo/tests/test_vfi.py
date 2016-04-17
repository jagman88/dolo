from dolo import *

from dolo import yaml_import
from dolo.algos.dtmscc.value_iteration import solve_policy

groot()
model = yaml_import("examples/models/rbc_mfga.yaml")

model.symbols

drv = solve_policy(model)

ap = model.options['approximation_space']
a = ap['a']
b = ap['b']
orders = ap['orders']

import numpy as np

kvec = np.linspace(a[0],b[0],orders[0])
vals = drv(0, kvec[:,None])

from matplotlib import pyplot as plt
# %matplotlib inline
plt.plot(kvec, vals)
