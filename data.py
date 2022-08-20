import numpy as np
from types import SimpleNamespace


def prepare_data(panda_inputs):
    """Build a training set of dummy encoded variables from existing input
    data"""
    cox = np.empty(panda_inputs.columns.size, dtype=np.object)
    constraints = np.empty_like(cox)
    for i in range(panda_inputs.columns.shape[0]):
        col = panda_inputs[panda_inputs.columns[i]]
        vals = np.minimum(col, 0)
        normal = col[(col >= 0)]
        scale = normal.max()
        if scale == 0 or np.isnan(scale):
            scale = 1
        normal /= scale
        un = np.unique(vals)
        un = un[un != 0]
        tab = np.zeros((un.shape[0] + 1, vals.shape[0]))
        tab[0] = np.maximum(col / scale, 0)
        med = np.empty(un.shape[0] + 1)
        MAD = np.empty_like(med)
        med[0] = np.median(normal)
        if np.isnan(med[0]):
            med[0] = 1
        MAD[0] = np.median(np.abs(normal - med[0]))
        if MAD[0] == 0:
            MAD[0] = 1.48 * np.std(normal)
        if not (MAD[0] > 0):
            MAD[0] == 1
        if np.isnan(MAD[0]):
            MAD[0] = 1
        tab[0, np.isnan(tab[0])] = 1
        MAD[0] = max(1e-4, MAD[0])
        # print (MAD[0])
        j = 1
        for u in un:
            tab[j] = (col == u)
            med[j] = np.mean(tab[j])
            MAD[j] = 1.48 * np.std(tab[j])  # Should be median
            if not (MAD[j] > 0):
                MAD[j] == 1e-4
            j += 1
        cox[i] = SimpleNamespace(name=panda_inputs.columns[i], total=tab,  # normal=normal,
                                 med=med, MAD=1.0 / MAD, unique=un, scale=scale)
    encoded = np.vstack(list(map(lambda x: x.total, cox))).T

    return encoded, cox, constraints
