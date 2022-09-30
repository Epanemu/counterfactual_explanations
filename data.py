import numpy as np
from types import SimpleNamespace


def prepare_data(panda_inputs):
    """Build a training set of dummy encoded variables from existing input
    data"""
    context = np.empty(panda_inputs.columns.size, dtype=np.object)
    constraints = np.empty_like(context)
    for i in range(panda_inputs.columns.shape[0]):
        col_data = panda_inputs[panda_inputs.columns[i]]
        discrete_vals = np.minimum(col_data, 0) # discrete values (0 represents continuous values if any)
        normalized = col_data[(col_data >= 0)] # take only continuous values, eventually normalized
        scale = normalized.max()
        if scale == 0 or np.isnan(scale): # if no continuous values at all
            scale = 1
        normalized /= scale # normalize data
        discrete_options = np.unique(discrete_vals) # get all indicator values
        discrete_options = discrete_options[discrete_options != 0] # get all that represent discrete values
        table_values = np.zeros((discrete_options.shape[0] + 1, discrete_vals.shape[0]))
        table_values[0] = np.maximum(col_data / scale, 0) # set normalized continuous values to the extra row for continuous
        table_values[0, np.isnan(table_values[0])] = 1 # replace nan values with 1

        median_vals = np.empty(discrete_options.shape[0] + 1)
        MAD = np.empty_like(median_vals) # median absolute deviation
        median_vals[0] = np.median(normalized) # set median value for continuous data
        if np.isnan(median_vals[0]): # not sure when this happens
            median_vals[0] = 1
        MAD[0] = np.median(np.abs(normalized - median_vals[0])) # set as median absolute deviation
        if MAD[0] == 0:
            MAD[0] = 1.48 * np.std(normalized) # makes it commensurate with MAD
        if not (MAD[0] > 0):
            MAD[0] == 1 # this line literally does not do anything, remained here only because it was here in the original
        if np.isnan(MAD[0]):
            MAD[0] = 1
        MAD[0] = max(1e-4, MAD[0]) # for numerical stability

        # setup discrete
        j = 1
        for d in discrete_options:
            table_values[j] = (col_data == d)
            median_vals[j] = np.mean(table_values[j])
            MAD[j] = 1.48 * np.std(table_values[j])  # Should be median
            if not (MAD[j] > 0):
                MAD[j] == 1e-4 # again, this line literally does not do anything
            j += 1
        context[i] = SimpleNamespace(name=panda_inputs.columns[i], values_table=table_values,  # normalized=normalized,
                                 median_vals=median_vals, inv_MAD=1.0 / MAD, disc_opts=discrete_options, scale=scale)
    # this whole thing is not consistent for solely discrete variables, that are set to have one of the values to 0,
    # taking the theoretical place of the continuous values, but the datapoints represented by 0 stays zero, instead of one hot encoding
    # in further encoding, though, this is acounted for.
    encoded = np.vstack(list(map(lambda x: x.values_table, context))).T # n_rows X all_variables

    return encoded, context, constraints
