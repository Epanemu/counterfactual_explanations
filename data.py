import numpy as np

from collections import namedtuple
Context = namedtuple('Context', ['name', 'values_table', 'median_vals', 'inv_MAD', "disc_opts", "scale"])

def prepare_data(panda_inputs):
    """
    Build a training set of dummy encoded variables from existing inputdata

    Assumes discrete values are negative (if variable is fully discrete, one value should still be 0)
    For continuous values are reserved all positive values (including 0, in case of continuous or mixed type variables)
    """
    context = np.empty(panda_inputs.columns.size, dtype=np.object)
    for i in range(panda_inputs.columns.shape[0]):
        col_data = panda_inputs[panda_inputs.columns[i]]

        discrete_vals = np.minimum(col_data, 0) # discrete values (0 represents continuous values if any)
        discrete_options = np.unique(discrete_vals) # get all indicator values

        # create table of values of the right size, regardless if first position will be another discrete or continuous
        table_values = np.zeros((discrete_options.shape[0], col_data.shape[0]))
        median_vals = np.empty(discrete_options.shape[0])
        MAD = np.empty_like(median_vals) # median absolute deviation

        continuous = col_data[(col_data >= 0)] # take only continuous values, eventually normalized
        scale = np.nanmax(continuous)
        if scale > 0: # if there are some continuous values
            normalized = continuous / scale # normalize data

            # set normalized continuous values to the extra first row for continuous
            table_values[0] = np.maximum(col_data / scale, 0)
            table_values[0, np.isnan(table_values[0])] = 1 # replace nan values with 1

            median_vals[0] = np.nanmedian(normalized) # set median value for continuous data
            MAD[0] = np.nanmedian(np.abs(normalized - median_vals[0])) # set as median absolute deviation
            if MAD[0] == 0:
                MAD[0] = 1.48 * np.nanstd(normalized) # makes it commensurate with MAD
            assert not np.isnan(MAD[0])
            MAD[0] = max(1e-4, MAD[0]) # for numerical stability

            # drop discrete option 0, since it is reserved for continuous if there are any
            discrete_options = discrete_options[discrete_options != 0]
            # start with discrete from position 1
            j = 1
        else:
            # 0 scale is the indicator of solely discrete variable
            scale = 0
            j = 0

        # setup discrete
        for d in discrete_options:
            table_values[j] = (col_data == d)
            median_vals[j] = np.nanmean(table_values[j])
            MAD[j] = 1.48 * np.nanstd(table_values[j])  # Should be median
            j += 1
        context[i] = Context(name=panda_inputs.columns[i], values_table=table_values,
                             median_vals=median_vals, inv_MAD=1.0 / MAD, disc_opts=discrete_options, scale=scale)

    encoded = np.vstack(list(map(lambda x: x.values_table, context))).T # n_rows X all_variables

    return encoded, context
