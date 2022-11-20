import numpy as np

from collections import namedtuple

class MixedEncoder:
    Context = namedtuple('Context', ['name', 'values_table', 'median_vals', 'inv_MAD', "disc_opts", "scale"])

    def __init__(self, pandas_dataframe):
        """
        Build a training set of dummy encoded variables from existing inputdata

        Assumes discrete values are negative (if variable is fully discrete, one value should still be 0)
        For continuous values are reserved all positive values (including 0, in case of continuous or mixed type variables)
        """
        self.n_vars = pandas_dataframe.columns.size
        self.context = np.empty(self.n_vars, dtype=np.object)
        for i in range(pandas_dataframe.columns.shape[0]):
            col_data = pandas_dataframe[pandas_dataframe.columns[i]]

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
            elif discrete_options.shape[0] == 1: # all values are 0, we assume it is continuous, happens in MNIST
                scale = 1
                table_values[0] = col_data
                table_values[0, np.isnan(table_values[0])] = 1 # replace nan values with 1
                MAD[0] = 1e-4 # for numerical stability
                discrete_options = np.array([])
            else:
                # 0 scale is the indicator of solely categorical variable
                scale = 0
                j = 0

            # setup discrete
            for d in discrete_options:
                table_values[j] = (col_data == d)
                median_vals[j] = np.nanmean(table_values[j])
                MAD[j] = 1.48 * np.nanstd(table_values[j])  # Should be median
                j += 1
            self.context[i] = MixedEncoder.Context(name=pandas_dataframe.columns[i], values_table=table_values,
                                median_vals=median_vals, inv_MAD=1.0 / MAD, disc_opts=discrete_options, scale=scale)

        self.encoding_size = sum(map(lambda x: x.median_vals.shape[0], self.context))

    def get_encoded_data(self):
        encoded = np.vstack(list(map(lambda x: x.values_table, self.context))).T # n_rows X all_variables
        return encoded

    def encode_datapoint(self, datapoint):
        encoded = np.zeros(self.encoding_size)
        index = 0
        for data, ctx in zip(datapoint, self.context):
            if ctx.scale == 0: # fully discrete
                val_i = (ctx.disc_opts == data).argmax()
                encoded[index + val_i] = 1
            else: # combined or fully continuous
                if data < 0: # discrete
                    val_i = (ctx.disc_opts == data).argmax()
                    encoded[index + val_i + 1] = 1
                else:
                    encoded[index] = data / ctx.scale
            index += ctx.median_vals.shape[0]
        return encoded
