import numpy as np

from collections import namedtuple


class MixedEncoder:
    Context = namedtuple('Context', ['name', 'values_table', 'median_vals', 'inv_MAD', "categ_opts", "scale", "categorical_ordered", "increasing", "epsilon", "purely_categ"])

    def __init__(self, pandas_dataframe, categorical_order={}, increasing_columns=[], causal_rels=[], epsilons={}):
        """
        Build a training set of dummy encoded variables from existing inputdata

        Assumes categorical values are negative (unless variable is purely categorical, then one value should still be 0)
        For continuous values are reserved all positive values (including 0, in case of continuous or mixed type variables)

        User can add column names of features that cannot decrease in the list increasing_columns
        and column names with lists representing an order of its categorical values
        and causal_rels is a list of 2-tuples with column names that have a causal relationship (so far only "if first gets higher, the second must as well" )
        Lastly, epsilons is a mapping of columns to the smallest increment value, if one can be defined. If not, it is computed from data.
        """
        columns = pandas_dataframe.columns.to_list()
        self.causal_rels = []
        for (c_from, c_to) in causal_rels:
            self.causal_rels.append((columns.index(c_from), columns.index(c_to)))

        self.n_vars = len(columns)
        self.context = np.empty(self.n_vars, dtype=object)
        for i, column in enumerate(columns):
            col_data = pandas_dataframe[column]

            categ_vals = np.minimum(col_data, 0)  # categorical values (0 represents continuous values if any)
            categ_options = np.unique(categ_vals)  # get all indicator values
            if column in categorical_order:
                ordered_categorical = ([0] if 0 in categ_options and 0 not in categorical_order[column] else []) + categorical_order[column]
                assert all(opt in ordered_categorical for opt in categ_options), "Some value in data is not in the ordering"
                categ_options = np.array(ordered_categorical)

            # create table of values of the right size, regardless if first position will be another categorical or continuous
            table_values = np.zeros((categ_options.shape[0], col_data.shape[0]))
            median_vals = np.empty(categ_options.shape[0])
            MAD = np.empty_like(median_vals)  # median absolute deviation

            continuous = col_data[(col_data >= 0)]  # take only continuous values, eventually normalized
            scale = np.nanmax(continuous)  # add also a shift here, if using some values shifted from 0 to better cover the interval [0,1]
            if scale > 0:  # if there are some continuous values
                if column not in epsilons:
                    cont_sorted = np.sort(continuous.copy().to_numpy())
                    diffs = cont_sorted[:-1] - cont_sorted[1:]
                    mindiff = diffs[diffs != 0].min()
                    epsilons[column] = mindiff

                normalized = continuous / scale  # normalize data
                epsilons[column] /= scale

                # set normalized continuous values to the extra first row for continuous
                table_values[0] = np.maximum(col_data / scale, 0)
                table_values[0, np.isnan(table_values[0])] = 1  # replace nan values with 1

                median_vals[0] = np.nanmedian(normalized)  # set median value for continuous data
                MAD[0] = np.nanmedian(np.abs(normalized - median_vals[0]))  # set as median absolute deviation
                if MAD[0] == 0:
                    MAD[0] = 1.48 * np.nanstd(normalized)  # makes it commensurate with MAD
                assert not np.isnan(MAD[0])
                MAD[0] = max(1e-4, MAD[0])  # for numerical stability

                # drop categorical option 0, since it is reserved for continuous if there are any
                categ_options = categ_options[categ_options != 0]
                # start with categorical from position 1
                j = 1
            elif categ_options.shape[0] == 1:  # all values are 0, we assume it is continuous, happens in MNIST
                scale = 1
                epsilons[column] = 0.00001  # we don't know anything about the domain
                table_values[0] = col_data
                table_values[0, np.isnan(table_values[0])] = 1  # replace nan values with 1
                MAD[0] = 1e-4  # for numerical stability
                categ_options = np.array([])
            else:
                # 0 scale is the indicator of purely categorical variable
                scale = 0
                epsilons[column] = 0  # irrelevant value
                j = 0

            # setup categorical
            for d in categ_options:
                table_values[j] = (col_data == d)
                median_vals[j] = np.nanmean(table_values[j])
                MAD[j] = 1.48 * np.nanstd(table_values[j])  # Should be median
                j += 1
            self.context[i] = MixedEncoder.Context(
                name=column, values_table=table_values, median_vals=median_vals,
                inv_MAD=1.0 / MAD, categ_opts=categ_options, scale=scale,
                categorical_ordered=column in categorical_order,
                increasing=column in increasing_columns,
                epsilon=epsilons[column],
                purely_categ=scale == 0,
            )

        self.encoding_size = sum(map(lambda x: x.median_vals.shape[0], self.context))

    def get_encoded_data(self):
        encoded = np.vstack(list(map(lambda x: x.values_table, self.context))).T  # n_rows X all_variables
        return encoded

    def encode_datapoint(self, datapoint):
        encoded = np.zeros(self.encoding_size)
        index = 0
        for data, ctx in zip(datapoint, self.context):
            if ctx.scale == 0:  # fully categorical
                val_i = (ctx.categ_opts == data).argmax()
                encoded[index + val_i] = 1
            else:  # combined or fully continuous
                if data < 0:  # categorical
                    val_i = (ctx.categ_opts == data).argmax()
                    encoded[index + val_i + 1] = 1
                else:
                    encoded[index] = data / ctx.scale
            index += ctx.median_vals.shape[0]
        return encoded
