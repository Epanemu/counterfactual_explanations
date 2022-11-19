from nn_model import NNModel
from counterfactual_generator import CounterfactualGenerator

from data import MixedEncoder
from objectives import compute_entropy, compute_wd

import numpy as np
import pandas as pd


frame = pd.read_csv('adult_frame.csv')
model_path = "model.pt"

# create target binary i.e. {0,1} variable to predict
target = np.asarray(frame['income'] == '>=50k')

# extract input variables used to make prediction
input_data = frame[frame.columns[0:8]]
encoder = MixedEncoder(input_data)
encoded = encoder.get_encoded_data()

# train the NN
model = NNModel(encoded.shape[1], hidden_sizes=[15, 10], output_size=1)
model.load(model_path)

# Create the explanation object and initialise it
cf_generator = CounterfactualGenerator(model, encoder)

data_size = input_data.shape[0]
# data to compute the objective on

classes = [True, False]
class_name = {
    True: ">=50k",
    False: "<50k"
}

scales = np.array([c.scale for c in encoder.context])
discrete_mask = scales == 0
scales[discrete_mask] = 1

data = []
cf_size_data = []
epsilons = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
for sample_size in [10, 50, 100, 500, 1000, 5000]:
    for epsilon in epsilons:
        # sample_size = int(data_size*obj_proportion)
        # print(f"PROPORTION:{obj_proportion} ({sample_size})")
        # print(f"EPSILON:{epsilon}")
        # obj_proportion = 0.001

        rng = np.random.default_rng(42)
        sample_idx = rng.integers(0, data_size, sample_size)
        input_set = input_data.iloc[sample_idx].values
        predictions = model.predict(encoded[sample_idx]).numpy().flatten() >= 0

        cf_set = cf_generator.explain_set(input_set, epsilon=epsilon, n_counterfactuals=100)
        cf_sizes = np.array([len(x) for x in cf_set])
        cf_size_data.append(cf_sizes)
        n_full = (cf_sizes >= 100).sum()
        print("Proportion of full sets:", n_full / sample_size)

        input_set = input_set / scales
        cf_diffs = []
        for fact, cfs in zip(input_set, cf_set):
            diffs = fact - cfs
            # each change in discrete is different, better representation needed, from -1 to -2 is not the same as from -3 to -4
            # except, the base is the same for all on which compute the entropy later, so I do not care at this point
            cf_diffs.append(diffs)
        cf_set = np.array(cf_set, dtype=object) # they are different length, must be treated as objects
        cf_diffs = np.array(cf_diffs, dtype=object) # they are different length, must be treated as objects

        for i, context in enumerate(encoder.context):
            # print(f"{context.name}: ")
            bin_edges = np.linspace(0, 1, 21) # stable bin sizes work better, no data is disregarded (all data used here is normalized to [0,1] range)
            # bin_edges = None
            for output_class in classes:
                # print(f"For class {class_name[output_class]}")
                mask = predictions == output_class

                data_entropy, bin_edges = compute_entropy(input_set[mask, i], is_categorical=(context.scale==0), bin_edges=bin_edges)
                # other
                cf_entropy, _ = compute_entropy(np.concatenate(cf_set[~mask])[:, i], is_categorical=(context.scale==0), bin_edges=bin_edges)
                # print(f"Difference of entropy of original and counterfactuals: {data_entropy - cf_entropy:.2f}")

                # this often collapses to entropy 0 for small samples, not enough variance in the data.
                cf_diff_entropy, _ = compute_entropy(np.concatenate(cf_diffs[mask])[:, i], is_categorical=(context.scale==0))
                # if cf_diff_entropy == 0:
                #     print(f"{context.name}: ")
                #     print(np.concatenate(cf_diffs[mask])[:, i])
                #     print(np.concatenate(cf_set[mask], axis=0)[:, i])
                #     print(input_set[mask, i])

                # wasserstein distance of distribution of countefactualy generated points to the distribution of the original points
                wasserstein = compute_wd(np.concatenate(cf_set[~mask])[:, i], input_set[mask, i], is_categorical=(context.scale==0), bin_edges=bin_edges)
                data.append({
                    "sample_size": sample_size,
                    "epsilon": epsilon,
                    "class": class_name[output_class],
                    "feature": context.name,
                    "diff_of_entropies": data_entropy - cf_entropy,
                    "entropy_of_changes": cf_diff_entropy,
                    "wasserstein_d": wasserstein,
                    "n_maxed_cfs": n_full
                })
                # print(f"Entropy of difference to factual: {cf_diff_entropy:.2f}")
                # print()
        #     print()
        # print()
    size_result = pd.DataFrame(cf_size_data)
    size_result.insert(0, "epsilons", epsilons)
    size_result.to_csv(f"export/cf_sizes_{sample_size}.csv", index=None)
    cf_size_data = []

result = pd.DataFrame(data)
result.to_csv("export/results_all.csv", index=None)
