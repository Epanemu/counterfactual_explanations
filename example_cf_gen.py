from nn_model import NNModel
from counterfactual_generator import CounterfactualGenerator
from textualizer import Textualizer

from data import MixedEncoder
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from tqdm import tqdm

frame = pd.read_csv('adult_frame.csv')
model_path = "model.pt"

# create target binary i.e. {0,1} variable to predict
target = np.asarray(frame['income'] == '>=50k', dtype=np.float32).reshape(-1, 1)

# extract input variables used to make prediction
input_data = frame[frame.columns[0:8]]
encoder = MixedEncoder(input_data)
encoded = encoder.get_encoded_data()

# partition into train and test, y ~ target, X ~ input data
X_train, X_test, y_train, y_test = train_test_split(encoded, target, test_size=0.2, random_state=42)

# train the NN
model = NNModel(encoded.shape[1], hidden_sizes=[15, 10], output_size=1)
to_train = False
if to_train:
    model.train(X_train, y_train, batch_size=128, epochs=50)
    model.save(model_path)
else:
    model.load(model_path)

# print("Train data:")
# model.test(X_train, y_train)
# print("Test data:")
# model.test(X_test, y_test)

# Create the explanation object and initialise it
cf_generator = CounterfactualGenerator(model, encoder)

string_vals = {'workclass': {0: 'Government', -3: 'Other/Unknown', -2: 'Private', -1: 'Self-Employed'},
               'education': {0: 'Assoc', -7: 'Bachelors', -6: 'Doctorate', -5: 'HS-grad', -4: 'Masters', -3: 'Prof-school', -2: 'School', -1: 'Some-college'},
               'marital_status': {0: 'Divorced', -4: 'Married', -3: 'Separated', -2: 'Single', -1: 'Widowed'},
               'occupation': {0: 'Blue-Collar', -5: 'Other/Unknown', -4: 'Professional', -3: 'Sales', -2: 'Service', -1: 'White-Collar'},
               'race': {0: 'Non-White', -1: 'White'},
               'gender': {0: 'Female', -1: 'Male'}}
textualizer = Textualizer(string_vals, encoder)


# index of datapoint to be explained
i = 0
in_data = input_data.iloc[i].values
print("Prediction:", int(model.predict(encoder.encode_datapoint(in_data)) >= 0))
print("True target:", int(target[i][0]))

# Direct approach to get n closest
n = 3
print(f"{n} best counterfactuals:")
counterfactuals = cf_generator.generate_n_counterfactuals(in_data, n)
for cf in textualizer.formulate_list(counterfactuals, cf_generator, labels=("BAD", "GOOD")):
    print(cf)

print()
# Get n closest with margin on the classifier output
margin = 1
print(f"{n} best counterfactuals with margin {margin} of the classifier value:")
counterfactuals = cf_generator.generate_n_counterfactuals(in_data, n, cf_margin=margin)
for cf in textualizer.formulate_list(counterfactuals, cf_generator, labels=("BAD", "GOOD")):
    print(cf)

get_by_distance = False
relative_distance_q = 1  # number of times the optimal value (~distance the closest counterfactual to the factual)
if get_by_distance:
    print()
    print("Counterfactuals close to optimum:")
    counterfactuals = cf_generator.generate_close_counterfactuals(in_data, relative_distance_q, n_limit=100, verbose=False)
    for cf in textualizer.formulate_list(counterfactuals, cf_generator, labels=("BAD", "GOOD")):
        print(cf)

explain_all = False
if explain_all:
    # THIS IMPLEMENTATION DOES NOT WORK RIGHT WITH TEXTUALIZER, it needs generator with original parameters
    # counterfactuals_list = cf_generator.explain_set(input_data.values, n_counterfactuals=5)
    # counterfactuals_list = cf_generator.explain_set(input_data.values, epsilon=1)
    textual_data = []
    for entry in tqdm(input_data.values):
        counterfactuals = cf_generator.generate_close_counterfactuals(entry, 1, n_limit=5)
        # counterfactuals = cf_generator.generate_n_counterfactuals(entry, 5))
        textual_data.append(textualizer.formulate_list(counterfactuals, cf_generator, labels=("BAD", "GOOD")))
    exp2 = pd.DataFrame(textual_data)
    exp2.to_csv('export/out_all.csv')

custom_change = False
if custom_change:
    in_data = in_data.astype(float)
    in_data[0] = 29.375
    print()
    print("Same data point with a custom change:")
    print("Prediction:", model.predict(encoder.encode_datapoint(in_data)))
    n = 1
    print(f"{n} best counterfactuals:")
    counterfactuals = cf_generator.generate_n_counterfactuals(in_data, n)
    for cf in textualizer.formulate_list(counterfactuals, cf_generator, labels=("BAD", "GOOD")):
        print(cf)
