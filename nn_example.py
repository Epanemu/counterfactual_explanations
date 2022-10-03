from nn_model import NNModel
from nn_counterfactual import NNExplanation

from data import prepare_data
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

frame = pd.read_csv('adult_frame.csv')

# create target binary i.e. {0,1} variable to predict
target = np.asarray(frame['income'] == '>=50k')

# extract input variables used to make prediction
input_data = frame[frame.columns[0:8]]
encoded, context = prepare_data(input_data)

# partition into train and test, y ~ target, X ~ input data
X_train, X_test, y_train, y_test = train_test_split(encoded, target, test_size=0.2, random_state=42)

# train the NN
model = NNModel(encoded.shape[1], hidden_sizes=[15, 10], output_size=1)
model.train(X_train, y_train, batch_size=128, epochs=50)
model.test(X_train, y_train)
model.test(X_test, y_test)

# Create the explanation object and initialise it
exp = NNExplanation(model, encoded.shape[1], context)

exp.string_vals = {'workclass': {0: 'Government', -3: 'Other/Unknown', -2: 'Private', -1: 'Self-Employed'},
                    'education': {0: 'Assoc', -7: 'Bachelors', -6: 'Doctorate', -5: 'HS-grad', -4: 'Masters', -3: 'Prof-school', -2: 'School', -1: 'Some-college'},
                    'marital_status': {0: 'Divorced', -4: 'Married', -3: 'Separated', -2: 'Single', -1: 'Widowed'},
                    'occupation': {0: 'Blue-Collar', -5: 'Other/Unknown', -4: 'Professional', -3: 'Sales', -2: 'Service', -1: 'White-Collar'},
                    'race': {0: 'Non-White', -1: 'White'},
                    'gender': {0: 'Female', -1: 'Male'}}


# index of datapoint to be explained
i = 409
in_data = input_data.iloc[i].values
print("Prediction:", model.predict(exp.mixed_encode(in_data)))

n = 10
print(f"{n} best counterfactuals:")
explanations = exp.generate_explanations(in_data, n, labels=("GOOD","BAD"))
for e in explanations:
    print(e)
