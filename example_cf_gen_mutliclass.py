from nn_model import NNModel
from counterfactual_generator import CounterfactualGenerator
from textualizer import Textualizer
from data import MixedEncoder

import numpy as np
import pandas as pd

from mnist import MNIST  # (this uses mnist-py package)

mnist_data = MNIST()
model_nn_path = "model_mnist_small.pt"

# partition into train and test, y ~ target, X ~ input data
X_train = mnist_data.train_set.images[:, ::10]  # take just every 10th pixel, I do not have license for big models right now
X_test = mnist_data.test_set.images[:, ::10]
y_train = np.argmax(mnist_data.train_set.labels, axis=1)
y_test = np.argmax(mnist_data.test_set.labels, axis=1)

# extract input variables used to make prediction
encoder = MixedEncoder(pd.DataFrame(X_train))  # MixedEncoder expects a dataframe
encoded = encoder.get_encoded_data()

# train the NN
model_nn = NNModel(encoded.shape[1], hidden_sizes=[15, 30], output_size=10)
to_train = False
if to_train:
    model_nn.train(X_train, y_train, batch_size=128, epochs=50)
    model_nn.save(model_nn_path)
else:
    model_nn.load(model_nn_path)

# print("Train data:")
# model_nn.test(X_train, y_train)
# print("Test data:")
# model_nn.test(X_test, y_test)

# Create the explanation object and initialise it
cf_generator = CounterfactualGenerator(encoder)

string_vals = {}
textualizer = Textualizer(string_vals, encoder)

# index of datapoint to be explained
i = 0
in_data = X_test[i]
# print("Prediction vals:", model_nn.predict(encoder.encode_datapoint(in_data)))
prediction = np.argmax(model_nn.predict(encoder.encode_datapoint(in_data)).numpy())
print("Prediction:", prediction)
print("True target:", y_test[i])

relative_distance_q = 0.1
print("Counterfactuals with any other class:")
# I have observed, that with margin 0 the optimum is at a draw (or almost draw in the numerical unstable way) between 2 or more classes
counterfactuals = cf_generator.generate_close_counterfactuals(in_data, model_nn, relative_distance_q, n_limit=100, verbose=False, cf_margin=0.1)
for cf in textualizer.formulate_list(counterfactuals, labels=[str(i) for i in range(10)]):
    print(cf)

generate_for_all_other_classses = False
if generate_for_all_other_classses:
    for goal_class in filter(lambda k: k != prediction, range(10)):
        print()
        print(f"Counterfactuals with class {goal_class}:")
        counterfactuals = cf_generator.generate_close_counterfactuals(in_data, model_nn, relative_distance_q, goal_class=goal_class, n_limit=100, verbose=False, cf_margin=0.1)
        for cf in textualizer.formulate_list(counterfactuals, labels=[str(i) for i in range(10)]):
            print(cf)
