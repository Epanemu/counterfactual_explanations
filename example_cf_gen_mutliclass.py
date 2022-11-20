from nn_model import NNModel
from counterfactual_generator import CounterfactualGenerator
from textualizer import Textualizer

from data import MixedEncoder
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from tqdm import tqdm

from mnist import MNIST # (this uses mnist-py package)

mnist_data = MNIST()
model_path = "model_mnist_toy.pt"

# partition into train and test, y ~ target, X ~ input data
X_train = mnist_data.train_set.images[:,::100] # take just every 10th pixel, I do not have license for big models right now
X_test = mnist_data.test_set.images[:,::100]
y_train = np.argmax(mnist_data.train_set.labels, axis=1)
y_test = np.argmax(mnist_data.test_set.labels, axis=1)

# extract input variables used to make prediction
encoder = MixedEncoder(pd.DataFrame(X_train)) # MixedEncoder expects a dataframe
encoded = encoder.get_encoded_data()

# train the NN
model = NNModel(encoded.shape[1], hidden_sizes=[], output_size=10)
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

string_vals = {}
textualizer = Textualizer(string_vals, encoder)

# index of datapoint to be explained
i = 0
in_data = X_test[i]
print("Prediction:", np.argmax(model.predict(encoder.encode_datapoint(in_data)).numpy()))
print("True target:", y_test[i])

relative_distance_q = 1
# print(f"Counterfactuals with any other class:")
# counterfactuals = cf_generator.generate_close_counterfactuals(in_data, relative_distance_q, n_limit=100, verbose=True)
# for cf in textualizer.formulate_list(counterfactuals, cf_generator, labels=[str(i) for i in range(10)]):
#     print(cf)


for goal_class in [7]:#range(10):
    print()
    print(f"Counterfactuals with class {goal_class}:")
    counterfactuals = cf_generator.generate_close_counterfactuals(in_data, relative_distance_q, goal_class=goal_class, n_limit=100, verbose=True)
    for cf in textualizer.formulate_list(counterfactuals, cf_generator, labels=[str(i) for i in range(10)]):
        print(cf)
