from model import LRModel
from counterfactual import LinearExplanation
# from visualizers import plot_coefficients
from data import prepare_data

import numpy as np
import pandas as pd

dataset = ['FICO', 'adult'][1]
data_dir = ''  # '../data/FICO/'

if dataset == 'FICO':
    # FICO case
    # load dataset
    frame = pd.read_csv(data_dir + "heloc_dataset_v1.csv")
    # partition into train and test
    train = np.arange(frame.shape[0]) % 10 != 0
    test = np.bitwise_not(train)
    # create target binary i.e. {0,1} variable to predict
    target = np.asarray(frame['RiskPerformance'] == 'Good')
    # extract input variables used to make prediction
    inputs = frame[frame.columns[1:]]
else:
    # Adult dataset
    # load dataset
    frame = pd.read_csv('adult_frame.csv')
    # partition into train and test
    from sklearn.model_selection import ShuffleSplit
    train, test = next(ShuffleSplit(
        test_size=0.20, random_state=17).split(frame))
    # create target binary i.e. {0,1} variable to predict
    target = np.asarray(frame['income'] == '>=50k')
    # extract input variables used to make prediction
    inputs = frame[frame.columns[0:8]]

encoded, context, constraints = prepare_data(inputs)
# train the logistic regression classifier
model = LRModel(encoded, target[train], train)
# if all the data is to be used for training use
# the following line instead
# LRModel(target)

# Create the explanation object and initialise it
exp = LinearExplanation(model, encoded, context, constraints)

if dataset == 'adult':  # Modify the pretty printer for special values.
    exp.special_val = {'workclass': {0: 'Government', -3: 'Other/Unknown', -2: 'Private', -1: 'Self-Employed'},
                       'education': {0: 'Assoc', -7: 'Bachelors', -6: 'Doctorate', -5: 'HS-grad', -4: 'Masters', -3: 'Prof-school', -2: 'School', -1: 'Some-college'},
                       'marital_status': {0: 'Divorced', -4: 'Married', -3: 'Separated', -2: 'Single', -1: 'Widowed'},
                       'occupation': {0: 'Blue-Collar', -5: 'Other/Unknown', -4: 'Professional', -3: 'Sales', -2: 'Service', -1: 'White-Collar'},
                       'race': {0: 'Non-White', -1: 'White'},
                       'gender': {0: 'Female', -1: 'Male'}}
# If dataset is 'Fico' exp.special_value is already correctly set up.
# Note that special values can either be handled differently for each named feature as in the above example or set generically.
# For example, FICO uses
#   special_val={-9.0:",i.e. No Bureau Record or No Investigation,",
#                 -8.0:",i.e. No Usable/Valid Accounts Trades or Inquiries,",
#                 -7.0:",i.e. Condition not Met ,"}
# As the same special values are consistent across all features


n = 10
print(f"Original variant (by Chris Russel) set to give upto {n} explanations")
i = 409  # Explain 410th test entry
text = exp.explain_entry(inputs.iloc[i].values, n)
# The explanation is an ordered list of text strings
# print them
for t in text:
    print(t)

# generate a full set of explanations for all data
explain_all = False
if explain_all:
    explanations = exp.explain_set(inputs.values, 12)
    exp2 = pd.DataFrame(explanations)
    exp2.to_csv('test.csv')
# visualise the set of linear weights ordered by their median contribution over the dataset.
# plot_coefficients(exp, med_weight=True)

print()
print(f"Improved variant ({n} best solutions):")
explanations = exp.explain_entry_better(inputs.iloc[i].values, n)
for e in explanations:
    print(e)
