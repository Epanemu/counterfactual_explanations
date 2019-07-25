#Load data
import numpy as np
import pandas as pd
method=['FICO','adult'][1]
data_dir=''#'../data/FICO/'
if method=='FICO':
    frame=pd.read_csv(data_dir+"heloc_dataset_v1.csv")
    train=np.arange(frame.shape[0])%10!=0
    test=np.bitwise_not(train)
    target=np.asarray(frame['RiskPerformance']=='Good')
    inputs=frame[frame.columns[1:]]
else:
    from sklearn.model_selection import ShuffleSplit
    frame = pd.read_csv('adult_frame.csv')
    train, test = next(ShuffleSplit(test_size=0.20, random_state=17).split(frame))
    target=np.asarray(frame['income']=='>=50k')
    inputs=frame[frame.columns[0:8]]
    
#Create the object and initialise it
from counterfactual import linear_explanation, plot_coefficients
exp=linear_explanation()
exp.encode_pandas(inputs)
#train the logistic regression classifier
exp.train_logistic(target[train],train)
if method =='adult': #Modify the pretty printer for special values.
    exp.special_val={'workclass': {0: 'Government', -3: 'Other/Unknown', -2: 'Private', -1: 'Self-Employed'}, 'education': {0: 'Assoc', -7: 'Bachelors', -6: 'Doctorate', -5: 'HS-grad', -4: 'Masters', -3: 'Prof-school', -2: 'School', -1: 'Some-college'}, 'marital_status': {0: 'Divorced', -4: 'Married', -3: 'Separated', -2: 'Single', -1: 'Widowed'}, 'occupation': {0: 'Blue-Collar', -5: 'Other/Unknown', -4: 'Professional', -3: 'Sales', -2: 'Service', -1: 'White-Collar'}, 'race': {0: 'Other', -1: 'White'}, 'gender': {0: 'Female', -1: 'Male'}}

#if only a subset of the data is to be used for training use
#the following line instead
#exp.train_logistic(target[train],train)

i =409 #Explain 410th test entry
text=exp.explain_entry(inputs.iloc[i].values,10)
#The explanation is an ordered list of text strings
#print them
for t in text:
    print (t)
    
#generate a full set of explanations for all data
explain_all=False
if explain_all:
    explanations=exp.explain_set(inputs.values,12)
    exp2=pd.DataFrame(explanations)
    exp2.to_csv('test.csv')
#visualise the set of linear weights ordered by their median contribution over the dataset.
plot_coefficients(exp,med_weight=True)
