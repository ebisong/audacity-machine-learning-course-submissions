
import numpy as np
import pandas as pd
from IPython.display import display

# %matplotlib inline

import visuals as vs


in_file =  'titanic_data.csv'
full_data = pd.read_csv(in_file)

display(full_data.head())

outcomes = full_data['Survived']

data = full_data.drop('Survived', axis=1)
display(data.head())


def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """

    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred):

        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean() * 100)

    else:
        return "Number of predictions does not match number of outcomes!"

# Test the 'accuracy_score' function
predictions = pd.Series(np.ones(5, dtype=int))
print(accuracy_score(outcomes[:5], predictions))


def predictions_0(data):
    """ Model with no features. Always predicts a passenger did not survive. """

    predictions = []
    for _, passenger in data.iterrows():
        # Predict the survival of 'passenger'
        predictions.append(0)

    # Return our predictions
    return pd.Series(predictions)


# Make the predictions
predictions = predictions_0(data)
print(accuracy_score(outcomes, predictions))

vs.survival_stats(data, outcomes, 'Sex')

def predictions_1(data):
    """ Model with one feature: 
            - Predict a passenger survived if they are female. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        if passenger['Sex'] == 'female':
            predictions.append(1)
        else:
            predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_1(data)
print (accuracy_score(outcomes, predictions))
vs.survival_stats(data, outcomes, 'Age', ["Sex == 'male'"])

def predictions_2(data):
    """ Model with two features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        if passenger['Sex'] == 'female':
            predictions.append(1)
        elif passenger['Age'] < 10:
            predictions.append(1)
        else:
            predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_2(data)
print (accuracy_score(outcomes, predictions))
vs.survival_stats(data, outcomes, 'Fare', ["Sex == 'male'", "Age > 10"])
vs.survival_stats(data, outcomes, 'SibSp', ["Sex == 'female'"])
def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        if passenger['Sex'] == 'female':
            if passenger['SibSp'] < 3:
                predictions.append(1)
            else:
                predictions.append(0)
        elif passenger['Age'] < 10:
            predictions.append(1)
        elif passenger['Fare'] > 500:
            predictions.append(1)
        else:
            predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_3(data)
print (accuracy_score(outcomes, predictions))
