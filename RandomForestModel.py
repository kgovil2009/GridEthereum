import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from GetAccountData import Paid_rate_division, AccountData
import matplotlib.pyplot as plt

"""
Random Forest training model is implemented here. The data is gathered from GetAccountData.
"""

class RandomForestEvaluation:
    def randomForestModel(self):
        features, mean_nonponzi_data, mean_ponzi_data = AccountData.getAccountData()

        print(features)
        # Labels are the values we want to predict
        labels = np.array(features['Ponzi'])
        # Remove the labels from the features
        # axis 1 refers to the columns
        features = features.drop('Ponzi', axis=1)
        features = features.drop('Address', axis=1)
        # Saving feature names for later use
        feature_list = list(features.columns)
        # Convert to numpy array
        features = np.array(features)

        # Split the data into training and testing sets
        train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                  random_state=42)

        print('Training Features Shape:', train.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test.shape)
        print('Testing Labels Shape:', test_labels.shape)
        # print(test_labels)

        train = train.astype('bool')
        train_labels = train_labels.astype('bool')
        test = test.astype('bool')
        test_labels = test_labels.astype('bool')

        # Instantiate model with 1000 decision trees
        rf = RandomForestClassifier(n_estimators=1000, random_state=42)
        # Train the model on training data
        rf.fit(train, train_labels)

        # Use the forest's predict method on the test data
        predictions = rf.predict(test)
        # print(predictions)
        # Calculate the absolute errors
        errors = abs(predictions ^ test_labels)
        # print(errors)

        # Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

        # Calculate mean absolute percentage error (MAPE)
        mape = 100 * round(np.mean(errors), 2)

        # Calculate and display accuracy
        accuracy = 100 - np.mean(mape)
        print('Accuracy:', round(accuracy, 2), '%.')

        n_nodes = []
        max_depths = []

        # Stats about the trees in random forest
        for ind_tree in rf.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)

        print(f'Average number of nodes {int(np.mean(n_nodes))}')
        print(f'Average maximum depth {int(np.mean(max_depths))}')

        # Training predictions (to demonstrate overfitting)
        train_rf_predictions = rf.predict(train)
        train_rf_probs = rf.predict_proba(train)[:, 1]

        # Testing predictions (to determine performance)
        rf_predictions = rf.predict(test)
        rf_probs = rf.predict_proba(test)[:, 1]

        # Plot formatting
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.size'] = 18

        return rf, mean_ponzi_data, mean_nonponzi_data, feature_list
