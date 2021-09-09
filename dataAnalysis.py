import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read in data, first row is our headers and data is sperated with a comma
data = pd.read_csv('data.csv', delimiter=',', header=0)

# Here we looking at 200 epochs, for each dataset size
# We want to see how training time changes
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(
    axis=0)[19, 29, 40, 50, 60][' simple_lstm_train_time'])
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(axis=0)[19, 29, 40, 50, 60]
         [' stacked_lstm_training_time'])
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(axis=0)[19, 29, 40, 50, 60]
         [' bidirectional_lstm_train_time'])
plt.xlabel("Dataset Size")
plt.ylabel("Seconds")
plt.legend(['Simple', 'Stacked', 'Bidirectional'])
plt.title("Dataset Size vs Training Time")
plt.show()

# Here we looking at 200 epochs, for each dataset size
# We want to see how prediction time changes
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(
    axis=0)[19, 29, 40, 50, 60][' simple_lstm_prediction_time'])
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(axis=0)[19, 29, 40, 50, 60]
         [' stacked_lstm_prediction_time'])
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(axis=0)[19, 29, 40, 50, 60]
         [' bidirectional_lstm_prediction_time'])
plt.xlabel("Dataset Size")
plt.ylabel("Seconds")
plt.legend(['Simple', 'Stacked', 'Bidirectional'])
plt.title("Dataset Size vs Prediction Time")
plt.show()

# Here we looking at 200 epochs, for each dataset size
# We want to see how test mae changes
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(
    axis=0)[19, 29, 40, 50, 60][' simple_test_mae'] * 1000000)
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(axis=0)[19, 29, 40, 50, 60]
         [' bidirectional_test_mae'] * 1000000)
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(axis=0)[19, 29, 40, 50, 60]
         [' stacked_test_mae'] * 1000000)
plt.xlabel("Dataset Size")
plt.ylabel("Mean Absolute Error")
plt.legend(['Simple', 'Bidirectional', 'Stacked'])
plt.title("Dataset Size vs Prediction Accuracy (MAE)")
plt.show()

# Here we looking at 200 epochs, for each dataset size
# We want to see how test mse changes
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(
    axis=0)[19, 29, 40, 50, 60][' simple_test_mse'] * 1000000)
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(axis=0)[19, 29, 40, 50, 60]
         [' bidirectional_test_mse'] * 1000000)
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(axis=0)[19, 29, 40, 50, 60]
         [' stacked_test_mse'] * 1000000)
plt.xlabel("Dataset Size")
plt.ylabel("Mean Squared Error")
plt.legend(['Simple', 'Bidirectional', 'Stacked'])
plt.title("Dataset Size vs Prediction Accuracy (MSE)")
plt.show()

# We know from looking at mse and mae predictions that 16000 is the sweet spot in terms of dataset size
# What number of epochs though?!
plt.plot(data.iloc[41:51][' epochs'], data.iloc[41:51]
         [' simple_test_mse'] * 1000000)
plt.plot(data.iloc[41:51][' epochs'], data.iloc[41:51]
         [' bidirectional_test_mse'] * 1000000)
plt.plot(data.iloc[41:51][' epochs'], data.iloc[41:51]
         [' stacked_test_mse'] * 1000000)
plt.xlabel("Number of Epochs")
plt.ylabel("Mean Squared Error")
plt.legend(['Simple', 'Bidirectional', 'Stacked'])
plt.title("Epochs vs Prediction Accuracy (MSE)")
plt.show()

plt.plot(data.iloc[41:51][' epochs'], data.iloc[41:51]
         [' simple_test_mae'] * 1000000)
plt.plot(data.iloc[41:51][' epochs'], data.iloc[41:51]
         [' bidirectional_test_mae'] * 1000000)
plt.plot(data.iloc[41:51][' epochs'], data.iloc[41:51]
         [' stacked_test_mae'] * 1000000)
plt.xlabel("Number of Epochs")
plt.ylabel("Mean Squared Error")
plt.legend(['Simple', 'Bidirectional', 'Stacked'])
plt.title("Epochs vs Prediction Accuracy (MAE)")
plt.show()

# Show that test mae is higher than the training mae
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(
    axis=0)[19, 29, 40, 50, 60][' simple_test_mse'] * 1000000)
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(axis=0)[19, 29, 40, 50, 60]
         [' bidirectional_test_mse'] * 1000000)
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(axis=0)[19, 29, 40, 50, 60]
         [' stacked_test_mse'] * 1000000)
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(
    axis=0)[19, 29, 40, 50, 60][' simple_train_mse'] * 1000000)
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(axis=0)[19, 29, 40, 50, 60]
         [' bidirectional_train_mse'] * 1000000)
plt.plot(data.loc(axis=0)[19, 29, 40, 50, 60]['dataset_size'], data.loc(axis=0)[19, 29, 40, 50, 60]
         [' stacked_train_mse'] * 1000000)
plt.xlabel("Dataset Size")
plt.ylabel("Mean Squared Error")
plt.legend(['Simple Prediction', 'Bidirectional Prediction', 'Stacked Prediction', 'Simple Training', 'Bidirectional Training', 'Stacked Training'])
plt.title("Comparing Training and Prediction MSE")
plt.show()

#Graph for mse and epochs
plt.plot(data.iloc[1:20][' epochs'], data.iloc[1:20]
         [' simple_test_mse'])
plt.plot(data.iloc[1:20][' epochs'], data.iloc[1:20]
         [' simple_train_mse'])
plt.plot(data.iloc[1:20][' epochs'], data.iloc[1:20]
         [' bidirectional_test_mse'])
plt.plot(data.iloc[1:20][' epochs'], data.iloc[1:20]
         [' bidirectional_train_mse'])
plt.plot(data.iloc[1:20][' epochs'], data.iloc[1:20]
         [' stacked_test_mse'])
plt.plot(data.iloc[1:20][' epochs'], data.iloc[1:20]
         [' stacked_train_mse'])
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.legend(['Simple Test', 'Simple Train', 'Bidirectional Test',
            'Bidirectional Train', 'Stacked Test', 'Stacked Train'])
plt.title("Number of Epochs vs Test and Training Mean Squared Error")
plt.show()
