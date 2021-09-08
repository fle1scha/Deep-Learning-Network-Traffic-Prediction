import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('data.csv', delimiter=',', header=0)
print(data.head())

# Just want to fetch the dataset of 1000 for now
# Graph for training time vs epochs
plt.plot(data.iloc[1:20][' epochs'], data.iloc[1:20][' simple_lstm_train_time'])
plt.plot(data.iloc[1:20][' epochs'], data.iloc[1:20]
         [' stacked_lstm_training_time'])
plt.plot(data.iloc[1:20][' epochs'], data.iloc[1:20]
         [' bidirectional_lstm_train_time'])
plt.xlabel("Epochs")
plt.ylabel("Seconds")
plt.legend(['Simple', 'Bidirectional', 'Stacked'])
plt.title("Number of Epochs vs Training Time")
plt.show()

#Graph for mae and epochs
plt.plot(data.iloc[21:31][' epochs'], data.iloc[21:31]
         [' simple_test_mae'])
plt.plot(data.iloc[21:31][' epochs'], data.iloc[21:31]
         [' simple_train_mae'])
plt.plot(data.iloc[21:31][' epochs'], data.iloc[21:31]
         [' bidirectional_test_mae'])
plt.plot(data.iloc[21:31][' epochs'], data.iloc[21:31]
         [' bidirectional_train_mae'])
plt.plot(data.iloc[21:31][' epochs'], data.iloc[21:31]
         [' stacked_test_mae'])
plt.plot(data.iloc[21:31][' epochs'], data.iloc[21:31]
         [' stacked_train_mae'])
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.legend(['Simple Test', 'Simple Train', 'Bidirectional Test', 'Bidirectional Train', 'Stacked Test', 'Stacked Train'])
plt.title("Number of Epochs vs Test and Training Mean Absolute Error")
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