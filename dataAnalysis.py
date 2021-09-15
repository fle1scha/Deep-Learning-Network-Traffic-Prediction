import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('data.csv', delimiter=',', header=0)
print(data)


# Graph for training time vs epochs
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' simple_lstm_train_time'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' simple_lstm_train_time'].loc[data[' neurons']==100] )
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' bidirectional_lstm_train_time'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' bidirectional_lstm_train_time'].loc[data[' neurons']==100] )
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' stacked_lstm_training_time'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' stacked_lstm_training_time'].loc[data[' neurons']==100] )
plt.legend(['Simple: neurons = 50', 'Simple: neurons = 100', 'Bidirectional: neurons = 50', 'Bidirectional: neurons = 100', 'Stacked: neurons = 50', 'Stacked: neurons = 100'])
plt.ylabel('Training Time (seconds)')
plt.xlabel('Epochs')
plt.xticks(np.arange(25, 150, step=25))  # Set label locations.
plt.title("Training Time vs Epochs")
plt.show()

#Simple Model MAE vs epochs and neurons
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' simple_train_mae'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' simple_val_mae'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' simple_train_mae'].loc[data[' neurons']==100] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' simple_val_mae'].loc[data[' neurons']==100] )
plt.legend(['Train: n = 50', 'Val: n = 50', 'Train: n = 100', 'Val: n = 100'])
plt.ylabel('MAE')
plt.xlabel('Epochs')
plt.xticks(np.arange(25, 150, step=25))  # Set label locations.
plt.title("Simple LSTM MAE vs Epochs")
plt.show()

#Simple Model MSE vs epochs and neurons
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' simple_train_mse'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' simple_val_mse'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' simple_train_mse'].loc[data[' neurons']==100] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' simple_val_mse'].loc[data[' neurons']==100] )
plt.legend(['Train: n = 50', 'Val: n = 50', 'Train: n = 100', 'Val: n = 100'])
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.xticks(np.arange(25, 150, step=25))  # Set label locations.
plt.title("Simple LSTM MSE vs Epochs")
plt.show()

#Simple Model R2 vs epochs and neurons
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' simple_r2'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' simple_val_r2'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' simple_r2'].loc[data[' neurons']==100] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' simple_val_r2'].loc[data[' neurons']==100] )
plt.legend(['Train: n = 50', 'Val: n = 50', 'Train: n = 100', 'Val: n = 100'])
plt.ylabel('R2')
plt.xlabel('Epochs')
plt.xticks(np.arange(25, 150, step=25))  # Set label locations.
plt.title("Simple LSTM R2 vs Epochs")
plt.show()

#Bidirectional Model MAE vs epochs and neurons
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' bidirectional_train_mae'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' bidirectional_val_mae'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' bidirectional_train_mae'].loc[data[' neurons']==100] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' bidirectional_val_mae'].loc[data[' neurons']==100] )
plt.legend(['Train: n = 50', 'Val: n = 50', 'Train: n = 100', 'Val: n = 100'])
plt.ylabel('MAE')
plt.xlabel('Epochs')
plt.xticks(np.arange(25, 150, step=25))  # Set label locations.
plt.title("Bidirectional LSTM MAE vs Epochs")
plt.show()

#bidirectional Model MSE vs epochs and neurons
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' bidirectional_train_mse'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' bidirectional_val_mse'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' bidirectional_train_mse'].loc[data[' neurons']==100] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' bidirectional_val_mse'].loc[data[' neurons']==100] )
plt.legend(['Train: n = 50', 'Val: n = 50', 'Train: n = 100', 'Val: n = 100'])
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.xticks(np.arange(25, 150, step=25))  # Set label locations.
plt.title("Bidirectional LSTM MSE vs Epochs")
plt.show()

#bidirectional Model R2 vs epochs and neurons
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' bidirectional_r2'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' bi_val_r2'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' bidirectional_r2'].loc[data[' neurons']==100] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' bi_val_r2'].loc[data[' neurons']==100] )
plt.legend(['Train: n = 50', 'Val: n = 50', 'Train: n = 100', 'Val: n = 100'])
plt.ylabel('R2')
plt.xlabel('Epochs')
plt.xticks(np.arange(25, 150, step=25))  # Set label locations.
plt.title("Bidirectional LSTM R2 vs Epochs")
plt.show()


#stacked Model MAE vs epochs and neurons
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' stacked_train_mae'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' stacked_val_mae'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' stacked_train_mae'].loc[data[' neurons']==100] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' stacked_val_mae'].loc[data[' neurons']==100] )
plt.legend(['Train: n = 50', 'Val: n = 50', 'Train: n = 100', 'Val: n = 100'])
plt.ylabel('MAE')
plt.xlabel('Epochs')
plt.xticks(np.arange(25, 150, step=25))  # Set label locations.
plt.title("stacked LSTM MAE vs Epochs")
plt.show()

#stacked Model MSE vs epochs and neurons
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' stacked_train_mse'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' stacked_val_mse'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' stacked_train_mse'].loc[data[' neurons']==100] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' stacked_val_mse'].loc[data[' neurons']==100] )
plt.legend(['Train: n = 50', 'Val: n = 50', 'Train: n = 100', 'Val: n = 100'])
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.xticks(np.arange(25, 150, step=25))  # Set label locations.
plt.title("stacked LSTM MSE vs Epochs")
plt.show()

#stacked Model R2 vs epochs and neurons
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' stacked_r2'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' stacked_val_r2'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' stacked_r2'].loc[data[' neurons']==100] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' stacked_val_r2'].loc[data[' neurons']==100] )
plt.legend(['Train: n = 50', 'Val: n = 50', 'Train: n = 100', 'Val: n = 100'])
plt.ylabel('R2')
plt.xlabel('Epochs')
plt.xticks(np.arange(25, 150, step=25))  # Set label locations.
plt.title("stacked LSTM R2 vs Epochs")
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