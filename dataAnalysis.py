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
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' simple_val_r2'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' simple_val_r2'].loc[data[' neurons']==100] )
plt.legend(['Val: n = 50','Val: n = 100'])
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
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' bi_val_r2'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' bi_val_r2'].loc[data[' neurons']==100] )
plt.legend(['Val: n = 50', 'Val: n = 100'])
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
plt.plot(data[' epochs'].loc[data[' neurons']==50] , data[' stacked_val_r2'].loc[data[' neurons']==50] )
plt.plot(data[' epochs'].loc[data[' neurons']==100] , data[' stacked_val_r2'].loc[data[' neurons']==100] )
plt.legend(['Val: n = 50', 'Val: n = 100'])
plt.ylabel('R2')
plt.xlabel('Epochs')
plt.xticks(np.arange(25, 150, step=25))  # Set label locations.
plt.title("stacked LSTM R2 vs Epochs")
plt.show()

print(data[' simple_lstm_train_time',' bidirectional_lstm_train_time', ' stacked_lstm_training_time'])