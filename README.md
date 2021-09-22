## Deep Learning for Network Traffic Prediction
The Deep Learning for Network Traffic Prediction (DL4NTP) system is split into two files: DL4NTP.py, which preprocceses the data and implements the models; and dataAnalysis.py, which extracts data and metrics from the models so that the results can be evaluated.

1. To run the DL4NTP.py and dataAnalysis.py program, a user should have the following packages installed:
    ```
    import tensorflow as tf
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    import datetime as dt
    import time
    import csv
    import seaborn as sns
    import keras.backend as K
    ```

2. To run either file, enter the following command in a terminal application:
    ```
    python3 DL4NTP.py
    ```
    or
    ```
    python3 dataAnalysis.py
    ```

    Note: the dataAnalysis file will not run succesfully if the DL4NTP program has not completed LSTM model training.

3. The user will be asked for prompts in the command line as the program run. Example answers to each input are provided below:
```
Would you like to view the preliminary statistical analysis plots? [Y/N]
Y
```
When 'Y' is inputted, various graphs will pop up as the preliminary statical analysis methods execute. Please close each figure to continue. 

```
View the split of training and test data? [Y/N]
Y
```
```
You will now be asked to define the ranges of the LSTM hyperparameter grid search.
Please enter the first epoch gridsearch hyperparameter value: 5

Please enter another epoch gridsearch hyperparameter value or type DONE to continue: DONE
```
As the model training process is only run once, the range of hyperparameters that the grid search uses are defined at the beginning the program, whilst the optimal model selection hyperparameters are defined through user input after training. 

```
Enter selected optimal simple LSTM epoch hyperparameter: 1 
Enter selected optimal simple LSTM neuron hyperparameter: 1000
Simple LSTM model succesfully defined.
```
The output of each model's prediction will be shown in a plot. Please close it to continue with the prediction of the next model. 

5. The dataAnalysis.py file sets up plots of MAE, MSE and R2 across the range of hyperparameters. The x-ticks of each plot have been coded according to the hyperparameter ranges of the actual project. If a graph is not showing on your hyperparameter ranges, you can adjust the ```plt.xticks()``` function to match the range of your inputs. 
