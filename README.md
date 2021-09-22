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