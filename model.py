import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, SGD

# Load dataset
file = open('training.pkl', 'rb')
data = pickle.load(file)
file.close()

# Get predictor and target
X = np.array([item[0] for item in data])
y = np.array([item[1] for item in data])

# Model
model = Sequential()
optimizer = Adam()

# Hidden Layer
model.add(Dense(
    units=32,
    input_shape=(10,),
    activation='relu'
))
model.add(Dense(
    units=32,
    input_shape=(10,),
    activation='relu'
))
model.add(Dense(
    units=32,
    input_shape=(10,),
    activation='relu'
))

# Output Layer
model.add(
    units=1,
    name='output_layer',
    activation='sigmoid'
)

# Evaluate Model
model.compile(
    loss='mean_absolute_error',
    optimizer=optimizer,
    metric=['mean_absolute_error']
)