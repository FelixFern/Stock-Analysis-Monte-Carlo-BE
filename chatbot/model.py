import pickle
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

# Load dataset
file = open('training.pkl', 'rb')
data = pickle.load(file=file)
file.close()

# Get predictor and target
X = np.array([item[0] for item in data])
y = np.array([item[1] for item in data])

# Model
model = Sequential(
    layers=(
    # Hidden Layer
    # 1
    Dense(units=128, input_shape=(len(X[0]),)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.25),

    # 2
    Dense(units=64),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.25),

    # 3
    Dense(units=64),
    BatchNormalization(),
    Activation('relu'),

    # Output Layer
    Dense(units=len(y[0])),
    Activation('softmax')
    )
)

# Optimization
optimizer = Adam(
    learning_rate=0.005,
    beta_1=0.9,
    beta_2=0.999
)

# Evaluate Model
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Fit & Save Model
hist = model.fit(
    X, y, 
    epochs=200,
    batch_size=5,
    verbose=1
)

model.save('chatbot_model.h5', hist)
print("Model has been saved successfully.")
