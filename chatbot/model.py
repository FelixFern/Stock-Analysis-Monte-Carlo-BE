import pickle
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

# Load dataset
file = open('chatbot/files/training.pkl', 'rb')
data = pickle.load(file=file)
file.close()

# Get predictor and target
X = np.array([item[0] for item in data])
y = np.array([item[1] for item in data])

# Define Layers
# Input Layer
inputs = Input(shape=(len(X[0]),))

# Hidden Layers
layer_1 = Dense(units=128)(inputs)
normalize_1 = BatchNormalization(axis=1)(layer_1)
activate_1 = Activation('relu')(normalize_1)
dropout_1 = Dropout(0.25)(activate_1)

layer_2 = Dense(units=64)(dropout_1)
normalize_2 = BatchNormalization(axis=1)(layer_2)
activate_2 = Activation('relu')(normalize_2)
dropout_2 = Dropout(0.25)(activate_2)

layer_3 = Dense(units=64)(dropout_2)
normalize_3 = BatchNormalization(axis=1)(layer_3)
activate_3 = Activation('relu')(normalize_3)

# Output Layer
final_layer = Dense(units=len(y[0]))(activate_3)
outputs = Activation('softmax')(final_layer)

# Final Model
model = Model(inputs=inputs, outputs=outputs, name="chatbot_model")

# Optimization
optimizer = Adam(
    learning_rate=5e-4,
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
    shuffle=True,
    verbose=1
)

model.save('chatbot/model/chatbot_model.h5', hist)
print("Model has been saved successfully.")
