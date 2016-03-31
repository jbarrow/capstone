from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import h5py

model = model_from_json(open('maps_lstm.json').read())
model.load_weights('maps_lstm.h5')

hf = h5py.File('data.h5')
X = hf['X'][24, :, :]
X_test = np.zeros((1,)+X.shape)
X_test[0] = X
p = model.predict(X_test, batch_size=1)

plt.imshow(p[0])
plt.show()
