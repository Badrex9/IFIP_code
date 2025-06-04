import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.utils import class_weight
import datetime

# --------------------- Début de script ---------------------
print(f"[{datetime.datetime.now()}] DÉBUT DU SCRIPT PYTHON", flush=True)

# --------------------- Pooling personnalisé ---------------------
class CustomAveragePooling(tf.keras.layers.Layer):
    def __init__(self, group_size, weights_matrix, **kwargs):
        super(CustomAveragePooling, self).__init__(**kwargs)
        self.group_size = group_size
        self.weights_matrix = tf.constant(weights_matrix, dtype=tf.float32)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        H, W, C = input_shape[-3], input_shape[-2], input_shape[-1]
        W_out = W // self.group_size
        W_truncated = W_out * self.group_size
        inputs = inputs[:, :, :W_truncated, :]
        inputs_grouped = tf.reshape(inputs, (-1, H, W_out, self.group_size, C))
        weights_matrix_expanded = tf.reshape(self.weights_matrix, (1, 1, 1, self.group_size, 1))
        outputs = tf.reduce_sum(inputs_grouped * weights_matrix_expanded, axis=-2)
        return outputs

# --------------------- Dataset avec shuffle simple ---------------------
def get_shuffled_dataset(X, Y, batch_size=256, seed=42):
    print(f"[{datetime.datetime.now()}] Shuffle simple des données", flush=True)
    indices = np.arange(len(X))
    np.random.seed(seed)
    np.random.shuffle(indices)

    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    ds = tf.data.Dataset.from_tensor_slices((X_shuffled, Y_shuffled))
    ds = ds.repeat()  # ← ajoute cette ligne pour permettre plusieurs époques
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# --------------------- Construction du modèle ---------------------
def build_model():
    print(f"[{datetime.datetime.now()}] Construction du modèle", flush=True)

    weights_matrix_first = [0.6, 0.3, 0.1]
    weights_matrix_second = [0.35, 0.25, 0.2, 0.1, 0.1]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (1, 3), activation='relu', input_shape=(82, 20, 1)))
    model.add(CustomAveragePooling(group_size=3, weights_matrix=weights_matrix_first))
    model.add(tf.keras.layers.Conv2D(16, (1, 2), activation='relu', padding='same'))
    model.add(CustomAveragePooling(group_size=5, weights_matrix=weights_matrix_second))
    model.add(tf.keras.layers.Conv2D(16, (1, 2), activation='relu', padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(15))  # À adapter si ≠ 15 classes

    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print(f"[{datetime.datetime.now()}] Modèle compilé", flush=True)
    return model

# --------------------- Chargement des données ---------------------
print(f"[{datetime.datetime.now()}] -------------------- Chargement des données --------------------", flush=True)
X_input = np.load('./X_input_split_train/X_input_0.npy')
print("Fichier 0 chargé", flush=True)

for i in range(1, 10):
    print(f"Chargement X_input_{i}", flush=True)
    X_input = np.concatenate((X_input, np.load(f'./X_input_split_train/X_input_{i}.npy')))
    print(f"Fichier {i} chargé", flush=True)

Y = np.load('./X_input_split_train/Y.npy')[:np.shape(X_input)[0]]

X_input = X_input.astype(np.float32)
Y = Y.astype(np.int32)

print("Taille de X_input :", X_input.shape, flush=True)
print("Taille de Y :", Y.shape, flush=True)
np.set_printoptions(threshold=np.inf, linewidth=1000)
for i in range(1,10):
	print("Valeur X_input: ", X_input[i, 10], flush=True)
# --------------------- Calcul des poids de classes ---------------------
#print(f"[{datetime.datetime.now()}] Calcul des class_weights...", flush=True)
#class_weights_array = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
#class_weights_dict = dict(enumerate(class_weights_array))
#print(f"class_weights : {class_weights_dict}", flush=True)

# --------------------- Préparation du dataset ---------------------
batch_size = 128
#ds = get_shuffled_dataset(X_input, Y, batch_size=batch_size)
#steps = len(X_input) // batch_size
#print(f"Dataset tf.data créé avec {steps} steps/epoch", flush=True)

# --------------------- Entraînement du modèle ---------------------
print(f"[{datetime.datetime.now()}] -------------------- Entrainement du modèle --------------------", flush=True)
model = build_model()
model.fit(X_input, Y, batch_size=batch_size, epochs=75)#, class_weight=class_weights_dict)
model.save('./models_training/model_seq_ip.h5')
print(f"[{datetime.datetime.now()}] Modèle sauvegardé dans ./models_training/model_seq_ip.h5", flush=True)
