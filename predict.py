import tensorflow as tf
import numpy as np
import os
import datetime

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

# --------------------- Début de script ---------------------
print(f"[{datetime.datetime.now()}] DÉBUT DU SCRIPT DE PRÉDICTION", flush=True)

# --------------------- Chargement du modèle ---------------------
model_path = './models_training/model_seq_ip.h5'
print(f"[{datetime.datetime.now()}] Chargement du modèle depuis {model_path}", flush=True)
model = tf.keras.models.load_model(model_path, custom_objects={"CustomAveragePooling": CustomAveragePooling})
print(f"[{datetime.datetime.now()}] Modèle chargé avec succès", flush=True)

# --------------------- Chargement des données de test ---------------------
test_dir = './X_input_split_test'
print(f"[{datetime.datetime.now()}] Chargement des fichiers depuis {test_dir}", flush=True)

X_test = np.load(os.path.join(test_dir, 'X_input_0.npy'))
print("Fichier X_input_0.npy chargé", flush=True)

for i in range(1, 20):
    file_path = os.path.join(test_dir, f'X_input_{i}.npy')
    print(f"Chargement {file_path}", flush=True)
    X_test = np.concatenate((X_test, np.load(file_path)))
    print(f"Fichier {i} chargé", flush=True)

X_test = X_test.astype(np.float32)
print(f"[{datetime.datetime.now()}] Données de test chargées avec shape : {X_test.shape}", flush=True)

# --------------------- Chargement des labels ---------------------
Y_test_path = os.path.join(test_dir, 'Y_test.npy')
Y_test = np.load(Y_test_path)
print(f"[{datetime.datetime.now()}] Y_test chargé avec shape : {Y_test.shape}", flush=True)

# --------------------- Prédictions ---------------------
print(f"[{datetime.datetime.now()}] Lancement des prédictions...", flush=True)
logits = model.predict(X_test, batch_size=128)
Y_pred = np.argmax(logits, axis=1)
print(f"[{datetime.datetime.now()}] Prédictions terminées", flush=True)

# --------------------- Sauvegarde ---------------------
np.save('./Y_prediction.npy', Y_pred)
print(f"[{datetime.datetime.now()}] Prédictions sauvegardées dans Y_prediction.npy", flush=True)
