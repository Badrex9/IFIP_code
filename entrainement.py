import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.models import load_model

class CustomAveragePooling(tf.keras.layers.Layer):
    def __init__(self, group_size, weights_matrix, **kwargs):
        """
        group_size : Taille du regroupement en colonnes pour le pooling
        weights_matrix : Matrice avec les pondérations
        """
        super(CustomAveragePooling, self).__init__(**kwargs)
        self.group_size = group_size
        # Convertir la matrice de poids en tensor pour s'assurer que c'est une constante TensorFlow
        self.weights_matrix = tf.constant(weights_matrix, dtype=tf.float32)

    def call(self, inputs):
        """
        Implémentation de la logique du pooling avec les poids sur les groupes de colonnes pour chaque canal.
        inputs : Entrées du modèle, tensor de forme [batch_size, height, width, channels]
        """
        # Obtenir la taille de l'entrée dynamique
        input_shape = tf.shape(inputs)
        H, W, C = input_shape[-3], input_shape[-2], input_shape[-1]  # Extraire dynamiquement H, W, et C

        # Vérifier dynamiquement que la largeur est divisible par la taille du groupe
        condition = tf.math.mod(W, self.group_size) == 0
        tf.debugging.assert_equal(condition, True, message="La largeur de l'entrée doit être divisible par group_size.")

        # Regrouper les colonnes par la taille du groupe pour appliquer le pooling
        W_out = W // self.group_size
        # Reshape pour créer les groupes sur la dimension des colonnes
        inputs_grouped = tf.reshape(inputs, (-1, H, W_out, self.group_size, C))  # [batch_size, H, W_out, group_size, channels]

        # Reshape la weights_matrix pour permettre le broadcasting
        # La pondération doit s'appliquer uniquement sur la dimension group_size, pas sur les canaux
        weights_matrix_expanded = tf.reshape(self.weights_matrix, (1, 1, 1, self.group_size, 1))  # [1, 1, 1, group_size, 1]

        # Appliquer la pondération avec broadcasting sur le dernier axe de la matrice
        outputs = tf.reduce_sum(inputs_grouped * weights_matrix_expanded, axis=-2)  # Somme sur la dimension group_size

        return outputs




def train_model(X,Y, epochs=20):
    # Définir la matrice de pondération
    weights_matrix_first = [0.6, 0.3, 0.1]
    weights_matrix_second = [0.35, 0.25, 0.2, 0.1, 0.1]
    # Ajout de la couche avec la pondération
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (1, 3), activation='relu', input_shape=(82, 20, 1)))
    model.add(CustomAveragePooling(group_size=3, weights_matrix=weights_matrix_first))
    model.add(tf.keras.layers.Conv2D(16, (1, 2), activation='relu'))
    model.add(CustomAveragePooling(group_size=5, weights_matrix=weights_matrix_second))
    model.add(tf.keras.layers.Conv2D(16, (1, 2), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(15))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    #print(sys.getsizeof(model))
    model.fit(X, Y, epochs=epochs, batch_size=256)#, batch_size=65536)
    model.save('./models_training/model_seq_ip.h5')
    return model

d_historique = 20

print("--------------------Chargement des données train--------------------")
X_input = np.load('./X_input_split_train/X_input_0.npy')
d_model = np.shape(X_input)[1]
for i in range(1, 20):
    print(i)
    X_input = np.concatenate((X_input, np.load('./X_input_split_train/X_input_' + str(i) + '.npy')))

Y = np.load('./X_input_split_train/Y.npy')[:np.shape(X_input)[0]]  # Ajuster la taille de Y

print("Taille de X_input :", X_input.shape)
print("Taille de Y :", Y.shape)

print("--------------------Fin du chargement des données--------------------")
print("--------------------Entrainement du modèle--------------------")
model = train_model(X_input, Y, epochs=75)
