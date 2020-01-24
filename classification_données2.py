from tabopen import *
import theano
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D, convolutional
# from keras.datasets import mnist

def create_one_hot_vec(tab, max_val) :
    """
    :param tab: tableau d'entiers
    :param max_val: {5, 100} Valeur maximale du tableau.
        Pour un tableau de valeurs continues, prendre max_val = 100,
        pour un tableau de valeurs allant de 0 à 4, prendre max_val = 5
    :return: tab_out : tableau en "one-hot vector".
    """
    tab_out = np.zeros(tab.size) #crée un tableau de la taille necessaire
    for i in range (len(tab)) :
        tab_out[i] = tab[i]-1
    tab_out = np_utils.to_categorical(tab_out, max_val) #Transformation en one-hot vector (que de '0' sauf un '1' à l'endroit voulu)
    return tab_out



tab_résultats = création_orga_tab(path="Scores_continuous.txt")
tab_données = création_orga_tab(path="Scores_discrete.txt")

#Les deux tableaux sont maintenant des vecteurs triées. Leur taille doit être réglée.
tab_données = tab_données.tolist()
tab_données = allonge_list(tab_données, tab_résultats.__len__())

tab_résultats = tab_résultats.tolist()

#Rangement aléatoire, mais de la même manière, des listes
random_tab = [i for i in range (len(tab_données))]
np.random.shuffle(random_tab)
tab_résultats = [tab_résultats[i] for i in random_tab]
tab_données = [tab_données[i] for i in random_tab]

tab_résultats = create_one_hot_vec(tab_résultats, 100)
tab_données = create_one_hot_vec(tab_données, 5)

model = Sequential([
    Dense(32, input_dim=100),
    Activation  ('relu'),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense (5),
    Activation('softmax'),
])

#Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=1, verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)

print("Score final : " + str(score))

print("Fin")