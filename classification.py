import numpy as np
# import nose
import theano

import tabopen
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D, convolutional
from keras.utils import np_utils
# from keras.datasets import mnist
print ("--------fin import--------")

print ("verion de numpy : " + str(np.__version__))
# print ("version de theano : " + str(theano.__version__))


#------ Creation tableaux : Q=Quality, D=Depth, C=Comfort  ------------------------------------------
#                           C=Continuous, 5=5 levels

Q_C_tab = np.zeros((60, 16))
Q_5_tab = np.zeros((60, 16))
D_C_tab = np.zeros((60, 16))
D_5_tab = np.zeros((60, 16))
C_C_tab = np.zeros((60, 16))
C_5_tab = np.zeros((60, 16))

tabopen.création_tab(quality_continuous_tab=Q_C_tab,
                     quality_5levels_tab=Q_5_tab,
                     depth_continuous_tab=D_C_tab,
                     depth_5levels_tab=D_5_tab)

#Choix valeurs limites utilisées dans choix_valeurs():
VALEUR1=20
VALEUR2=40
VALEUR3=60
VALEUR4=80

def choix_valeurs(val) :
    """
    Donne le score dans l'echelle de 5 niveaux, allant de 0 à 4
    Paramètre
    ----------
    val: int
        valeur allant de 0 à 100 qui va être convertie en int allant de 0 à 4
    """
    choix = 0
    if val <= VALEUR1 :
        choix = 0
    elif val <= VALEUR2 :
        choix = 1
    elif val <= VALEUR3 :
        choix = 2
    elif val <= VALEUR4 :
        choix = 3
    else :
        choix = 4
    return (choix)

def résultats_attendus(tab):
    """
    Crée le tableau de résultats attendus
    :param tab: tableau de valeurs allant de 0 à 99
    :return: tableau allant de 0 à 4 (definition de la modification dans choix_valeurs()

    """
    tab_resultats = tab.copy()
    for i in range(tab_resultats.shape[0]) :
        for j in range(tab_resultats.shape[1]) :
            tab_resultats[i,j] = choix_valeurs(tab[i,j])
    return (tab_resultats)

def create_one_hot_vec(tab, max_val) :
    """
    :param tab: tableau d'entiers
    :param max_val: {5, 100} Valeur maximale du tableau.
        Pour un tableau de valeurs continues, prendre max_val = 100,
        pour un tableau de valeurs allant de 0 à 4, prendre max_val = 5
    :return: tab_out : tableau en "one-hot vector".
    """
    tab_out = np.zeros(tab.size) #crée un tableau de la taille necessaire
    for line_nb in range (tab.shape[0]) :
        for value_nb in range(tab.shape[1]) :
            tab_out[line_nb*tab.shape[1] + value_nb] = tab[line_nb, value_nb]
    tab_out = np_utils.to_categorical(tab_out, max_val) #Transformation en one-hot vector (que de '0' sauf un '1' à l'endroit voulu)
    return tab_out

#tableau d'entrainement, tableaux de tests


tab_entrainement = Q_C_tab[0:40,:] #les 40 premières lignes sur 60 seront réservées à l'entrainement
tab_tests = Q_C_tab[41:,:] #On garde les 20 dernières pour le test
(X_train, y_train), (X_test, y_test) = (tab_entrainement, résultats_attendus(tab_entrainement)), (tab_tests,résultats_attendus(tab_tests))

#--------------Début Deep Learning--------------------------------------------------------------------
print ("Début deep learning")
np.random.seed(124)

#On change le format des données pour que cela fasse des vecteurs "one-hot"

X_train = create_one_hot_vec(X_train, 100)
X_test = create_one_hot_vec(X_test, 100)
y_train = create_one_hot_vec(y_train, 5)
y_test = create_one_hot_vec(y_test, 5)
print("Tableaux créés")

#Architecture du modèle
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

