import numpy as np
import nose
#import theano

import tabopen
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D, convolutional
# from keras.utils import np_utils
# from keras.datasets import mnist
print ("--------fin import--------")

print ("verion de numpy : " + str(np.__version__))
print ("version de theano : " + str(theano.__version__))


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

def choix_valeurs(val) :
    choix = 1
    if val <= 20 :
        choix = 1
    elif val <= 40 :
        choix = 2
    elif val <= 60 :
        choix = 3
    elif val <= 80 :
        choix = 4
    else :
        choix = 5
    return (choix)

def résultats_attendus(tab):
    tab_resultats = tab.copy()
    for i in range(tab_resultats.shape[0]) :
        for j in range(tab_resultats.shape[1]) :
            tab_resultats[i,j] = choix_valeurs(tab[i,j])
    return (tab_resultats)

#tableau d'entrainement, tableaux de tests
tab_entrainement = Q_C_tab[0:40,:] #les 40 premières lignes sur 60 seront réservées à l'entrainement
tab_tests = Q_C_tab[41:,:] #On garde les 20 dernières pour le test
(X_train, y_train), (X_test, y_test) = (tab_entrainement, résultats_attendus(tab_entrainement)), (tab_tests,résultats_attendus(tab_tests))

#--------------Début Deep Learning--------------------------------------------------------------------
print ("Début deep learning")
np.random.seed(123)
