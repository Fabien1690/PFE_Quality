from keras.backend import batch_dot

from tabopen import *
import theano
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D, convolutional
# from keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib.collections as collections
print ("----Fin import----")


def create_one_hot_vec(tab, max_val) :
    """
    :param tab: tableau d'entiers
    :param max_val: {5, 100} Valeur maximale du tableau.
        Pour un tableau de valeurs continues, prendre max_val = 100,
        pour un tableau de valeurs allant de 0 à 4, prendre max_val = 5
    :return: tab_out : tableau en "one-hot vector".
    """
    if (type(tab) == type([])) :
        tab_out = np.zeros(len(tab))
    else :
        tab_out = np.zeros(tab.size) #crée un tableau de la taille necessaire
    for i in range (len(tab)) :
        tab_out[i] = tab[i]-1
    tab_out = np_utils.to_categorical(tab_out,dtype='int32') #Transformation en one-hot vector (que de '0' sauf un '1' à l'endroit voulu)
    return tab_out



tab_données = création_orga_tab(path="Scores_continuous.txt")
tab_résultats = création_orga_tab(path="Scores_discrete.txt")

#Les deux tableaux sont maintenant des vecteurs triées. Leur taille doit être réglée.
tab_données = tab_données.tolist()
tab_résultats = tab_résultats.tolist()
tab_résultats = allonge_list(tab_résultats, tab_données.__len__())
tab_résultats = np.sort(tab_résultats)

#Rangement aléatoire, mais de la même manière, des listes
random_tab = [i for i in range (len(tab_résultats))]
np.random.shuffle(random_tab)

tab_données = [tab_données[i] for i in random_tab]
tab_résultats = [tab_résultats[i] for i in random_tab]

demarcation = int(tab_données.__len__()*0.7)
tab_résultats_entrainement = tab_résultats[:demarcation]
tab_données_entrainement = tab_données[:demarcation]
tab_résultats_verification = tab_résultats[demarcation:]
tab_données_verification = tab_données[demarcation:]

tab_données_entrainement = create_one_hot_vec(tab_données_entrainement, 100)
tab_résultats_entrainement = create_one_hot_vec(tab_résultats_entrainement, 5)
tab_données_verification = create_one_hot_vec(tab_données_verification, 100)
tab_résultats_verification = create_one_hot_vec(tab_résultats_verification, 5)
print("----Tableaux créés----")

#----------------------Model------------------------
model = Sequential([
    Dense(64, input_dim=100),
    Activation('relu'),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense (5),
    Activation('softmax'),
])

#Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#----------Modification des poids du modèle-------------
print("----Repartition des poids dans le réseau----")
model.load_weights("Sauvegarde_model_test2.h5")
# model.fit(tab_données_entrainement, tab_résultats_entrainement, batch_size=64, epochs=15, verbose=2)
# model.save_weights("Sauvegarde_model_test2.h5")
# score = model.evaluate(tab_données_verification, tab_résultats_verification, verbose=2)
# print("Score final : " + str(score))

#-----------Prediction---------------------
pred_input = [i for i in range(1,101)]
pred_input__one_hot = create_one_hot_vec(pred_input, 100)
rounded_predictions = model.predict_classes(pred_input__one_hot, batch_size=10)
predictions_precise = model.predict(pred_input__one_hot, batch_size=10)
#----------Tracé de la courbe------------------
print("----Tracé de la courbe----")
fig, ax=plt.subplots()
ax.plot(pred_input,rounded_predictions, marker='.', linewidth=1)
missing_data = [0]
for i in range (100):
    if i in tab_résultats:
        missing_data = missing_data + [0]
    else:
        missing_data = missing_data + [1]
print(missing_data)
collection = collections.BrokenBarHCollection.span_where(
    pred_input, ymin=0, ymax=4, where=np.array(missing_data)==0,facecolor='red', alpha=0.6)
ax.add_collection(collection)
collection2 = collections.BrokenBarHCollection.span_where(
    pred_input, ymin=0, ymax=4, where=np.array(missing_data)==1,facecolor='green', alpha=0.2)
ax.add_collection(collection2)
plt.grid()
plt.show()

print("Fin")