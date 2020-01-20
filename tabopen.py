import numpy as np
from numpy.core.multiarray import ndarray

def tofloat(value):
    """Prend une valeur du tableau de données et transforme les ',' en '.' et change
    la valeur en float """
    value_fin = ""
    for nb in value :
        if nb == ',':
            nb = "."
        value_fin += nb
    return float(value_fin)

def création_tab(
        quality_continuous_tab = np.zeros((60, 16)),
        quality_5levels_tab= np.zeros((60, 16)),
        depth_continuous_tab= np.zeros((60, 16)),
        depth_5levels_tab= np.zeros((60, 16)),
        comfort_continuous_tab= np.zeros((60, 16)),
        comfort_5levels_tab= np.zeros((60, 16)),
):
    scores_file = open("Scores_3DHigh_NN.txt")

    nb_lignes = 60
    nb_colonnes = 16
    # Creation des tableaux de valeurs

    line_index=0
    for line in scores_file :
        #print("line nb", line_nb, " : ", line)

    #    if line_nb > 4 and line_nb < 65 :
    #        print("line nb", line_nb, " : ", line)
    #    if line_nb == 5 :
    #        numbers_line_5 = line.split()
    #        for nb in numbers_line_5:
    #            print(nb)

        values = line.split()
        value_index = 0
        for value in values[3:] : #3 correspond au nombre de valeurs qui n'ont pas d'utilité ici (age, sexe)
            if line_index < nb_lignes : # evaluation de facon continue
                if value_index < nb_colonnes :
                    quality_continuous_tab[line_index, value_index] = tofloat(value)
                elif value_index < 2*nb_colonnes :
                    depth_continuous_tab[line_index, value_index - nb_colonnes ] = tofloat(value)
                else :
                    comfort_continuous_tab[line_index, value_index - 2*nb_colonnes ] = tofloat(value)

            elif line_index < nb_lignes*2+1:  # evaluation sur 5 niveaux
                if value_index < nb_colonnes  :
                    quality_5levels_tab[int((line_index-1)-nb_lignes), value_index] = tofloat(value)
                elif value_index < 2*nb_colonnes  :
                    depth_5levels_tab[int((line_index-1)-nb_lignes), value_index - nb_colonnes ] = tofloat(value)
                else :
                    comfort_5levels_tab[int((line_index-1)-nb_lignes), value_index - 2*nb_colonnes ] = tofloat(value)
            #On peut garder les 3 derniers tableaux ici
            value_index += 1
        line_index += 1

