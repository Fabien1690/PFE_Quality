import numpy as np
from numpy.core.multiarray import ndarray

scores_file = open("Scores_3DHigh_NN.txt")

# Creation des tableaux de valeurs
quality_continuous_tab = np.zeros((59, 16))
quality_5levels_tab = np.zeros((59, 16))
depth_continuous_tab = np.zeros((59, 16))
depth_5levels_tab = np.zeros((59, 16))

lines_to_keep = range(4, 6)

values_1st_array = range(3,19)
values_2nd_array = range(21, 37)
values_3rd_array = range(39, 55)

line_nb=0
for line in scores_file:
    #print("line nb", line_nb, " : ", line)

#    if line_nb > 4 and line_nb < 65 :
#        print("line nb", line_nb, " : ", line)
#    if line_nb == 5 :
#        numbers_line_5 = line.split()
#        for nb in numbers_line_5:
#            print(nb)


    if line_nb in lines_to_keep :
        values = line.split()
        print("line nb " + line_nb.__str__() + " : ", end= '')
        #On regarde les valeurs de la ligne si on l'a gardé
        value_nb = 0
        for value in values :
            if value_nb in values_1st_array :

                #quality_continuous_tab[line_nb, value_nb-3] = float(value)
                print(value, end=' ' )

            if value_nb in values_2nd_array :
                print(value, end=' ' )

            if value_nb in values_3rd_array :
                print(value, end=' ' )
            value_nb += 1

        print() #saut de ligne après avoir affiché toutes les valeurs de la ligne

    line_nb += 1
#print(quality_continuous_tab)
#print(scores_file.read()) #read(n) : donne les n premiers caractères de la chaine

#scores_file.close()