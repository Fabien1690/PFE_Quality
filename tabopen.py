import numpy as np

def tofloat(value):
    """Prend une valeur du tableau de données et transforme les ',' en '.' et change
    la valeur en float """
    value_fin = ""
    for nb in value :
        if nb == ',':
            nb = "."
        value_fin += nb
    try :
        return float(value_fin)
    except :
        print("Erreur : Valeur \"" + value + "\" non entière.")
        return 0


def création_tab(
        quality_continuous_tab = np.zeros((60, 16)),
        quality_5levels_tab= np.zeros((60, 16)),
        depth_continuous_tab= np.zeros((60, 16)),
        depth_5levels_tab= np.zeros((60, 16)),
        comfort_continuous_tab= np.zeros((60, 16)),
        comfort_5levels_tab= np.zeros((60, 16)),
):
    """
    Importe les tableaux du fichier Scores_3DHigh_NN dans les 6 (au max) tableaux correspondants
    Les valeurs vont de 0 à 99
    """
    scores_file = open("Scores_3DHigh_NN.txt")

    nb_lignes = 60
    nb_colonnes = 16

    # Creation des tableaux de valeurs
    line_index=0
    for line in scores_file :
        values = line.split()
        value_index = 0
        for value in values[3:] : #3 correspond au nombre de valeurs qui n'ont pas d'utilité ici (age, sexe)
            if line_index < nb_lignes : # evaluation de facon continue
                if value_index < nb_colonnes :
                    quality_continuous_tab[line_index, value_index] = tofloat(value)*0.99
                elif value_index < 2*nb_colonnes :
                    depth_continuous_tab[line_index, value_index - nb_colonnes ] = tofloat(value)*0.99
                else :
                    comfort_continuous_tab[line_index, value_index - 2*nb_colonnes ] = tofloat(value)*0.99
            elif line_index < nb_lignes*2+1:  # evaluation sur 5 niveaux
                if value_index < nb_colonnes  :
                    quality_5levels_tab[int((line_index-1)-nb_lignes), value_index] = tofloat(value)*0.99
                elif value_index < 2*nb_colonnes  :
                    depth_5levels_tab[int((line_index-1)-nb_lignes), value_index - nb_colonnes ] = tofloat(value)*0.99
                else :
                    comfort_5levels_tab[int((line_index-1)-nb_lignes), value_index - 2*nb_colonnes ] = tofloat(value)*0.99
            #On peut garder les 3 derniers tableaux ici
            value_index += 1
        line_index += 1
    scores_file.close()

def fill_table(tab_in = [], lines_to_remove = [], line_min=0, line_max=0):
    """
    Crée une liste contenant les valeurs de tab_in, passées en une seule liste d'entiers
    :param tab_in: tableau de valeurs, provenant d'un fichier
    :param lines_to_remove: liste de lignes à ne pas considérer
    :return tableau rempli
    """
    tab_out = []
    line_index = 0
    for line in tab_in:
        if line_index > line_min and line_index < line_max:
            values = line.split()
            if line_index not in lines_to_remove:  # On retire les lignes séparatrices
                for value in values:
                    tab_out = np.concatenate((tab_out, [tofloat(value)]))
        line_index = line_index + 1
    return tab_out

def création_orga_tab(path, min_line=0, max_line=1) :
    """
    Crée la table à partir des données n°2
    :param path: ["Scores_continuous.txt","Scores_discrete.txt"]chemin vers la source
    :return tab_resultats:tableau où seront stockées les données
    """
    #Ouverture des tables
    # scores_continuous = open("Scores_continuous.txt")
    # scores_discrete = open("Scores_discrete.txt")
    if path not in ["Scores_continuous.txt", "Scores_discrete.txt"] :
        raise FileNotFoundError("Chemin de fichier non valide.")
    Score = open(path)
    if path == "Scores_continuous.txt" :
        lines_to_remove = [0, 53, 112, 173]
    else :
        lines_to_remove = [0, 52, 105, 156]
    tab_resultats = fill_table(tab_in=Score, lines_to_remove=lines_to_remove, line_min= min_line, line_max=max_line)
    tab_resultats = np.sort(tab_resultats)

    Score.close()
    return tab_resultats

def allonge_list(list, longueur_voulue) :
    try :
        assert (len(list)<longueur_voulue)
    except AssertionError :
        print("La longueur souhaitée doit être supérieure à celle de la liste entrée")
    list_out = []
    longueur= len(list)
    add_ratio = longueur_voulue/longueur
    Qte_nombre = []
    for number in range(int(np.max(list))+ 1):
        Qte_nombre.append(number-100) #On fait attention que le numéro ne soit pas le max
        Qte_nombre.append(list.count(number))
        for i in range(int(list.count(number)*add_ratio)):
            list_out.append(number)
    while (list_out.__len__()<longueur_voulue) : #On rajoute une valeur pour les numéros les plus nombreux
        m = Qte_nombre.index(max(Qte_nombre)) # m-1 : numero le plus représenté (+- 100), m : la qantité du nombre
        list_out.append(Qte_nombre[m-1]+100)
        Qte_nombre.remove(Qte_nombre[m-1])
        Qte_nombre.remove(Qte_nombre[m-1])
    return(list_out)


#Tests


# random_tab = [i for i in range()]
# np.random.shuffle(random_tab)



