
#importation des bibliothèques à utiliser
import os
import matplotlib.pyplot as plt #Traçage des figures
import numpy as np #Opérations mathématiques
import pandas as pd  #Manipulation de la data
from sklearn.model_selection import train_test_split  #Créer training and testing sets
from sklearn.model_selection import cross_val_score #évaluer la perfomance de l'estimateur
from sklearn.model_selection import KFold #méthode d'entrainement du data en la divisant en k-1 subsets
from sklearn.pipeline import Pipeline #faciliter la recherche des paramètres
from sklearn.preprocessing import MinMaxScaler  #mise à échelle et ajustement des données
from tensorflow.python.keras.models import Sequential #Construction des couches (Layers)
from tensorflow.python.keras.layers import Dense #application de la fonction d'activation
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor #Régression pour obtenir des valeurs continues

# Fonction de Prédiction
def PRED(X):
#PHASE D APPRENTISSAGE
    path="C:/Users/HP NoteBoooK/Desktop/work/Centralisé/Prediction"#emplacement des fichiers DATA
    os.chdir(path) #changement de l'environnement de travail vers cet emplacement
    X=[X]#adaptation au type des entrées manipulées par le réseau de neurones
    dataset=np.loadtxt("DATA.txt") #extraire la data depuis le fichier texte: DATA
    x=dataset[:,:18]#Sélection des inputs
    y=dataset[:,22:40]#Sélection des outputs
    y=np.reshape(y, (-1,18)) #inversion de la matrice output
    scaler_x = MinMaxScaler()#échelle des inputs
    scaler_y = MinMaxScaler()#échelle des outputs
    print(scaler_x.fit(x))
    xscale=scaler_x.transform(x)#Mise à échelle des inputs
    print(scaler_y.fit(y))
    yscale=scaler_y.transform(y)#échelle des outputs
    print(xscale.shape,yscale.shape)
    X_train, X_test, y_train, y_test = train_test_split(xscale, yscale) #phase du training
    model = Sequential() #Création du réseau de neurones
    model.add(Dense(12, input_dim=18, kernel_initializer='normal', activation='relu'))#Création de la Première couche
    model.add(Dense(18, activation='linear'))#Création de la Deuxième couche
    model.summary()#Finalisation du modèle du réseau de neurones
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae']) #Compilation du code avec la fonction d'erreur est la mean square error
    history = model.fit(X_train, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=0.2) #Génération des résultats à chaque reprise

#PHASE DE TEST
    Xnew = np.array(list(X)) #Données de Test
    Xnew= scaler_x.transform(Xnew) #Mise à échelle
    ynew= model.predict(Xnew) #exécution sous le réseau de neurones de prédiction
    #inversion de la mise à échelle (Restitution des valeurs originales)
    ynew = scaler_y.inverse_transform(ynew)
    Xnew = scaler_x.inverse_transform(Xnew)
    return ynew[0] #Résultat de prédiction

def CONTR(X):
#PHASE D APPRENTISSAGE

    path="C:/Users/HP NoteBoooK/Desktop/work/Centralisé/Control"#emplacement des fichiers DATA
    os.chdir(path) #changement de l'environnement de travail vers cet emplacement
    dataset=np.loadtxt("DATA_C_C.txt") #extraire la data depuis le fichier texte: DATA
    X=[X]#adaptation au type des entrées manipulées par le réseau de neurones
    x=dataset[:,3:21]#Sélection des inputs
    y=dataset[:,:3]#Sélection des outputs
    y=np.reshape(y, (-1,3)) #inversion de la matrice output
    scaler_x = MinMaxScaler()#Mise à échelle des inputs
    scaler_y = MinMaxScaler()#Mise à échelle des outputs
    print(scaler_x.fit(x))
    xscale=scaler_x.transform(x)#Mise à échelle des inputs
    print(scaler_y.fit(y))
    yscale=scaler_y.transform(y)#échelle des outputs
    print(xscale.shape,yscale.shape)
    X_train, X_test, y_train, y_test = train_test_split(xscale, yscale) #phase du training
    model = Sequential() #Création du réseau de neurones
    model.add(Dense(12, input_dim=18, kernel_initializer='normal', activation='relu'))#Création de la Première couche
    model.add(Dense(3, activation='linear'))#Création de la Deuxième couche
    model.summary()#Finalisation du modèle du réseau de neurones
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae']) #Compilation du code avec la fonction d'erreur est la mean square error
    history = model.fit(X_train, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)#Génération des résultats à chaque reprise

#PHASE DE TEST
    Xnew = np.array(list(X)) #Données de Test
    Xnew= scaler_x.transform(Xnew)#Mise à échelle
    ynew= model.predict(Xnew)#exécution sous le réseau de neurones de prédiction
    #inversion de la mise à échelle (Restitution des valeurs originales)
    ynew = scaler_y.inverse_transform(ynew)
    Xnew = scaler_x.inverse_transform(Xnew)
    return ynew[0] #Résultat du Contrôle