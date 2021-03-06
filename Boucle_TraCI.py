#importation des bibliothèques à utiliser
import os #Biblio Operating System de python pour manipuler les fichiers
import sys #Faciliter l'exécution du code
import optparse #exécution sous la command line

from sumolib import checkBinary  # checks if the files are binary
import traci #Bibliothèque du Traffic Control
path="C:/Users/HP NoteBoooK/Desktop/Simulation"
os.chdir(path)

path="C:/Users/HP NoteBoooK/Desktop/Simulation" #emplacement du fichier de simulation
os.chdir(path) #changement de l'environnement de travail vers cet emplacement

# Importation des modules de  $SUMO_HOME/tools directory (outils qui ont été installés automatiquement avec le pack SUMO)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


#manipulation de la command line
def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


t=traci._trafficlight.TrafficLightDomain() #spécification du domaine traffic lights que nous utiliserons par la suite
t2=traci._lane.LaneDomain() #spécification du domaine:lanes(routes) que nous utiliserons par la suite
TLS=['2240228358', '455732938', '455732939'] #les IDs des feux de circulation dans notre réseau routier

# contains TraCI control loop
def run():
    step = 0  #initiation
    while step < 100: #Réglage du temps de simulation ici: 100 secondes
        traci.simulationStep()  #Lancement de la simulation sous SUMO
        if step==0: #Au début de la simulation
            LPI=[list(dict.fromkeys(list(t.getControlledLanes(tlsID=str(i))))) for i in TLS]#génération d'une liste qui comporte les lanes per intersection, càd les routes controllées par chacun des feux de circulation (l'intégration de la  méthode 'dict.fromkeys' a pour but d'éliminer les doublons dans les listes générées)
            LPI=sum(LPI,[]) #transformation en une liste à une dimension
            print(LPI,len(LPI))#Visualisation de la liste générée (peut être éliminée)
        if step%60==0: #Horizon de la prédiction =60secondes
            PREDICT=[[(t2.getLastStepOccupancy(laneID=str(i))*t2.getLength(laneID=str(i))),t2.getWaitingTime(laneID=str(i))] for i in LPI]#génération du Queue Lenght et Queue time suivant chaque Lane
            PREDICT=sum(PREDICT,[]) #Transformation en une liste d'une seule dimension
            print (PREDICT, len(PREDICT)) #Visualisation (Peut être éliminée)
            PREDICTION=PRED(PREDICT) #exécution de l'opération de prédiction avec la fonction PRED prédéfinie
            print(PREDICTION) #Visualisation des prédictions générées
            CONTROL=CONTR(PREDICTION) #exécution de l'opération du contrôle avec la fonction prédéfinie CONTR
            print(CONTROL) #Visualisation des résultats du contrôle
            t.setPhaseDuration(tlsID='2240228358',phaseDuration=CONTROL[0]) #Changement de la phase du premier feu de circulation
            t.setPhaseDuration(tlsID='455732938',phaseDuration=CONTROL[1])#Changement de la phase du deuxième feu de circulation
            t.setPhaseDuration(tlsID='455732939',phaseDuration=CONTROL[2])#Changement de la phase du troisième feu de circulation
            AFTER=[[t2.getWaitingTime(laneID=str(i))] for i in LPI] #Génération des temps d'attente moyens au niveau de chaque lane
            print("AFTER=%s"%(AFTER)) #Affichage de ces résultats
        step += 1 #incrémentation de la boucle
    traci.close() # Arrêt de la simulation
    sys.stdout.flush() #assurer la visualisation des outputs au moment de l'exécution du code


# Début de l'exécution du code
if __name__ == "__main__":
    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    traci.start([sumoBinary, "-c", "simulation.sumocfg"])#établir la connection entre TraCI et SUMO
    run()

