import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


###################################### Importation des données : ######################################

data_root_path = "C:/Users/kroki/Dropbox/Caroline/Documents/Cours/1 - Master/4ème semestre/Machine learning/Data/"
# data_root_path = "/Users/a_clouet/Documents/Master_2022-2024/M2/S2/MachineLearning/Projet/"

# Test Data : 
#test_away_player_statistics = pd.read_csv(data_root_path + "Test_Data/test_away_player_statistics_df.csv", sep=",", index_col=0)
#test_away_team_statistics = pd.read_csv(data_root_path + "Test_Data/test_away_team_statistics_df.csv", sep=",", index_col=0)
#test_home_player_statistics = pd.read_csv(data_root_path + "Test_Data/test_home_player_statistics_df.csv", sep=",", index_col=0)
#test_home_team_statistics = pd.read_csv(data_root_path + "Test_Data/test_home_team_statistics_df.csv", sep=",", index_col=0)

# Train data : 
#train_away_player_statistics = pd.read_csv(data_root_path + "Train_Data/train_away_player_statistics_df.csv", sep=",", index_col=0)
#train_away_team_statistics = pd.read_csv(data_root_path + "Train_Data/train_away_team_statistics_df.csv", sep=",", index_col=0)
#train_home_player_statistics = pd.read_csv(data_root_path + "Train_Data/train_home_player_statistics_df.csv", sep=",", index_col=0)
#train_home_team_statistics = pd.read_csv(data_root_path + "Train_Data/train_home_team_statistics_df.csv", sep=",", index_col=0)

# Extra data : 
# Y_train_supp = pd.read_csv(data_root_path + "benchmark_and_extras/Y_train_supp.csv", sep=",", index_col=0)

# Données de tests : 
Y_test = pd.read_csv(data_root_path + "Y_test_random.csv", sep=",", index_col=0)
Y_train = pd.read_csv(data_root_path + "Y_train.csv", sep=",", index_col=0) # scores 



######################################## Analyse des données : ########################################

#print("train_away_player_statistics")
#print(train_away_player_statistics.describe())

#print("train_away_team_statistics")
#print(train_away_team_statistics.describe())

#print("train_home_player_statistics")
#print(train_home_player_statistics.describe())

#print("train_home_team_statistics")
#print(train_home_team_statistics.describe())

print("Test")
print(Y_test.describe())

print("Train")
print(Y_train.describe())

######################################### Données manquantes : ########################################

# Graphique données manquantes (cours) : 
def graph_missing_data(df:pd.DataFrame) :
    plt.figure(figsize=(16, 4))
    (df.count() - df.shape[1]).plot.bar()
    plt.show()

graph_missing_data(test_home_team_statistics)




# Y_train = Y_train.replace({np.inf:np.nan,-np.inf:np.nan})

