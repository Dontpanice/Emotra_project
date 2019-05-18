import pickle
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#                           Load the data into dataframes
# =============================================================================
with open('./Dataset.pickle', 'rb') as f:
    Dataset = pickle.load(f)
    
Area = Dataset[['Area','Area2','Area3','Area4','Area5','Area6','Area7']].reset_index(drop=True)
Amplitude = Dataset[['Amplitude','Amplitude2','Amplitude3','Amplitude4','Amplitude5','Amplitude6','Amplitude7']].reset_index(drop=True)
Risetime = Dataset[['Risetime','Risetime2','Risetime3','Risetime4','Risetime5','Risetime6','Risetime7']].reset_index(drop=True)
Settlingetime = Dataset[['Settlingtime','Settlingtime2','Settlingtime3','Settlingtime4','Settlingtime5','Settlingtime6','Settlingtime7']].reset_index(drop=True)
Arclength = Dataset[['Arclength1','Arclength2','Arclength3','Arclength4','Arclength5','Arclength6','Arclength7']].reset_index(drop=True)
HeartRate = Dataset[['HeartRate','HeartRate2','HeartRate3','HeartRate4','HeartRate5','HeartRate6','HeartRate7']].reset_index(drop=True)
#relative_means = Dataset['relative_mean'].reset_index(drop=True)
means = Dataset['means'].reset_index(drop=True)
N_amp = Dataset['N_amp'].reset_index(drop=True)
Label = Dataset['Label'].reset_index(drop=True)

#%%
# =============================================================================
#                       make boxplots of dataset
# =============================================================================




#separate hypo and reactive
DatasetR = Dataset.loc[Dataset['Label'] == 0]
DatasetH = Dataset.loc[Dataset['Label'] == 1]

#extract features from Reactive
Area = DatasetR[['Area','Area2','Area3','Area4','Area5','Area6','Area7']].reset_index(drop=True)
Amplitude = DatasetR[['Amplitude','Amplitude2','Amplitude3','Amplitude4','Amplitude5','Amplitude6','Amplitude7']].reset_index(drop=True)
Risetime = DatasetR[['Risetime','Risetime2','Risetime3','Risetime4','Risetime5','Risetime6','Risetime7']].reset_index(drop=True)
Settlingetime = DatasetR[['Settlingtime','Settlingtime2','Settlingtime3','Settlingtime4','Settlingtime5','Settlingtime6','Settlingtime7']].reset_index(drop=True)
Arclength = DatasetR[['Arclength1','Arclength2','Arclength3','Arclength4','Arclength5','Arclength6','Arclength7']].reset_index(drop=True)
HeartRate = DatasetR[['HeartRate','HeartRate2','HeartRate3','HeartRate4','HeartRate5','HeartRate6','HeartRate7']].reset_index(drop=True)
#relative_means = Dataset['relative_mean'].reset_index(drop=True)
means = DatasetR['means'].reset_index(drop=True)
N_amp = DatasetR['N_amp'].reset_index(drop=True)
Label = DatasetR['Label'].reset_index(drop=True)

#calcuate the mean of values across all segments and apply log to result
R_Area_log = [np.log(np.mean(rad)) for rad in Area.values]
R_Amplitude_log = [np.log(np.mean(rad)) for rad in Amplitude.values]
R_Risetime_log = [np.log(np.mean(rad)) for rad in Risetime.values]
R_Settlingetime_log = [np.log(np.mean(rad)) for rad in Settlingetime.values]
R_Arclength_log = [np.log(np.mean(rad)) for rad in Arclength.values]
R_HeartRate_log = [np.log(np.mean(rad)) for rad in HeartRate.values]
R_means_log = [np.log(rad) for rad in means.values]
R_N_amp_log = [np.log(rad) for rad in N_amp.values]

#extract features from Hypo
Area = DatasetH[['Area','Area2','Area3','Area4','Area5','Area6','Area7']].reset_index(drop=True)
Amplitude = DatasetH[['Amplitude','Amplitude2','Amplitude3','Amplitude4','Amplitude5','Amplitude6','Amplitude7']].reset_index(drop=True)
Risetime = DatasetH[['Risetime','Risetime2','Risetime3','Risetime4','Risetime5','Risetime6','Risetime7']].reset_index(drop=True)
Settlingetime = DatasetH[['Settlingtime','Settlingtime2','Settlingtime3','Settlingtime4','Settlingtime5','Settlingtime6','Settlingtime7']].reset_index(drop=True)
Arclength = DatasetH[['Arclength1','Arclength2','Arclength3','Arclength4','Arclength5','Arclength6','Arclength7']].reset_index(drop=True)
HeartRate = DatasetH[['HeartRate','HeartRate2','HeartRate3','HeartRate4','HeartRate5','HeartRate6','HeartRate7']].reset_index(drop=True)
#relative_means = Dataset['relative_mean'].reset_index(drop=True)
means = DatasetH['means'].reset_index(drop=True)
N_amp = DatasetH['N_amp'].reset_index(drop=True)
Label = DatasetH['Label'].reset_index(drop=True)

#calcuate the mean of values across all segments and apply log to result
H_Area_log = [np.log(np.mean(rad)) for rad in Area.values]
H_Amplitude_log = [np.log(np.mean(rad)) for rad in Amplitude.values]
H_Risetime_log = [np.log(np.mean(rad)) for rad in Risetime.values]
H_Settlingetime_log = [np.log(np.mean(rad)) for rad in Settlingetime.values]
H_Arclength_log = [np.log(np.mean(rad)) for rad in Arclength.values]
H_HeartRate_log = [np.log(np.mean(rad)) for rad in HeartRate.values]
H_means_log = [np.log(rad) for rad in means]
H_N_amp_log = [np.log(rad) for rad in N_amp]




#%%


Groups1 = [H_Area_log,R_Area_log,H_Amplitude_log,R_Amplitude_log,H_Risetime_log,R_Risetime_log,H_Settlingetime_log,R_Settlingetime_log]
Groups2 = [H_Arclength_log,R_Arclength_log,H_HeartRate_log,R_HeartRate_log,H_means_log,R_means_log,H_N_amp_log,R_N_amp_log]   


R_Groups = [R_Area_log,R_Amplitude_log,R_Risetime_log,R_Settlingetime_log,R_Arclength_log,R_HeartRate_log,R_means_log,R_N_amp_log]
H_Groups = [H_Area_log,H_Amplitude_log,H_Risetime_log,H_Settlingetime_log,H_Arclength_log,H_HeartRate_log,H_means_log,H_N_amp_log]



fig, ax = plt.subplots(figsize = (10,10))
pos = np.array(range(len(Groups1)))+1
bp = ax.boxplot(Groups1, sym='k+', positions=pos,
                notch=1, bootstrap=5000)

ax.set_xlabel('Boxplots')
ax.set_ylabel('(area log(values)')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.title('')
#plt.xticks(np.arange(1,9,1), ['H_AE','R_AE','H_AM','R_AM','H_RT','R_RT','H_ST','R_ST'])
plt.xticks(np.arange(1.5,9,2), ['Area','Amplitude','Risetime','Settlingtime'])
plt.show()   

#%%
fig, ax = plt.subplots(figsize = (10,10))
pos = np.array(range(len(Groups2)))+1
bp = ax.boxplot(Groups2, sym='k+', positions=pos,
                notch=1, bootstrap=5000)

ax.set_xlabel('Boxplots')
ax.set_ylabel('(area log(values)')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.title('')
#plt.xticks(np.arange(1,9,1), ['H_AE','R_AE','H_AM','R_AM','H_RT','R_RT','H_ST','R_ST'])
plt.xticks(np.arange(1.5,9,2), ['Arc length','Heart rate','Mean','NS.SCR'])
plt.show()   



