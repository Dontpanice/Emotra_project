from my_functions_remake2 import my_svm_model,train_hmm,calc_settling_risetime,find_arclen,get_area_and_peak,calc_arclen,get_lin,korrekt_seq,cut_important,get_important_indexses,smooth,plot_individuals_with_sound_reaction,plot_individuals_with_sound,classifier1,classifier,baseline_classifier_median,baseline_classifier_mean,plot_all,separate_skinC,remove_at_indexes,plot_individuals_in_segment,compare_similar_means,cut_segment_of_df,merging,extract_signal,extract_signal2, extract_labels, separate, get_median_and_means, find_index_sound
import numpy as np
import pickle
import matplotlib.pyplot as plt
#import random 
import pandas as pd
#import copy
#import time

#from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from sklearn import datasets
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from itertools import combinations



#%%
# =============================================================================
#                             Extract data                              
# =============================================================================
# --------------------------------------------  Extract all labels from files, save in "label_data"
df_labels = extract_labels('./Rapporter_cleaned_2/')
# --------------------------------------------- Extract all Highgrade-signaldata from files, save in "signal_data"
signal_data_list, lost_signal_data_list = extract_signal('./Eudor-a_data_2/')
# --------------------------------------------- Extract all mediumgrade-signaldata from files, save in "signal_data2"   
signal_data_list2 = extract_signal2('./Eud_data/')
# ------------------------------------------  MERGING LABELS WITH DATA, comparing ID betwen files and merging dataframe with label.
lost_matches,found_matches = merging(df_labels,signal_data_list)
lost_matches2,found_matches2 = merging(df_labels,signal_data_list2)
All_data = signal_data_list + signal_data_list2       
# =============================================================================
#                     Remove non label-match Dataframe
# =============================================================================
remove = []
for idx,minilista in enumerate(All_data):
    ID = minilista[0]
    df = minilista[1]
#    for idx,df in enumerate(minilista):
    if 'label' in df.columns:
        continue
    else:
        remove.append(idx)
remove_at_indexes(All_data,remove)
# =============================================================================
#                  Cut Dataframe into orientation segments
# =============================================================================
thrown_out2,check2,remove2 = cut_segment_of_df(All_data,2,9)
remove_at_indexes(All_data,remove2)
#%%

with open('D:/Master_thesis_data/Emotra_preprocessed/All_data_ready.pickle', 'rb') as f:
    All_data = pickle.load(f)
    
    

# =============================================================================
#                           Smooth Dataframe 
# =============================================================================
smooth(All_data,50)  
# =============================================================================
#                       Cut Dataframe into 4s segments
# =============================================================================
indexes_to_cut = get_important_indexses(All_data)
all_segments = cut_important(All_data,indexes_to_cut)

# =============================================================================
#   Correct the sequences  and make all negative values in sequence positive equal to 1e-08
#               + Get Area and peak for these sequences and (remove areas caused by 1e-08 convertion)
# =============================================================================
#Format as all_segments [ ID, [data]]   ---> [data] = [df,df,df,df,df,df,df]
R_sequences,H_sequences,R_area,R_peak,H_area,H_peak,H_area_List,H_peak_List,R_area_List,R_peak_List = get_area_and_peak(all_segments)

#H_area = [seg1,seg2,seg3] 

# =============================================================================
#   calculate arclength --- normalize y axis, calc arclength, amplify numbers so above value of 0
# =============================================================================
# feed list of sequences. list = [sequence1], [sequence2] , returns list

#normalisera sekvenser
H_sequences_N = []
for lista in H_sequences:
    minilista = []
    for sequence in lista:
        sequence = sequence.reshape(1, -1)
        nomalized = preprocessing.normalize(sequence, norm ='l2')
        
        #bump up so value is above 0
        nomalized_bumped = [n * 1000 for n in nomalized]
        nomalized_bumped = np.array(nomalized_bumped)
        nomalized_bumped = nomalized_bumped.reshape(-1, 1)
        minilista.append(nomalized_bumped)
        
    H_sequences_N.append(minilista)
    
R_sequences_N = []
for lista in R_sequences:
    minilista = []
    for sequence in lista:
        sequence = sequence.reshape(1, -1)
        nomalized = preprocessing.normalize(sequence, norm ='l2')
        
        #bump up so value is above 0
        nomalized_bumped = [n * 1000 for n in nomalized]
        nomalized_bumped = np.array(nomalized_bumped)
        nomalized_bumped = nomalized_bumped.reshape(-1, 1)
        minilista.append(nomalized_bumped)
    R_sequences_N.append(minilista)
    
    
H_arc = find_arclen(H_sequences_N)
R_arc = find_arclen(R_sequences_N)

# =============================================================================
#                       calculate settling time & rise time measures
# =============================================================================
# feed list of sequences. list = [sequence1], [sequence2] , returns list
H_risetime_L,H_amplitud_L,H_settlingtime_L = calc_settling_risetime(H_sequences)
R_risetime_L,R_amplitud_L,R_settlingtime_L = calc_settling_risetime(R_sequences)



# =============================================================================
#                       Get heart rate from segments
# =============================================================================

H_HT = []
R_HT = []

for lista in all_segments:
    dfs = lista[1]
    seven_HT = []
    for df in dfs:
#        print(df.columns)
        label = df['label'].iloc[0]
        HT = df['Heart Rate']  
        seven_HT.append(HT)
        
    if label == 0:        
        R_HT.append(np.array(seven_HT))
    elif label == 1:        
        H_HT.append(np.array(seven_HT))
    
# =============================================================================
#                        Get mean and median for each sequence 
# =============================================================================
#%%


H_segmentmeans = [np.log(np.mean(item)) for item in H_sequences]
R_segmentmeans = [np.log(np.mean(item)) for item in R_sequences]

Groups = [H_segmentmeans,R_segmentmeans ]

fig, ax = plt.subplots(figsize = (10,10))
pos = np.array(range(len(Groups))) + 1
bp = ax.boxplot(Groups, sym='k+', positions=pos,
                notch=1, bootstrap=5000)


ax.set_ylabel('(area log(values)')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.title('7segments')
ax.set_xlabel('Boxplots')
plt.xticks([1, 2, 3,4,5,6,7], ['Hypo', 'reactive', 'bin2','bin3','bin4', 'bin5', 'bin6'])
plt.show()


# calculate means for sequences
R_means = [np.mean(item)for item in R_segmentmeans]
H_means = [np.mean(item) for item in H_segmentmeans]

#calculate relative means
H_mean = np.mean(H_segmentmeans)
R_mean = np.mean(R_segmentmeans)
R_relativ_means = [abs(item)/H_mean for item in R_segmentmeans]
H_relativ_means = [abs(item)/R_mean for item in H_segmentmeans]

Groups = [H_relativ_means,R_relativ_means ]

fig, ax = plt.subplots(figsize = (10,10))
pos = np.array(range(len(Groups))) + 1
bp = ax.boxplot(Groups, sym='k+', positions=pos,
                notch=1, bootstrap=5000)

ax.set_xlabel('Boxplots')
ax.set_ylabel('(area log(values)')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.title('7segments')
plt.xticks([1, 2, 3,4,5,6,7], ['Hypo', 'reactive', 'bin2','bin3','bin4', 'bin5', 'bin6'])
plt.show()






#%%
# =============================================================================
#                        Merge Data together into one dataframe
# =============================================================================

# Create Dataset
Dataset_H = pd.DataFrame(columns=['Area','Area2','Area3','Area4','Area5','Area6','Area7','Amplitude','Amplitude2','Amplitude3','Amplitude4','Amplitude5','Amplitude6','Amplitude7','Risetime','Risetime2','Risetime3','Risetime4','Risetime5','Risetime6','Risetime7','Settlingtime','Settlingtime2','Settlingtime3','Settlingtime4','Settlingtime5','Settlingtime6','Settlingtime7','Arclength1','Arclength2','Arclength3','Arclength4','Arclength5','Arclength6','Arclength7','HeartRate','HeartRate2','HeartRate3','HeartRate4','HeartRate5','HeartRate6','HeartRate7','Label'])
for i in range(0,len(H_area)):
    LA  = H_area[i]
    LP = H_peak[i]
    RT = H_risetime_L[i]
    ST = H_settlingtime_L[i]
    ARC = H_arc[i]
    HT = H_HT[i]

    
    A_L = []
    P_L = []
    R_L = []
    S_L = []
    ARC_L = []
    l2 = []
    HT_L = []
    
    for j in range(0,7):
        
        area = LA[j]
        peak = LP[j]
        risetime = RT[j]
        settlintime = ST[j]
        ARClen = ARC[j][0]
        Heart_Rate = HT[j][0]
        
        A_L.append(area)
        P_L.append(peak)
        R_L.append(risetime)
        S_L.append(settlintime)
        ARC_L.append(ARClen)
        HT_L.append(Heart_Rate)

    l2.append(1)
#    print(len(l2))
    data = A_L + P_L + R_L + S_L + ARC_L + HT_L + l2
    Dataset_H.loc[i] = data        

Dataset_R = pd.DataFrame(columns=['Area','Area2','Area3','Area4','Area5','Area6','Area7','Amplitude','Amplitude2','Amplitude3','Amplitude4','Amplitude5','Amplitude6','Amplitude7','Risetime','Risetime2','Risetime3','Risetime4','Risetime5','Risetime6','Risetime7','Settlingtime','Settlingtime2','Settlingtime3','Settlingtime4','Settlingtime5','Settlingtime6','Settlingtime7','Arclength1','Arclength2','Arclength3','Arclength4','Arclength5','Arclength6','Arclength7','HeartRate','HeartRate2','HeartRate3','HeartRate4','HeartRate5','HeartRate6','HeartRate7','Label'])
for i in range(0,len(R_area)):
    LA  = R_area[i]
    LP = R_peak[i]
    RT = R_risetime_L[i]
    ST = R_settlingtime_L[i]
    ARC = R_arc[i]
    HT = R_HT[i]

    
    A_L = []
    P_L = []
    R_L = []
    S_L = []
    l2 = []
    ARC_L = []
    HT_L = []
    
    for j in range(0,7):
        
        area = LA[j]
        peak = LP[j]
        risetime = RT[j]
        settlintime = ST[j]
        ARClen = ARC[j][0]
        Heart_Rate = HT[j][0]
        
        A_L.append(area)
        P_L.append(peak)
        R_L.append(risetime)
        S_L.append(settlintime)
        ARC_L.append(ARClen)
        HT_L.append(Heart_Rate)
        
    l2.append(0)
#    print(len(l2))
    data = A_L + P_L + R_L + S_L + ARC_L + HT_L  + l2
    Dataset_R.loc[i] = data                


Dataset = pd.DataFrame(columns=['Area','Area2','Area3','Area4','Area5','Area6','Area7','Amplitude','Amplitude2','Amplitude3','Amplitude4','Amplitude5','Amplitude6','Amplitude7','Risetime','Risetime2','Risetime3','Risetime4','Risetime5','Risetime6','Risetime7','Settlingtime','Settlingtime2','Settlingtime3','Settlingtime4','Settlingtime5','Settlingtime6','Settlingtime7','Arclength1','Arclength2','Arclength3','Arclength4','Arclength5','Arclength6','Arclength7','HeartRate','HeartRate2','HeartRate3','HeartRate4','HeartRate5','HeartRate6','HeartRate7','Label'])

# add means
Dataset_H['means'] = [np.mean(item) for item in H_segmentmeans]
Dataset_R['means'] = [np.mean(item)for item in R_segmentmeans]

# add relative means
Dataset_R['relative_mean'] = R_relativ_means
Dataset_H['relative_mean'] = H_relativ_means

Dataset = Dataset.append(Dataset_R)
Dataset = Dataset.append(Dataset_H)

#%%   

# =============================================================================
#       ------------------------ MODELS---------SVM--------------
# =============================================================================

# =============================================================================
#   Change here to consider different features in dataset before feeding SVM
# =============================================================================


Area = Dataset.iloc[:, np.arange(0,7)].reset_index(drop=True)
Amplitude = Dataset.iloc[:, np.arange(7,14)].reset_index(drop=True)
Risetime = Dataset.iloc[:, np.arange(14,21)].reset_index(drop=True)
Settlingetime = Dataset.iloc[:, np.arange(21,28)].reset_index(drop=True)
Arclength = Dataset.iloc[:, np.arange(28,35)].reset_index(drop=True)
HeartRate = Dataset.iloc[:, np.arange(35,42)].reset_index(drop=True)

relative_means = Dataset['relative_mean'].reset_index(drop=True)
means = Dataset['means'].reset_index(drop=True)


dfs = [Area,Amplitude,Risetime,Settlingetime,Arclength,HeartRate,relative_means,means]
dfs_idx = [0,1,2,3,4,5,6,7]
#dfs_names = ['Area','Amplitude','Risetime','Settlingetime','Arclength']
dfs_names = ['AE','AM','RT','ST','ARC','HR','Rel_M','M']


All_comb_res = []
All_avgs = []
for N_comb in range (2,len(dfs_idx)+1):
# Get all combinations of length 2 
    comb = combinations(dfs_idx, N_comb) 
    
    
    all_comb = [] 
    for combinationss in list(comb):
        df = pd.DataFrame( index = np.arange(0,1012))
        name = ''
        for idx in combinationss:
            df = df.join(dfs[idx])
            name = name + ' + ' + dfs_names[idx]
    #        print(idx)
    #        print(df.join(dfs[idx]))
            
        all_comb.append([df,name])
        
    All_res = []
    Avg_res_df = pd.DataFrame(columns=['accuracy','precision','recall','F1','Combination'])  
    for idx,combination in enumerate(all_comb):
        df = combination[0]
        name = combination[1]
        
        X = df
        y = Dataset['Label']
        print('\n')
        print('This is test for : ',name)
        results_df = my_svm_model(X,y)
        results_df['Combination'] = [name]*5
        All_res.append( results_df)
        
        
        #calculate avg results from cross validation
        
        accuracy = np.mean(results_df['accuracy'])
        presicion = np.mean(results_df['precision'])
        recall = np.mean(results_df['recall'])
        F1 = np.mean(results_df['F1'])
        
        Avg_res_df.loc[idx] = [accuracy,presicion,recall,F1,name]
        
        
    
    All_comb_res.append(All_res)
    All_avgs.append(Avg_res_df)
        
    
    
# ============================================================================
#                       Save results in xlsx format
# =============================================================================
    
result_df_export = pd.DataFrame(columns=['accuracy','precision','recall','F1','Combination'])  
for lista in All_comb_res:
    for res in lista: 
    #    print(res)
        result_df_export= result_df_export.append(res)

result_df_export.to_excel(excel_writer = 'SVM_results/SVM_results_2.xlsx')
#%%

result_df_export_avg = pd.DataFrame(columns=['accuracy','precision','recall','F1','Combination'])  
for df in All_avgs:
    result_df_export_avg = result_df_export_avg.append(df)
#    print(result_df_export_avg.append(df))
    

#print(result_df_export_avg) 
#result_df_export_avg = result_df_export_avg.T

#print(result_df_export_avg[4]) 
result_df_export_avg.to_excel(excel_writer = 'SVM_results/SVM_results_avg_2.xlsx')


#%%
from sklearn.model_selection import cross_val_score

clf = SVC(kernel='rbf', C=1,gamma='auto')
scores = cross_val_score(clf, X,y, cv=5)

np.average(scores)                                              
  
#%%





fig, ax = plt.subplots()
ax.hist(Dataset_H['Area'], ls='dashed', alpha = 1, lw=3, color= 'red', label= 'Hypo')
ax.hist(Dataset_R['Area'],  ls='dotted', alpha = 0.5, lw=3, color= 'green',label='Reactive',bins=70)
ax.legend()
plt.ylabel('N')
plt.xlabel('area ')
plt.title('Histogram of area')
plt.show()

fig, ax = plt.subplots()
ax.hist(Dataset_H['Amplitude'], ls='dashed', alpha = 1, lw=3, color= 'red', label= 'Hypo')
ax.hist(Dataset_R['Amplitude'],  ls='dotted', alpha = 0.5, lw=3, color= 'green',label='Reactive',bins=70)
ax.legend()
plt.ylabel('N')
plt.xlabel('area ')
plt.title('Histogram of area')
plt.show()


#%%
plt.hist(Dataset_R['Area'])
plt.show()
#Dataset_R['Area']

plt.hist(Dataset_H['Area'])
plt.show()
#Dataset_R['Area']
#%%

# Visualising the Training set results
#from matplotlib.colors import ListedColormap
#X_set, y_set = X_train, y_train
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('SVM Classifier (Training set border visualization) RBF kernel')
#plt.xlabel('Area')
#plt.ylabel('Amplitude length')
#plt.legend()
#plt.show()


#%%




A = all_segments[:10]

before = []
after = []

for lista in A:
    Id = lista[0]
    DFs = lista[1]
    for df in DFs:
#        Id = lista2[0]
        signal = df['skin conductance']
        before.append(signal)
        
        df_roll = pd.DataFrame(signal)
        
        window = 
        smoothed = df_roll.rolling(window=window).mean().values
        after.append(smoothed)

#%%
        
        
        
        
        
        
        
    

