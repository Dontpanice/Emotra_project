from my_functions_emotra import get_important_indexses2,my_svm_model,train_hmm,calc_settling_risetime,find_arclen,get_area_and_peak,calc_arclen,get_lin,korrekt_seq,cut_important,get_important_indexses,smooth,plot_individuals_with_sound_reaction,plot_individuals_with_sound,classifier1,classifier,baseline_classifier_median,baseline_classifier_mean,plot_all,separate_skinC,remove_at_indexes,plot_individuals_in_segment,compare_similar_means,cut_segment_of_df,merging,extract_signal,extract_signal2, separate, get_median_and_means, find_index_sound
import numpy as np
import pickle
import matplotlib.pyplot as plt

#from my_functions_emotra import get_important_indexses2,my_svm_model,train_hmm,calc_settling_risetime,find_arclen,get_area_and_peak,calc_arclen,get_lin,korrekt_seq,cut_important,get_important_indexses,smooth,plot_individuals_with_sound_reaction,plot_individuals_with_sound,classifier1,classifier,baseline_classifier_median,baseline_classifier_mean,plot_all,separate_skinC,remove_at_indexes,plot_individuals_in_segment,compare_similar_means,cut_segment_of_df,merging,extract_signal,extract_signal2, separate, get_median_and_means, find_index_sound

#import random 
import pandas as pd
#import copy
#import time

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from sklearn import datasets
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix



#import cPickle as cpickle

#import neurokit


#%%
# =============================================================================
#                             Extract data                              
# =============================================================================
# --------------------------------------------  Extract all labels from files, save in "label_data"
df_labels = extract_labels('../Rapporter_cleaned_2/')
# --------------------------------------------- Extract all Highgrade-signaldata from files, save in "signal_data"
signal_data_list, lost_signal_data_list = extract_signal('../Eudor-a_data_2/')
# --------------------------------------------- Extract all mediumgrade-signaldata from files, save in "signal_data2"   
signal_data_list2 = extract_signal2('../Eud_data/')
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


#with open('Picled_data.pickle', 'wb') as f:
#    pickle.dump(All_data, f)

#%%
# =============================================================================
#    Remove SCL with Moving window technique  (must happen before segemnting)
# =============================================================================


with open('Processed_EDA.pickle', 'wb') as f:
    pickle.dump(processed_signals, f)

processed_signals = []

for lista in All_data:
      df = lista[1]
      EDA  = df['skin conductance']
#      label  = df['label'][0]
      
      
      
      processed_eda = neurokit.eda_process(EDA, sampling_rate=195, alpha=0.0008, gamma=0.01,  scr_method='makowski', scr_treshold=0.1)
      
##      df = processed_eda['df']
      phasic = processed_eda['df']['EDA_Phasic']
#      EDA = processed_eda['df']['EDA_Raw']
##      Filtered = processed_eda['df']['EDA_Filtered']
      tonic = processed_eda['df']['EDA_Tonic']

      processed_signals.append(processed_eda)   
#      processed_signals.append(EDA)





#%%
# =============================================================================
#                 plot a couple of whole sequences POST-CVXalgorithm
# =============================================================================


for idx,lista in enumerate(All_data[:10]):
      df = lista[1]
      tonic = df['EDA_tonic']
      phasic = df['EDA_Phasic']

      #plot
      plt.figure()
      plt.plot(phasic)
      plt.plot(tonic)
      plt.xlabel('Time(195 samples per second) ')
      plt.ylabel('Skin conductance μS')
      plt.savefig('D:/Master_thesis_data/Emotra_preprocessed/Emotra_project/Algorithm_extraction_plot/' + 'plot_' + str(idx) + '.png')
      






#%%
# =============================================================================
#    plot a couple of mini sequences after segmenting with POST-CVXalgorithm
# =============================================================================


for idx,lista in enumerate(all_segments[:50]):
      minilista = lista[1]
      for idx2,df in enumerate(minilista):
            tonic = df['EDA_tonic'].reset_index(drop=True)
            phasic = df['EDA_Phasic'].reset_index(drop=True)

            #Plot
            plt.figure()
            plt.plot(phasic)
            plt.xlabel('Time(195 samples per second) ')
            plt.ylabel('Skin conductance μS')
            plt.savefig('D:/Master_thesis_data/Emotra_preprocessed/Emotra_project/miniplots_afterALG/' + 'plot_' + str(idx) +' - '+ str(idx2) + '.png')

#%%
            
# =============================================================================
#    plot a couple of mini sequences after segmenting with PRE-CVXalgorithm
# =============================================================================
            
for idx,lista in enumerate(all_segments[:50]):
      minilista = lista[1]
      for idx2,df in enumerate(minilista):
            SC = df['skin conductance'].reset_index(drop=True)

            #Plot
            plt.figure()
            plt.plot(SC)
            plt.xlabel('Time(195 samples per second) ')
            plt.ylabel('Skin conductance μS')

            plt.savefig('D:/Master_thesis_data/Emotra_preprocessed/Emotra_project/miniplots_beforeALG/' + 'plot_' + str(idx) +' - '+ str(idx2) + '.png')
#%%
# =============================================================================
#                       SAVE Necessary Datasets for Analysis
# =============================================================================
#with open('all_segments_4s.pickle', 'wb') as f:
#    pickle.dump(all_segments, f)
#    
#with open('all_segments_8s.pickle', 'wb') as f:
#    pickle.dump(all_segments2, f)
    
    
#%% 
# =============================================================================
#                       Load Necessary Datasets for Analysis
# =============================================================================
with open('./Picled_data.pickle', 'rb') as f:
    All_data = pickle.load(f)
    
with open('./Processed_EDA.pickle', 'rb') as f:
    Processed_EDA = pickle.load(f)

#with open('all_segments_4s.pickle', 'rb') as f:
#    all_segments_4s = pickle.load(f)
#
#with open('all_segments_8s.pickle', 'rb') as f:
#    all_segments_8s = pickle.load(f)

#
# =============================================================================
#                 Merge All data with tonic and phasic values.
# =============================================================================

for idx,element in enumerate(Processed_EDA):
     
      df = element['df']
      EDA_stats = element['EDA']
      
      #Extract data from algorithm
      N_amplitudes = len(EDA_stats['SCR_Peaks_Amplitudes'])
      SCR_Peaks = df['SCR_Peaks']
      SCR_Recoveries = df['SCR_Recoveries']
      SCR_Onsets = df['SCR_Onsets']
      Phasic = df['EDA_Phasic']
      tonic = df['EDA_Tonic']

      # merge with All_data frame
      All_df = All_data[idx][1]
      All_df['Nr_amp'] = N_amplitudes
      All_df['SCR_Peaks'] = SCR_Peaks.values
      All_df['SCR_Recoveries'] = SCR_Recoveries.values
      All_df['SCR_Onsets'] = SCR_Onsets.values
      All_df['EDA_tonic'] = tonic.values
      All_df['EDA_Phasic'] = Phasic.values
#%%
# =============================================================================
#                       Cut Dataframe into 4s segments
# =============================================================================
#indexes_to_cut = get_important_indexses(All_data)
#all_segments = cut_important(All_data,indexes_to_cut)
#%
# =============================================================================
#                       Cut Dataframe into 8s segments
# =============================================================================
indexes_to_cut_big = get_important_indexses2(All_data) 
all_segments = cut_important(All_data,indexes_to_cut_big)      
#%%
#for lista in all_segments[:2]:
#      minilista = lista[1]
#      for df in minilista:
#            EDA = df['skin conductance']
#            processed_eda = neurokit.eda_process(EDA, sampling_rate=195, alpha=0.0008, gamma=0.01,  scr_method='makowski', scr_treshold=0.1)
#            plt.figure()
#            plt.plot(EDA)
#            plt.show()
#      
#      
#      
#      ##      df = processed_eda['df']
##            phasic = processed_eda['df']['EDA_Phasic']
#      #      EDA = processed_eda['df']['EDA_Raw']
#      ##      Filtered = processed_eda['df']['EDA_Filtered']
#            tonic = processed_eda['df']['EDA_Tonic']
#            plt.figure()
#            plt.plot(tonic)
#            plt.show()

#            processed_signals.append(processed_eda)   
#      processed_signals.append(EDA)



#%%
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
#                             Get peak count from segment 
# =============================================================================

H_N_amp = []
R_N_amp = []      
        
for lista in all_segments:
    dfs = lista[1]
    seven_N_amp = []
    for df in dfs:
#        print(df.columns)
        label = df['label'].iloc[0]
        N_amp = df['Nr_amp'].iloc[0] 

        
    if label == 0:        
        R_N_amp.append(N_amp)
    elif label == 1:        
        H_N_amp.append(N_amp)    
# =============================================================================
#                        Get mean and median for each sequence 
# =============================================================================



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

##calculate relative means
#H_mean = np.mean(H_segmentmeans)
#R_mean = np.mean(R_segmentmeans)

#R_relativ_means = [abs(item)/H_mean for item in R_segmentmeans]
#H_relativ_means = [abs(item)/R_mean for item in H_segmentmeans]


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

Dataset_H['N_amp'] = H_N_amp
Dataset_R['N_amp'] = R_N_amp

# add relative means
#Dataset_R['relative_mean'] = R_relativ_means
#Dataset_H['relative_mean'] = H_relativ_means

Dataset = Dataset.append(Dataset_R)
Dataset = Dataset.append(Dataset_H)
import sklearn.utils
Dataset = sklearn.utils.shuffle(Dataset)
#

# =============================================================================
#       ------------------------ MODELS---------SVM--------------
# =============================================================================

# =============================================================================
#   Change here to consider different features in dataset before feeding SVM
# =============================================================================


# =============================================================================
#1. Load Dataset / get all combinations of features 
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


dfs = [Area,Amplitude,Risetime,Settlingetime,Arclength,HeartRate,means,N_amp]
dfs_idx = [0,1,2,3,4,5,6,7]
#dfs_names = ['Area','Amplitude','Risetime','Settlingetime','Arclength']
dfs_names = ['AE','AM','RT','ST','ARC','HR','M','N.amp']
dfs_data_n_names = list(zip(dfs,dfs_names))


from itertools import combinations
# Get all combinations of dataframes and corresponding names. 

all_comb = []
for i in range (1,len(dfs_data_n_names)+1):
# Get all combinations of length 2 
    comb = list(combinations(dfs_data_n_names,i))
    all_comb.append(comb)

check = []
All_comb_res = []
All_avgs = []
DF_sets = []       
for combinationss in all_comb:
    for combi in combinationss:
        
        df = pd.DataFrame( index = np.arange(0,len(all_segments)))
        name = ''
        set_of_comb=[]
        
        for pair in combi:
           df = df.join(pair[0])
           name = name + ' + ' + pair[1]
            
        DF_sets.append([df,name])

 

# =============================================================================
#                   Parameter optimization for following all features:
# =============================================================================
#

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score  
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import sklearn.metrics
# =============================================================================
#            Calculate results for each feature in set. 
# =============================================================================
All_res = []        

param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['rbf']}
#initiate dataframe for results values.

#[rec, f1, acc, pre,best_parameters, rec2, name]
Avg_res_df = pd.DataFrame(columns=['recall','f1','precision','accuracy','parameters','recall2','name'])  

for idx,set_ in enumerate(DF_sets):
    
    df = set_[0]
    name = set_[1]
    
    X = df
    y = Label
    print('\n')
    print('This is test for : ',name)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # set to cv=5 and n_jobs=4 
    grid = GridSearchCV(SVC() ,param_grid,refit = True, verbose=2, scoring = 'f1')
    
    grid.fit(X_train,y_train)
    
    print("The best parameters are %s with a recall score of %0.2f"
          % (grid.best_params_, grid.best_score_))
     
    
    y_pred = grid.predict(X_test)
    
    rec2 = grid.best_score_
    best_parameters = grid.best_params_
    rec = recall_score(y_pred,y_test)
    f1 = f1_score(y_pred,y_test)
    acc = accuracy_score(y_pred,y_test)
    pre = precision_score(y_pred,y_test)
    results = [rec, f1, acc, pre,best_parameters, rec2, name]
    

    Avg_res_df.loc[idx] = [rec, f1, acc, pre,best_parameters, rec2, name]
    
    

All_comb_res.append(All_res)
All_avgs.append(Avg_res_df)
#%%        
# ============================================================================
#                       Save results in xlsx format
# =============================================================================
    
result_df_export = pd.DataFrame(columns=['accuracy','precision','recall','F1','Combination'])  
for lista in All_comb_res:
    for res in lista: 
        result_df_export= result_df_export.append(res)

result_df_export.to_excel(excel_writer = 'SVM_results/SVM_results_cross.xlsx')

result_df_export_avg = pd.DataFrame(columns=['accuracy','precision','recall','F1','Combination'])  
for df in All_avgs:
    result_df_export_avg = result_df_export_avg.append(df)


result_df_export_avg.to_excel(excel_writer = 'SVM_results/SVM_results_avg.xlsx')











#%%
# ================================================================================================================================================
#                        Kmeans  on raw segmented data. 
# ================================================================================================================================================
 
#%%



# =============================================================================
#                   Extract sequences for clustering
# =============================================================================
H_seqs = []
R_seqs = []

for lista in All_data:
    label = lista[1]['label'].iloc[0]
    EDA = lista[1]['skin conductance']
    if label == 0:
        R_seqs.append(EDA)
    elif label == 1:
        H_seqs.append(EDA)
        
        
        

import numpy
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler       
seed = 0
numpy.random.seed(seed)
# =============================================================================
#              project difference between clusters?
# =============================================================================
#assign id to each individual
for idx,lista in enumerate(All_data):
    lista.append(idx)

A_seqs = []
for lista in All_data:
    ID = lista[2]
    EDA = lista[1]['skin conductance']
    A_seqs.append([ID,EDA])

A_seqs_noid = [n[1] for n in A_seqs]

#cut into same shapes
A_seqs_noid = [n[:63398] for n in A_seqs_noid ]
        
A_seqs_noid = np.array(A_seqs_noid)


X_train = A_seqs_noid
#length of sequence, 63398/195hz = 325 ponts per secons resample to that sieze
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)  # Keep only 50 time series
X_train = TimeSeriesResampler(sz=325).fit_transform(X_train)  # Make time series shorter
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=9, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

plt.figure()
for yi in range(9):
    plt.subplot(3, 3, yi + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    if yi == 1:
        plt.title("Euclidean $k$-means")



# get position from number cluster postion belonging to 1
id_list1 =  [idx for idx,n in enumerate(y_pred) if n == 1]
id_list6 =  [idx for idx,n in enumerate(y_pred) if n == 6]

cluster1 = []
for ID in id_list1:
    cluster1.append(All_data[ID])
    
cluster6 = []
for ID in id_list6:
    cluster6.append(All_data[ID])



count1 = [n[1]['label'].iloc[0] for n in cluster6]
count1 = np.array(count1)
np.count_nonzero(count1 == 1)


count2 = [n[1]['label'].iloc[0] for n in cluster1]
count2 = np.array(count2)
np.count_nonzero(count2 == 1)


#%%
# =============================================================================
#               cut and make sequences into correct chapr for Kmeans
# =============================================================================
        
lengths = [len(n) for n in H_seqs]
lens = np.min(lengths) #63398

#cut into same shapes
H_seqs = [n[:63398] for n in H_seqs ]

    
lengths = [len(n) for n in R_seqs]
lens = np.min(lengths) #63398

#cut into same shapes
R_seqs = [n[:63398] for n in R_seqs ]


A_seqs = R_seqs + H_seqs

R_seqs = np.array(R_seqs)
H_seqs = np.array(H_seqs)
A_seqs = np.array(A_seqs)



#%%
#X = R_seqs
#Y = np.array([1]*749).reshape(749,1)

X = R_seqs
Xtrain = X
#Y = np.array([1]*749).reshape(749,1)

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)



#length of sequence, 63398/195hz = 325 ponts per secons resample to that sieze

X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)  # Keep only 50 time series
X_train = TimeSeriesResampler(sz=325).fit_transform(X_train)  # Make time series shorter
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=9, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

plt.figure()
for yi in range(9):
    plt.subplot(3, 3, yi + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    if yi == 1:
        plt.title("Euclidean $k$-means")






#%%        
# DBA-k-means
print("DBA k-means")
dba_km = TimeSeriesKMeans(n_clusters=3, n_init=2, metric="dtw", verbose=True, max_iter_barycenter=10, random_state=seed)
y_pred = dba_km.fit_predict(X_train)

for yi in range(3):
    plt.subplot(3, 3, 4 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    if yi == 1:
        plt.title("DBA $k$-means")

# Soft-DTW-k-means
print("Soft-DTW k-means")
sdtw_km = TimeSeriesKMeans(n_clusters=3, metric="softdtw", metric_params={"gamma_sdtw": .01},
                           verbose=True, random_state=seed)
y_pred = sdtw_km.fit_predict(X_train)

for yi in range(3):
    plt.subplot(3, 3, 7 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    if yi == 1:
        plt.title("Soft-DTW $k$-means")

plt.tight_layout()
plt.show()

#%%

# =============================================================================
#                   concatinate all segemnts and perform clustering
# =============================================================================
R = []
H = []
A = []
for lista in all_segments:
    df= lista[1]
    label = df[1]['label'].iloc[0]
    
#    concated = [n['skin conductance'].tolist() for n in df]
#    concated = sum(concated,[])
#    A.append(np.array(concated))    
    
    
    if label == 0:
        
        concated = [n['skin conductance'].tolist() for n in df]
        concated = sum(concated,[])
        R.append(np.array(concated))
    elif label == 1:
        concated = [n['skin conductance'].tolist() for n in df]
        concated = sum(concated,[])
        H.append(np.array(concated))


#R = np.array(R).reshape(263,1)
del H[250]
A = H + R
A = np.array(A)
H = np.array(H)
R = np.array(R)


#%%
#perform clustering
#data for hypo
#X = H
#Y = np.array([1]*262).reshape(262,1)


#data for Reactive
#X = R
#Y = np.array([0]*749).reshape(749,1)

# data for all concatinated
X = A
Y = np.array([0]*1011).reshape(1011,1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)




X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)  # Keep only 50 time series
X_train = TimeSeriesResampler(sz=325).fit_transform(X_train)  # Make time series shorter
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=6, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

plt.figure()
for yi in range(6):
    plt.subplot(3, 3, yi + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    if yi == 1:
        plt.title("Euclidean $k$-means")
        

#%%
# =============================================================================
#                   individual segemnts and perform clustering
# =============================================================================
        
        
R = []
H = []
for lista in all_segments:
    df= lista[1]
    label = df[1]['label'].iloc[0]
    if label == 0:
        
        concated = [n['skin conductance'].tolist() for n in df]
#        concated = sum(concated,[])
        R.append(np.array(concated))
    elif label == 1:
        concated = [n['skin conductance'].tolist() for n in df]
#        concated = sum(concated,[])
        H.append(np.array(concated))

hypo = []
for n in H:
    for n2 in n:
        hypo.append(n2)
        
reactive = []
for n in R:
    for n2 in n:
        reactive.append(n2)

        
X = np.array(reactive)
Y = np.array([0]*5243).reshape(5243,1)        
        
#X = np.array(hypo)
#Y = np.array([1]*1842).reshape(1842,1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)




X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)  # Keep only 50 time series
X_train = TimeSeriesResampler(sz=325).fit_transform(X_train)  # Make time series shorter
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=6, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

plt.figure()
for yi in range(6):
    plt.subplot(3, 3, yi + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    if yi == 1:
        plt.title("Euclidean $k$-means")        
































#%%             Baseline classifier for 7 segements

# ================================================================================================================================================
#                       PLOT BASELINE
# ================================================================================================================================================

H_segmentmeans_log = [np.log(np.mean(item)) for item in H_sequences]
R_segmentmeans_log = [np.log(np.mean(item)) for item in R_sequences]


bins = np.arange(np.min(R_segmentmeans_log), np.max(R_segmentmeans_log), 0.4)

plt.hist(R_segmentmeans_log, bins =bins,alpha = 1, color = 'green' )
plt.hist(H_segmentmeans_log, bins =bins,alpha = 0.5,color = 'red' )
plt.ylabel('Frequency')
plt.xlabel('Log(µS)')
plt.title('Frequency of sequences with a specific µS mean')
plt.show() 
    

#%%
from sklearn.metrics import accuracy_score, recall_score,f1_score,precision_score
# ============================================================================
#                               classify baseline
# =============================================================================
limitH = np.mean(H_segmentmeans_log)
limitR = np.mean(R_segmentmeans_log)


means_log = Dataset['means']
label = Dataset['Label']

classified = [0 if n > limitH else 1 for n in means_log]

print('Mean classifier')

print('Accuracy',accuracy_score(label, classified, normalize=True))
print('Prec    ', precision_score(label,classified))
print('Recall  ', recall_score(label, classified))
print('f1      ',f1_score(label,classified))

limitH = np.median(H_segmentmeans_log)
limitR = np.median(R_segmentmeans_log)

classified = [0 if n > limitH else 1 for n in means_log]

print('Median classifier')

print('Accuracy',accuracy_score(label, classified, normalize=True))
print('Prec    ', precision_score(label,classified))
print('Recall  ', recall_score(label, classified))
print('f1      ',f1_score(label,classified))


#accuracy_score(label, classified, normalize=False)

