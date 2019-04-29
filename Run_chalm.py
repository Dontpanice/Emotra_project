from my_functions_remake2 import train_hmm,calc_settling_risetime,find_arclen,get_area_and_peak,calc_arclen,get_lin,korrekt_seq,cut_important,get_important_indexses,smooth,plot_individuals_with_sound_reaction,plot_individuals_with_sound,classifier1,classifier,baseline_classifier_median,baseline_classifier_mean,plot_all,separate_skinC,remove_at_indexes,plot_individuals_in_segment,compare_similar_means,cut_segment_of_df,merging,extract_signal,extract_signal2, extract_labels, separate, get_median_and_means, find_index_sound
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random 
import pandas as pd
import copy
from pomegranate import HiddenMarkovModel,NormalDistribution
import time

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import KFold


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

with open('All_data_ready.pickle', 'rb') as f:
    All_data = pickle.load(f)
    
    
#%%
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

# list of segment sequences                                            list = [sequence1], [sequence2]
#R_sequences
#H_sequences
# list of segment sequences individual Areas  and peaks                sequence = [area1,area2,area3....]
#R_area
#R_peak
# list of segment sequences concatinated Areas and peaks               sequence = [area] 
#H_area_List
#H_peak_List
      
# =============================================================================
#                       calculate arclength
# =============================================================================
# feed list of sequences. list = [sequence1], [sequence2] , returns list
#output = find_arclen(H_sequences)
#output = find_arclen(R_sequences)

# =============================================================================
#                       calculate settling time & rise time measures
# =============================================================================
# feed list of sequences. list = [sequence1], [sequence2] , returns list
risetime_L,amplitud_L,settlingtime_L = calc_settling_risetime(H_sequences)
risetime_L,amplitud_L,settlingtime_L = calc_settling_risetime(R_sequences)





# =============================================================================
#       ---------------------------MODELS---------HMM--------------
# =============================================================================

#%%
# combine small segments for HMM analysis:
def concat_segments(sequences):
    sequences_con = []
    concated_segs = np.array([])
    for segments_list in sequences:
        concated_segs = np.append(concated_segs,segments_list)
        sequences_con.append(pd.Series(concated_segs))
        concated_segs = np.array([])
    return sequences_con

H_concat_sequences = concat_segments(H_sequences)
R_concat_sequences = concat_segments(R_sequences)
 
#test1 = H_concat_sequences[:10]
#test2 = R_concat_sequences[:10]

# =============================================================================
#                         Train HMM models and save results in Dataframe ""
# =============================================================================


#del H_concat_sequences[250] 
#%%
Nfold = 5
# saved_models_hypo = [ [model,test] , [model2,test2] ...]
saved_models_hypo = train_hmm(H_concat_sequences,Nfold)
saved_models_reac = train_hmm(R_concat_sequences,Nfold)



models = list(zip(saved_models_hypo,saved_models_reac))
results = pd.DataFrame(columns=['accuracy','precision','recall','TP','FP','TN','FN'])
for idx,model in enumerate(models):
    hypo_mod = model[0][0]
    reac_mod = model[1][0]
    
    h_test = model[0][1]
    r_test = model[1][1]
    
    print(r_test)
    
    pred_reac, acc = classifier(r_test,reac_mod,hypo_mod,'r')
    
    N = len(r_test) 
    TP = N * acc
    FP = N * (1-acc)
    
    pred_hypo,acc = classifier(h_test,reac_mod,hypo_mod,'h')
    
    N = len(h_test) 
    TN = N * acc
    FN = N * (1-acc)
    
    # control
    if TP == 0:
        TP = 1
    elif FP == 0:
        FP = 1
    elif TN == 0:
        TN = 1
    elif FN == 0:
        FN = 1
    
    accuracy = (TP+TN) / (TP+TN+FP+FN)
    precision = (TP) / (TP+FP)
    recall = (TP) / (TP+FN)
    
    print('\naccuracy : ', accuracy)
    print('precision : ', precision) 
    print('recall : ', recall)
    print('\nTP = ', TP,'\nTN = ', TN,'\nFP = ', FP,'\nFN = ', FN)
    
    data = [accuracy,precision,recall,TP,FP,TN,FN]
#    df = pd.DataFrame(columns=['accuracy','precision','recall','TP','FP','TN','FN'],data=data)
    
    results.loc[idx] = data
    
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
#       ------------------------ MODELS---------SVM--------------
# =============================================================================

# =============================================================================
#                         Wrong , concatinead
# =============================================================================
# Create Dataset
Dataset_H = pd.DataFrame(columns=['Area','Amplitude','Label'])
for idx,value in enumerate(H_area_List):
    area = H_area_List[idx]
    amplitude = H_peak_List[idx]
    Dataset_H.loc[idx] = [area,amplitude,1]
Dataset_R = pd.DataFrame(columns=['Area','Amplitude','Label'])
for idx,value in enumerate(R_area_List):
    area = R_area_List[idx]
    amplitude = R_peak_List[idx]
    Dataset_R.loc[idx] = [area,amplitude,0]
Dataset = pd.DataFrame(columns=['Area','Amplitude','Label'])
Dataset = Dataset.append(Dataset_H)
Dataset = Dataset.append(Dataset_R)

Dataset.Area = Dataset.Area.astype(float)
Dataset.Amplitude = Dataset.Amplitude.astype(float)
Dataset.Label = Dataset.Label.astype(int)

#%%
# =============================================================================
#                        separately
# =============================================================================

# Create Dataset
Dataset_H = pd.DataFrame(columns=['Area','Amplitude','Area2','Amplitude2','Area3','Amplitude3','Area4','Amplitude4','Area5','Amplitude5','Area6','Amplitude6','Area7','Amplitude7','Label'])
for i in range(0,len(H_area)):
    LA  = H_area[i]
    LP = H_peak[i]
    l2 = []
    for j in range(0,7):
        area = LA[j]
        peak = LP[j]
        l2.append(area)
        l2.append(peak)

    l2.append(1)
#    print(len(l2))
    Dataset_H.loc[i] = l2        


Dataset_R = pd.DataFrame(columns=['Area','Amplitude','Area2','Amplitude2','Area3','Amplitude3','Area4','Amplitude4','Area5','Amplitude5','Area6','Amplitude6','Area7','Amplitude7','Label'])
for i in range(0,len(R_area)):
    LA  = R_area[i]
    LP = R_peak[i]
    l2 = []
    for j in range(0,7):
        area = LA[j]
        peak = LP[j]
        l2.append(area)
        l2.append(peak)

    l2.append(0)
#    print(len(l2))
    Dataset_R.loc[i] = l2                


Dataset = pd.DataFrame(columns=pd.DataFrame(columns=['Area','Amplitude','Area2','Amplitude2','Area3','Amplitude3','Area4','Amplitude4','Area5','Amplitude5','Area6','Amplitude6','Area7','Amplitude7','Label']))
Dataset = Dataset.append(Dataset_H)
Dataset = Dataset.append(Dataset_R)


#%%   
X = Dataset.iloc[:, np.arange(0,14)]
y = Dataset.iloc[:, 14]

#X = Dataset.iloc[:, [0,1]]
#y = Dataset.iloc[:, 2]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting classifier to the Training set
from sklearn.svm import SVC

classifier = SVC(kernel ='rbf', random_state = 0)
classifier.fit(X_train,y_train) 

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#%%

from sklearn.model_selection import cross_val_score

clf = SVC(kernel='rbf', C=1,gamma='auto')
scores = cross_val_score(clf, X,y, cv=5)

np.average(scores)                                              
  

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
