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



#%%
# =============================================================================
#                        Merge Data together
# =============================================================================

# Create Dataset
Dataset_H = pd.DataFrame(columns=['Area','Area2','Area3','Area4','Area5','Area6','Area7','Amplitude','Amplitude2','Amplitude3','Amplitude4','Amplitude5','Amplitude6','Amplitude7','Risetime','Risetime2','Risetime3','Risetime4','Risetime5','Risetime6','Risetime7','Settlingtime','Settlingtime2','Settlingtime3','Settlingtime4','Settlingtime5','Settlingtime6','Settlingtime7','Arclength1','Arclength2','Arclength3','Arclength4','Arclength5','Arclength6','Arclength7','Label'])
for i in range(0,len(H_area)):
    LA  = H_area[i]
    LP = H_peak[i]
    RT = H_risetime_L[i]
    ST = H_settlingtime_L[i]
    ARC = H_arc[i]
    
    A_L = []
    P_L = []
    R_L = []
    S_L = []
    ARC_L = []
    l2 = []
    
    for j in range(0,7):
        
        area = LA[j]
        peak = LP[j]
        risetime = RT[j]
        settlintime = ST[j]
        ARClen = ARC[j][0]
        
        A_L.append(area)
        P_L.append(peak)
        R_L.append(risetime)
        S_L.append(settlintime)
        ARC_L.append(ARClen)

    l2.append(1)
#    print(len(l2))
    data = A_L + P_L + R_L + S_L + ARC_L + l2
    Dataset_H.loc[i] = data        

Dataset_R = pd.DataFrame(columns=['Area','Area2','Area3','Area4','Area5','Area6','Area7','Amplitude','Amplitude2','Amplitude3','Amplitude4','Amplitude5','Amplitude6','Amplitude7','Risetime','Risetime2','Risetime3','Risetime4','Risetime5','Risetime6','Risetime7','Settlingtime','Settlingtime2','Settlingtime3','Settlingtime4','Settlingtime5','Settlingtime6','Settlingtime7','Arclength1','Arclength2','Arclength3','Arclength4','Arclength5','Arclength6','Arclength7','Label'])
for i in range(0,len(R_area)):
    LA  = R_area[i]
    LP = R_peak[i]
    RT = R_risetime_L[i]
    ST = R_settlingtime_L[i]
    ARC = R_arc[i]
    
    A_L = []
    P_L = []
    R_L = []
    S_L = []
    l2 = []
    ARC_L = []
    
    for j in range(0,7):
        
        area = LA[j]
        peak = LP[j]
        risetime = RT[j]
        settlintime = ST[j]
        ARClen = ARC[j][0]
        
        A_L.append(area)
        P_L.append(peak)
        R_L.append(risetime)
        S_L.append(settlintime)
        ARC_L.append(ARClen)
        
    l2.append(0)
#    print(len(l2))
    data = A_L + P_L + R_L + S_L + ARC_L + l2
    Dataset_R.loc[i] = data                


Dataset = pd.DataFrame(columns=['Area','Area2','Area3','Area4','Area5','Area6','Area7','Amplitude','Amplitude2','Amplitude3','Amplitude4','Amplitude5','Amplitude6','Amplitude7','Risetime','Risetime2','Risetime3','Risetime4','Risetime5','Risetime6','Risetime7','Settlingtime','Settlingtime2','Settlingtime3','Settlingtime4','Settlingtime5','Settlingtime6','Settlingtime7','Arclength1','Arclength2','Arclength3','Arclength4','Arclength5','Arclength6','Arclength7','Label'])
Dataset = Dataset.append(Dataset_H)
Dataset = Dataset.append(Dataset_R)


#%%   


from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


# =============================================================================
#             Change here to consider different features in dataset
# =============================================================================
# Area + amplitude
#X = Dataset.iloc[:, np.arange(0,14)]
#y = Dataset.iloc[:, -1]

# Area + amplitude + raiseing time 
#X = Dataset.iloc[:, np.arange(0,21)]
#y = Dataset.iloc[:, -1]

# Area + amplitude + raiseing time +settlingtime
#X = Dataset.iloc[:, np.arange(0,28)]
#y = Dataset.iloc[:, -1]

# Area + amplitude + raiseing time +settlingtime + ARClength
X = Dataset.iloc[:, np.arange(0,35)]
y = Dataset.iloc[:, -1]


results_df = pd.DataFrame(columns=['accuracy','precision','recall','F1'])
results = []

kf = KFold(n_splits=5, shuffle = True)
cm_L = []
for train_index,test_index in kf.split(X):
#   print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    # Fitting classifier to the Training set

    
    classifier = SVC(kernel ='rbf', random_state = 0)
    classifier.fit(X_train,y_train) 
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    
    cm_L.append(cm)
    
    TN, FP, FN, TP = cm.ravel()
#
    accuracy = (TP+TN) / (TP+TN+FP+FN)
    precision = (TP) / (TP+FP)
    recall = (TP) / (TP+FN)
    F1 = 2*((precision*recall)/(precision+recall))
    
    
    data = [accuracy,precision,recall,F1]
    results.append(data)
    
    

for idx,lista in enumerate(results):
#    print(lista)
    results_df.loc[idx] = lista 
#%%
accuracy = np.mean(results_df['accuracy'])
presicion = np.mean(results_df['precision'])
recall = np.mean(results_df['recall'])
F1 = np.mean(results_df['F1'])

print('AVG accuracy  : ',accuracy) 
print('AVG presicion : ',presicion) 
print('AVG recall    : ',recall) 
print('AVG F1        : ',F1)   

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
