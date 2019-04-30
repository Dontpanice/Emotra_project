from my_functions_emotra import my_hmm,train_hmm,calc_settling_risetime,find_arclen,get_area_and_peak,calc_arclen,get_lin,korrekt_seq,cut_important,get_important_indexses,smooth,plot_individuals_with_sound_reaction,plot_individuals_with_sound,classifier1,classifier,baseline_classifier_median,baseline_classifier_mean,plot_all,separate_skinC,remove_at_indexes,plot_individuals_in_segment,compare_similar_means,cut_segment_of_df,merging,extract_signal,extract_signal2, extract_labels, separate, get_median_and_means, find_index_sound
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

with open('D:/Master_thesis_data/Emotra_preprocessed/All_data_ready.pickle', 'rb') as f:
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
 

#remove id250 because wrong length
del H_concat_sequences[250]


# =============================================================================
#                         Train HMM models and save results in Dataframe ""
# =============================================================================
#%%
# =============================================================================
#                         Without cross validation
# =============================================================================
# saved_models_hypo = [ [model,test] , [model2,test2] ...]

H_concat_sequences = [np.array(n) for n in H_concat_sequences]
R_concat_sequences = [np.array(n) for n in R_concat_sequences]

#model_Hypo = HiddenMarkovModel.from_samples(NormalDistribution, n_components=5, X=H_concat_sequences)

saved_models_hypo = my_hmm(H_concat_sequences)
saved_models_reac = my_hmm(R_concat_sequences)


hypo_mod = saved_models_hypo[0][0]
h_test = saved_models_hypo[0][1]
#
reac_mod = saved_models_reac[0][0]
r_test = saved_models_reac[0][1]

pred_reac, R_acc = classifier(r_test,reac_mod,hypo_mod,'r')
pred_hypo,H_acc = classifier(h_test,reac_mod,hypo_mod,'h')






#%%
# =============================================================================
#                       With cross validation
# =============================================================================
Nfold = 5
saved_models_hypo = train_hmm(H_concat_sequences,Nfold)
saved_models_reac = train_hmm(R_concat_sequences,Nfold)

finished_training = []

models = list(zip(saved_models_hypo,saved_models_reac))
results = pd.DataFrame(columns=['accuracy','precision','recall','TP','FP','TN','FN'])
for idx,model in enumerate(models):
    hypo_mod = model[0][0]
    reac_mod = model[1][0]
    
    h_test = model[0][1]
    r_test = model[1][1]
    
    print(r_test)
    
    pred_reac, R_acc = classifier(r_test,reac_mod,hypo_mod,'r')
    pred_hypo,H_acc = classifier(h_test,reac_mod,hypo_mod,'h')
    
    finished_training.append([pred_reac, R_acc])
    finished_training.append([pred_hypo, H_acc])
    
    
    
    #%%
#    N = len(r_test) 
#    TP = N * acc
#    FP = N * (1-acc)
#    
#    
#    
#    N = len(h_test) 
#    TN = N * acc
#    FN = N * (1-acc)
#    
#    # control
#    if TP == 0:
#        TP = 1
#    elif FP == 0:
#        FP = 1
#    elif TN == 0:
#        TN = 1
#    elif FN == 0:
#        FN = 1
#    
#    accuracy = (TP+TN) / (TP+TN+FP+FN)
#    precision = (TP) / (TP+FP)
#    recall = (TP) / (TP+FN)
#    
#    print('\naccuracy : ', accuracy)
#    print('precision : ', precision) 
#    print('recall : ', recall)
#    print('\nTP = ', TP,'\nTN = ', TN,'\nFP = ', FP,'\nFN = ', FN)
#    
#    data = [accuracy,precision,recall,TP,FP,TN,FN]
##    df = pd.DataFrame(columns=['accuracy','precision','recall','TP','FP','TN','FN'],data=data)
#    
#    results.loc[idx] = data
    