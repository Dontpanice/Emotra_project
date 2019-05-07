# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:50:37 2019

@author: Arnaud
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#for converting .docx file into text-file
import docx2txt
#for maniulating strings and iterate over windows folder/files
import os
import re
import time
#import json
import scipy as sp
from scipy import stats
import logging
import random 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import KFold
from pomegranate import HiddenMarkovModel,NormalDistribution
from sklearn import preprocessing
#from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def my_svm_model(X,y):
    
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
    
    accuracy = np.mean(results_df['accuracy'])
    presicion = np.mean(results_df['precision'])
    recall = np.mean(results_df['recall'])
    F1 = np.mean(results_df['F1'])
    
    print('AVG accuracy  : ',accuracy) 
    print('AVG presicion : ',presicion) 
    print('AVG recall    : ',recall) 
    print('AVG F1        : ',F1)
    return results_df

def findmin(lista):
    minimum = 99999999
    for element in lista:
        if len(element) < minimum:
            minimum = len(element)
        else:
            continue
    return minimum

def reshape(lista,minimum):
    reshaped = []
    for element in lista:
        re = element[:minimum]
        reshaped.append(re)
    return reshaped

def train_hmm(Train_hypo,Nfold):
    kf = KFold(n_splits=Nfold,shuffle = True)
    
    saved_models_hypo = []
    
    
#    minimum2 = findmin(Train_hypo)
#    reshaped_hypo = reshape(Train_hypo, minimum2)      
    scaler2 = preprocessing.StandardScaler().fit(Train_hypo)
    X_train_transformed_hypo = scaler2.transform(Train_hypo)
    random.seed( 808 )
    
    #print(kf.get_n_splits(x_Reac_train))
    for train,test in kf.split(X_train_transformed_hypo):
    #    hypo_eaxmples = random.sample(X_train_transformed_hypo, 10)
#        train = X_train_transformed_hypo[train]
#        test = X_train_transformed_hypo[test]
        hypo_eaxmples = X_train_transformed_hypo[:10]  
        model_Hypo = HiddenMarkovModel()
        #inititiera modellens parametrar från ett exempel
        model_Hypo = HiddenMarkovModel.from_samples(NormalDistribution, n_components=5, X=hypo_eaxmples)  
        #train the model
        start_time = time.time()
        model_Hypo.fit(train, algorithm='baum-welch')
        print("--- %s seconds ---" % (time.time() - start_time))
        #save the model
        saved_models_hypo.append([model_Hypo,test])
#        accompan_test_hypo.append(test)
        
    return saved_models_hypo



def my_hmm(Train_hypo):
    random.seed( 808 )
#    kf = KFold(n_splits=Nfold,shuffle = True)
#    Train_hypo = list(Train_hypo)
    saved_models_hypo = []
    

#    scaler2 = preprocessing.StandardScaler().fit(Train_hypo)
#    X_train_transformed_hypo = scaler2.transform(Train_hypo)
    
    train,test= train_test_split(Train_hypo, shuffle = True)
    
# Feature Scaling
    sc = StandardScaler()
    train = sc.fit_transform(train)
    test = sc.transform(test)
    
    
    # Fitting classifier to the Training set
    model_Hypo = HiddenMarkovModel()
    #inititiera modellens parametrar från ett exempel
    hypo_eaxmples = random.sample(Train_hypo, 10)
    model_Hypo = HiddenMarkovModel.from_samples(NormalDistribution, n_components=5, X=hypo_eaxmples)  
    #train the model
    start_time = time.time()
    model_Hypo.fit(train, algorithm='baum-welch')
    print("--- %s seconds ---" % (time.time() - start_time))
    #save the model
    saved_models_hypo.append([model_Hypo,test])
    
    # Predicting the Test set results
#    y_pred = classifier.predict(X_test)
    
    
    # Making the Confusion Matrix
#    cm = confusion_matrix(y_test, y_pred)
        

    
    
    



        
    return saved_models_hypo

def calc_settling_risetime(H_sequences):
    #reaction_measurements = []
    RT_L = []
    A_L = []
    ST_L = []
    for idx,lista in enumerate(H_sequences):
        risetime_L = []
        amplitud_L = []
        settlingtime_L = []
        for idx2,array in enumerate(lista):
            
            # to get settling and rise time
            array = array.tolist()
            amplitud = max(array)
            Peak_index = array.index(max(array))
            end = 0
            start = 0
            
            for i in range (Peak_index, len(array)):
                if array[i] < (amplitud/2) and Peak_index != 0:
                    end =  i
                    break
                else:       
                    continue     
    #        array.reverse()
            for i in range (0, len(array)):
                if array[Peak_index-i] < 0.000001:
                    start = Peak_index - i 
                    break
                else:
                    continue     
    #        print('start:', start , 'peak:',Peak_index ,'end:',end)
                    
                
        
            risetime =  Peak_index - start
            settlingtime = end - Peak_index
            
            risetime_L.append(risetime)
            amplitud_L.append(amplitud)
            settlingtime_L.append(settlingtime)
    
        RT_L.append(risetime_L)
        A_L.append(amplitud_L)
        ST_L.append(settlingtime_L)
    
    
    return RT_L,A_L,ST_L


def merging(df_labels,signal_data_list):       
    lost_matches = []
    found_matches = []
      
    for Id_label in df_labels.loc[:,'Id']:
        for signal_data in signal_data_list:
            Id_signal = signal_data[0]
            signal_df = signal_data[1]
            if Id_signal == Id_label:
                signal_df.loc[:,'label'] = df_labels.loc[Id_label,'label']
                found_matches.append(Id_signal)
            else:
                lost_matches.append(Id_signal)
                continue
            
    return lost_matches,found_matches



def extract_labels(path):
    hypo_list = []
    re_list = []
    
    for file in os.scandir(path):
        # ex: BARI1(1bb2c6c4-55f0-4628-bea5-abf7ed6c4385)
        document_text = docx2txt.process(path + file.name)
        doc_split = document_text.split("\t")
        Id = re.sub('\.docx', '', file.name)
        
        for idx,line in enumerate(doc_split):
                if 'Test Result:' in line:
                       activity = doc_split[idx+1].lower()
                       if 'hypo' in activity:
                            hypo_list.append([str(Id),1])
                               
                       elif 'reactive' in activity:
                            re_list.append([str(Id),0])
                       else:
                            print('didnt find result asnwer ' + Id )
                            break

                else:
#                        print('didnt find result: part')
                        continue
    #make performance oriented dataframe
    Columns = ["Id","label" ]
    label_data = hypo_list + re_list
    df_labels = pd.DataFrame.from_records(label_data, columns = Columns)
    df_labels.index = df_labels['Id']  
    
    return df_labels



#doc_split = document_text.split("\t")
#for idx,line in enumerate(doc_split):
#        if 'Test Result:' in line:
#               activity = doc_split[idx+1]
#               if 'Hypo' in activity:
#                       
##               print(activity)
#        else:
#                continue
#
#%%

    
#hypo_list = []
#re_list = []
#document_text = docx2txt.process("Rapporter_cleaned_2/afe1c0b7-8a9d-471b-9634-b8a8b7845036.docx")
#doc_split = document_text.split("\t")
#Id = re.sub('\.docx', '', "Rapporter_cleaned_2/afe1c0b7-8a9d-471b-9634-b8a8b7845036.docx")
#for idx,line in enumerate(doc_split):
#                if 'Test Result:' in line:
#                       activity = doc_split[idx+1].lower()
#                       if 'hypo' in activity:
#                               hypo_list.append([str(Id),1])
#                               
#                       elif 'reactive' in activity:
#                            re_list.append([str(Id),0])
#                       else:
#                            print('didnt find result asnwer')
#
#                else:
##                        print('didnt find result: part')
#                        continue
                
                




def extract_signal(path):
    start_time = time.time()
    signal_data_list = []
    lost_signal_data_list = []
    
    for idx, folder in enumerate(os.scandir(path)):
    #    print(idx, folder.name)
        folder_name = folder.name
        for file in os.scandir(path + folder_name):
    #        print(file)
            if file.name == 'signaldata.tsv':
    #            print("something")
                signal_data = pd.read_csv(path + folder_name +'/'+ file.name,header=None, sep='\t')
                 #RENAME COLUMN NAMES and remove excess data.
                signal_data = signal_data.rename(columns={10:'sound_stimuli', 8: 'skin conductance',11:'Event Markers', 3: 'Heart Rate', 5: 'Pressure1',6:'Pressure2',7:'amplification'})
                signal_data = signal_data.drop(columns = [0,1,2,4,9,12]) 
    #            signal_data.append([folder_name,signal])
                signal_data_list.append([folder.name,signal_data])
            else:
                lost_signal_data_list.append(folder_name)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return signal_data_list, lost_signal_data_list

#
#  11 – Event Markers – Inputs from the test leader to indicate that the patient did something that deviated from the normal state of waiting and listening. E.g. if the patient sneezes or starts to cough.
#> 3 – Heart Rate (AC) – Gives an approximation of the heart rate signal from the patient.
#> 4 – Heart Rate (DC) - Gives an approximation of the heart rate signal from the patient.
#> 5 – Pressure indicator 1 – Indicates pressure on one of the electrodes
#> 6 – Pressure indicator 2 - Indicates pressure on the other electrode


def extract_signal2(path):
    
    start_time = time.time()
    signal_data_list = []
    lost_signal_data_list = []
    
    for idx, folder in enumerate(os.scandir(path)):
    #    print(idx, folder.name)
        folder_name = folder.name
        for file in os.scandir(path + folder_name):
            try:
                if file.name.split(')')[-1] == 'signaldata.tsv':
        #            print("something")
                    signal_data = pd.read_csv(path + folder_name +'/'+ file.name,header=None, sep='\t')
                    
                    #RENAME COLUMN NAMES and remove excess data.
                    signal_data = signal_data.rename(columns={10:'sound_stimuli', 8: 'skin conductance',11:'Event Markers', 3: 'Heart Rate', 5: 'Pressure1',6:'Pressure2',7:'amplification'})
                    signal_data = signal_data.drop(columns = [0,1,2,4,9,12])          
                    
        #            signal_data.append([folder_name,signal])
                    signal_data_list.append([folder.name,signal_data])
                else:
                    lost_signal_data_list.append(folder_name)
            except:
                print('error')
                
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return signal_data_list


#--------------------------------------------------------- SEPARATE HYPO DATA WITH REACTIVE DATA ---------------------------------------------------
# I want to keep id if errors occur
def separate(signal_data_list):
    
    hypo_sequence = []
    Reac_sequence = []
    for idx, lista in enumerate(signal_data_list):
        try:
            #reset indexing for following code to work
            ID = lista[0]
            dataframe = lista[1].reset_index(drop=True)
            print('Length is : ',len(dataframe))
            if dataframe.loc[1,'label'] == 1:
                hypo_sequence.append([dataframe.loc[:,'skin conductance'], dataframe.loc[:,'sound_stimuli']])
            elif dataframe.loc[1,'label'] == 0:
                Reac_sequence.append([dataframe.loc[:,'skin conductance'], dataframe.loc[:,'sound_stimuli']])
        except:
            print(str(idx) + " error, missing matching report/label from before, ID : " + '\n' + ID)
            
    return Reac_sequence, hypo_sequence


def plot_all(data,binss,title,up_limit,colore,N):
    fig, ax = plt.subplots(figsize=(10,7.5))
    plt.xlabel('Skin-conductance (µS)')
    plt.ylabel('Frequency')
    mu = np.mean(data)
    median = np.median(data)
    sigma = np.std(data)
    textstr = '\n'.join((
    r'$\mu=%.2f$' % (mu, ),
    r'$\mathrm{median}=%.2f$' % (median, ),
    r'$\sigma=%.2f$' % (sigma, )))

    plt.hist(data ,bins=binss, color=colore,range= (0,up_limit))
    plt.title('Density plot of' + title + ' (N ='+str(N)+')')

    # text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top',horizontalalignment ='right', bbox=props)
#    ax.set_axis_off()
    plt.show()
    fig.savefig('Density_Groups/Density_'+title+'.png')


def separate_skinC(signal_data_list):
    
    hypo_sequence = []
    Reac_sequence = []
    for idx, lista in enumerate(signal_data_list):
        try:
            #reset indexing for following code to work
            ID = lista[0]
            dataframe = lista[1].reset_index(drop=True)
#            print('Length is : ',len(dataframe))
            if dataframe.loc[1,'label'] == 1:
                hypo_sequence.append(dataframe.loc[:,'skin conductance'])
            elif dataframe.loc[1,'label'] == 0:
                Reac_sequence.append(dataframe.loc[:,'skin conductance'])
        except:
            print(str(idx) + " error, missing matching report/label from before, ID : " + '\n' + ID)
            
    return Reac_sequence, hypo_sequence



            
def find_index_sound(lista):    
    index_list = []
    for idx,element in enumerate(lista):
        if element == 1:
            index_list.append(idx)
        else:
            continue 
        
    index_list_refined = index_list[0::196]
    return index_list_refined

def get_median_and_means(lista):
    median_list = []
    mean_list = []
    for sequence in lista:
        median_list.append(np.median(sequence))
        mean_list.append(np.mean(sequence)) 
        
    return median_list, mean_list


def cut_segment_of_df(lista,start_sound,end_sound):
    thrown_out = []
    check = []
    remove_later = []
    for idx,minilista in enumerate(lista):
        #ta df och klipp efter 3dje sound stimuli
        ID = minilista[0] 
        dataframe = minilista[1]
        try:
            if len(dataframe) < 165000 or len(dataframe) > 190000:
#                print('dropped too small or too big dataframe')
                thrown_out.append([ID,dataframe])
                #Fy, ta aldrig bort element i det du loopar, din idiot. Du har spenderat flera timmar på att lösa det här. 
#                del lista[idx]
                remove_later.append(idx)
                
            else:
                soundstimuli = dataframe['sound_stimuli']
                reaction_frame = dataframe['sound_stimuli']
                
                #add soundstimuli reaction frame
                shift_right = [0]*702
                reaction_frame = shift_right + dataframe['sound_stimuli'].tolist()
                reaction_frame = reaction_frame[:len(reaction_frame)-702]
                dataframe['reaction limit'] = reaction_frame
                
                
                indices_of_1 = find_index_sound(soundstimuli)
                #klipp från 3dje stimlui till 5te stimuli. 
                start = indices_of_1[start_sound]
                # adding 780 datapoint to the end since we want to register the reaction after 4s of last stimuli , equivalent to 780 datapoints + 195 for adjusting to end point of the 1 s sound stimuli.  
                end = indices_of_1[end_sound] +780 +195
                #save location which reaction must appear.
                                 
                dataframe.drop(dataframe.index[:start-391],inplace=True)
                dataframe.drop(dataframe.index[end-start:],inplace=True)
                
                check.append([ID,len(dataframe)])
    #            minilista[1] = modified
        except BaseException as e:
            print('Failed to do something: ' + str(e) + ' - ' + str(ID))
    print(' - Ended - , returning too small dataframes which were throw out ')
    return thrown_out,check,remove_later


def join_all_signaldata(lista):
    all_signaldata = []
    for signaldata in lista:
        all_signaldata += signaldata.tolist()
    return all_signaldata



def find_arclen(H_sequences):
    arc_lengths = []
    for idx,lista in enumerate(H_sequences):
        minilista = []
        for idx2,array in enumerate(lista):
            
            #Choose n=1 for as accurate arclength as possible.
            n = 1
            arc = calc_arclen(array,n)
            minilista.append(arc)
        
        arc_lengths.append(minilista)
    return arc_lengths



def get_area_and_peak (all_segments):

    #summed Area and Peak for each 9 segemnts of sequence and in format [ [summed Area for 9segments], [etc], [etc]  ]
    H_area_List = []
    H_peak_List = []
    R_area_List = []
    R_peak_List = []
    R_sequences = []
    H_sequences = []
    
    # separate area and amplitude
    R_area = []
    R_peak = []
    
    H_area = []
    H_peak = []
    
#    H_lost_counter = 0
#    R_lost_counter = 0
#    H_counter = 0
#    R_counter = 0
    for idx,serie in enumerate(all_segments):
        R_nine_seq_A = []
        R_nine_seq_P = []
        R_nine_seq = []
        
        
        H_nine_seq_P = []
        H_nine_seq_A = []
        H_nine_seq = []    
        
        
        
        for idx2,df in enumerate(serie[1]):
            try:
                df = df.reset_index(drop = True)
                SC = df['skin conductance']
                label = df['label'].iloc[0]
                SC = korrekt_seq(SC)
                SC = np.array(SC)
                # make negtaive values into positive
                array_pos = np.array([n if n > 0 else 0.00000001 for n in SC])
                area = np.trapz(array_pos)
                peak = max(array_pos)
                
                if label == 0:
    #                corr_sequences_R.append(SC)
                    R_nine_seq_A.append(area)
                    R_nine_seq_P.append(peak)
                    R_nine_seq.append(array_pos)

                elif label == 1:
    #                H_counter +=1
                    H_nine_seq_A.append(area)
                    H_nine_seq_P.append(peak)
                    H_nine_seq.append(array_pos)
                else:
                    print('error')
                    continue
    #                print('error -', peak, '<-- peak - area ---> ', area )
                    
            except BaseException as e:
                print('Failed to do something: ' + str(e) + ' - ' + str(idx))
    
        label_s = serie[1][0]['label'].iloc[0]
    #    print(R_nine_seq_A)
    #    print(sum(R_nine_seq_A))
        R_summed_Area = sum(R_nine_seq_A) 
        R_summed_Peak = sum(R_nine_seq_P)
        H_summed_Area = sum(H_nine_seq_A)
        H_summed_Peak = sum(H_nine_seq_P)
        
        if label_s == 0:
            R_area_List.append(np.array(R_summed_Area))
            R_peak_List.append(np.array(R_summed_Peak))
            
            R_area.append(np.array(R_nine_seq_A))
            R_peak.append(np.array(R_nine_seq_P))
            
            R_sequences.append(R_nine_seq)
            
        elif label_s == 1:
            H_area.append(np.array(H_nine_seq_A))
            H_peak.append(np.array(H_nine_seq_P))
            H_area_List.append(np.array(H_summed_Area))
            H_peak_List.append(np.array(H_summed_Peak))
            H_sequences.append(H_nine_seq)
            
        else:
            print('error')
            continue
    
    #print('lost ', R_lost_counter, ' many Reactive areas and peaks', R_counter+H_lost_counter, 'which is', R_lost_counter/(R_counter+R_lost_counter),'%')
    #print('lost ', H_lost_counter, ' many Hypo areas and peaks of ', H_counter+H_lost_counter, 'which is', H_lost_counter/(H_counter+H_lost_counter),'%')
        
    return R_sequences,H_sequences,R_area,R_peak,H_area,H_peak,H_area_List,H_peak_List,R_area_List,R_peak_List




def calc_arclen(array,n):
    y = array
    x1 = np.arange(0,len(array),n)
    plist = []
    for index in x1:
        p = y[index]
        plist.append(p)
        
    lines = []    
    for i in range(0,len(plist)):
        if i == len(plist)-1:
            break
        else:
            y2 = plist[i+1]
            y1 = plist[i]
            Dy = y2-y1
            Dx = n
            line = np.sqrt(np.square(Dy) + np.square(Dx))
            lines.append(line)
        
    arc_legth = sum(lines)
        
    
    return arc_legth

def compare_similar_means(lista,lista2):
    comparable_list = []
    for signaldata in lista:
        mean1 = np.mean(signaldata)
        for signaldata2 in lista2:
         mean2 = np.mean(signaldata2)
         if mean1 - mean2 < 0.2 and mean1 - mean2 > -0.2:
             comparable_list.append([signaldata,signaldata2])
             break
         else:
             continue
    return comparable_list



def remove_at_indexes(data,lista):
    lista.sort()
    lista.reverse()    
    for index in lista:
        del data[index]
        



def plot_individuals_in_segment(lista,segment_name,colour):
    plt.ioff() 
    for i in range(0,len(lista)): 
        signal_data = lista[i][0]
        sound_stimuli = lista[i][1]
        
        fig, ax1 = plt.subplots(figsize=(30,15))
        plt.ylabel('Skin-conductance (µS)')
        plt.xlabel('Time (195Hz) , 1s = 195 samples')
        ax2 = ax1.twinx()
        
        curve, = ax1.plot(signal_data, label='curve - Reactive', color=colour)
        sound, = ax2.plot(sound_stimuli, label='sound interval - ' + segment_name, color='black', alpha=0.4)
        
        plt.legend(handles = (curve,sound), loc='upper right')
        plt.xlabel('Time (195Hz) , 1s = 195 samples')
        plt.title('Plot of reactive individual')
    
        fig.savefig('Individual_Hypo_Reactive_plots/'+segment_name+'/graph_reactive '+str(i)+'.png')
        plt.clf()
        plt.cla()
        plt.close()
    print('Finished,\n Please check Individual_Hypo_Reactive_plots/' +segment_name+'/' )
    
    
def plot_individuals_with_sound(data):
    plt.ioff() 
    for i in range(0,len(data)):
        try:
            ID = data[i][0]
            dataframe = data[i][1]
            label = stats.mode(dataframe['label'])[0].tolist()[0]
            signal = dataframe['skin conductance']
            sound = dataframe['sound_stimuli'] 
            fig, ax1 = plt.subplots(figsize=(30,15))
            plt.ylabel('Skin-conductance (µS)')
            plt.xlabel('Time (195Hz) , 1s = 195 samples')
            ax2 = ax1.twinx()

            sound, = ax2.plot(sound, label='sound interval' , color='black', alpha=0.4)
            if label == 0:
                curve, = ax1.plot(signal, label='curve - Reactive', color='green')
                plt.legend(handles = (curve,sound), loc='upper right')
                plt.xlabel('Time (195Hz) , 1s = 195 samples')
                plt.title('Plot of reactive individual')
                fig.savefig('Individual_plots/Individual_plots/graph_reactive '+str(i)+'.png')
                plt.clf()
                plt.cla()
                plt.close()
            elif label == 1:
                curve, = ax1.plot(signal, label='curve - Hypo Reactive', color='red')
                plt.legend(handles = (curve,sound), loc='upper right')
                plt.xlabel('Time (195Hz) , 1s = 195 samples')
                plt.title('Plot of Hypo Reactive individual')
                fig.savefig('Individual_plots/Individual_plots/graph_Hypo_Reactive '+str(i)+'.png')
                plt.clf()
                plt.cla()
                plt.close()
        except BaseException as e:
            print('Failed to do something: ' + str(e) + ' - ' + str(i) +' - ' + str(ID))
    print('Finished' )
    
    
def plot_individuals_with_sound_reaction(data):
    plt.ioff() 
    for i in range(0,len(data)):
        try:
            ID = data[i][0]
            dataframe = data[i][1]
            label = stats.mode(dataframe['label'])[0].tolist()[0]
            signal = dataframe['skin conductance']
            sound = dataframe['sound_stimuli']
            reaction = dataframe['reaction limit']
            
            fig, ax1 = plt.subplots(figsize=(30,15))
            plt.ylabel('Skin-conductance (µS)')
            plt.xlabel('Time (195Hz) , 1s = 195 samples')
            ax2 = ax1.twinx()
            ax3 = ax1.twinx()
            sound, = ax2.plot(sound, label='sound interval' , color='black', alpha=0.4)
            reaction, = ax3.plot(reaction, label='reaction interval' , color='orange', alpha=0.4)
            if label == 0:
                curve, = ax1.plot(signal, label='curve - Reactive', color='green')
                plt.legend(handles = (curve,sound,reaction), loc='upper right')
                plt.xlabel('Time (195Hz) , 1s = 195 samples')
                plt.title('Plot of reactive individual')
                fig.savefig('Individual_plots/Individual_plots/graph_reactive '+str(i)+'.png')
                plt.clf()
                plt.cla()
                plt.close()
            elif label == 1:
                curve, = ax1.plot(signal, label='curve - Hypo Reactive', color='red')
                plt.legend(handles = (curve,sound,reaction), loc='upper right')
                plt.xlabel('Time (195Hz) , 1s = 195 samples')
                plt.title('Plot of Hypo Reactive individual')
                fig.savefig('Individual_plots/Individual_plots/graph_Hypo_Reactive '+str(i)+'.png')
                plt.clf()
                plt.cla()
                plt.close()
        except BaseException as e:
            print('Failed to do something: ' + str(e) + ' - ' + str(i) +' - ' + str(ID))
    print('Finished' )


# Format Data = [ID,[Dataframe]]
def cut_important(data,indexes_to_cut):
    all_segments= []
    for idx,minilista in enumerate(data):
        ID = minilista[0]
        df = minilista[1]
        cut= indexes_to_cut[idx]
        segments= []
        for minicut in cut:
            start = minicut[0]
            end = minicut[1]
            
            minisegment = df[start:end]
            segments.append(minisegment)
            
        all_segments.append([ID,segments])
    
    return all_segments

from numpy import ones,vstack
from numpy.linalg import lstsq

def get_lin(p1,p2):
    points = [p1,p2]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
#    print("Line Solution is y = {m}x + {c}".format(m=m,c=c))
    return m, c

def  korrekt_seq(sequence):
    SC = sequence
#    SC = SC.reset_index(drop=True)
#    
#    plt.plot(test)
#    plt.show()
    
    SC2 = SC.tolist()
    start = SC2[0]       
    end = SC2[-1]
    
    p1 = (1,start)
    p2 = (len(SC2),end)
       
    
    k,m = get_lin(p1,p2)
    
    #print(test)
    #plt.plot(test)
    #plt.show()
    
    transformed = []
    for idx,value in enumerate(SC):
        transformed.append(value - (k*idx)) 
    
    #plt.plot(transformed)
    #plt.show()
    
    
    
    
    pull_down = []
    first_val = transformed[0]
    for value in transformed:
        pulled = value - first_val
        pull_down.append(pulled)
    
    return pull_down
        
def get_important_indexses(lista):
        indexes_to_cut = []
        for idx,minilista in enumerate(lista):
            try:
                df = minilista[1]
                ID = minilista[0]
                sound = df["sound_stimuli"]
                # get index of all 1 of soundstimuli in list
                indexlist1 = []
                for idx,soundlabel  in enumerate (sound):
                #    print(soundlabel)
                    if soundlabel == 1:
                #        print("found here ",idx)
                        indexlist1.append(idx)
                        continue
                    else:
                        continue
                
                # Now get appropriate starting poin to cut and appropriate endingpoint to cut
                idx = 0
                startindex = []
                endindex = []
                while idx < len(indexlist1)-2:
                    idx += 1
                #    print(test[idx])
                    if (indexlist1[idx+1] - indexlist1[idx]) > 3:
                        #take end of sound iddex -117 steps because of 78 of the first 195 signal stimuli not being relevant
#                        startindexx = indexlist1[idx]-117 -390
                        startindexx = indexlist1[idx]-117 

#                        sound_start = indexlist1[idx]-117 -390
                        startindex.append(startindexx)
                        #add remaining 4 seconds (780 datapoints) to starting intervall as end indexes.
#                        endindex.append(startindexx + 780 +390)
                        endindex.append(startindexx + 780)
                    else:
                        continue
                    
                indexes = list(zip(startindex,endindex))
                indexes_to_cut.append(indexes)
            except BaseException as e:
                print('Failed to do something: ' + str(e) + ' - ' + str(ID))
        
        return indexes_to_cut    



def get_important_indexses2(lista):
        indexes_to_cut = []
        for idx,minilista in enumerate(lista):
            try:
                df = minilista[1]
                ID = minilista[0]
                sound = df["sound_stimuli"]
                # get index of all 1 of soundstimuli in list
                indexlist1 = []
                for idx,soundlabel  in enumerate (sound):
                #    print(soundlabel)
                    if soundlabel == 1:
                #        print("found here ",idx)
                        indexlist1.append(idx)
                        continue
                    else:
                        continue
                
                # Now get appropriate starting poin to cut and appropriate endingpoint to cut
                idx = 0
                startindex = []
                endindex = []
                while idx < len(indexlist1)-2:
                    idx += 1
                #    print(test[idx])
                    if (indexlist1[idx+1] - indexlist1[idx]) > 3:
                        #take end of sound iddex -117 steps because of 78 of the first 195 signal stimuli not being relevant
#                        startindexx = indexlist1[idx]-117 -390
                        startindexx = indexlist1[idx]-1000

#                        sound_start = indexlist1[idx]-117 -390
                        startindex.append(startindexx)
                        #add remaining 4 seconds (780 datapoints) to starting intervall as end indexes.
#                        endindex.append(startindexx + 780 +390)
                        endindex.append(startindexx + 780)
                    else:
                        continue
                    
                indexes = list(zip(startindex,endindex))
                indexes_to_cut.append(indexes)
            except BaseException as e:
                print('Failed to do something: ' + str(e) + ' - ' + str(ID))
        
        return indexes_to_cut        
  
def smooth(data,window):
    for i in range(0,len(data)):
        try:
            ID = data[i][0]
            dataframe = data[i][1]
#            label = stats.mode(dataframe['label'])[0].tolist()[0]
            signal = dataframe['skin conductance']
#            sound = dataframe['sound_stimuli']
#            reaction = dataframe['reaction limit']
            df_roll = pd.DataFrame(signal)
            smoothed = df_roll.rolling(window=window).mean().values
            dataframe['skin conductance'] = smoothed            
            
        except BaseException as e:
            print('Failed to do something: ' + str(e) + ' - ' + str(i) +' - ' + str(ID))
        
    
    
# Take the one with highest MEAN or MEDIAN and assign it as Reactive and vice versa. Check Accuracy
#From Density plot of both groups, classify by Mean or Median as a baseline classifier      
def baseline_classifier_mean(lista, limit):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for minilista in lista:
        try:
    #        print('s')
            df = minilista[1]
            df = df.reset_index(drop=True)
            label = df.loc[0,'label']
            signal = df['skin conductance']
                
            
            # mean of hypo is 1.33 and median 0.81
            if np.mean(signal) <= limit and label == 1:
                TP += 1
            elif np.mean(signal) > limit and label == 0:
                TN += 1
            elif np.mean(signal) <= limit and label == 0:
                FP += 1
            elif np.mean(signal) > limit and label == 1:
                FN += 1
            else:
                print('error')                   
        except:
            print('error - missing label for signal')
            
            
    return TP,TN,FP,FN




def baseline_classifier_median(lista, limit):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for minilista in lista:
        try:
    #        print('s')
            df = minilista[1]
            df = df.reset_index(drop=True)
            label = df.loc[0,'label']
            signal = df['skin conductance']
                
            
            # mean of hypo is 1.33 and median 0.81
            if np.median(signal) <= limit and label == 1:
                TP += 1
            elif np.median(signal) > limit and label == 0:
                TN += 1
            elif np.median(signal) <= limit and label == 0:
                FP += 1
            elif np.median(signal) > limit and label == 1:
                FN += 1
            else:
                print('error')                   
        except:
            print('error - missing label for signal')
            
            
    return TP,TN,FP,FN


def classifier1(lista,model_Reac,model_Hypo):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for minilista in lista:
        try:
    #        print('s')
            df = minilista[1]
            df = df.reset_index(drop=True)
            label = df.loc[0,'label']
            signal = df['skin conductance']
            
            prob_reac = model_Reac.log_probability(signal)
            prob_hypo = model_Hypo.log_probability(signal)
            
            if prob_hypo > prob_reac  and label == 1:
                TP += 1
            elif prob_hypo > prob_reac and label == 0:
                FP += 1
            elif prob_hypo <= prob_reac and label == 0:
                TN += 1
            elif prob_hypo <= prob_reac and label == 1:
                FN += 1
        except:
            print('error')
    return TP,TN,FP,FN




def classifier(test_list,model_Reac,model_Hypo,typee):  
    count_reac = 0
    count_hypo = 0    
    predicted = []
#    TP = 0
#    TN = 0
#    FP = 0
#    FN = 0
    for sequence in test_list:
#        print(sequence)

        prob_reac = model_Reac.log_probability(list(sequence))
        prob_hypo = model_Hypo.log_probability(list(sequence))
        
#        print(prob_reac)
#        print(prob_hypo)
        
        # check accuracy
        if prob_reac > prob_hypo:
            count_reac += 1
            predicted.append(0)
        else:
            count_hypo +=1
            predicted.append(1)
    if typee == 'H' or typee == 'h':
        print('accuracy for Hypo test : ' , count_hypo/len(test_list))
        return predicted, count_hypo/len(test_list)
    else:
        print('accuracy for Reactive test : '  , count_reac/len(test_list))
        return predicted, count_reac/len(test_list) 
    

