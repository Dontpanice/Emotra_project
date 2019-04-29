# Working folder - D:\Master_thesis_data\Emotra_preprocessed\Rapporter_cleaned

from my_functions_remake2 import findmin,reshape,calc_arclen,get_lin,korrekt_seq,cut_important,get_important_indexses,smooth,plot_individuals_with_sound_reaction,plot_individuals_with_sound,classifier1,classifier,baseline_classifier_median,baseline_classifier_mean,plot_all,separate_skinC,remove_at_indexes,plot_individuals_in_segment,compare_similar_means,cut_segment_of_df,merging,extract_signal,extract_signal2, extract_labels, separate, get_median_and_means, find_index_sound
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random 
import pandas as pd
import copy
from pomegranate import HiddenMarkovModel,NormalDistribution
import time
#%% --------------------------------------------  Extract all labels from files, save in "label_data"
df_labels = extract_labels('./Rapporter_cleaned_2/')

# --------------------------------------------- Extract all Highgrade-signaldata from files, save in "signal_data"
signal_data_list, lost_signal_data_list = extract_signal('./Eudor-a_data_2/')
# --------------------------------------------- Extract all mediumgrade-signaldata from files, save in "signal_data2"   
signal_data_list2 = extract_signal2('./Eud_data/')

# ------------------------------------------  MERGING LABELS WITH DATA, comparing ID betwen files and merging dataframe with label.

lost_matches,found_matches = merging(df_labels,signal_data_list)
lost_matches2,found_matches2 = merging(df_labels,signal_data_list2)


All_data = signal_data_list + signal_data_list2



#%%
with open('All_data_ready.pickle', 'wb') as f:
    pickle.dump(All_data, f)




#%%                              Remove those without label-match

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
    


#%%
#Stats_Reactive = []
#Stats_Hypo = []
#Stats_df_hypo = pd.DataFrame(columns = ['sig_mean','sig_median','sig_std','ht_mean','ht_median','ht_std','pr_mean','pr_median','pr_std','pr2_mean','pr2_median','pr2_std'] )
#Stats_df_reac = pd.DataFrame( columns = ['sig_mean','sig_median','sig_std','ht_mean','ht_median','ht_std','pr_mean','pr_median','pr_std','pr2_mean','pr2_median','pr2_std'] )
#
#for idx,minilista in enumerate(All_data_mini):
#    try:
#        ID = minilista[0]
#        df = minilista[1]
#        
#        label = df['label'][0]
#        
#        signal = df['skin conductance']
#        
#        pressure = df['Pressure1']
#        pressure2 = df['Pressure2']
#        
#        Heart_rate = df['Heart Rate']
#        
#        #signal stats
#        sig_mean =   np.mean(signal)
#        sig_median = np.median(signal)
#        sig_std =    np.std(signal)
#        
#        #Heart_rate stats
#        ht_mean =   np.mean(Heart_rate)
#        ht_median = np.median(Heart_rate)
#        ht_std =    np.std(Heart_rate)
#        
#    
#        #pressure stats
#        pr_mean =   np.mean(pressure)
#        pr_median = np.median(pressure)
#        pr_std =    np.std(pressure)
#        
#        #pressure2 stats
#        pr2_mean =   np.mean(pressure2)
#        pr2_median = np.median(pressure2)
#        pr2_std =    np.std(pressure2)
#        
#        data = [sig_mean,sig_median,sig_std,ht_mean,ht_median,ht_std,pr_mean,pr_median,pr_std,pr2_mean,pr2_median,pr2_std]
#    
#        if label == 0:
#             Stats_df_reac.loc[idx] = data
#            
#        elif label == 1:
#             Stats_df_hypo.loc[idx] = data
#        
#        else:
#            print('error label')
#    except BaseException as e:
#        print('Failed to do something: ' + str(e) + ' - ' + str(idx))




#%%  PLOT individual SEGMENT DATA WITH SOUND-SIGNAL 
# will be saved in folder "Individual plots", rename folder manually later to not get overwritten next time you run this code.
#plot_individuals_with_sound(All_data)


#%%
thrown_out2,check2,remove2 = cut_segment_of_df(All_data,2,9)
remove_at_indexes(All_data,remove2)



#%%             LOAD CHECKPOINT
with open('All_data_ready.pickle', 'rb') as f:
    All_data = pickle.load(f)
    
#%%
smooth(All_data,150)  


#%%  
# =============================================================================
#                plotta sekvenser som når 0 mätvärden. 
# =============================================================================

#Index(['Heart Rate', 'Pressure1', 'Pressure2', 'skin conductance',
#       'sound_stimuli', 'Event Markers', 'label', 'reaction limit'],
#      dtype='object')

for idx,minilista in enumerate(All_data):
#    try:
    ID = minilista[0]
    df = minilista[1]
    label = df['label'].iloc[0]
    signal = df['skin conductance']
    sound = df['sound_stimuli']
    reaction = df['reaction limit']
    event = df['Event Markers']
    
    
    for value in signal:
#        print(value)
        if value < 0.1:
            
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax3 = ax1.twinx()
            ax4 = ax1.twinx()
            
            sound, = ax2.plot(sound, label='sound interval' , color='black', alpha=0.8)
            reaction, = ax3.plot(reaction, label='reaction' , color='yellow', alpha=0.8)
            event, = ax4.plot(event, label='event' , color='pink', alpha=0.8)

            plt.plot(signal)
            plt.savefig('Individual_plots/Rensa_0/' + str(idx)+'.png')
            plt.clf()
            plt.cla()
            plt.close()
            break
        else:
            continue



#%%                      
# =============================================================================
#                      TA BORT OUTLIERS HÄR
# =============================================================================
listofout = [107,159,1103,1155,813,892,861]

lista = []
for i in listofout:
    lista.append(All_data[i])
    
for i in listofout:
    del All_data[i]



#%%
            
indexes_to_cut = get_important_indexses(All_data)
all_segments = cut_important(All_data,indexes_to_cut)



#%%       D         Check for eventmarkers in data

#10 – The sound stimuli.  This is used to indicate when a patient receives stimuli and the start of the stimuli is used to calculate response times of the other signals.
#> 8 – The skin conductance, this is the primary input of the method. How this signal reacts to a sound stimuli is the central part of the method.
#> 7 – Skin conductance amplification. This signal normally doesn’t hold any interest since the analysist are only interesting in shapes and responses in a small time window, while the amplification is normally regarded to be more a question of a patients overall conductive properties(Patients with dry skin for example are overall less conductive.). I will however put this signal in this group since it is closely related to the most important signal.
#>  
#> Secondary signals for the EDOR-Test method:
#> This signals are normally not used when analyzing but can be used to explain deviant behaviors.
#> 11 – Event Markers – Inputs from the test leader to indicate that the patient did something that deviated from the normal state of waiting and listening. E.g. if the patient sneezes or starts to cough.
#> 3 – Heart Rate (AC) – Gives an approximation of the heart rate signal from the patient.
#> 4 – Heart Rate (DC) - Gives an approximation of the heart rate signal from the patient.
#> 5 – Pressure indicator 1 – Indicates pressure on one of the electrodes
#> 6 – Pressure indicator 2 - Indicates pressure on the other electrode

markers_list = []
for idx,lista in enumerate(all_segments):
        ID = lista[0]
        dfs = lista[1]
        for idx2,df in enumerate(dfs):
                event = df['Event Markers']
                x = sum(event)
                if x > 0:
                        markers_list.append([ID,idx,idx2])
                        print("found events at lista " + str(idx) +' datafram nr: '+ str(idx2) )
                else:
                        continue



#%%                                       PLOT Minisegments
plt.ioff()
for idx,serie in enumerate(all_segments[:10]):
    for idx2,minicut in enumerate(serie):
        try:
            minicut = minicut.reset_index(drop = True)
            SC = minicut['skin conductance']
            label = minicut['label'].iloc[0]
            SC = korrekt_seq(SC)
            plt.ylabel('Skin-conductance (µS)')
            plt.xlabel('Time (195Hz) , 1s = 195 samples')
            if label == 0:
                plt.title('Corrected plot of reactive individual')
                plt.plot(SC, color = 'Green')
#                plt.legend(handles = (curve,sound), loc='upper left')
                plt.xlabel('Time (195Hz) , 1s = 195 samples')
                plt.axis([1,len(SC),0,max(SC)+(max(SC)*0.2)])
                plt.savefig('Individual_plots/Miniplots' + '/RE_' +str(idx)+'-'+str(idx2)+'.png')
            else:
                plt.title('Corrected plot of hypo individual')
#                curve, = ax1.plot(SC, label='curve - Hypo Reactive', color='red')
                plt.xlabel('Time (195Hz) , 1s = 195 samples')
                plt.axis([1,len(SC),0,max(SC)+(max(SC)*0.2)])
                plt.plot(SC, color = 'Red')     
#                plt.legend(handles = (curve,sound), loc='upper left')
                plt.savefig('Individual_plots/Miniplots' + '/HY_' +str(idx)+'-'+str(idx2)+'.png')
            plt.clf()
            plt.cla()
            plt.close()
        except BaseException as e:
            print('Failed to do something: ' + str(e) + ' - ' + str(idx))




#%%                   STANDARDIZE THE VALUES    AND TURN DF TO SIMPLE LISTS


test_ =  copy.deepcopy(all_segments) 

#%%    
#from sklearn import preprocessing
#from sklearn import datasets
#
#standardized_reactive = []
#standardized_hypo = []
#
#for idx,lista in enumerate(test_):
#    for idx2,df in enumerate(lista):
#        try:
#            df = df.reset_index(drop = True)
#            SC = df['skin conductance']
#            label = df['label'].tolist()[0]
#            
##            SC = SC.tolist()
#            SC = np.array(SC)
##            SC = SC.reshape(-1, 1)
##            minimum2 = findmin(SC)
##            reshaped_hypo = reshape(SC, minimum2) 
##            scaler2 = preprocessing.StandardScaler().fit(SC)
##            SC_scaled = scaler2.transform(SC)
#            if label == 0:
#                standardized_reactive.append(SC)
#            else:
#                standardized_hypo.append(SC)
##            df.drop('skin conductance', axis = 1, inplace = True)
##            lista[idx2]df['skin conductance'] = SC_scaled
#            
#        except BaseException as e:
#            print('Failed to do something: ' + str(e) + ' - ' + str(idx))


#%%   # CORRECT THE SEQUENCES

#corr_sequences_R = []
#corr_sequences_H = []    
#
#for idx,lista in enumerate(test_):
#    for idx2,df in enumerate(lista[1]):
#        try:
#            df = df.reset_index(drop = True)
#            SC = df['skin conductance']
#            label = df['label'].iloc[0]
#            
#            SC = korrekt_seq(SC)
#            SC = np.array(SC)
#            if label == 0:
#                corr_sequences_R.append(SC)
#            else:
#                corr_sequences_H.append(SC)
#        except BaseException as e:
#            print('Failed to do something: ' + str(e) + ' - ' + str(idx))
#

#%%
# =============================================================================
#   Correct the sequences  and make all negative values in sequence positive equal to 0. 
#               + Get Area and peak for these sequences and (remove areas caused by 0.00000001 convertion)
# =============================================================================


#summed Area and Peak for each 9 segemnts of sequence and in format [ [summed Area for 9segments], [etc], [etc]  ]
H_area_List = []
H_peak_List = []
R_area_List = []
R_peak_List = []
R_sequences = []
H_sequences = []

R_area = []
R_peak = []

H_area = []
H_peak = []

H_lost_counter = 0
R_lost_counter = 0
H_counter = 0
R_counter = 0
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
            
            

            
            
#            rise_time = array_pos.index(max(array_pos))
#            settling_time =
        

            # remove area and peak if peak is 1e-08, since that means there is no area in the graph
            if label == 0:
#                corr_sequences_R.append(SC)
                R_nine_seq_A.append(area)
                R_nine_seq_P.append(peak)
                R_nine_seq.append(array_pos)
#                R_counter +=1    
#                R_area_List.append(area)
#                R_peak_List.append(peak)
            elif label == 1:
#                H_counter +=1
                H_nine_seq_A.append(area)
                H_nine_seq_P.append(peak)
                H_nine_seq.append(array_pos)
#                corr_sequences_H.append(SC)
#                H_nine_seq.append(array_pos)
#                 H_peak_List.append(peak)
#                 H_area_List.append(area)
#            elif label == 0:
#                R_lost_counter += 1
#            elif label == 1:
#                H_lost_counter += 1
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
    
#%% 
# =============================================================================
#                    PLOT Corrected sequences    
# =============================================================================
    
   
for idx,lista in enumerate(H_sequences):
    for idx2,array in enumerate(lista):
        
        plt.plot(array)
        plt.savefig('Individual_plots/corrected_plots/' + str(idx)+'-'+ str(idx2) +'.png')
        plt.clf()
        plt.cla()
        plt.close()
        
#%%
        
# =============================================================================
#                       calculate arclength
# =============================================================================
arc_lengths = []
for idx,lista in enumerate(H_sequences):
    for idx2,array in enumerate(lista):
        
        #Choose n=1 for as accurate arclength as possible.
        n = 1
        arc = calc_arclen(array,n)
        arc_lengths.append(arc)
        
#%%        
# =============================================================================
#                       calculate settling time & rise time measures
# =============================================================================


#reaction_measurements = []
risetime_L = []
amplitud_L = []
settlingtime_L = []

for idx,lista in enumerate(H_sequences):
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
    
#    reaction_measurements.append([risetime,amplitud,settlingtime])
        
        

plt.hist(risetime_L)
plt.title('Rising time')
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()

plt.hist(amplitud_L)
plt.title('amplitude time')
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()

plt.hist(settlingtime_L)
plt.title('Settling time')
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()



risetime_L_log = [np.log(n) for n in risetime_L]       
amplitud_L_log = [np.log(n) for n in amplitud_L]
settlingtime_L_log = [np.log(n) for n in settlingtime_L] 
H_area_List_log = [np.log(n) for n in H_area_List] 

Groups = [risetime_L_log, amplitud_L_log, settlingtime_L_log, H_area_List_log]

fig, ax = plt.subplots()
pos = np.array(range(len(Groups))) + 1
bp = ax.boxplot(Groups, sym='k+', positions=pos,
                notch=1, bootstrap=5000)

ax.set_xlabel('Boxplots')
ax.set_ylabel('log(values)')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.title('Hypo reactive group')
plt.xticks([1, 2, 3,4], ['risetime', 'amplitude', 'settlingtime','area'])
plt.show()

# =============================================================================
#                               Now for Reactive
# =============================================================================
#reaction_measurements = []
risetime_L = []
amplitud_L = []
settlingtime_L = []

for idx,lista in enumerate(R_sequences):
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
    
#    reaction_measurements.append([risetime,amplitud,settlingtime])
        
        

plt.hist(risetime_L)
plt.title('Rising time')
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()

plt.hist(amplitud_L)
plt.title('amplitude time')
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()

plt.hist(settlingtime_L)
plt.title('Settling time')
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()



risetime_L_log_R = [np.log(n) for n in risetime_L]       
amplitud_L_log_R = [np.log(n) for n in amplitud_L]
settlingtime_L_log_R = [np.log(n) for n in settlingtime_L] 
R_area_List_log_R = [np.log(n) for n in R_area_List] 

Groups = [risetime_L_log_R, amplitud_L_log_R, settlingtime_L_log_R, R_area_List_log_R]

fig, ax = plt.subplots()
pos = np.array(range(len(Groups))) + 1
bp = ax.boxplot(Groups, sym='k+', positions=pos,
                notch=1, bootstrap=5000)

ax.set_xlabel('Boxplots')
ax.set_ylabel('log(values)')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.title('Reactive group')
plt.xticks([1, 2, 3,4], ['risetime', 'amplitude', 'settlingtime','area'])
plt.show()

    

Groups = [risetime_L_log_R, amplitud_L_log_R, settlingtime_L_log_R, R_area_List_log_R, risetime_L_log, amplitud_L_log, settlingtime_L_log, H_area_List_log     ]

fig, ax = plt.subplots(figsize = (10,10))
pos = np.array(range(len(Groups))) + 1
bp = ax.boxplot(Groups, sym='k+', positions=pos,
                notch=1, bootstrap=5000)

ax.set_xlabel('Boxplots')
ax.set_ylabel('log(values)')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.title('reactive and hypo group')
plt.xticks([1, 2, 3,4,5,6,7,8], ['R_risetime', 'R_amplitude', 'R_settlingtime','R_area','H_risetime', 'H_amplitude', 'H_settlingtime','H_area'])
plt.show()        


#%%



# =============================================================================
#            Area boxplots for individual segments    
# =============================================================================

# =============================================================================
#                            H_peak
# =============================================================================
#risetime_L_log_R = [np.log(n) for n in risetime_L]       
H_peak_log = [np.log(n) for n in H_peak]   
        
hp_bin0 = []
hp_bin1 = []
hp_bin2 = []
hp_bin3 = []
hp_bin4 = []
hp_bin5 = []
hp_bin6 = []

for array in H_peak_log:
    for idx,value in enumerate(array):
        if idx == 0:
            hp_bin0.append(value)
        elif idx == 1:
            hp_bin1.append(value)
        elif idx == 2:
            hp_bin2.append(value) 
        elif idx == 3:
            hp_bin3.append(value)
        elif idx == 4:
            hp_bin4.append(value)
        elif idx == 5:
            hp_bin5.append(value)
        else:
            hp_bin6.append(value)
#            print('whooot')
  
            
Groups = [hp_bin0,hp_bin1,hp_bin2,hp_bin3,hp_bin4,hp_bin5,hp_bin6]

fig, ax = plt.subplots(figsize = (10,10))
pos = np.array(range(len(Groups))) + 1
bp = ax.boxplot(Groups, sym='k+', positions=pos,
                notch=1, bootstrap=5000)

ax.set_xlabel('Boxplots')
ax.set_ylabel('(peak log(values)')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.title('hypo 7 segment bins')
plt.xticks([1, 2, 3,4,5,6,7], ['hp_bin0', 'hp_bin1', 'hp_bin2','hp_bin3','hp_bin4', 'hp_bin5', 'hp_bin6'])
plt.show()                  

# =============================================================================
#                            H_area
# =============================================================================

H_area_log = [np.log(n) for n in H_area]   
        
bin0 = []
bin1 = []
bin2 = []
bin3 = []
bin4 = []
bin5 = []
bin6 = []

for array in H_area_log:
    for idx,value in enumerate(array):
        if idx == 0:
            bin0.append(value)
        elif idx == 1:
            bin1.append(value)
        elif idx == 2:
            bin2.append(value) 
        elif idx == 3:
            bin3.append(value)
        elif idx == 4:
            bin4.append(value)
        elif idx == 5:
            bin5.append(value)
        else:
            bin6.append(value)
#            print('whooot')
  
            
Groups = [bin0,bin1,bin2,bin3,bin4,bin5,bin6]

fig, ax = plt.subplots(figsize = (10,10))
pos = np.array(range(len(Groups))) + 1
bp = ax.boxplot(Groups, sym='k+', positions=pos,
                notch=1, bootstrap=5000)

ax.set_xlabel('Boxplots')
ax.set_ylabel('(area log(values)')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.title('hypo 7 segment bins')
plt.xticks([1, 2, 3,4,5,6,7], ['bin0', 'bin1', 'bin2','bin3','bin4', 'bin5', 'bin6'])
plt.show()   



# =============================================================================
#                            R_peak
# =============================================================================


#risetime_L_log_R = [np.log(n) for n in risetime_L]       
R_peak_log = [np.log(n) for n in R_peak]   
        
rp_bin0 = []
rp_bin1 = []
rp_bin2 = []
rp_bin3 = []
rp_bin4 = []
rp_bin5 = []
rp_bin6 = []

for array in R_peak_log:
    for idx,value in enumerate(array):
        if idx == 0:
            rp_bin0.append(value)
        elif idx == 1:
            rp_bin1.append(value)
        elif idx == 2:
            rp_bin2.append(value) 
        elif idx == 3:
            rp_bin3.append(value)
        elif idx == 4:
            rp_bin4.append(value)
        elif idx == 5:
            rp_bin5.append(value)
        else:
            rp_bin6.append(value)
#            print('whooot')
  
            
Groups = [rp_bin0,rp_bin1,rp_bin2,rp_bin3,rp_bin4,rp_bin5,rp_bin6]

fig2, ax2 = plt.subplots(figsize = (10,10))
pos = np.array(range(len(Groups))) + 1
bp = ax.boxplot(Groups, sym='k+', positions=pos,
                notch=1, bootstrap=5000)

ax.set_xlabel('Boxplots')
ax.set_ylabel('(peak log(values)')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.title('Reactive 7 segment bins')
plt.xticks([1, 2, 3,4,5,6,7], ['rp_bin0', 'rp_bin1', 'rp_bin2','rp_bin3','rp_bin4', 'rp_bin5', 'rp_bin6'])
plt.show()                  

# =============================================================================
#                            R_area
# =============================================================================

R_area_log = [np.log(n) for n in R_area]   
        
bin0 = []
bin1 = []
bin2 = []
bin3 = []
bin4 = []
bin5 = []
bin6 = []

for array in R_area_log:
    for idx,value in enumerate(array):
        if idx == 0:
            bin0.append(value)
        elif idx == 1:
            bin1.append(value)
        elif idx == 2:
            bin2.append(value) 
        elif idx == 3:
            bin3.append(value)
        elif idx == 4:
            bin4.append(value)
        elif idx == 5:
            bin5.append(value)
        else:
            bin6.append(value)
#            print('whooot')
  
            
Groups = [bin0,bin1,bin2,bin3,bin4,bin5,bin6]

fig, ax = plt.subplots(figsize = (10,10))
pos = np.array(range(len(Groups))) + 1
bp = ax.boxplot(Groups, sym='k+', positions=pos,
                notch=1, bootstrap=5000)

ax.set_xlabel('Boxplots')
ax.set_ylabel('(area log(values)')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.title('reactive 7 segment bins')
plt.xticks([1, 2, 3,4,5,6,7], ['rp_bin0', 'rp_bin1', 'rp_bin2','rp_bin3','rp_bin4', 'rp_bin5', 'rp_bin6'])
plt.show()   





# =============================================================================
#                 together
# =============================================================================

#%%

Groups = [rp_bin0,rp_bin1,rp_bin2,rp_bin3,rp_bin4,rp_bin5,rp_bin6,hp_bin0,hp_bin1,hp_bin2,hp_bin3,hp_bin4,hp_bin5,hp_bin6]

fig2, ax2 = plt.subplots(figsize = (10,10))
pos = np.array(range(len(Groups))) + 1
bp = ax.boxplot(Groups, sym='k+', positions=pos,
                notch=1, bootstrap=5000)

ax.set_xlabel('Boxplots')
ax.set_ylabel('(peak log(values)')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.title('Reactive 7 segment bins')
plt.xticks([1, 2, 3,4,5,6,7,8, 9, 10,11,12,13,14], ['bin0', 'bin1', 'bin2','bin3','bin4', 'bin5', 'bin6','hp_bin0', 'hp_bin1', 'hp_bin2','hp_bin3','hp_bin4', 'hp_bin5', 'hp_bin6'])
plt.show()                  


#%%

Groups = [rp_bin0]

fig2, ax2 = plt.subplots(figsize = (10,10))
pos = np.array(range(len(Groups))) + 1
bp = ax.boxplot(Groups, sym='k+', positions=pos,
                notch=1, bootstrap=5000)

ax.set_xlabel('Boxplots')
ax.set_ylabel('(peak log(values)')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.title('Reactive 7 segment bins')
plt.xticks([1,2], ['rp_bin0'])
plt.show()                  




#%%
# =============================================================================
#                               scatterplot OF Area and Amplitude
# =============================================================================      
fig, ax = plt.subplots()

ax.scatter(R_area_List, R_peak_List, alpha=0.2, c='green',label='Reactive')
ax.scatter(H_area_List, H_peak_List, alpha=0.2, c='red',label='Hypo')
ax.legend()
plt.ylabel('Peak length ')
plt.xlabel('Area')
plt.title('Area + peaklength of reactive and hypo segments')
plt.show()      

fig, ax = plt.subplots()
ax.hist(H_peak_List, ls='dashed', alpha = 1, lw=3, color= 'red', label= 'Hypo')
ax.hist(R_peak_List,  ls='dotted', alpha = 0.5, lw=3, color= 'green',label='Reactive')
ax.legend()
plt.ylabel('N')
plt.xlabel('peak lenght ')
plt.title('Histogram of peak lengths')

plt.show()

fig, ax = plt.subplots()
ax.hist(H_area_List, ls='dashed', alpha = 1, lw=3, color= 'red',label='Hypo')
ax.hist(R_area_List,  ls='dotted', alpha = 0.5, lw=3, color= 'green',label='Reactive')
ax.legend()
plt.ylabel('N')
plt.xlabel('Area')
plt.title('Histogram of area')

plt.show()


#%%










#%%
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import KFold
# Separate Reactive & Hypo reactive
Train_reac, Train_hypo = separate_skinC(All_data)

#Train_reac_labels = [0]*len(Train_reac)
#Train_hypo_labels = [1]*len(Train_hypo)

#x_Reac_train, x_Reac_test, y_Reac_train, y_Reac_test = train_test_split(Train_reac, Train_reac_labels, test_size=0.2 , random_state=0)
#x_Hypo_train, x_Hypo_test, y_Hypo_train, y_Hypo_test = train_test_split(Train_hypo, Train_hypo_labels, test_size=0.2, random_state=0)

#%% HMM_Reactive

#from sklearn.pipeline import make_pipeline
#scaler2 = preprocessing.StandardScaler().fit(reshaped_hypo)
#X_train_transformed_hypo = scaler2.transform(reshaped_hypo)
    
#%%   Find minimum length of series in list and shape it into that shaoe for all series. (needed for standardscaler)
    

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



kf = KFold(n_splits=5)

saved_models_hypo = []
saved_models_reac = []


minimum2 = findmin(Train_hypo)
reshaped_hypo = reshape(Train_hypo, minimum2)      
scaler2 = preprocessing.StandardScaler().fit(reshaped_hypo)
X_train_transformed_hypo = scaler2.transform(reshaped_hypo)
accompan_test_hypo=[]

random.seed( 808 )

#print(kf.get_n_splits(x_Reac_train))
for train,test in kf.split(X_train_transformed_hypo):
#    hypo_eaxmples = random.sample(X_train_transformed_hypo, 10)
    train = X_train_transformed_hypo[train]
    test = X_train_transformed_hypo[test]
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
#    accompan_test_hypo.append(test)

# KÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖR Ikväll 
minimum = findmin(Train_reac)    
reshaped_reac = reshape(Train_reac, minimum) 
scaler = preprocessing.StandardScaler().fit(reshaped_reac)
X_train_transformed_reac = scaler.transform(reshaped_reac)  
accompan_test_reac=[]

    
for train,test in kf.split(X_train_transformed_reac):
    
    train = X_train_transformed_reac[train]
    test = X_train_transformed_reac[test]
    reac_eaxmples = X_train_transformed_reac[:10]
#    reac_eaxmples = random.sample(X_train_transformed_reac, 10)

    model_Reac = HiddenMarkovModel()
    #inititiera modellens parametrar från ett exempel
    model_Reac = HiddenMarkovModel.from_samples(NormalDistribution, n_components=5, X=reac_eaxmples)
    
    #train model
    start_time = time.time()
    model_Reac.fit(train, algorithm='baum-welch')
    print("--- %s seconds ---" % (time.time() - start_time))
    
    #save the model
    saved_models_reac.append(model_Reac)
    accompan_test_reac.append(test)


#%%               # Hypo beeing in position 0 of list typles
models = list(zip(saved_models_hypo,saved_models_reac))

for model in models:
    hypo_mod = model[0][0]
    reac_mod = model[1][0]
    
    h_test = model[0][1]
    r_test = model[1][1]


    pred_reac, acc = classifier(r_test,reac_mod,hypo_mod,'r')
    N = 151
    
    TP = N * acc
    FP = N * (1-acc)
    
    pred_hypo,acc = classifier(h_test,reac_mod,hypo_mod,'h')
    N = 52
    
    TN = N * acc
    FN = N * (1-acc)
    
    print('\naccuracy : ', (TP+TN) / (TP+TN+FP+FN))
    print('precision : ', (TP) / (TP+FP)) 
    print('recall : ', (TP) / (TP+FN))
    print('\nTP = ', TP,'\nTN = ', TN,'\nFP = ', FP,'\nFN = ', FN)
    
    
#%%  












#%%
    
#    print("%s %s" % (train, test))
 
reac_eaxmples = X_train_transformed_reac[:10]
hypo_eaxmples = X_train_transformed_hypo[:10]  

model_Reac = HiddenMarkovModel()
#inititiera modellens parametrar från ett exempel
model_Reac = HiddenMarkovModel.from_samples(NormalDistribution, n_components=5, X=reac_eaxmples)

model_Hypo = HiddenMarkovModel()
#inititiera modellens parametrar från ett exempel
model_Hypo = HiddenMarkovModel.from_samples(NormalDistribution, n_components=5, X=hypo_eaxmples)

#%% --------------------------------------------------------- TRAIN MODELS ---------------------------------------------------
start_time = time.time()
model_Reac.fit(x_Reac_train, algorithm='baum-welch')
print("--- %s seconds ---" % (time.time() - start_time)) 

start_time = time.time()
model_Hypo.fit(x_Hypo_train, algorithm='baum-welch')
print("--- %s seconds ---" % (time.time() - start_time))

#%% --------------------------------------------------------- TRAIN small MODELS ---------------------------------------------------
start_time = time.time()
model_Reac.fit(Train_reac[0:5], algorithm='baum-welch')
print("--- %s seconds ---" % (time.time() - start_time)) 

start_time = time.time()
model_Hypo.fit(Train_hypo[0:5], algorithm='baum-welch')
print("--- %s seconds ---" % (time.time() - start_time))

#%% --------------------------------------------------------- SAVE/PICKLE MODELS ---------------------------------------------------
#  COVERT MODELS TO JSON 
save_model_reac = model_Reac.to_json()
save_model_hypo = model_Hypo.to_json()

#save pickle	
with open('save_model_reac_blue_n_red_standard.pickle', 'wb') as f:
    pickle.dump(save_model_reac, f)
with open('save_model_hypo_blue_n_red_standard.pickle', 'wb') as f:
    pickle.dump(save_model_hypo, f)

#%%    
save_model_reac = model_Reac.to_json()
save_model_hypo = model_Hypo.to_json()    
#save pickle	
with open('save_model_reac_blue_norm.pickle', 'wb') as f:
    pickle.dump(save_model_reac, f)
with open('save_model_hypo_blue_norm.pickle', 'wb') as f:
    pickle.dump(save_model_hypo, f)
    
#%% --------------------------------------------------------- LOAD MODELS ---------------------------------------------------
with open('save_model_reac_blue.pickle', 'rb') as f:
    model_1_jason = pickle.load(f)
with open('save_model_hypo_blue.pickle', 'rb') as f:
    model_2_jason = pickle.load(f)       

model_reac_blue = model_Reac.from_json(model_1_jason)
model_hypo_blue = model_Hypo.from_json(model_2_jason)              



    


