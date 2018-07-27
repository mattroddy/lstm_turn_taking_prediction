# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os

#%% Load file
file_list_path = './data/splits/complete.txt'
va_annotations_path = './data/extracted_annotations/voice_activity/'

# structure: hold_shift.hdf5/50ms-250ms-500ms/hold_shift-stats/seq_num/g_f_predict/[index,i]
if 'file_overlap' in locals():
    if isinstance(file_overlap,h5py.File):   # Just HDF5 files
        try:
            file_overlap.close()
        except:
            pass # Was already closed

if not(os.path.exists('./data/datasets/')):
    os.makedirs('./data/datasets/')

file_overlap = h5py.File('./data/datasets/overlaps.hdf5','w')

#pause_str_list = ['50ms','250ms','500ms']
#pause_length_list = [1,5,10]
#length_of_future_window = 20 # (one second) find instances where only one person speaks after the pause within this future window

#%% get Annotations
# Load list of test files
complete_file_list = list(pd.read_csv(file_list_path,header=None)[0])
annotations_dict = {}
frame_times_dict = {}
for file_name in complete_file_list:
    data_f = pd.read_csv(va_annotations_path+'/'+file_name+'.f.csv')
    data_g = pd.read_csv(va_annotations_path+'/'+file_name+'.g.csv')
    annotations = np.column_stack([np.array(data_g)[:,1].astype(bool),np.array(data_f)[:,1].astype(bool)])
    annotations_dict[file_name] = annotations
    frame_times_dict[file_name] = np.array(data_g)[:,0]

    
overlap_count = 0
hold_count,shift_count,same_count = 0,0,0

ex_hold_count,ex_shift_count,ex_same_count = 0,0,0
short_count = 0
long_count = 0

    
start_speech = 28 # 1.4 seconds of speech before overlap
start_gaps_allowed = 5 # 5
overlap_min = 2 # 100 ms of speech minimum for overlap to fall into either category
short_class_length = 20 # we define a short utterance as 1 second
eval_length = 10
skip_window = short_class_length-overlap_min
#eval_window = 60 # 3 seconds of speech (should change this to reflect backchannels and previous def of short)
#short_max_extra = 10 # if utterance is a a maximum of 500ms longer after the 500ms overlap it is Short
#short_no_speech = 100 # 5 seconds of required silence after short utterances
#long_min_extra = 50 # 2.5 seconds of minimurequired speech after overlap to be defined as LONG

g_overlaps_list = []
f_overlaps_list = []
g_overlaps_list_ex = []
f_overlaps_list_ex = []

first_spkr_distribution,second_spkr_distribution = [],[]

for seq in complete_file_list:
    annotations = annotations_dict[seq]
    g_overlaps = []
    f_overlaps = []
    g_overlaps_ex = []
    f_overlaps_ex = []
        
    #%% overlaps
    overlapping_frames = (annotations[:,0] & annotations[:,1]) 
    
    for g_f,f_g in zip([0,1],[1,0]):
        for indx in range(start_speech,len(annotations)-short_class_length-eval_length):
            if (np.sum(1*(annotations[indx-start_speech:indx,g_f]))>=(start_speech- start_gaps_allowed)) and all(~(annotations[indx-start_speech:indx,f_g])) \
            and all(overlapping_frames[indx:indx+overlap_min]):
                overlap_count = overlap_count+1
                eval_indx = indx+overlap_min
#                first_spkr_distribution.append(annotations[indx+overlap_min:indx+overlap_min+eval_length,g_f])
#                second_spkr_distribution.append(annotations[indx+overlap_min:indx+overlap_min+eval_length,f_g])
                # to evaluate check which speaker has greater prob of speaking over 20 frames from eval_indx
                if sum(1*annotations[indx+short_class_length:indx+short_class_length+eval_length,g_f])> \
                sum(1*annotations[indx+short_class_length:indx+short_class_length+eval_length,f_g]):
                    hold_count +=1
                    if g_f == 0: # 0 is g, 1 is f
                        g_overlaps.append([eval_indx,0]) # [index of overlap, hold(0) or shift(1)]
                    else:
                        f_overlaps.append([eval_indx,0])
                elif sum(1*annotations[indx+short_class_length:indx+short_class_length+eval_length,g_f])< \
                sum(1*annotations[indx+short_class_length:indx+short_class_length+eval_length,f_g]):
                    shift_count +=1
                    if g_f == 0: # 0 is g, 1 is f
                        g_overlaps.append([eval_indx,1]) # [index of overlap, hold(0) or shift(1)]
                    else:
                        f_overlaps.append([eval_indx,1])
                else:
                    same_count +=1
                # exclusive version
                if (sum(1*annotations[indx+short_class_length:indx+short_class_length+eval_length,g_f])>0) and\
                (sum(1*annotations[indx+short_class_length:indx+short_class_length+eval_length,f_g])==0):
                    ex_hold_count +=1
                    if g_f == 0: # 0 is g, 1 is f
                        g_overlaps_ex.append([eval_indx,0]) # [index of overlap, hold(0) or shift(1)]
                    else:
                        f_overlaps_ex.append([eval_indx,0])
                elif (sum(1*annotations[indx+short_class_length:indx+short_class_length+eval_length,f_g])>0) and\
                (sum(1*annotations[indx+short_class_length:indx+short_class_length+eval_length,g_f])==0 ):
                    ex_shift_count +=1
                    if g_f == 0: # 0 is g, 1 is f
                        g_overlaps_ex.append([eval_indx,1]) # [index of overlap, hold(0) or shift(1)]
                    else:
                        f_overlaps_ex.append([eval_indx,1])
                else:
                    ex_same_count +=1
                
    g_overlaps_list.append(g_overlaps)
    f_overlaps_list.append(f_overlaps)
    g_overlaps_list_ex.append(g_overlaps_ex)
    f_overlaps_list_ex.append(f_overlaps_ex)
#first_spkr=np.sum(np.array(first_spkr_distribution), axis=0)
#second_spkr= np.sum(np.array(second_spkr_distribution), axis=0)

print('num of instances: {}'.format(overlap_count))
print('num hold_count: {}'.format(hold_count))
print('num shift_count: {}'.format(shift_count))  
print('num exclusive hold_count: {}'.format(ex_hold_count))
print('num exclusive shift_count: {}'.format(ex_shift_count))

#%% Write to file
# for seq,indx in zip(seq_list,range(0,len(seq_list))):
for seq,indx in zip(complete_file_list,range(len(complete_file_list))):
    file_overlap.create_dataset('/overlap_hold_shift/'+seq+'/'+'g',data=np.array(g_overlaps_list[indx]))
    file_overlap.create_dataset('/overlap_hold_shift/'+seq+'/'+'f',data=np.array(f_overlaps_list[indx]))
    file_overlap.create_dataset('/overlap_hold_shift_exclusive/'+seq+'/'+'g',data=np.array(g_overlaps_list_ex[indx]))
    file_overlap.create_dataset('/overlap_hold_shift_exclusive/'+seq+'/'+'f',data=np.array(f_overlaps_list_ex[indx]))

file_overlap.close()



  