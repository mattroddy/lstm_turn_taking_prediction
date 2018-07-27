# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import h5py
import os


#%% Load file
file_list_path = './data/splits/complete.txt'
va_annotations_path = './data/extracted_annotations/voice_activity/'

# structure: hold_shift.hdf5/50ms-250ms-500ms/hold_shift-stats/seq_num/g_f_predict/[index,i]
if 'file_onset' in locals():
    if isinstance(file_onset,h5py.File):   # Just HDF5 files
        try:
            file_onset.close()
        except:
            pass # Was already closed

if not(os.path.exists('./data/datasets/')):
    os.makedirs('./data/datasets/')
file_onset = h5py.File('./data/datasets/onsets.hdf5','w')

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
    
onset_count = 0
short_count = 0
long_count = 0

    
start_silence = 30 # 1.5 seconds of silence before onset
onset_min = 10 # 500 ms of speech minimum for onset to fall into either category
short_max_extra = 10 # if utterance is a a maximum of 500ms longer after the 500ms onset it is Short
short_no_speech = 100 # 5 seconds of required silence after short utterances
long_min_extra = 50 # 2.5 seconds of minimurequired speech after onset to be defined as LONG

g_onsets_list = []
f_onsets_list = []

for seq in complete_file_list:
    annotations = annotations_dict[seq]
    g_onsets = []
    f_onsets = []
        
    #%% find pauses
    pauses = ~(annotations[:,0] | annotations[:,1]) 
    # False is when someone is speaking
    
    for g_f,f_g in zip([0,1],[1,0]):
        for indx in range(start_silence,len(annotations)-(short_no_speech+onset_min)):
            if all(annotations[indx:indx+onset_min,g_f]) and not(any(annotations[indx-start_silence:indx,g_f])):
                onset_count = onset_count + 1
                # onset frame is 50ms after the current index
                onset_frame = indx + onset_min
                # check that selection is right length and includes right info
                selection = annotations[indx+onset_min-1:indx+onset_min+short_max_extra-1,g_f]
#                if all(annotations[indx:indx+onset_min+long_min_extra,g_f]):
                if annotations[indx,g_f] and annotations[indx+onset_min+long_min_extra,g_f] and \
                    sum(1*annotations[indx:indx+onset_min+long_min_extra,g_f])>50:
                    long_count = long_count + 1
                    if g_f == 0: # 0 is g, 1 is f
                        g_onsets.append([onset_frame,1]) # [index of onset, short(0) or long(1)]
                    else:
                        f_onsets.append([onset_frame,1])
                elif not(len(np.where(selection[:-1]!=selection[1:])[0])==0):
                    change_indx = np.where(selection[:-1]!=selection[1:])[0][0]
                    if not(any(annotations[indx+onset_min+change_indx+1:indx+onset_min+change_indx+long_min_extra+1,g_f])):
                        short_count = short_count + 1
                        if g_f == 0: # 0 is g, 1 is f
                            g_onsets.append([onset_frame,0]) # [index of onset, short(0) or long(1)]
                        else:
                            f_onsets.append([onset_frame,0])

    g_onsets_list.append(g_onsets)
    f_onsets_list.append(f_onsets)

print('num of instances: {}'.format(onset_count))  
print('num short_count: {}'.format(short_count))
print('num long_count: {}'.format(long_count))

for seq,indx in zip(complete_file_list,range(len(complete_file_list))):
    file_onset.create_dataset('/short_long/'+seq+'/'+'g',data=np.array(g_onsets_list[indx]))
    file_onset.create_dataset('/short_long/'+seq+'/'+'f',data=np.array(f_onsets_list[indx]))

file_onset.close()



  