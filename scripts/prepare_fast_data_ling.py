# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import h5py 
#import pickle
from itertools import zip_longest
import time as t
import multiprocessing
# load data: hold_shift, onsets, overlaps
# structure: hold_shift.hdf5/50ms-250ms-500ms/hold_shift-stats/seq_num/g_f_predict/[index,i]
if 'of_split' in locals():
    if isinstance(of_split, h5py.File):   # Just HDF5 files
        try:
            of_split.close()
        except:
            pass # Was already closed

def find_max_len(df_list):
    max_len = 0
    for df in df_list:
        max_len=max(max_len,len(df))
    return max_len

def fill_array(arr, seq):
    if arr.ndim == 1:
        try:
            len_ = len(seq)
        except TypeError:
            len_ = 0
        arr[:len_] = seq
        arr[len_:] = np.nan
    else:
        for subarr, subseq in zip_longest(arr, seq, fillvalue=()):
            fill_array(subarr, subseq)
            
data_select_dict = {0:['f','g'],
                    1:['c1','c2']}
time_label_select_dict = {0:'frame_time', # gemaps
                    1:'timestamp',# openface
                    2:'frame_time'} # annotations
#%% Settings
num_workers = 4
data_select = 0
time_label_select = 2
annotations_dir = './data/extracted_annotations/voice_activity'
file_list = list(pd.read_csv('./data/splits/complete.txt',header=None,dtype=str)[0])

time_scale_folder = 'words_advanced_100'
output_name = 'words_split'
features_list = ['frame_time','word']

if os.path.exists('./datasets/'+output_name+'.hdf5'):
    os.remove('./datasets/'+output_name+'.hdf5')

out_split = h5py.File('./data/extracted_annotations/'+output_name+'.hdf5','w')
folder_path = os.path.join('./data/extracted_annotations',time_scale_folder)
# structure: of_split/file/[f,g][x,x_i]/feature/matrix[T,4]

t_1=t.time()

def file_run(file_name):
    print(file_name)
    annot_f = pd.read_csv(annotations_dir+'/'+file_name+'.'+data_select_dict[data_select][0]+'.csv',delimiter=',')
    data_f_temp = pd.read_csv(folder_path+'/'+file_name+'.'+data_select_dict[data_select][0]+'.csv')
    data_g_temp = pd.read_csv(folder_path+'/'+file_name+'.'+data_select_dict[data_select][1]+'.csv')
    
    split_indices=[]
    for f_time in annot_f['frame_time'][1:]:
        split_indices.append(np.max(np.where(data_f_temp[time_label_select_dict[time_label_select]] < f_time))+1)
    data_f_split_list = np.split(data_f_temp,split_indices)
    data_g_split_list = np.split(data_g_temp,split_indices)
    
    max_len = 1
    data_dict_f = {'x':{},'x_i':{}}
    data_dict_g = {'x':{},'x_i':{}}
    for feature_name in features_list:    
        data_dict_f['x'][feature_name] = np.zeros((len(annot_f['frame_time']),max_len))
        data_dict_f['x_i'][feature_name] = np.zeros((len(annot_f['frame_time']),max_len))
        data_dict_g['x'][feature_name] = np.zeros((len(annot_f['frame_time']),max_len))
        data_dict_g['x_i'][feature_name] = np.zeros((len(annot_f['frame_time']),max_len))
        for ind, (data_f_split, data_g_split) in enumerate(zip(data_f_split_list,data_g_split_list)):
            data_dict_f['x'][feature_name][ind,:len(data_f_split)] = data_f_split[feature_name]
            if data_dict_f['x'][feature_name][ind,:len(data_f_split)]:
                data_dict_f['x_i'][feature_name][ind,:len(data_f_split)] = 1 
            data_dict_g['x'][feature_name][ind,:len(data_g_split)] = data_g_split[feature_name]
            if data_dict_g['x'][feature_name][ind,:len(data_g_split)]:
                data_dict_g['x_i'][feature_name][ind,:len(data_g_split)] = 1 
            
    data_fg = {file_name:{data_select_dict[data_select][0]:data_dict_f, data_select_dict[data_select][1]:data_dict_g } }
    return data_fg

if __name__=='__main__':
    p = multiprocessing.Pool(num_workers)
    results = p.map(file_run,(file_list),chunksize=1) 
    for file_name,data_fg in zip(file_list,results):
        for f_g in data_select_dict[data_select]:
            for x_type in ['x','x_i']:
                for feat in features_list:
                    out_split.create_dataset(file_name+'/'+f_g+'/'+x_type+'/'+feat ,data=np.array(data_fg[file_name][f_g][x_type][feat]))
out_split.close()
