# -*- coding: utf-8 -*-

# This file preprocesses the features using the pooled mean and variance of the entire dataset to calculate z-scores.

import os
import time as t
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import scipy.io as io
from multiprocessing import Pool

num_workers = 4

input_gemaps_dir = './data/features/gemaps_features/'
output_gemaps_dir = './data/features/gemaps_features_processed/'

gemaps_feat_name_list = [ 'F0semitoneFrom27.5Hz', 'jitterLocal', 'F1frequency',
       'F1bandwidth', 'F2frequency', 'F3frequency', 'Loudness',
       'shimmerLocaldB', 'HNRdBACF', 'alphaRatio', 'hammarbergIndex',
       'spectralFlux', 'slope0-500', 'slope500-1500', 'F1amplitudeLogRelF0',
       'F2amplitudeLogRelF0', 'F3amplitudeLogRelF0', 'mfcc1', 'mfcc2', 'mfcc3',
       'mfcc4']

csv_files=os.listdir(input_gemaps_dir)
voice_activity_files = [file.split('.')[0]+'.'+file.split('.')[1]+'.csv' for file in csv_files]

if not(os.path.exists(output_gemaps_dir)):
    os.makedirs(output_gemaps_dir)
for feature_set in ['raw','znormalized']:
    if not(os.path.exists(output_gemaps_dir+'/'+feature_set)):
        os.mkdir(output_gemaps_dir+'/'+feature_set)

#%% loop through files
curTime = t.time()
missing_count = 0

def process_func_one(data):

    target_file, annotation_file = data
    mean_list, max_list, min_list, std_list,num_vals = [],[],[],[],[]
    target_csv_gemaps = pd.read_csv(input_gemaps_dir+target_file,delimiter=',')
    target_csv_gemaps.fillna(0)
    num_vals.append( len(target_csv_gemaps))
    
    # raw features
    temp_dict = {}
    temp_dict['frame_time'] = target_csv_gemaps['frameTime']
    for feature in gemaps_feat_name_list:
        temp_dict[feature] = target_csv_gemaps[feature]
    outputcsv = pd.DataFrame(temp_dict)
    outputcsv[list(temp_dict.keys())].to_csv(output_gemaps_dir+'raw/'+target_file,
                     float_format = '%.10f', sep=',', index=False,header=True)
  
    # znormalized
    temp_dict = {}
    gemaps_std_list,gemaps_mean_list,gemaps_max_list,gemaps_min_list = [],[],[],[]
    temp_dict['frame_time'] = target_csv_gemaps['frameTime']
    for feature in gemaps_feat_name_list:
        # if feature in skip_normalization_list:
        #     temp_dict[feature] = target_csv_gemaps[feature]
        # else:
        try:
            temp_dict[feature] = preprocessing.scale(target_csv_gemaps[feature] )
        except ValueError:
            print('Error in feature:'+feature+' in file: '+target_file)
                
                    
            # gemaps_feat_name_list.append(feature)
        gemaps_std_list.append(np.std(target_csv_gemaps[feature],axis=0))
        gemaps_mean_list.append(np.mean(target_csv_gemaps[feature],axis=0))
        gemaps_max_list.append(np.max(target_csv_gemaps[feature],axis=0))
        gemaps_min_list.append(np.min(target_csv_gemaps[feature],axis=0))
            
    mean_list.append(gemaps_mean_list)
    std_list.append(gemaps_std_list)
    min_list.append(gemaps_min_list)
    max_list.append(gemaps_max_list)
            
    outputcsv = pd.DataFrame(temp_dict)
    outputcsv[list(temp_dict.keys())].to_csv(output_gemaps_dir+'znormalized/'+target_file,
                     float_format = '%.10f', sep=',', index=False,header=True)
    
    return std_list,mean_list,num_vals

def process_func_two(data):
    target_file, annotation_file,variance_pd,mean_pd = data
    target_csv_gemaps = pd.read_csv(input_gemaps_dir+target_file,delimiter=',')

    # znormalized
    temp_dict = {}
    temp_dict['frame_time'] = target_csv_gemaps['frameTime']
    for feature in gemaps_feat_name_list:
        # if feature in skip_normalization_list:
        #     temp_dict[feature] = np.array(target_csv_gemaps[feature])
        # else:
        temp_dict[feature] =  (np.array(target_csv_gemaps[feature]) - np.array(mean_pd[feature]))/np.array(variance_pd[feature])
            
    outputcsv = pd.DataFrame(temp_dict)
    outputcsv = outputcsv[['frame_time']+gemaps_feat_name_list]
    outputcsv = outputcsv.fillna(0)
    outputcsv.to_csv(output_gemaps_dir+'znormalized_pooled/'+target_file,
                     float_format = '%.10f', sep=',', index=False,header=True)



mean_list, max_list, min_list, std_list,num_vals = [],[],[],[],[]


if __name__ == '__main__':

    my_data_one = []
    for target_file, annotation_file in zip(csv_files,voice_activity_files):
        my_data_one.append([target_file,annotation_file])
    p = Pool(num_workers)
    multi_output=p.map(process_func_one,my_data_one)
    std_list,mean_list,num_vals = [],[],[]
    for l in multi_output:
        std_list.append(l[0][0])
        mean_list.append(l[1][0])
        num_vals.append(l[2][0])        
    
    totalTime = t.time() - curTime
    print('Time taken:')
    print(totalTime)
    
    #%% Reprocess values
    for feature_set in ['znormalized_pooled']:
        if not(os.path.exists(output_gemaps_dir+'/'+feature_set)):
            os.mkdir(output_gemaps_dir+'/'+feature_set)
    
    numerator_variance = np.sum(np.tile(np.array(num_vals)-1,[np.array(std_list).shape[1],1]).transpose() * np.array(std_list)**2,axis=0)
    pooled_variance = np.sqrt(numerator_variance/(sum(num_vals)-len(num_vals)))
    
    numerator_mean = np.sum(np.tile(np.array(num_vals),[np.array(mean_list).shape[1],1]).transpose() * np.array(mean_list),axis=0)
    pooled_mean = numerator_mean/(sum(num_vals))
    
    mean_pd,variance_pd = {},{}
    for feat_name,p_mean,p_var in zip(gemaps_feat_name_list,pooled_mean,pooled_variance):
        mean_pd[feat_name] = p_mean
        variance_pd[feat_name] = p_var
    
    my_data_two = []
    for target_file, annotation_file in zip(csv_files,voice_activity_files):
        my_data_two.append([target_file,annotation_file,variance_pd,mean_pd])
    
    p.map(process_func_two,my_data_two)
    # process_func_two(my_data_two[)
    
