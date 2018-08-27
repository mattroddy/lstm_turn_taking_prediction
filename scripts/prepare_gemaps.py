# -*- coding: utf-8 -*-
import os
import time as t
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import scipy.io as io
from multiprocessing import Pool
import sys
num_workers = 4

# takes about 5 mins for 10ms, 1 min for 50ms
if len(sys.argv)==2:
    speed_setting = int(sys.argv[1])
else:
    speed_setting = 0 # 0 for 50ms, 1 for 10ms

if speed_setting == 0:
    input_gemaps_dir = './data/signals/gemaps_features_50ms/'
    output_gemaps_dir = './data/signals/gemaps_features_processed_50ms/'
elif speed_setting == 1:
    input_gemaps_dir = './data/signals/gemaps_features_10ms/'
    output_gemaps_dir = './data/signals/gemaps_features_processed_10ms/'
    shift_back_amount = 4 # The window_size is 50ms long so needs to be shifted back to avoid looking into future.

annotation_dir = './data/extracted_annotations/voice_activity/'

csv_files=os.listdir(input_gemaps_dir)
voice_activity_files = [file.split('.')[0]+'.'+file.split('.')[1]+'.csv' for file in csv_files]
skip_normalization_list = []
if not(os.path.exists(output_gemaps_dir)):
    os.mkdir(output_gemaps_dir)
for feature_set in ['raw','znormalized','/znormalized_pooled']:
    if not(os.path.exists(output_gemaps_dir+'/'+feature_set)):
        os.mkdir(output_gemaps_dir+'/'+feature_set)

frequency_features_list = ['F0semitoneFrom27.5Hz','jitterLocal','F1frequency', 
                           'F1bandwidth','F2frequency','F3frequency']
frequency_mask_list = [0,0,0,
                       0,0,0]
energy_features_list = ['Loudness','shimmerLocaldB', 'HNRdBACF']
energy_mask_list = [0,0,0]
spectral_features_list = ['alphaRatio','hammarbergIndex','spectralFlux', 
                          'slope0-500', 'slope500-1500','F1amplitudeLogRelF0',
                          'F2amplitudeLogRelF0','F3amplitudeLogRelF0','mfcc1', 
                          'mfcc2','mfcc3', 'mfcc4']
spectral_mask_list = [0,0,0,
                      0,0,-201,
                      -201,-201,0,
                      0,0,0]

gemaps_full_feature_list = frequency_features_list + energy_features_list + spectral_features_list
gemaps_feat_name_list = frequency_features_list + energy_features_list + spectral_features_list
#full_mask_list = frequency_mask_list + energy_mask_list + spectral_mask_list
#%% get the names of features and test the alignment of features

test_covarep = pd.read_csv(input_gemaps_dir+csv_files[0],delimiter=',')
test_gemaps = pd.read_csv(input_gemaps_dir+csv_files[0],delimiter=',')



#%% loop through files
curTime = t.time()
missing_count = 0


def loop_func_one(data):
    #%%
    target_file, annotation_file = data

    # Get files and annotations
#    target_mat_covarep = io.loadmat(input_gemaps_dir+target_file)
#    target_csv_gemaps = target_mat_covarep['features']
    target_csv_gemaps = pd.read_csv(input_gemaps_dir+target_file,delimiter=',')
    mean_list, max_list, min_list, std_list, num_vals = [], [], [], [], []
    num_vals.append( len(target_csv_gemaps))
    # raw features
    temp_dict = {}
    temp_dict['frame_time'] = target_csv_gemaps['frameTime']
    for feature in gemaps_full_feature_list:
        if speed_setting ==1:
            tmp = np.zeros(target_csv_gemaps[feature].shape)
            tmp[:-shift_back_amount] = target_csv_gemaps[feature][shift_back_amount:]
            temp_dict[feature] = tmp
        else:
            temp_dict[feature] = target_csv_gemaps[feature]
    outputcsv = pd.DataFrame(temp_dict)
    outputcsv[list(temp_dict.keys())].to_csv(output_gemaps_dir+'raw/'+target_file,
                     float_format = '%.10f', sep=',', index=False,header=True)
  
    # znormalized
    temp_dict = {}
    covarep_std_list,covarep_mean_list,covarep_max_list,covarep_min_list,gemaps_feat_name_list = [],[],[],[],[]
    temp_dict['frame_time'] = target_csv_gemaps['frameTime']
    for feature in gemaps_full_feature_list:
        if feature in skip_normalization_list:
            if speed_setting == 1:
                temp_dict[feature] = np.zeros(target_csv_gemaps[feature].shape)
                temp_dict[feature][:-shift_back_amount] = target_csv_gemaps[feature][shift_back_amount:]
            else:
                temp_dict[feature] = target_csv_gemaps[feature]
        else:
            if speed_setting == 1:
                temp_dict[feature] = np.zeros(target_csv_gemaps[feature].shape)
                temp_dict[feature][:-shift_back_amount] = preprocessing.scale(target_csv_gemaps[feature])[shift_back_amount:]
            else:
                temp_dict[feature] = preprocessing.scale(target_csv_gemaps[feature] )

            gemaps_feat_name_list.append(feature)
            covarep_std_list.append(np.std(target_csv_gemaps[feature],axis=0))
            covarep_mean_list.append(np.mean(target_csv_gemaps[feature],axis=0))
            covarep_max_list.append(np.max(target_csv_gemaps[feature],axis=0))
            covarep_min_list.append(np.min(target_csv_gemaps[feature],axis=0))
            
    mean_list.append(covarep_mean_list)
    std_list.append(covarep_std_list)
    min_list.append(covarep_min_list)
    max_list.append(covarep_max_list)
            
    outputcsv = pd.DataFrame(temp_dict)
    outputcsv[list(temp_dict.keys())].to_csv(output_gemaps_dir+'znormalized/'+target_file,
                     float_format = '%.10f', sep=',', index=False,header=True)
    
    return std_list,mean_list,num_vals

#for target_file, annotation_file in zip(csv_files,voice_activity_files):
def loop_func_two(data):
    target_file, annotation_file,variance_pd,mean_pd = data
    target_csv_gemaps = pd.read_csv(input_gemaps_dir+target_file,delimiter=',')

    # znormalized_pooled
    temp_dict = {}
#    covarep_std_list,covarep_mean_list,covarep_max_list,covarep_min_list,gemaps_feat_name_list = [],[],[],[],[]
    temp_dict['frame_time'] = target_csv_gemaps['frameTime']
    for feature in gemaps_full_feature_list:
        if feature in skip_normalization_list:
            if speed_setting == 1:
                temp_dict[feature] = np.zeros(target_csv_gemaps[feature].shape)
                temp_dict[feature][:-shift_back_amount] = target_csv_gemaps[shift_back_amount:]
            else:
                temp_dict[feature] = np.array(target_csv_gemaps[feature])
        else:
            if speed_setting ==1:
                tmp = np.zeros(target_csv_gemaps[feature].shape)
                tmp2 = (np.array(target_csv_gemaps[feature]) - np.array(mean_pd[feature]))/np.array(variance_pd[feature])
                tmp[:-shift_back_amount] = tmp2[shift_back_amount:]
                temp_dict[feature] = tmp
            else:
                temp_dict[feature] =  (np.array(target_csv_gemaps[feature]) - np.array(mean_pd[feature]))/np.array(variance_pd[feature])

            
    outputcsv = pd.DataFrame(temp_dict)
    outputcsv = outputcsv[['frame_time']+gemaps_full_feature_list]
    outputcsv = outputcsv.fillna(0)
    outputcsv.to_csv(output_gemaps_dir+'znormalized_pooled/'+target_file,
                     float_format = '%.10f', sep=',', index=False,header=True)

    
#loop_func_one(my_data_one[0])


mean_list, max_list, min_list, std_list,num_vals = [],[],[],[],[]
if __name__ == '__main__':
    my_data_one = []
    for target_file, annotation_file in zip(csv_files,voice_activity_files):
        my_data_one.append([target_file,annotation_file])
    p = Pool(num_workers)
    multi_output=p.map(loop_func_one,my_data_one)
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
            
    #max_pd=pd.DataFrame(columns=gemaps_feat_name_list)
    #max_pd.loc[0] = np.transpose(np.max(np.array(max_list),axis=0))
    #min_pd=pd.DataFrame(columns=gemaps_feat_name_list)
    #min_pd.loc[0] = np.transpose(np.min(np.array(max_list),axis=0))
    
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
    
    p.map(loop_func_two,my_data_two)
    
#mean_pd=pd.DataFrame(columns=gemaps_feat_name_list)
#mean_pd.loc[0] = np.transpose(pooled_mean)
#variance_pd=pd.DataFrame(columns=gemaps_feat_name_list)
#variance_pd.loc[0] = np.transpose(pooled_variance)





