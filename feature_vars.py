# -*- coding: utf-8 -*-

import os.path
import pickle

# %% Setup dictionaries of files and features to send to dataloader
# structure of list sent to dataloaders should be [{'folder_path':'path_to_folder','features':['feat_1','feat_2']},{...}]     
gemaps_features_list = ['F0semitoneFrom27.5Hz', 'jitterLocal', 'F1frequency',
                        'F1bandwidth', 'F2frequency', 'F3frequency', 'Loudness',
                        'shimmerLocaldB', 'HNRdBACF', 'alphaRatio', 'hammarbergIndex',
                        'spectralFlux', 'slope0-500', 'slope500-1500', 'F1amplitudeLogRelF0',
                        'F2amplitudeLogRelF0', 'F3amplitudeLogRelF0', 'mfcc1', 'mfcc2', 'mfcc3',
                        'mfcc4']

gemaps_10ms_dict_list = [
    {'folder_path': './data/datasets/gemaps_split.hdf5',
     'features': gemaps_features_list,
     'modality': 'acous',
     'is_h5_file': True,
     'uses_master_time_rate': False,
     'time_step_size': 5,
     'is_irregular': False,
     'short_name': 'gmaps10'}]

gemaps_10ms_dict_list_csv = [
    {'folder_path': './data/signals/gemaps_features_processed_10ms/znormalized',
     'features': gemaps_features_list,
     'modality': 'acous',
     'is_h5_file': False,
     'uses_master_time_rate': False,
     'time_step_size': 5,
     'is_irregular': False,
     'short_name': 'gmaps10'}]

gemaps_50ms_dict_list = [
    {'folder_path': './data/signals/gemaps_features_processed_50ms/znormalized',
     'features': gemaps_features_list,
     'modality': 'acous',
     'is_h5_file': False,
     'uses_master_time_rate': True,
     'time_step_size': 1,
     'is_irregular': False,
     'short_name': 'gmaps50'}]


va_50ms_dict_list = [
    {'folder_path': './data/extracted_annotations/voice_activity',
     'features': ['val'],
     'modality': 'acous',
     'is_h5_file': False,
     'uses_master_time_rate': True,
     'time_step_size': 1,
     'is_irregular': False,
     'short_name': 'va50'
     }]
va_10ms_dict_list = [
    {'folder_path': './data/extracted_annotations/voice_activity_10ms',
     'features': ['val'],
     'modality': 'acous',
     'is_h5_file': False,
     'uses_master_time_rate': False,
     'time_step_size': 5,
     'is_irregular': False,
     'short_name': 'va10'
     }]


# %% linguistic features

# word_embed_in_dim = len(pickle.load(open('./data/extracted_annotations/glove_embed_table.p', 'rb')))
word_embed_in_dim_50ms = len(pickle.load(open('./data/extracted_annotations/set_dict_50ms.p', 'rb')))
word_embed_in_dim_10ms = len(pickle.load(open('./data/extracted_annotations/set_dict_10ms.p', 'rb')))
word_embed_out_dim_no_glove = 64
# word_embed_out_dim_glove = 300
word_features_list = ['word']

word_reg_dict_list_visual = [
    {'folder_path': './data/extracted_annotations/words_advanced_50ms_averaged/',
     'features': word_features_list,
     'modality': 'visual',
     'is_h5_file': False,
     'uses_master_time_rate': True,
     'time_step_size': 1,
     'is_irregular': False,
     'short_name': 'wrd_reg',
     'title_string': '_word',
     'embedding': True,
     'embedding_num': word_embed_in_dim_50ms,
     'embedding_in_dim': len(word_features_list),
     'embedding_out_dim': word_embed_out_dim_no_glove,
     'embedding_use_func': True,
     'use_glove': False,
     'glove_embed_table':''
     }]

word_reg_dict_list_acous = [
    {'folder_path': './data/extracted_annotations/words_advanced_50ms_averaged/',
     'features': word_features_list,
     'modality': 'acous',
     'is_h5_file': False,
     'uses_master_time_rate': True,
     'time_step_size': 1,
     'is_irregular': False,
     'short_name': 'wrd_reg',
     'title_string': '_word',
     'embedding': True,
     'embedding_num': word_embed_in_dim_50ms,
     'embedding_in_dim': len(word_features_list),
     'embedding_out_dim': word_embed_out_dim_no_glove,
     'embedding_use_func': True,
     'use_glove': False,
     'glove_embed_table':''
     }]

word_reg_dict_list_10ms_acous = [
    {'folder_path': './data/datasets/words_split_10ms_5_chunked.hdf5',
     'features': word_features_list,
     'modality': 'acous',
     'is_h5_file': True,
     'uses_master_time_rate': False,
     'time_step_size': 5,
     'is_irregular': False,
     'short_name': 'wrd_reg_10ms',
     'title_string': '_word',
     'embedding': True,
     'embedding_num': word_embed_in_dim_10ms,
     'embedding_in_dim': len(word_features_list),
     'embedding_out_dim': word_embed_out_dim_no_glove,
     'embedding_use_func': True,
     'use_glove': False,
     'glove_embed_table':''
     }]

word_reg_dict_list_10ms_visual = [
    {'folder_path': './data/datasets/words_split_10ms_5_chunked.hdf5',
     'features': word_features_list,
     'modality': 'visual',
     'is_h5_file': True,
     'uses_master_time_rate': False,
     'time_step_size': 5,
     'is_irregular': False,
     'short_name': 'wrd_reg_10ms',
     'title_string': '_word',
     'embedding': True,
     'embedding_num': word_embed_in_dim_10ms,
     'embedding_in_dim': len(word_features_list),
     'embedding_out_dim': word_embed_out_dim_no_glove,
     'embedding_use_func': True,
     'use_glove': False,
     'glove_embed_table':''
     }]

word_irreg_dict_list = [
    {'folder_path': './data/datasets/words_split_irreg_50ms.hdf5',
     'features': word_features_list,
     'modality': 'visual',
     'is_h5_file': True,
     'uses_master_time_rate': False,
     'time_step_size': 2,
     'is_irregular': True,
     'short_name': 'wrd_irreg',
     'title_string': '_word',
     'embedding': True,
     'embedding_num': word_embed_in_dim_50ms,
     'embedding_in_dim': len(word_features_list),
     'embedding_out_dim': word_embed_out_dim_no_glove,
     'embedding_use_func': True,
     'use_glove': False,
     'glove_embed_table':''
     }]

word_irreg_fast_dict_list = [
    {'folder_path': './data/datasets/words_split_50ms.hdf5',
     'features': word_features_list,
     'modality': 'visual',
     'is_h5_file': True,
     'uses_master_time_rate': True,
     'time_step_size': 1,
     'is_irregular': True,
     'short_name': 'wrd_irreg_fast',
     'title_string': '_word',
     'embedding': True,
     'embedding_num': word_embed_in_dim_50ms,
     'embedding_in_dim': len(word_features_list),
     'embedding_out_dim': word_embed_out_dim_no_glove,
     'embedding_use_func': True,
     'use_glove': False,
     'glove_embed_table':''
     }]


