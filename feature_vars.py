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


gemaps_10ms_dict_list = [{'folder_path': './data/datasets/gemaps_split.hdf5',
                          'features': gemaps_features_list,
                          'modality': 'acous',
                          'is_h5_file': True,
                          #                     'uses_lstm_cell': False,
                          'uses_master_time_rate': False,
                          'time_step_size': 5,
                          'is_irregular': False,
                          'short_name': 'gmaps10'}]

gemaps_50ms_dict_list = [{'folder_path': './data/features/gemaps_features_processed/znormalized',
                          'features': gemaps_features_list,
                          'modality': 'acous',
                          'is_h5_file': False,
                          'uses_master_time_rate': True,
                          #                     'uses_lstm_cell':False,
                          'time_step_size': 1,
                          'is_irregular': False,
                          'short_name': 'gmaps50'}]


va_50ms_dict_list = [{'folder_path': './data/extracted_annotations/voice_activity',
                      'features': ['val'],
                      'modality': 'acous',
                      'is_h5_file': False,
                      #                     'uses_lstm_cell': False,
                      'uses_master_time_rate': True,
                      'time_step_size': 1,
                      'is_irregular': False,
                      'short_name': 'va50'
                      }]
va_10ms_dict_list = [{'folder_path': './data/extracted_annotations/voice_activity_10ms',
                      'features': ['val'],
                      'modality': 'acous',
                      'is_h5_file': False,
                      #                     'uses_lstm_cell': False,
                      'uses_master_time_rate': False,
                      'time_step_size': 5,
                      'is_irregular': False,
                      'short_name': 'va10'
                      }]

# %% linguistic features
# words
# if os.path.split(os.path.abspath('../'))[-1] == 'lstm_turn_taking_prediction':
word_to_ix = pickle.load(open('./data/extracted_annotations/word_to_ix.p', 'rb'))
word_embed_in_dim = len(word_to_ix) + 1
word_embed_out_dim = 64
word_embed_out_dim_glove = 300
word_features_list = ['word']

word_irreg_dict_list = [{'folder_path': './data/extracted_annotations/words_split.hdf5',
                         'features': word_features_list,
                         'modality': 'visual',
                         'is_h5_file': True,
                         'uses_master_time_rate': True,
                         'time_step_size': 1,
                         'is_irregular': True,
                         'short_name': 'wrd_irreg',
                         'title_string': '_word',
                         'embedding': True,
                         'embedding_num': word_embed_in_dim,
                         'embedding_in_dim': len(word_features_list),
                         'embedding_out_dim': word_embed_out_dim,
                         'embedding_use_func': True,
                         'use_glove': False}]

word_reg_dict_list = [{'folder_path': './data/extracted_annotations/words_split.hdf5',
                       'features': word_features_list,
                       'modality': 'visual',
                       'is_h5_file': True,
                       'uses_master_time_rate': True,
                       #                     'uses_lstm_cell':False,
                       'time_step_size': 1,
                       'is_irregular': False,
                       'short_name': 'wrd_reg',
                       'title_string': '_word',
                       'embedding': True,
                       'embedding_num': word_embed_in_dim,
                       'embedding_in_dim': len(word_features_list),
                       'embedding_out_dim': word_embed_out_dim,
                       'embedding_use_func': True,
                       'use_glove': False}]

#    
#    word_dict_list = [{'folder_path': '../extracted_annotations/words_advanced_100',
#                     'features':word_features_list,
#                     'modality':'visual',
#                     'is_h5_file': False,
#                     'uses_master_time_rate':True,
#                     'uses_lstm_cell':True,
#                     'time_step_size': 1,
#                     'title_string': '_word',
#                     'embedding':True,
#                     'embedding_num': word_embed_in_dim,
#                     'embedding_in_dim': len(word_features_list),
#                     'embedding_out_dim': word_embed_out_dim,
#                     'embedding_use_func':True}]

# part of speech
#    pos_embed_in_dim = 60
#    pos_embed_out_dim = 64
#    pos_features_list = ['POS_'+str(pos_indx) for pos_indx in list(range(pos_embed_in_dim))]
#    pos_dict_list = [{'folder_path': '../extracted_annotations/POS_advanced_100',
#                     'features':pos_features_list,
#                     'title_string': '_word',
#                     'embedding':True,
#                     'embedding_num': pos_embed_in_dim,
#                     'embedding_in_dim': len(pos_features_list),
#                     'embedding_out_dim': pos_embed_out_dim,
#                     'embedding_use_func':False}]

