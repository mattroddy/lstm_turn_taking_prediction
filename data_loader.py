# -*- coding: utf-8 -*-
import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
from itertools import zip_longest
import h5py

import warnings

warnings.filterwarnings('ignore')
use_cuda = torch.cuda.is_available()

print('Use CUDA: ' + str(use_cuda))
if use_cuda:
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
    p_memory = False
else:
    dtype = torch.FloatTensor
    dtype_long = torch.cuda.LongTensor
    p_memory = False

if 'h_data' in locals():
    if isinstance(h_data, h5py.File):
        try:
            h_data.close()
        except:
            pass
data_select_dict = {0: ['f', 'g'],
                    1: ['c1', 'c2']}
time_label_select_dict = {0: 'frame_time',  # gemaps
                          1: 'timestamp'}  # openface


class TurnPredictionDataset(Dataset):

    def __init__(self, feature_dict_list, annotations_dir, file_list, seq_length, prediction_length=60, set_type='test', data_select=0):
        # inputs: feature_dict_list is string path of folder with extracted data csv files
        # file_list is string with path to a .txt file with file list in it
        # todo: option to select features, correct docstring
        self.len = 0
        self.seq_length = seq_length
        self.feature_dict_list = feature_dict_list
        self.annotations_dir = annotations_dir
        self.file_list = pd.read_csv(file_list, header=None, dtype=str)[0]
        self.set_type = set_type
        self.dataset = list()
        self.prediction_length = prediction_length
        self.results_lengths = dict()
        self.data_select = data_select
        self.time_step_size = {}
        self.active_modalities = []
        self.uses_master_time_rate_bool = {'acous': True, 'visual': True}
        self.time_step_size = {'acous': 1, 'visual': 1}
        self.is_irregular = {'acous': False, 'visual': False}

        # %% embedding stuff
        self.feature_size = 0

        # %% Train
        if self.set_type == 'train':
            print('loading training data into memory')
            for filename in self.file_list:
                # !!! Should at some point combine all the fs and gs so that there is less code duplication
                # split data into data_f and data_g
                # !!! data is dictionary in form: data_f['modality']['x_type']['feature']
                data_f = {'acous': {'x': {}, 'x_i': {}},
                          'visual': {'x': {}, 'x_i': {}}}
                data_g = {'acous': {'x': {}, 'x_i': {}},
                          'visual': {'x': {}, 'x_i': {}}}
                self.total_embed_in_dim = {'acous': 0, 'visual': 0}
                self.total_embed_out_dim = {'acous': 0, 'visual': 0}
                self.feature_name_list = {'acous': [], 'visual': []}
                self.embedding_info = {'acous': [], 'visual': []}  # is set every file loop, probably some other more efficient way
                annot_f = pd.read_csv(self.annotations_dir + '/' + filename + '.' + data_select_dict[data_select][0] + '.csv', delimiter=',')
                annot_g = pd.read_csv(self.annotations_dir + '/' + filename + '.' + data_select_dict[data_select][1] + '.csv', delimiter=',')

                for feature_dict in self.feature_dict_list:

                    # Get settings for the modality
                    if not (feature_dict['modality'] in self.active_modalities):
                        self.active_modalities.append(feature_dict['modality'])
                        self.time_step_size[feature_dict['modality']] = feature_dict['time_step_size']
                        self.is_irregular[feature_dict['modality']] = feature_dict['is_irregular']  # uses time_bools to change lstm input or not
                        self.uses_master_time_rate_bool[feature_dict['modality']] = feature_dict['uses_master_time_rate']  # is 50ms or 10ms sampled

                    if not (feature_dict['is_h5_file']):
                        data_f_temp = pd.read_csv(feature_dict['folder_path'] + '/' + filename + '.' + data_select_dict[data_select][0] + '.csv')
                        data_g_temp = pd.read_csv(feature_dict['folder_path'] + '/' + filename + '.' + data_select_dict[data_select][1] + '.csv')
                        if 'embedding' in feature_dict and feature_dict['embedding'] == True:
                            embed_info = {}
                            #                        embed_info['features'] = feature_dict['features'] # need to add 'f_' and 'g_' do this at end
                            for embed_key in ['features', 'embedding', 'title_string', 'embedding_num', 'embedding_in_dim', 'embedding_out_dim', 'embedding_use_func', 'use_glove']:
                                embed_info[embed_key] = feature_dict[embed_key]
                            self.embedding_info[feature_dict['modality']].append(embed_info)
                            self.total_embed_in_dim[feature_dict['modality']] = self.total_embed_in_dim[feature_dict['modality']] + embed_info['embedding_in_dim']
                            self.total_embed_out_dim[feature_dict['modality']] = self.total_embed_out_dim[feature_dict['modality']] + embed_info['embedding_out_dim']
                            self.embedding_info[feature_dict['modality']][-1]['emb_indices'] = [(len(self.feature_name_list[feature_dict['modality']]), len(self.feature_name_list[feature_dict['modality']]) + embed_info['embedding_in_dim'])]

                        for feature_name in feature_dict['features']:
                            data_f[feature_dict['modality']]['x'][feature_name] = data_f_temp[feature_name]
                            data_g[feature_dict['modality']]['x'][feature_name] = data_g_temp[feature_name]
                        self.feature_name_list[feature_dict['modality']] += feature_dict['features']  # this makes other code for name lists superfluous

                    else:

                        h_data = h5py.File(feature_dict['folder_path'], 'r')
                        for feature_name in feature_dict['features']:
                            data_f[feature_dict['modality']]['x'][feature_name] = h_data[filename + '/' + data_select_dict[data_select][0] + '/x/' + feature_name]
                            data_f[feature_dict['modality']]['x_i'][feature_name] = h_data[filename + '/' + data_select_dict[data_select][0] + '/x_i/' + feature_name]
                            data_g[feature_dict['modality']]['x'][feature_name] = h_data[filename + '/' + data_select_dict[data_select][1] + '/x/' + feature_name]
                            data_g[feature_dict['modality']]['x_i'][feature_name] = h_data[filename + '/' + data_select_dict[data_select][1] + '/x_i/' + feature_name]

                        if 'embedding' in feature_dict and feature_dict['embedding'] == True:
                            embed_info = {}
                            #                        embed_info['features'] = feature_dict['features'] # need to add 'f_' and 'g_' do this at end
                            for embed_key in ['features', 'embedding', 'title_string', 'embedding_num', 'embedding_in_dim', 'embedding_out_dim', 'embedding_use_func', 'use_glove']:
                                embed_info[embed_key] = feature_dict[embed_key]
                            self.embedding_info[feature_dict['modality']].append(embed_info)
                            self.total_embed_in_dim[feature_dict['modality']] = self.total_embed_in_dim[feature_dict['modality']] + embed_info['embedding_in_dim']
                            self.total_embed_out_dim[feature_dict['modality']] = self.total_embed_out_dim[feature_dict['modality']] + embed_info['embedding_out_dim']
                            self.embedding_info[feature_dict['modality']][-1]['emb_indices'] = [(len(self.feature_name_list[feature_dict['modality']]), len(self.feature_name_list[feature_dict['modality']]) + embed_info['embedding_in_dim'])]

                        self.feature_name_list[feature_dict['modality']] += feature_dict['features']

                self.num_feat_per_person = {'acous': len(data_f['acous']['x'].keys()),
                                            'visual': len(data_f['visual']['x'].keys())}  # this is half the dimension of the output of dataloader
                self.num_feat_for_lstm = {'acous': 2 * (self.num_feat_per_person['acous'] - self.total_embed_in_dim['acous'] + self.total_embed_out_dim['acous']),
                                          'visual': 2 * (self.num_feat_per_person['visual'] - self.total_embed_in_dim['visual'] + self.total_embed_out_dim['visual'])}

                #### split features into batches 
                file_dur = len(annot_f)
                num_batches = int(np.floor(file_dur / self.seq_length))
                self.results_lengths[filename] = num_batches * self.seq_length
                data_f_np_times = np.array(annot_f['frame_time'])
                data_g_np_times = np.array(annot_g['frame_time'])

                predict_f_np = np.array([np.roll(annot_f['val'], -roll_indx) for roll_indx in range(1, prediction_length + 1)]).transpose()
                predict_g_np = np.array([np.roll(annot_g['val'], -roll_indx) for roll_indx in range(1, prediction_length + 1)]).transpose()

                data_f_np_dict, data_g_np_dict, data_f_np_dict_list, data_g_np_dict_list, data_f_np_bools, data_g_np_bools = {}, {}, {}, {}, {}, {}
                #                for modality in self.feature_name_list.keys():
                for modality in self.active_modalities:
                    #                    print(modality)
                    #                    print(self.is_irregular[modality])
                    if not (self.is_irregular[modality]):
                        data_f_np_dict_list[modality] = list()
                        data_g_np_dict_list[modality] = list()
                        for feature_name in self.feature_name_list[modality]:
                            data_f_np_dict_list[modality].append(np.squeeze(np.array(data_f[modality]['x'][feature_name])))  # !!!
                            data_g_np_dict_list[modality].append(np.squeeze(np.array(data_g[modality]['x'][feature_name])))

                        data_f_np_dict[modality] = np.asarray(data_f_np_dict_list[modality]).reshape([len(data_f_np_dict_list[modality]), len(data_f_np_dict_list[modality][0]), self.time_step_size[modality]])
                        data_g_np_dict[modality] = np.asarray(data_g_np_dict_list[modality]).reshape([len(data_g_np_dict_list[modality]), len(data_g_np_dict_list[modality][0]), self.time_step_size[modality]])

                    else:
                        data_f_np_dict_list[modality] = list()
                        data_g_np_dict_list[modality] = list()

                        for feature_name in self.feature_name_list[modality]:
                            data_f_np_dict_list[modality].append(np.array(data_f[modality]['x'][feature_name]))
                            data_g_np_dict_list[modality].append(np.array(data_g[modality]['x'][feature_name]))
                        data_f_np_dict[modality] = np.zeros([len(data_f_np_dict_list[modality]), len(data_f_np_dict_list[modality][0]), self.time_step_size[modality]])
                        data_g_np_dict[modality] = np.zeros([len(data_g_np_dict_list[modality]), len(data_g_np_dict_list[modality][0]), self.time_step_size[modality]])
                        if self.data_select == 0:
                            data_f_np_dict[modality][:, :, :data_f_np_dict_list[modality][0].shape[-1]] = np.asarray(data_f_np_dict_list[modality]).reshape([len(data_f_np_dict_list[modality]), len(data_f_np_dict_list[modality][0]), self.time_step_size[modality]])
                            data_g_np_dict[modality][:, :, :data_g_np_dict_list[modality][0].shape[-1]] = np.asarray(data_g_np_dict_list[modality]).reshape([len(data_g_np_dict_list[modality]), len(data_g_np_dict_list[modality][0]), self.time_step_size[modality]])
                        else:
                            data_f_np_dict[modality][:, :, :data_f_np_dict_list[modality][0].shape[-1]] = np.asarray(data_f_np_dict_list[modality])
                            data_g_np_dict[modality][:, :, :data_g_np_dict_list[modality][0].shape[-1]] = np.asarray(data_g_np_dict_list[modality])
                        # !!!
                        # get bool indices
                        data_f_np_bools[modality] = np.zeros([data_f[modality]['x_i'][feature_name].shape[0], self.time_step_size[modality]])
                        data_g_np_bools[modality] = np.zeros([data_g[modality]['x_i'][feature_name].shape[0], self.time_step_size[modality]])
                        data_f_np_bools[modality][:, :data_f[modality]['x_i'][feature_name].shape[-1]] = \
                            np.asarray(data_f[modality]['x_i'][feature_name], dtype=np.float32) + np.asarray(data_g[modality]['x_i'][feature_name], dtype=np.float32)  # note: all features should have the same bool matrix in current implementation
                        data_g_np_bools[modality][:, :data_g[modality]['x_i'][feature_name].shape[-1]] = \
                            np.asarray(data_g[modality]['x_i'][feature_name], dtype=np.float32) + np.asarray(data_f[modality]['x_i'][feature_name], dtype=np.float32)

                # features for g  
                for i in range(1, num_batches + 1):
                    datapoint, data_temp_x, data_temp_x_i = {}, {}, {}
                    for modality in self.active_modalities:
                        if not (self.is_irregular[modality]):
                            data_temp_x[modality] = np.empty([2 * self.num_feat_per_person[modality], self.seq_length, self.time_step_size[modality]], dtype=np.float32)
                            data_temp_x[modality][0:self.num_feat_per_person[modality], :, :] = data_g_np_dict[modality][:, (i - 1) * self.seq_length:i * self.seq_length, :]
                            data_temp_x[modality][self.num_feat_per_person[modality]:, :, :] = data_f_np_dict[modality][:, (i - 1) * self.seq_length:i * self.seq_length, :]
                        else:
                            data_temp_x[modality] = np.zeros([2 * self.num_feat_per_person[modality], self.seq_length, self.time_step_size[modality]], dtype=np.float32)
                            data_temp_x_i[modality] = np.zeros([self.seq_length, self.time_step_size[modality]], dtype=np.float32)
                            data_temp_x_i[modality][:self.seq_length, :self.time_step_size[modality]] = \
                                data_g_np_bools[modality][(i - 1) * self.seq_length:i * self.seq_length, :self.time_step_size[modality]]
                            data_temp_x[modality][0:self.num_feat_per_person[modality], :, :self.time_step_size[modality]] = \
                                data_g_np_dict[modality][:, (i - 1) * self.seq_length:i * self.seq_length]
                            data_temp_x[modality][self.num_feat_per_person[modality]:, :, :self.time_step_size[modality]] = \
                                data_f_np_dict[modality][:, (i - 1) * self.seq_length:i * self.seq_length]
                    reset_states_flag = i == 1  # during training the states are reset in the run_json code anyway so this value is unimportant
                    datapoint['x'] = data_temp_x  # note: might need to treat modalites is separate 'x's
                    datapoint['y'] = predict_g_np[(i - 1) * self.seq_length:i * self.seq_length]
                    datapoint['info'] = {
                        'reset_states_flag': reset_states_flag,
                        #                            'g_f':'g',
                        'g_f': data_select_dict[data_select][1],
                        'file_names': filename,
                        'time_indices': np.array([(i - 1) * self.seq_length, i * self.seq_length]),
                        'time_frames': np.array(data_g_np_times[(i - 1) * self.seq_length:i * self.seq_length]),
                        'batch_num': i - 1,
                    }
                    datapoint['time_bools'] = data_temp_x_i
                    self.dataset.append(datapoint)
                    self.len += 1

                # features for f
                for i in range(1, num_batches + 1):
                    datapoint, data_temp_x, data_temp_x_i = {}, {}, {}
                    for modality in self.active_modalities:
                        if not (self.is_irregular[modality]):
                            data_temp_x[modality] = np.empty([2 * self.num_feat_per_person[modality], self.seq_length, self.time_step_size[modality]], dtype=np.float32)
                            data_temp_x[modality][0:self.num_feat_per_person[modality], :, :] = data_f_np_dict[modality][:, (i - 1) * self.seq_length:i * self.seq_length, :]
                            data_temp_x[modality][self.num_feat_per_person[modality]:, :, :] = data_g_np_dict[modality][:, (i - 1) * self.seq_length:i * self.seq_length, :]
                        else:
                            data_temp_x[modality] = np.zeros([2 * self.num_feat_per_person[modality], self.seq_length, self.time_step_size[modality]], dtype=np.float32)
                            data_temp_x_i[modality] = np.zeros([self.seq_length, self.time_step_size[modality]], dtype=np.float32)
                            data_temp_x_i[modality][:self.seq_length, :self.time_step_size[modality]] = \
                                data_f_np_bools[modality][(i - 1) * self.seq_length:i * self.seq_length, :self.time_step_size[modality]]
                            data_temp_x[modality][0:self.num_feat_per_person[modality], :, :self.time_step_size[modality]] = \
                                data_f_np_dict[modality][:, (i - 1) * self.seq_length:i * self.seq_length]
                            data_temp_x[modality][self.num_feat_per_person[modality]:, :, :self.time_step_size[modality]] = \
                                data_g_np_dict[modality][:, (i - 1) * self.seq_length:i * self.seq_length]
                    reset_states_flag = i == 1  # during training the states are reset in the run_json code anyway so this value is unimportant
                    datapoint['x'] = data_temp_x  # note: might need to treat modalites is separate 'x's
                    datapoint['y'] = predict_f_np[(i - 1) * self.seq_length:i * self.seq_length]
                    datapoint['info'] = {
                        'reset_states_flag': reset_states_flag,
                        #                            'g_f':'f',
                        'g_f': data_select_dict[self.data_select][0],
                        'file_names': filename,
                        'time_indices': np.array([(i - 1) * self.seq_length, i * self.seq_length]),
                        'time_frames': np.array(data_f_np_times[(i - 1) * self.seq_length:i * self.seq_length]),
                        'batch_num': i - 1,
                        #                            'feature_names':data_g_feature_names + data_f_feature_names
                        #                            'time_bools':data_temp_x_i
                    }
                    datapoint['time_bools'] = data_temp_x_i
                    self.dataset.append(datapoint)
                    self.len += 1

        # %%  Load test data
        # if test is selected, the dataloader batch size should be set to one,drop_last=False. Data will 
        elif self.set_type == 'test':

            print('loading test data into memory')
            # Sort the conversations by length and divide into sections of length seq_length
            self.dataset = list()
            self.results_lengths = dict()
            data_f_list_np, predict_f_list, data_f_list_ft, data_g_list_ft, predict_g_list, data_g_list_np = [], [], [], [], [], []
            seq_length_list, data_f_np_bools_list, data_g_np_bools_list = [], [], []
            self.feature_size = 0

            for filename, conv_indx_i in zip(self.file_list, range(len(self.file_list))):
                data_f = {'acous': {'x': {}, 'x_i': {}},
                          'visual': {'x': {}, 'x_i': {}}}
                data_g = {'acous': {'x': {}, 'x_i': {}},
                          'visual': {'x': {}, 'x_i': {}}}
                self.total_embed_in_dim = {'acous': 0, 'visual': 0}
                self.total_embed_out_dim = {'acous': 0, 'visual': 0}
                self.feature_name_list = {'acous': [], 'visual': []}
                self.embedding_info = {'acous': [], 'visual': []}  # is set every file loop, probably some other more efficient way
                annot_f = pd.read_csv(self.annotations_dir + '/' + filename + '.' + data_select_dict[data_select][0] + '.csv', delimiter=',')
                annot_g = pd.read_csv(self.annotations_dir + '/' + filename + '.' + data_select_dict[data_select][1] + '.csv', delimiter=',')

                for feature_dict in self.feature_dict_list:
                    # Get settings for the modality
                    if not (feature_dict['modality'] in self.active_modalities):
                        self.active_modalities.append(feature_dict['modality'])
                        self.time_step_size[feature_dict['modality']] = feature_dict['time_step_size']
                        self.is_irregular[feature_dict['modality']] = feature_dict['is_irregular']
                        self.uses_master_time_rate_bool[feature_dict['modality']] = feature_dict['uses_master_time_rate']

                    if not (feature_dict['is_h5_file']):
                        data_f_temp = pd.read_csv(feature_dict['folder_path'] + '/' + filename + '.' + data_select_dict[data_select][0] + '.csv')
                        data_g_temp = pd.read_csv(feature_dict['folder_path'] + '/' + filename + '.' + data_select_dict[data_select][1] + '.csv')
                        if 'embedding' in feature_dict and feature_dict['embedding'] == True:
                            embed_info = {}
                            for embed_key in ['features', 'embedding', 'title_string', 'embedding_num', 'embedding_in_dim', 'embedding_out_dim', 'embedding_use_func', 'use_glove']:
                                embed_info[embed_key] = feature_dict[embed_key]
                            self.embedding_info[feature_dict['modality']].append(embed_info)
                            self.total_embed_in_dim[feature_dict['modality']] = self.total_embed_in_dim[feature_dict['modality']] + embed_info['embedding_in_dim']
                            self.total_embed_out_dim[feature_dict['modality']] = self.total_embed_out_dim[feature_dict['modality']] + embed_info['embedding_out_dim']
                            self.embedding_info[feature_dict['modality']][-1]['emb_indices'] = [(len(self.feature_name_list[feature_dict['modality']]), len(self.feature_name_list[feature_dict['modality']]) + embed_info['embedding_in_dim'])]

                        for feature_name in feature_dict['features']:
                            data_f[feature_dict['modality']]['x'][feature_name] = data_f_temp[feature_name]
                            data_g[feature_dict['modality']]['x'][feature_name] = data_g_temp[feature_name]
                        self.feature_name_list[feature_dict['modality']] += feature_dict['features']  # this makes other code for name lists superfluous
                    else:

                        h_data = h5py.File(feature_dict['folder_path'], 'r')
                        for feature_name in feature_dict['features']:
                            data_f[feature_dict['modality']]['x'][feature_name] = h_data[filename + '/' + data_select_dict[data_select][0] + '/x/' + feature_name]
                            data_f[feature_dict['modality']]['x_i'][feature_name] = h_data[filename + '/' + data_select_dict[data_select][0] + '/x_i/' + feature_name]
                            data_g[feature_dict['modality']]['x'][feature_name] = h_data[filename + '/' + data_select_dict[data_select][1] + '/x/' + feature_name]
                            data_g[feature_dict['modality']]['x_i'][feature_name] = h_data[filename + '/' + data_select_dict[data_select][1] + '/x_i/' + feature_name]

                        if 'embedding' in feature_dict and feature_dict['embedding'] == True:
                            embed_info = {}
                            for embed_key in ['features', 'embedding', 'title_string', 'embedding_num', 'embedding_in_dim', 'embedding_out_dim', 'embedding_use_func', 'use_glove']:
                                embed_info[embed_key] = feature_dict[embed_key]
                            self.embedding_info[feature_dict['modality']].append(embed_info)
                            self.total_embed_in_dim[feature_dict['modality']] = self.total_embed_in_dim[feature_dict['modality']] + embed_info['embedding_in_dim']
                            self.total_embed_out_dim[feature_dict['modality']] = self.total_embed_out_dim[feature_dict['modality']] + embed_info['embedding_out_dim']
                            self.embedding_info[feature_dict['modality']][-1]['emb_indices'] = [(len(self.feature_name_list[feature_dict['modality']]), len(self.feature_name_list[feature_dict['modality']]) + embed_info['embedding_in_dim'])]

                        self.feature_name_list[feature_dict['modality']] += feature_dict['features']

                predict_g_list.append(np.array([np.roll(annot_g['val'], -roll_indx) for roll_indx in range(1, prediction_length + 1)]).transpose())
                predict_f_list.append(np.array([np.roll(annot_f['val'], -roll_indx) for roll_indx in range(1, prediction_length + 1)]).transpose())

                data_f_np_dict, data_g_np_dict, data_f_np_dict_list, data_g_np_dict_list, data_f_np_bools, data_g_np_bools = {}, {}, {}, {}, {}, {}
                #                for modality in self.feature_name_list.keys():
                for modality in self.active_modalities:
                    if not (self.is_irregular[modality]):
                        data_f_np_dict_list[modality] = list()
                        data_g_np_dict_list[modality] = list()
                        for feature_name in self.feature_name_list[modality]:
                            data_f_np_dict_list[modality].append(np.squeeze(np.array(data_f[modality]['x'][feature_name])))
                            data_g_np_dict_list[modality].append(np.squeeze(np.array(data_g[modality]['x'][feature_name])))
                        data_f_np_dict[modality] = np.asarray(data_f_np_dict_list[modality]).reshape([len(data_f_np_dict_list[modality]), len(data_f_np_dict_list[modality][0]), self.time_step_size[modality]])
                        data_g_np_dict[modality] = np.asarray(data_g_np_dict_list[modality]).reshape([len(data_g_np_dict_list[modality]), len(data_g_np_dict_list[modality][0]), self.time_step_size[modality]])

                    else:
                        data_f_np_dict_list[modality] = list()
                        data_g_np_dict_list[modality] = list()

                        for feature_name in self.feature_name_list[modality]:
                            data_f_np_dict_list[modality].append(np.array(data_f[modality]['x'][feature_name]))
                            data_g_np_dict_list[modality].append(np.array(data_g[modality]['x'][feature_name]))
                        data_f_np_dict[modality] = np.zeros([len(data_f_np_dict_list[modality]), len(data_f_np_dict_list[modality][0]), self.time_step_size[modality]])
                        data_g_np_dict[modality] = np.zeros([len(data_g_np_dict_list[modality]), len(data_g_np_dict_list[modality][0]), self.time_step_size[modality]])
                        if data_select == 0:
                            data_f_np_dict[modality][:, :, :data_f_np_dict_list[modality][0].shape[-1]] = np.asarray(data_f_np_dict_list[modality]).reshape([len(data_f_np_dict_list[modality]), len(data_f_np_dict_list[modality][0]), self.time_step_size[modality]])
                            data_g_np_dict[modality][:, :, :data_f_np_dict_list[modality][0].shape[-1]] = np.asarray(data_g_np_dict_list[modality]).reshape([len(data_g_np_dict_list[modality]), len(data_g_np_dict_list[modality][0]), self.time_step_size[modality]])
                        else:
                            data_f_np_dict[modality][:, :, :data_f_np_dict_list[modality][0].shape[-1]] = np.asarray(data_f_np_dict_list[modality])
                            data_g_np_dict[modality][:, :, :data_f_np_dict_list[modality][0].shape[-1]] = np.asarray(data_g_np_dict_list[modality])
                        # get bool indices
                        data_f_np_bools[modality] = np.zeros([data_f[modality]['x_i'][feature_name].shape[0], self.time_step_size[modality]])
                        data_g_np_bools[modality] = np.zeros([data_g[modality]['x_i'][feature_name].shape[0], self.time_step_size[modality]])
                        data_f_np_bools[modality][:, :data_f[modality]['x_i'][feature_name].shape[-1]] = \
                            np.asarray(data_f[modality]['x_i'][feature_name], dtype=np.float32) + np.asarray(data_g[modality]['x_i'][feature_name], dtype=np.float32)  # note: all features should have the same bool matrix in current implementation
                        data_g_np_bools[modality][:, :data_g[modality]['x_i'][feature_name].shape[-1]] = \
                            np.asarray(data_g[modality]['x_i'][feature_name], dtype=np.float32) + np.asarray(data_f[modality]['x_i'][feature_name], dtype=np.float32)

                data_f_list_np.append(data_f_np_dict)
                data_g_list_np.append(data_g_np_dict)
                data_f_np_bools_list.append(data_f_np_bools)
                data_g_np_bools_list.append(data_g_np_bools)
                data_f_list_ft.append(np.array(annot_f['frame_time']))
                data_g_list_ft.append(np.array(annot_g['frame_time']))

                # Find out how many batches there are in 
                seq_length_list.append((int(np.floor(len(data_f_list_ft[-1]) / self.seq_length)), conv_indx_i, filename))
                file_dur = len(annot_f)
                num_batches = int(np.floor(file_dur / self.seq_length))
                self.results_lengths[filename] = num_batches * self.seq_length

                self.num_feat_per_person = {'acous': len(data_f['acous']['x'].keys()),
                                            'visual': len(data_f['visual']['x'].keys())}  # this is half the dimension of the output of dataloader
                self.num_feat_for_lstm = {'acous': 2 * (self.num_feat_per_person['acous'] - self.total_embed_in_dim['acous'] + self.total_embed_out_dim['acous']),
                                          'visual': 2 * (self.num_feat_per_person['visual'] - self.total_embed_in_dim['visual'] + self.total_embed_out_dim['visual'])}

            # sort lists
            seq_length_list = sorted(seq_length_list, reverse=True)
            length_list = [seq_length_list[row][0] for row in range(len(seq_length_list))]
            cindx_list = [seq_length_list[row][1] for row in range(len(seq_length_list))]
            cfile_list = [seq_length_list[row][2] for row in range(len(seq_length_list))]

            max_length = length_list[0]

            for step_indx in range(max_length):
                batch_len_count = 0
                batch_list_y = list()
                batch_list_ft = list()
                batch_list_time_indices = list()
                batch_list_gf = list()
                batch_list_file_name = list()

                conv_indx_list = list()

                batch_list_x = {}
                bool_list = {}

                for modality in self.active_modalities:
                    batch_list_x[modality] = []
                    bool_list[modality] = []

                data_temp_x, data_temp_x_i, datapoint = {}, {}, {}

                for conv_indx, conv_length, filename in zip(cindx_list, length_list, cfile_list):
                    if conv_length > step_indx:
                        for modality in self.active_modalities:
                            if not (self.is_irregular[modality]):
                                data_temp_x[modality] = np.empty([2 * self.num_feat_per_person[modality], self.seq_length, self.time_step_size[modality]], dtype=np.float32)
                                data_temp_x[modality][0:self.num_feat_per_person[modality], :, :] = data_g_list_np[conv_indx][modality][:, (step_indx) * self.seq_length:(step_indx + 1) * self.seq_length, :]
                                data_temp_x[modality][self.num_feat_per_person[modality]:, :, :] = data_f_list_np[conv_indx][modality][:, (step_indx) * self.seq_length:(step_indx + 1) * self.seq_length, :]
                            else:
                                data_temp_x[modality] = np.zeros([2 * self.num_feat_per_person[modality], self.seq_length, self.time_step_size[modality]], dtype=np.float32)
                                data_temp_x_i[modality] = np.zeros([self.seq_length, self.time_step_size[modality]], dtype=np.float32)
                                data_temp_x_i[modality][:self.seq_length, :self.time_step_size[modality]] = \
                                    data_g_np_bools_list[conv_indx][modality][(step_indx) * self.seq_length:(step_indx + 1) * self.seq_length, :self.time_step_size[modality]]
                                data_temp_x[modality][0:self.num_feat_per_person[modality], :, :self.time_step_size[modality]] = \
                                    data_g_list_np[conv_indx][modality][:, (step_indx) * self.seq_length:(step_indx + 1) * self.seq_length]
                                data_temp_x[modality][self.num_feat_per_person[modality]:, :, :self.time_step_size[modality]] = \
                                    data_f_list_np[conv_indx][modality][:, (step_indx) * self.seq_length:(step_indx + 1) * self.seq_length]
                                bool_list[modality].append(data_temp_x_i[modality])

                            batch_list_x[modality].append(data_temp_x[modality])

                        batch_list_y.append(predict_g_list[conv_indx][step_indx * self.seq_length:(step_indx + 1) * self.seq_length])
                        batch_list_ft.append(data_g_list_ft[conv_indx][step_indx * self.seq_length:(step_indx + 1) * self.seq_length])
                        batch_list_time_indices.append(np.array([step_indx * self.seq_length, (step_indx + 1) * self.seq_length]))
                        batch_list_gf.append(data_select_dict[data_select][1])
                        batch_list_file_name.append(filename)
                        conv_indx_list.append(conv_indx)
                        batch_len_count += 1
                        for modality in self.active_modalities:
                            if not (self.is_irregular[modality]):
                                data_temp_x[modality] = np.empty([2 * self.num_feat_per_person[modality], self.seq_length, self.time_step_size[modality]], dtype=np.float32)
                                data_temp_x[modality][0:self.num_feat_per_person[modality], :, :] = data_f_list_np[conv_indx][modality][:, (step_indx) * self.seq_length:(step_indx + 1) * self.seq_length, :]
                                data_temp_x[modality][self.num_feat_per_person[modality]:, :, :] = data_g_list_np[conv_indx][modality][:, (step_indx) * self.seq_length:(step_indx + 1) * self.seq_length, :]
                            else:
                                data_temp_x[modality] = np.zeros([2 * self.num_feat_per_person[modality], self.seq_length, self.time_step_size[modality]], dtype=np.float32)
                                data_temp_x_i[modality] = np.zeros([self.seq_length, self.time_step_size[modality]], dtype=np.float32)
                                data_temp_x_i[modality][:self.seq_length, :self.time_step_size[modality]] = \
                                    data_f_np_bools_list[conv_indx][modality][(step_indx) * self.seq_length:(step_indx + 1) * self.seq_length, :self.time_step_size[modality]]
                                data_temp_x[modality][0:self.num_feat_per_person[modality], :, :self.time_step_size[modality]] = \
                                    data_f_list_np[conv_indx][modality][:, (step_indx) * self.seq_length:(step_indx + 1) * self.seq_length]
                                data_temp_x[modality][self.num_feat_per_person[modality]:, :, :self.time_step_size[modality]] = \
                                    data_g_list_np[conv_indx][modality][:, (step_indx) * self.seq_length:(step_indx + 1) * self.seq_length]
                                bool_list[modality].append(data_temp_x_i[modality])

                            batch_list_x[modality].append(data_temp_x[modality])

                        batch_list_y.append(predict_f_list[conv_indx][step_indx * self.seq_length:(step_indx + 1) * self.seq_length])
                        batch_list_ft.append(data_f_list_ft[conv_indx][step_indx * self.seq_length:(step_indx + 1) * self.seq_length])
                        batch_list_time_indices.append(np.array([step_indx * self.seq_length, (step_indx + 1) * self.seq_length]))
                        batch_list_gf.append(data_select_dict[data_select][0])
                        batch_list_file_name.append(filename)
                        conv_indx_list.append(conv_indx)
                        #                        batch_list_feature_names.append(data_g_feature_names+data_f_feature_names)
                        batch_len_count += 1

                datapoint['info'] = {
                    'batch_size': batch_len_count,
                    'g_f': batch_list_gf,
                    'file_names': batch_list_file_name,
                    'time_indices': batch_list_time_indices,
                    'time_frames': batch_list_ft,
                    'batch_num': step_indx,
                    'conv_indices': conv_indx_list,
                }
                # !!! dimensions: datapoint['x'][modality][batch,feats,seq,times]
                datapoint['x'] = {}
                datapoint['time_bools'] = {}
                for modality in self.active_modalities:
                    datapoint['x'][modality] = np.array(batch_list_x[modality])
                    datapoint['time_bools'][modality] = np.array(bool_list[modality])
                datapoint['y'] = np.array(batch_list_y)
                self.dataset.append(datapoint)
                self.len += 1

            # pickle.dump(test_set,open('test_set.p','wb'))
            print('done loading testing data')
        else:
            raise ValueError('error loading data set. need to specify "train" or "test"')
        for mod in ['acous', 'visual']:
            for embed_indx in range(len(self.embedding_info[mod])):
                self.embedding_info[mod][embed_indx]['emb_indices'].append((self.embedding_info[mod][embed_indx]['emb_indices'][0][0] \
                                                                            + self.num_feat_per_person[mod], self.embedding_info[mod][embed_indx]['emb_indices'][0][1] + self.num_feat_per_person[mod]))

    def get_results_lengths(self):
        return self.results_lengths

    def get_feature_size_dict(self):
        return self.num_feat_for_lstm

    def get_embedding_info(self):
        return self.embedding_info

    def get_lstm_settings_dict(self, input_settings_dict):
        self.lstm_settings_dict = input_settings_dict
        for modality in self.active_modalities:
            self.lstm_settings_dict['active_modalities'] = self.active_modalities
            self.lstm_settings_dict['uses_master_time_rate'][modality] = self.uses_master_time_rate_bool[modality]
            self.lstm_settings_dict['time_step_size'][modality] = self.time_step_size[modality]
            self.lstm_settings_dict['is_irregular'][modality] = self.is_irregular[modality]
        return self.lstm_settings_dict

    def get_master_time_rates(self):
        return self.uses_master_time_rate_bool

    def get_lstm_cell_usage(self):
        return self.is_irregular

    def get_active_modalities(self):
        return self.active_modalities

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        output_list = []
        for mod in ['acous', 'visual']:
            if mod in self.active_modalities:

                if self.is_irregular[mod]:
                    output_list.append(torch.squeeze(torch.FloatTensor(self.dataset[idx]['x'][mod])))

                elif not (self.uses_master_time_rate_bool[mod]) and not (self.is_irregular[mod]):
                    in_shape = self.dataset[idx]['x'][mod].shape
                    out_shape = in_shape[:-2] + tuple([in_shape[-1] * in_shape[-2]])
                    #                    output_list.append(torch.FloatTensor(self.dataset[idx]['x'][mod]).view([-1,in_shape[-1]*in_shape[-2]]))
                    output_list.append(torch.FloatTensor(self.dataset[idx]['x'][mod]).view(out_shape))
                #                    if self.set_type == 'test':
                #                        print('debug me')
                else:
                    output_list.append(torch.squeeze(torch.FloatTensor(self.dataset[idx]['x'][mod])))
            else:
                output_list.append([])

            if self.is_irregular[mod]:  # check this
                output_list.append(torch.FloatTensor(self.dataset[idx]['time_bools'][mod]).transpose(-2, -1))
            else:
                output_list.append([])

        return output_list[0], output_list[1], output_list[2], output_list[3], torch.FloatTensor(self.dataset[idx]['y']).transpose(-2, -1), self.dataset[idx]['info']
