# -*- coding: utf-8 -*-
import torch
import pickle
from torch.autograd import Variable
# from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# import torch.multiprocessing as mp
import numpy as np
# import asynch_lstm


use_cuda = torch.cuda.is_available()

if use_cuda:
    #    torch.cuda.device(randint(0,1))
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
    p_memory = True
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor
    p_memory = True

# # Glove embeddings
# glove_embed_table =


# %% LSTM Class

# lstm axes: [sequence,minibatch,features]
class LSTMPredictor(nn.Module):

    def __init__(self, lstm_settings_dict, feature_size_dict={'acous': 0, 'visual': 0},
                 batch_size=32, seq_length=200, prediction_length=60, embedding_info=[]):
        super(LSTMPredictor, self).__init__()

        # General model settings
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.feature_size_dict = feature_size_dict
        self.prediction_length = prediction_length

        # lstm_settings_dict
        self.lstm_settings_dict = lstm_settings_dict
        self.feature_size_dict['master'] = 0
        if self.lstm_settings_dict['no_subnets']:
            for act_mod in self.lstm_settings_dict['active_modalities']:
                self.feature_size_dict['master'] += self.feature_size_dict[act_mod]
        else:
            for act_mod in self.lstm_settings_dict['active_modalities']:
                self.feature_size_dict['master'] += self.lstm_settings_dict['hidden_dims'][act_mod]
        self.num_layers = lstm_settings_dict['layers']

        # embedding settings
        self.embedding_info = embedding_info
        self.embeddings = {'acous': [], 'visual': []}
        self.embedding_indices = {'acous': [], 'visual': []}
        self.embed_delete_index_list = {'acous': [], 'visual': []}
        self.embed_data_types = {'acous': [], 'visual': []}
        self.len_output_of_embeddings = {'acous': 0, 'visual': 0}
        self.embedding_flags = {}

        for modality in self.embedding_info.keys():
            self.embedding_flags[modality] = bool(len(self.embedding_info[modality]))
            if self.embedding_flags[modality]:
                for embedding in self.embedding_info[modality]:
                    self.len_output_of_embeddings[modality] += 2 * embedding['embedding_out_dim']
                for emb_func_indx in range(len(self.embedding_info[modality])):
                    if self.embedding_info[modality][emb_func_indx]['embedding_use_func']:
                        self.embeddings[modality].append( nn.Embedding(
                            self.embedding_info[modality][emb_func_indx]['embedding_num'],
                            self.embedding_info[modality][emb_func_indx]['embedding_out_dim']
                            ).type(dtype))
                        self.embedding_func = self.embeddings[modality][-1]
                        self.embed_data_types[modality].append(dtype_long)
                    elif self.embedding_info[modality][emb_func_indx]['use_glove']:
                        embed_tab_path = self.embedding_info[modality][emb_func_indx]['glove_embed_table']
                        glove_embed_table = pickle.load(open(embed_tab_path, 'rb'))
                        glove_embed_table[0] = np.random.normal(0,1e5,300) # need this to deal with BCE error
                        self.embeddings[modality].append( nn.Embedding.from_pretrained(
                            torch.FloatTensor(glove_embed_table).type(dtype),
                            freeze=self.lstm_settings_dict['freeze_glove']))
                        self.embedding_func = self.embeddings[modality][-1]
                        self.embed_data_types[modality].append(dtype_long)
                        print('using glove embeddings')
                    else:
                        self.embeddings[modality].append(
                            nn.Linear(self.embedding_info[modality][emb_func_indx]['embedding_num'],
                                      self.embedding_info[modality][emb_func_indx]['embedding_out_dim'],
                                      bias=True).type(dtype))
                        self.embedding_linear = self.embeddings[modality][-1]
                        self.embed_data_types[modality].append(dtype)
                    self.embedding_indices[modality].append(
                        self.embedding_info[modality][emb_func_indx]['emb_indices'])  # two tuples for start and end
                for emb_func_indx in range(len(self.embedding_info[modality])):
                    self.embed_delete_index_list[modality] += list(
                        range(self.embedding_indices[modality][emb_func_indx][0][0],
                              self.embedding_indices[modality][emb_func_indx][0][1]))
                    self.embed_delete_index_list[modality] += list(
                        range(self.embedding_indices[modality][emb_func_indx][1][0],
                              self.embedding_indices[modality][emb_func_indx][1][1]))

        # Initialize LSTMs
        self.lstm_dict= {}
        if self.lstm_settings_dict['no_subnets']:
            if not (len(self.lstm_settings_dict['active_modalities']) == 1):
                raise ValueError('Can only have one modality if no subnets')
            else:
                self.lstm_settings_dict['is_irregular']['master'] = self.lstm_settings_dict['is_irregular'][
                    self.lstm_settings_dict['active_modalities'][0]]
                if self.lstm_settings_dict['is_irregular']['master']:
                    # self.lstm_dict['master'] = nn.LSTMCell(self.feature_size_dict['master'],
                    #                                        self.lstm_settings_dict['hidden_dims']['master']).type(dtype)
                    self.lstm_dict['master'] = nn.LSTMCell(self.feature_size_dict['master'],
                                                           self.lstm_settings_dict['hidden_dims']['master']).type(dtype)
                    self.lstm_master = self.lstm_dict['master']
                else:
                    self.lstm_dict['master'] = nn.LSTM(self.feature_size_dict['master'],
                                                       self.lstm_settings_dict['hidden_dims']['master']).type(dtype)
                    self.lstm_master = self.lstm_dict['master']
        else:
            self.lstm_settings_dict['is_irregular']['master'] = False
            self.lstm_dict['master'] = nn.LSTM(self.feature_size_dict['master'],
                                               self.lstm_settings_dict['hidden_dims']['master']).type(dtype)
            self.lstm_master = self.lstm_dict['master']
            for lstm in self.lstm_settings_dict['active_modalities']:
                if self.lstm_settings_dict['is_irregular'][lstm]:
                    # self.lstm_dict[lstm] = nn.LSTMCell(self.feature_size_dict[lstm],
                    #                                    self.lstm_settings_dict['hidden_dims'][lstm]).type(dtype)
                    self.lstm_dict[lstm] = nn.LSTMCell(self.feature_size_dict[lstm],
                                                       self.lstm_settings_dict['hidden_dims'][lstm]).type(dtype)
                    if lstm == 'acous':
                        self.lstm_cell_acous = self.lstm_dict[lstm]
                    else:
                        self.lstm_cell_visual = self.lstm_dict[lstm]
                else:
                    self.lstm_dict[lstm] = nn.LSTM(self.feature_size_dict[lstm],
                                                   self.lstm_settings_dict['hidden_dims'][lstm]).type(dtype)
                    if lstm == 'acous':
                        self.lstm_acous = self.lstm_dict[lstm]
                    else:
                        self.lstm_visual = self.lstm_dict[lstm]

        # init dropout layers
        self.dropout_dict = {}
        for drop_key,drop_val in self.lstm_settings_dict['dropout'].items():
            self.dropout_dict[drop_key] = nn.Dropout(drop_val)
            setattr(self,'dropout_'+str(drop_key),self.dropout_dict[drop_key])

        self.out = nn.Linear(self.lstm_settings_dict['hidden_dims']['master'], prediction_length).type(dtype)
        self.init_hidden()

    def init_hidden(self):
        self.hidden_dict = {}
        for lstm in self.lstm_dict.keys():
            if self.lstm_settings_dict['is_irregular'][lstm]:
                self.hidden_dict[lstm] = (
                Variable(torch.zeros(self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(dtype),
                Variable(torch.zeros(self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(dtype))
            else:
                self.hidden_dict[lstm] = (Variable(
                    torch.zeros(self.num_layers, self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(
                    dtype), Variable(torch.zeros(
                        self.num_layers, self.batch_size,self.lstm_settings_dict['hidden_dims'][lstm])).type(dtype))

    def change_batch_size_reset_states(self, batch_size):
        self.batch_size = int(batch_size)
        for lstm in self.lstm_dict.keys():
            if self.lstm_settings_dict['is_irregular'][lstm]:
                self.hidden_dict[lstm] = (
                Variable(torch.zeros(self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(dtype),
                Variable(torch.zeros(self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(dtype))
            else:
                self.hidden_dict[lstm] = (Variable(
                    torch.zeros(self.num_layers, self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(
                    dtype), Variable(torch.zeros(
                        self.num_layers, self.batch_size, self.lstm_settings_dict['hidden_dims'][lstm])).type(dtype))

    def change_batch_size_no_reset(self, new_batch_size):
        for lstm in self.lstm_dict.keys():
            if self.lstm_settings_dict['is_irregular'][lstm]:
                self.hidden_dict[lstm] = (
                Variable(self.hidden_dict[lstm][0][:new_batch_size, :].data.contiguous().type(dtype)),
                Variable(self.hidden_dict[lstm][1][:new_batch_size, :].data.contiguous().type(dtype)))
            else:
                self.hidden_dict[lstm] = (
                Variable(self.hidden_dict[lstm][0][:, :new_batch_size, :].data.contiguous().type(dtype)),
                Variable(self.hidden_dict[lstm][1][:, :new_batch_size, :].data.contiguous().type(dtype)))
        self.batch_size = new_batch_size

    def weights_init(self, init_std):
        # init bias to zero recommended in http://proceedings.mlr.press/v37/jozefowicz15.pdf
        nn.init.normal(self.out.weight.data, 0, init_std)
        nn.init.constant(self.out.bias, 0)

        for lstm in self.lstm_dict.keys():
            if self.lstm_settings_dict['is_irregular'][lstm]:
                nn.init.normal(self.lstm_dict[lstm].weight_hh, 0, init_std)
                nn.init.normal(self.lstm_dict[lstm].weight_ih, 0, init_std)
                nn.init.constant(self.lstm_dict[lstm].bias_hh, 0)
                nn.init.constant(self.lstm_dict[lstm].bias_ih, 0)

            else:
                nn.init.normal(self.lstm_dict[lstm].weight_hh_l0, 0, init_std)
                nn.init.normal(self.lstm_dict[lstm].weight_ih_l0, 0, init_std)
                nn.init.constant(self.lstm_dict[lstm].bias_hh_l0, 0)
                nn.init.constant(self.lstm_dict[lstm].bias_ih_l0, 0)

    def embedding_helper(self, in_data, modality):
        embeds_one = []
        embeds_two = []

        for emb_func_indx in range(len(self.embeddings[modality])):
            embeds_one_tmp = self.embeddings[modality][emb_func_indx](
                Variable(in_data[:, :, self.embedding_indices[modality][emb_func_indx][0][0]:
                                       self.embedding_indices[modality][emb_func_indx][0][1]] \
                         .data.type(self.embed_data_types[modality][emb_func_indx]).squeeze()))

            embeds_two_tmp = self.embeddings[modality][emb_func_indx](
                Variable(in_data[:, :, self.embedding_indices[modality][emb_func_indx][1][0]:
                                       self.embedding_indices[modality][emb_func_indx][1][1]] \
                         .data.type(self.embed_data_types[modality][emb_func_indx]).squeeze()))

            if not (self.lstm_settings_dict['uses_master_time_rate'][modality]) and self.lstm_settings_dict['is_irregular'][modality]:
                embeds_one_tmp = embeds_one_tmp.transpose(2,3)
                embeds_two_tmp = embeds_two_tmp.transpose(2,3)

            embeds_one.append(embeds_one_tmp)
            embeds_two.append(embeds_two_tmp)

        non_embeddings = list(set(list(range(in_data.shape[2]))).difference(
            set(self.embed_delete_index_list[modality])))  # !!! is shape[2] correct?
        
        if len(non_embeddings) != 0:
            in_data = in_data[:, :, non_embeddings]
            for emb_one, emb_two in zip(embeds_one, embeds_two):
                in_data = torch.cat((in_data, emb_one), 2)
                in_data = torch.cat((in_data, emb_two), 2)

        else:
            for emb_one, emb_two in zip(embeds_one, embeds_two):
                in_data = torch.cat((in_data, emb_one), 2)
                in_data = torch.cat((in_data, emb_two), 2)
            embed_keep = list(set(list(range(in_data.shape[2]))).difference(set(self.embed_delete_index_list[modality])))
            in_data = in_data[:, :, embed_keep]
        return in_data

    def forward(self, in_data):
        x, i, h = {}, {}, {}
        h_list = []
        x['acous'], i['acous'], x['visual'], i['visual'] = in_data

        if not (self.lstm_settings_dict['no_subnets']):
            for mod in self.lstm_settings_dict['active_modalities']:
                if self.embedding_flags[mod]:
                    x[mod] = self.embedding_helper(x[mod], mod)

                # Apply dropout input layers
                x[mod] = self.dropout_dict[mod + '_in'](x[mod])

                cell_out_list = []
                if not(self.lstm_settings_dict['is_irregular'][mod]) and self.lstm_settings_dict['uses_master_time_rate'][mod]:
                    h[mod], self.hidden_dict[mod] = self.lstm_dict[mod](x[mod], self.hidden_dict[mod])

                elif not (self.lstm_settings_dict['is_irregular'][mod]) and not(self.lstm_settings_dict['uses_master_time_rate'][mod]):
                    h_acous_temp, self.hidden_dict[mod] = self.lstm_dict[mod](x[mod], self.hidden_dict[mod])
                    # h[mod] = h_acous_temp[0::self.lstm_settings_dict['time_step_size'][mod]]
                    h[mod] = h_acous_temp[self.lstm_settings_dict['time_step_size'][mod]-1::self.lstm_settings_dict['time_step_size'][mod]]
                elif self.lstm_settings_dict['is_irregular'][mod] and self.lstm_settings_dict['uses_master_time_rate'][mod]:
                    for seq_indx in range(self.seq_length):
                        changed_indices = np.where(i[mod][seq_indx])[0].tolist()
                        if len(changed_indices) > 0:
                            h_l, c_l = self.lstm_dict[mod](x[mod][seq_indx][changed_indices], (self.hidden_dict[mod][0][changed_indices], self.hidden_dict[mod][1][changed_indices]))
                            h_l_copy = self.hidden_dict[mod][0].clone()
                            c_l_copy = self.hidden_dict[mod][1].clone()
                            h_l_copy[changed_indices] = h_l
                            c_l_copy[changed_indices] = c_l
                            self.hidden_dict[mod] = (h_l_copy,c_l_copy)
                            # self.hidden_dict[mod][0][changed_indices] = h_l
                            # self.hidden_dict[mod][1][changed_indices] = c_l
                        cell_out_list.append(self.hidden_dict[mod][0])
                    h[mod] = torch.stack(cell_out_list)

                elif bool(self.lstm_settings_dict['is_irregular'][mod]) and not (
                self.lstm_settings_dict['uses_master_time_rate'][mod]):  # for ling and visual data
                    for seq_indx in range(self.seq_length):
                        for step_indx in range(x[mod].shape[-1]):
                            changed_indices = np.where(i[mod][seq_indx, :, step_indx])[0].tolist()
                            if len(changed_indices) > 0:
                                h_l, c_l = self.lstm_dict[mod](x[mod][seq_indx][:][changed_indices][:, :, step_indx], (self.hidden_dict[mod][0][changed_indices], self.hidden_dict[mod][1][changed_indices]))
                                h_l_copy = self.hidden_dict[mod][0].clone()
                                c_l_copy = self.hidden_dict[mod][1].clone()
                                h_l_copy[changed_indices] = h_l
                                c_l_copy[changed_indices] = c_l
                                self.hidden_dict[mod] = (h_l_copy, c_l_copy)
                                # self.hidden_dict[mod][0][changed_indices] = h_l
                                # self.hidden_dict[mod][1][changed_indices] = c_l
                        cell_out_list.append(self.hidden_dict[mod][0])
                    h[mod] = torch.stack(cell_out_list)
                else:
                    raise ValueError('problem in forward pass')

                # apply dropout
                h[mod] = self.dropout_dict[str(mod)+'_out'](h[mod])

                h_list.append(h[mod])
            lstm_out, self.hidden_dict['master'] = self.lstm_dict['master'](torch.cat(h_list, 2),self.hidden_dict['master'])
            lstm_out = self.dropout_dict['master_out'](lstm_out)

        else:  # For no subnets...

            if not (len(self.lstm_settings_dict['active_modalities']) == 1):
                raise ValueError('need to have only one modality when there are no subnets')

            mod = self.lstm_settings_dict['active_modalities'][0]
            if self.embedding_flags[mod]:
                x[mod] = self.embedding_helper(x[mod], mod)

            x[mod] = self.dropout_dict['master_in'](x[mod])

            cell_out_list = []
            # get outputs of lstm['acous']
            if not (self.lstm_settings_dict['is_irregular'][mod]) and \
                    self.lstm_settings_dict['uses_master_time_rate'][mod]:
                lstm_out, self.hidden_dict['master'] = self.lstm_dict['master'](x[mod], self.hidden_dict['master'])

            elif not (self.lstm_settings_dict['is_irregular'][mod]) and not (self.lstm_settings_dict['uses_master_time_rate'][mod]):
                h_acous_temp, self.hidden_dict['master'] = self.lstm_dict['master'](x[mod],self.hidden_dict['master'])
                # lstm_out = h_acous_temp[0::self.lstm_settings_dict['time_step_size'][mod]] # <- example of bad index
                lstm_out = h_acous_temp[self.lstm_settings_dict['time_step_size'][mod]-1::self.lstm_settings_dict['time_step_size'][mod]] # <-correct indexing

            elif self.lstm_settings_dict['is_irregular'][mod] and self.lstm_settings_dict['uses_master_time_rate'][mod]:
                for seq_indx in range(self.seq_length):
                    changed_indices = np.where(i[mod][seq_indx])[0].tolist()
                    if len(changed_indices) > 0:
                        h_l, c_l = self.lstm_dict['master'](
                                                            x[mod][seq_indx][changed_indices],
                                                            (
                                                                self.hidden_dict['master'][0][changed_indices],
                                                                self.hidden_dict['master'][1][changed_indices]
                                                            )
                                                            )
                        h_l_copy = self.hidden_dict['master'][0].clone()
                        c_l_copy = self.hidden_dict['master'][1].clone()
                        h_l_copy[changed_indices] = h_l
                        c_l_copy[changed_indices] = c_l
                        self.hidden_dict['master'] = (h_l_copy,c_l_copy)
                        # self.hidden_dict['master'][0][changed_indices] = h_l
                        # self.hidden_dict['master'][1][changed_indices] = c_l
                    cell_out_list.append(self.hidden_dict['master'][0])
                lstm_out = torch.stack(cell_out_list)

            elif bool(self.lstm_settings_dict['is_irregular'][mod]) and not (self.lstm_settings_dict['uses_master_time_rate'][mod]):  # for ling and visual data
                for seq_indx in range(self.seq_length):
                    for step_indx in range(x[mod].shape[-1]):
                        changed_indices = np.where(i[mod][seq_indx])[0].tolist()
                        if len(changed_indices) > 0:
                            h_l, c_l = self.lstm_dict['master'](
                                                                x[mod][seq_indx][:][changed_indices][:, :, step_indx],
                                                                (
                                                                    self.hidden_dict['master'][0][changed_indices],
                                                                    self.hidden_dict['master'][1][changed_indices]
                                                                )
                                                                )
                            h_l_copy = self.hidden_dict['master'][0].clone()
                            c_l_copy = self.hidden_dict['master'][1].clone()
                            h_l_copy[changed_indices] = h_l
                            c_l_copy[changed_indices] = c_l
                            self.hidden_dict['master'] = (h_l_copy,c_l_copy)
                            # self.hidden_dict['master'][0][changed_indices] = h_l
                            # self.hidden_dict['master'][1][changed_indices] = c_l
                    cell_out_list.append(self.hidden_dict['master'][0])
                lstm_out = torch.stack(cell_out_list)
            else:
                raise ValueError('problem in forward pass')

            lstm_out = self.dropout_dict[str(mod)+'_out'](lstm_out)

        # sigmoid_out = F.sigmoid(self.out(lstm_out))
        sigmoid_out = self.out(lstm_out)

        return sigmoid_out
