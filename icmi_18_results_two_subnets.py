# -*- coding: utf-8 -*-
import json
# from subprocess import Popen,PIPE
import subprocess
import torch.multiprocessing as multiprocessing

import sys
import pandas as pd
import time as t
import platform
import os
import pickle
import numpy as np
import feature_vars as feat_dicts
from time import gmtime, strftime
# import shutil
from random import randint

seq_length = 600
no_subnets = False

experiment_top_path = './two_subnets/'

plat = platform.linux_distribution()[0]
# plat = 'not_arch'
if plat == 'arch':
    print('platform: arch')
    py_env = '/home/matt/anaconda3/bin/python'
elif plat == 'debian':
    py_env = '../../../anaconda3/bin/python'
else:
    print('platform: ' + plat)
    py_env = '/home/mroddy/anaconda3/envs/py36/bin/python'


# %% Common settings for all experiments
num_epochs = 1500
early_stopping = True
patience = 10
slow_test = True
train_list_path = './data/splits/training.txt'
test_list_path = './data/splits/testing.txt'
# train_list_path = './data/splits/training_dev_small.txt'
# test_list_path = './data/splits/testing_dev_small.txt'

# %% Experiment settings

# note: master is the one that needs to be changed in all cases for the no_subnet experiments
Acous_50ms_Ling_50ms = {
    'lr': 0.01,
    'l2_dict':
        {'emb': 0.0001,
         'out': 0.00001,
         'master': 0.00001,
         'acous': 0.0001,
         'visual': 0.00001
         },
    'dropout_dict': {
        'master_out': 0.25,
        'master_in': 0.5,
        'acous_in': 0.25,
        'acous_out': 0.25,
        'visual_in': 0.25,
        'visual_out': 0.25
    },
    'hidden_nodes_master': 50,
    'hidden_nodes_acous': 50,
    'hidden_nodes_visual': 50
}

Acous_10ms_Ling_50ms = {
    'lr': 0.01,
    'l2_dict':
        {'emb': 0.0001,
         'out': 0.00001,
         'master': 0.00001,
         'acous': 0.0001,
         'visual': 0.00001
         },
    'dropout_dict': {
        'master_out': 0.25,
        'master_in': 0.5,
        'acous_in': 0.25,
        'acous_out': 0.25,
        'visual_in': 0.25,
        'visual_out': 0.
    },
    'hidden_nodes_master': 50,
    'hidden_nodes_acous': 50,
    'hidden_nodes_visual': 50
}

Acous_50ms_Ling_Asynch = {
    'lr': 0.01,
    'l2_dict':
        {'emb': 0.001,
         'out': 0.00001,
         'master': 0.00001,
         'acous': 0.0001,
         'visual': 0.0001
         },
    'dropout_dict': {
        'master_out': 0,
        'master_in': 0.5,
        'acous_in': 0.,
        'acous_out': 0.25,
        'visual_in': 0.25,
        'visual_out': 0.
    },
    'hidden_nodes_master': 50,
    'hidden_nodes_acous': 50,
    'hidden_nodes_visual': 50
}

Acous_10ms_Ling_Asynch = {
    'lr': 0.01,
    'l2_dict':
        {'emb': 0.001,
         'out': 0.00001,
         'master': 0.00001,
         'acous': 0.0001,
         'visual': 0.00001
         },
    'dropout_dict': {
        'master_out': 0.25,
        'master_in': 0.25,
        'acous_in': 0.25,
        'acous_out': 0,
        'visual_in': 0.25,
        'visual_out': 0
    },
    'hidden_nodes_master': 50,
    'hidden_nodes_acous': 50,
    'hidden_nodes_visual': 50
}

Acous_10ms_Ling_10ms = {
    'lr': 0.01,
    'l2_dict':
        {'emb': 0.0001,
         'out': 0.00001,
         'master': 0.00001,
         'acous': 0.0001,
         'visual': 0.00001
         },
    'dropout_dict': {
        'master_out': 0.,
        'master_in': 0.5,
        'acous_in': 0.25,
        'acous_out': 0.25,
        'visual_in': 0.25,
        'visual_out': 0.
    },
    'hidden_nodes_master': 50,
    'hidden_nodes_acous': 50,
    'hidden_nodes_visual': 50
}


# %% Experiments list

gpu_select = 0
test_indices = [0,1,2]

experiment_name_list = [
    '1_Acous_50ms_Ling_50ms',
    '2_Acous_10ms_Ling_50ms',
    '3_Acous_50ms_Ling_Asynch',
    '4_Acous_10ms_Ling_Asynch',
    '5_Acous_10ms_Ling_10ms',
]

experiment_features_lists = [
    feat_dicts.gemaps_50ms_dict_list + feat_dicts.word_reg_dict_list_visual,
    feat_dicts.gemaps_10ms_dict_list + feat_dicts.word_reg_dict_list_visual,
    feat_dicts.gemaps_50ms_dict_list + feat_dicts.word_irreg_fast_dict_list,
    feat_dicts.gemaps_10ms_dict_list + feat_dicts.word_irreg_fast_dict_list,
    feat_dicts.gemaps_10ms_dict_list + feat_dicts.word_reg_dict_list_10ms_visual

]

experiment_settings_list = [
    Acous_50ms_Ling_50ms,
    Acous_10ms_Ling_50ms,
    Acous_50ms_Ling_Asynch,
    Acous_10ms_Ling_Asynch,
    Acous_10ms_Ling_10ms
]

eval_metric_list = ['f_scores_50ms', 'f_scores_250ms', 'f_scores_500ms', 'f_scores_overlap_hold_shift',
                    'f_scores_overlap_hold_shift_exclusive', 'f_scores_short_long', 'train_losses',
                    'test_losses', 'test_losses_l1']

if not (os.path.exists(experiment_top_path)):
    os.mkdir(experiment_top_path)


def run_trial(parameters):
    experiment_name, experiment_features_list, exp_settings = parameters

    trial_path = experiment_top_path + experiment_name

    test_path = trial_path + '/test/'

    if not (os.path.exists(trial_path)):
        os.mkdir(trial_path)

    if not (os.path.exists(test_path)):
        os.mkdir(test_path)

    best_master_node_size = exp_settings['hidden_nodes_master']
    best_acous_node_size = exp_settings['hidden_nodes_acous']
    best_visual_node_size = exp_settings['hidden_nodes_visual']
    l2_dict = exp_settings['l2_dict']
    drp_dict = exp_settings['dropout_dict']
    best_lr = exp_settings['lr']
    #    best_l2 = l2_list[0]
    # Run full test
    # Run full test number_of_tests times
    test_fold_list = []
    for test_indx in test_indices:
        name_append_test = str(test_indx) + '_' + experiment_name + \
                           '_m_' + str(best_master_node_size) + \
                           '_a_' + str(best_acous_node_size) + \
                           '_v_' + str(best_visual_node_size) + \
                           '_lr_' + str(best_lr)[2:] + \
                           '_l2e_' + str(l2_dict['emb'])[2:] + \
                           '_l2o_' + str(l2_dict['out'])[2:] + \
                           '_l2m_' + str(l2_dict['master'])[2:] + \
                           '_l2a_' + str(l2_dict['acous'])[2:] + \
                           '_l2v_' + str(l2_dict['visual'])[2:] + \
                           '_dmo_' + str(drp_dict['master_out'])[2:] + \
                           '_dmi_' + str(drp_dict['master_in'])[2:] + \
                           '_dao_' + str(drp_dict['acous_out'])[2:] + \
                           '_dai_' + str(drp_dict['acous_in'])[2:] + \
                           '_dvo_' + str(drp_dict['visual_out'])[2:] + \
                           '_dvi_' + str(drp_dict['visual_in'])[2:] + \
                           '_seq_' + str(seq_length)
        test_fold_list.append(os.path.join(test_path, name_append_test))
        if not (os.path.exists(os.path.join(test_path, name_append_test))) and not (
        os.path.exists(os.path.join(test_path, name_append_test, 'results.p'))):
            json_dict = {'feature_dict_list': experiment_features_list,
                         'results_dir': test_path,
                         'name_append': name_append_test,
                         'no_subnets': no_subnets,
                         'hidden_nodes_master': best_master_node_size,
                         'hidden_nodes_acous': best_acous_node_size,
                         'hidden_nodes_visual': best_visual_node_size,
                         'learning_rate': best_lr,
                         'sequence_length': seq_length,
                         'num_epochs': num_epochs,
                         'early_stopping': early_stopping,
                         'patience': patience,
                         'slow_test': slow_test,
                         'train_list_path': train_list_path,
                         'test_list_path': test_list_path,
                         'use_date_str': False,
                         'freeze_glove_embeddings': False,
                         'grad_clip_bool': False,
                         'l2_dict': l2_dict,
                         'dropout_dict': drp_dict
                         }
            json_dict = json.dumps(json_dict)
            arg_list = [json_dict]
            my_env = {'CUDA_VISIBLE_DEVICES': str(gpu_select)}
            command = [py_env, './run_json.py'] + arg_list
            print(command)
            print('\n *** \n')
            print(test_path + name_append_test)
            print('\n *** \n')
            response = subprocess.run(command, stderr=subprocess.PIPE, env=my_env)
            print(response.stderr)
            #            sys.stderr.write(response.stderr)
            #                    sys.stdout.write(line)
            #                    sys.stdout.flush()
            if not (response.returncode == 0):
                raise (ValueError('error in test subprocess: ' + name_append_test))

    best_vals_dict, best_vals_dict_array, last_vals_dict, best_fscore_array = {}, {}, {}, {}
    for eval_metric in eval_metric_list:
        best_vals_dict[eval_metric] = 0
        last_vals_dict[eval_metric] = 0
        best_vals_dict_array[eval_metric] = []
        best_fscore_array[eval_metric] = []


    for test_run_indx in test_indices:
        test_run_folder = str(test_run_indx) + '_' + experiment_name + \
                          '_m_' + str(best_master_node_size) + \
                          '_a_' + str(best_acous_node_size) + \
                          '_v_' + str(best_visual_node_size) + \
                          '_lr_' + str(best_lr)[2:] + \
                          '_l2e_' + str(l2_dict['emb'])[2:] + \
                          '_l2o_' + str(l2_dict['out'])[2:] + \
                          '_l2m_' + str(l2_dict['master'])[2:] + \
                          '_l2a_' + str(l2_dict['acous'])[2:] + \
                          '_l2v_' + str(l2_dict['visual'])[2:] + \
                          '_dmo_' + str(drp_dict['master_out'])[2:] + \
                          '_dmi_' + str(drp_dict['master_in'])[2:] + \
                          '_dao_' + str(drp_dict['acous_out'])[2:] + \
                          '_dai_' + str(drp_dict['acous_in'])[2:] + \
                          '_dvo_' + str(drp_dict['visual_out'])[2:] + \
                          '_dvi_' + str(drp_dict['visual_in'])[2:] + \
                          '_seq_' + str(seq_length )

        test_results = pickle.load(open(os.path.join(test_path, test_run_folder, 'results.p'), 'rb'))
        total_num_epochs = len(test_results['test_losses'])
        best_loss_indx = np.argmin(test_results['test_losses'])

        # get average and lists
        for eval_metric in eval_metric_list:
            best_vals_dict[eval_metric] += float(test_results[eval_metric][best_loss_indx]) * (
                        1.0 / float(len(test_indices)))
            last_vals_dict[eval_metric] += float(test_results[eval_metric][-1]) * (1.0 / float(len(test_indices)))
            best_vals_dict_array[eval_metric].append(float(test_results[eval_metric][best_loss_indx]))
            best_fscore_array[eval_metric].append(float(np.amax(test_results[eval_metric])))

    report_dict = {'experiment_name': experiment_name,
                   'best_vals': best_vals_dict,
                   'last_vals': last_vals_dict,
                   'best_vals_array': best_vals_dict_array,
                   'best_fscore_array': best_fscore_array,
                   'best_fscore_500_average': np.mean(best_fscore_array['f_scores_500ms']),
                   'best_test_loss_average': np.mean(best_vals_dict['test_losses']),
                   'best_indx': int(best_loss_indx),
                   'num_epochs_total': int(total_num_epochs),
                   'selected_lr': best_lr,
                   'selected_master_node_size': int(best_master_node_size)
                   }

    json.dump(report_dict, open(trial_path + '/report_dict.json', 'w'), indent=4, sort_keys=True)


# create folder within loop number

# %% run multiprocessing

param_list = []
for experiment_name, experiment_features_list, experiment_settings in zip(experiment_name_list,
                                                                          experiment_features_lists,
                                                                          experiment_settings_list):
    param_list.append([experiment_name, experiment_features_list, experiment_settings])

# if __name__=='__main__':
#    p = multiprocessing.Pool(num_workers)
#    p.map(run_trial,param_list)   
for params in param_list:
    run_trial(params)


