# -*- coding: utf-8 -*-
import json
import subprocess
import platform
import os
import pickle
import numpy as np

gpu_select = 0
test_indices = [0, 1, 2, 3, 4]
num_workers = 1
num_epochs = 1500
lr_list = [0.001]
dropout_visual_p = 0.0
dropout_acous_p = 0.0
word_embed_out_dim = 64
seq_length = 200
experiment_top_path = './icmi_results/'

plat = platform.platform()
py_env =  '/home/matt/anaconda3/bin/python'

# Acoustic features
gemaps_features_list = ['F0semitoneFrom27.5Hz', 'jitterLocal', 'F1frequency',
                        'F1bandwidth', 'F2frequency', 'F3frequency', 'Loudness',
                        'shimmerLocaldB', 'HNRdBACF', 'alphaRatio', 'hammarbergIndex',
                        'spectralFlux', 'slope0-500', 'slope500-1500', 'F1amplitudeLogRelF0',
                        'F2amplitudeLogRelF0', 'F3amplitudeLogRelF0', 'mfcc1', 'mfcc2', 'mfcc3',
                        'mfcc4']

gemaps_50ms_dict_list = [{'folder_path': './data/features/gemaps_features_processed/znormalized',
                          'features': gemaps_features_list,
                          'modality': 'acous',
                          'is_h5_file': False,
                          'uses_master_time_rate': True,
                          #                     'uses_lstm_cell':False,
                          'time_step_size': 1,
                          'is_irregular': False,
                          'short_name': 'gmaps50'}]

# Words
word_to_ix = pickle.load(open('./data/extracted_annotations/word_to_ix.p', 'rb'))
word_embed_in_dim = len(word_to_ix) + 1
word_embed_out_dim = 64
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

# Combined linguistic features

# %% Common settings for all experiments
early_stopping = True
patience = 10
slow_test = True
train_list_path = './data/splits/training.txt'
test_list_path = './data/splits/testing.txt'

# %% Experiment settings
Acous_20Hz = {
    'l2_dict':
        {'out': 0.0001,
         'master': 0.0001,
         'acous': 0.0001,
         'visual': 0.0001},
    'hidden_nodes_master': 60,
    'hidden_nodes_acous': 0,
    'hidden_nodes_visual': 0,
    'no_subnets': True
}
Ling_20Hz = {
    'l2_dict':
        {'out': 0.0001,
         'master': 0.001,
         'acous': 0.0001,
         'visual': 0.0001},
    'hidden_nodes_master': 60,
    'hidden_nodes_acous': 0,
    'hidden_nodes_visual': 0,
    'no_subnets': True
}
Ling_asynch = {
    'l2_dict':
        {'out': 0.0001,
         'master': 0.01,
         'acous': 0.0001,
         'visual': 0.0001},
    'hidden_nodes_master': 60,
    'hidden_nodes_acous': 0,
    'hidden_nodes_visual': 0,
    'no_subnets': True
}

Acous_20Hz_Ling_20Hz = {
    'l2_dict':
        {'out': 0.0001,
         'master': 0.0001,
         'acous': 0.0001,
         'visual': 0.0001},
    'hidden_nodes_master': 60,
    'hidden_nodes_acous': 0,
    'hidden_nodes_visual': 0,
    'no_subnets': True
}

Acous_20Hz_Ling_20Hz = {
    'l2_dict':
        {'out': 0.0001,
         'master': 0.0001,
         'acous': 0.0001,
         'visual': 0.0001},
    'hidden_nodes_master': 10,
    'hidden_nodes_acous': 20,
    'hidden_nodes_visual': 30,
    'no_subnets': False
}

Acous_20Hz_Ling_asynch = {
    'l2_dict':
        {'out': 0.0001,
         'master': 0.0001,
         'acous': 0.0001,
         'visual': 0.01},
    'hidden_nodes_master': 10,
    'hidden_nodes_acous': 20,
    'hidden_nodes_visual': 30,
    'no_subnets': False
}

# %% Experiments list

experiment_name_list = [
    '0_Acous_20Hz',
    '1_Ling_20Hz',
    '3_Ling_asynch',
    '4_Acous_20Hz_Ling_20Hz'
    '11_Acous_20Hz_Ling_20Hz',
    '13_Acous_20Hz_Ling_asynch',
]
experiment_features_lists = [
    gemaps_50ms_dict_list,
    word_reg_dict_list,
    word_irreg_dict_list,
    gemaps_50ms_dict_list + word_reg_dict_list,
    gemaps_50ms_dict_list + word_reg_dict_list,
    gemaps_50ms_dict_list + word_irreg_dict_list,
]

experiment_settings_list = [
    Acous_20Hz,
    Ling_20Hz,
    Ling_asynch,
    Acous_20Hz_Ling_20Hz,
    Acous_20Hz_Ling_20Hz,
    Acous_20Hz_Ling_asynch,
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
    no_subnets = exp_settings['no_subnets']
    best_lr = lr_list[0]

    test_fold_list = []
    for test_indx in test_indices:
        name_append_test = str(test_indx) + '_' + experiment_name + '_' + 'master_' + str(best_master_node_size) + '_acous_' + str(best_acous_node_size) + \
                           '_visual_' + str(best_visual_node_size) + '_lr_' \
                           + str(best_lr)[2:] + '_l2m_' + str(l2_dict['master'])[2:] + '_l2a_' + str(l2_dict['acous'])[2:] + '_l2v_' + str(l2_dict['visual'])[2:] + '_l2o_' + str(l2_dict['out'])[2:] + \
                           '_drop_a_' + str(dropout_acous_p)[2:] + '_drop_v_' + str(dropout_visual_p)[2:]
        test_fold_list.append(os.path.join(test_path, name_append_test))
        if not (os.path.exists(os.path.join(test_path, name_append_test))) and not (os.path.exists(os.path.join(test_path, name_append_test, 'results.p'))):
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
                         'dropout_acous_p': dropout_acous_p,
                         'dropout_visual_p': dropout_visual_p,
                         'l2_dict': l2_dict
                         }
            json_dict = json.dumps(json_dict)
            arg_list = [json_dict]
            my_env = {'CUDA_VISIBLE_DEVICES': str(gpu_select)}
            command = [py_env, './run_json.py'] + arg_list
            # command = ['python', './run_json.py'] + arg_list
            print(command)
            print('\n *** \n')
            print(test_path + '/' + name_append_test)
            print('\n *** \n')
            response = subprocess.run(command, stderr=subprocess.PIPE, env=my_env)
            print(response.stderr)
            if not (response.returncode == 0):
                raise (ValueError('error in test subprocess: ' + name_append_test))

    best_vals_dict, best_vals_dict_array, last_vals_dict, best_fscore_array = {}, {}, {}, {}
    for eval_metric in eval_metric_list:
        best_vals_dict[eval_metric] = 0
        last_vals_dict[eval_metric] = 0
        best_vals_dict_array[eval_metric] = []
        best_fscore_array[eval_metric] = []

    for test_run_indx in test_indices:
        test_run_folder = str(test_run_indx) + '_' + experiment_name + '_' + 'master_' + str(best_master_node_size) + '_acous_' + str(best_acous_node_size) + \
                          '_visual_' + str(best_visual_node_size) + '_lr_' \
                          + str(best_lr)[2:] + '_l2m_' + str(l2_dict['master'])[2:] + '_l2a_' + str(l2_dict['acous'])[2:] + '_l2v_' + str(l2_dict['visual'])[2:] + '_l2o_' + str(l2_dict['out'])[2:] + \
                          '_drop_a_' + str(dropout_acous_p)[2:] + '_drop_v_' + str(dropout_visual_p)[2:]
        test_results = pickle.load(open(os.path.join(test_path, test_run_folder, 'results.p'), 'rb'))
        total_num_epochs = len(test_results['test_losses'])
        best_loss_indx = np.argmin(test_results['test_losses'])

        # get average and lists
        for eval_metric in eval_metric_list:
            best_vals_dict[eval_metric] += float(test_results[eval_metric][best_loss_indx]) * (1.0 / float(len(test_indices)))
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
                   'selected_master_node_size': int(best_master_node_size),
                   'selected_acous_node_size': int(best_acous_node_size),
                   'selected_visual_node_size': int(best_visual_node_size)}

    json.dump(report_dict, open(trial_path + '/report_dict.json', 'w'), indent=4, sort_keys=True)


param_list = []
for experiment_name, experiment_features_list, experiment_settings in zip(experiment_name_list, experiment_features_lists, experiment_settings_list):
    param_list.append([experiment_name, experiment_features_list, experiment_settings])

# if __name__=='__main__':
#    p = multiprocessing.Pool(num_workers)
#    p.map(run_trial,param_list)   
for params in param_list:
    run_trial(params)
