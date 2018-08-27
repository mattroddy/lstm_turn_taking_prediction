import xml.etree.ElementTree
import os
import numpy as np
import pandas as pd
import time as t
import pickle
import sys
import nltk

# select settings for 50ms (0) or 10ms (1) features
# takes 1.5mins for 50ms, 3 mins for 10ms setting
if len(sys.argv)==2:
    speed_setting = int(sys.argv[1])
else:
    speed_setting = 0 # 0 for 50ms, 1 for 10ms

if speed_setting == 0:
    path_to_features = './data/signals/gemaps_features_processed_50ms/znormalized/'
    path_to_orig_embeds = './data/extracted_annotations/words_advanced_50ms_raw/'
    path_to_extracted_annotations = './data/extracted_annotations/words_advanced_50ms_averaged/'
    output_set_dict = './data/extracted_annotations/set_dict_50ms.p'

elif speed_setting == 1:
    path_to_features = './data/signals/gemaps_features_processed_10ms/znormalized/'
    path_to_orig_embeds = './data/extracted_annotations/words_advanced_10ms_raw/'
    path_to_extracted_annotations = './data/extracted_annotations/words_advanced_10ms_averaged/'
    output_set_dict = './data/extracted_annotations/set_dict_10ms.p'


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
t_1 = t.time()

if not(os.path.exists(path_to_extracted_annotations)):
    os.mkdir(path_to_extracted_annotations)

files_feature_list = os.listdir(path_to_orig_embeds)
files_annotation_list = list()
files_output_list = list()
for file in files_feature_list:
    base_name = os.path.basename(file)
    files_annotation_list.append(os.path.splitext(base_name)[0]+'.timed-units.xml')
    files_output_list.append(os.path.splitext(base_name)[0]+'.csv')

word_to_ix = pickle.load(open('./data/extracted_annotations/word_to_ix.p','rb'))
# glove_embed_table = pickle.load(open('./extracted_annotations/glove_embed_table.p','rb'))

#%% Create delayed frame annotations
max_len = 0
total_list = []
for i in range(0,len(files_feature_list)):

    print('percent done files create:'+str(i/len(files_feature_list))[0:4])
    orig_file = pd.read_csv(path_to_orig_embeds+files_feature_list[i],delimiter=',')
    combins = np.array(orig_file[orig_file.columns[1:]])[list(set(np.nonzero(np.array(orig_file[orig_file.columns[1:]]))[0]))]
    local_set = [frozenset(i) for i in combins]
    total_list.extend(local_set)

total_set = set(total_list)

#%% create new averaged glove embedding dict (can maybe try different approaches apart from averaging in future)
set_dict , glove_embed_dict_50ms = {},{}
for indx, glove_combination in enumerate(total_set):
    # combin_list = []
    # for val in list(glove_combination):
    #     if not(val == 0):
    #         combin_list.append(glove_embed_table[int(val)])
    # glove_embed_dict_50ms[indx+1] = np.mean( combin_list, axis=0 )
    set_dict[glove_combination] = indx+1

# glove_embed_dict_50ms[0] = np.zeros(glove_embed_dict_50ms[1].shape)
set_dict[frozenset([0])] = 0
# glove_embed_table_50ms = np.zeros([len(glove_embed_dict_50ms),glove_embed_dict_50ms[1].shape[0]])
# for n in range(len(glove_embed_dict_50ms)):
#     glove_embed_table_50ms[n,:] = glove_embed_dict_50ms[n]

#%% get new word_reg annotations for new embedding dict
for i in range(0,len(files_feature_list)):

    print('percent done files create:'+str(i/len(files_feature_list))[0:4])
    orig_file = pd.read_csv(path_to_orig_embeds+files_feature_list[i],delimiter=',')
    frame_times = orig_file['frameTimes']
    word_annotations = np.zeros(frame_times.shape)
    indices=list(set(np.nonzero(np.array(orig_file[orig_file.columns[1:]]))[0]))
    for indx in indices:
        word_annotations[indx] = set_dict[frozenset(np.array(orig_file[orig_file.columns[1:]])[indx])]
    output = pd.DataFrame(np.vstack([frame_times, word_annotations]).transpose())
    output.columns = ['frameTimes','word']
    output.to_csv(path_to_extracted_annotations + files_output_list[i], float_format='%.6f', sep=',', index=False,header=True)

# pickle.dump(glove_embed_table_50ms,open(output_glove_embed_table,'wb'))
pickle.dump(set_dict,open(output_set_dict,'wb'))
print('total_time: '+str(t.time()-t_1))

