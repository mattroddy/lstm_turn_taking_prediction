import xml.etree.ElementTree
import os
import numpy as np
import pandas as pd
import time as t
import pickle
import sys
import nltk

nltk.download('punkt')

# takes 36 seconds for 50ms version, 2 mins for 10ms version

# select settings for 50ms (0) or 10ms (1) features
if len(sys.argv)==2:
    speed_setting = int(sys.argv[1])
else:
    speed_setting = 0 # 0 for 50ms, 1 for 10ms

if speed_setting == 0:
    path_to_features = './data/signals/gemaps_features_processed_50ms/znormalized/'
    path_to_extracted_annotations = './data/extracted_annotations/words_advanced_50ms_raw/'
    frame_delay = 2  # word should only be output 100 ms after it is said
    max_len_setting = 2 # using 2 for the moment for the purpose of speed
elif speed_setting ==1:
    path_to_features = './data/signals/gemaps_features_processed_10ms/znormalized/'
    path_to_extracted_annotations = './data/extracted_annotations/words_advanced_10ms_raw/'
    frame_delay = 10
    max_len_setting = 2


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
t_1 = t.time()

path_to_annotations='./data/maptaskv2-1/Data/timed-units/'
word_to_ix = pickle.load(open('./data/extracted_annotations/word_to_ix.p','rb'))

if not(os.path.exists(path_to_extracted_annotations)):
    os.mkdir(path_to_extracted_annotations)
files_feature_list = os.listdir(path_to_features)
files_annotation_list = list()
files_output_list = list()
for file in files_feature_list:
    base_name = os.path.basename(file)
    files_annotation_list.append(os.path.splitext(base_name)[0]+'.timed-units.xml')
    files_output_list.append(os.path.splitext(base_name)[0]+'.csv')


#%% Create delayed frame annotations

#delete these three
lengths_list = []
longer_than_one_list = []
added_to_end_list = 0

max_len = 0
for i in range(0,len(files_feature_list)):

    print('percent done files create:'+str(i/len(files_feature_list))[0:4])
    frame_times=np.array(pd.read_csv(path_to_features+files_feature_list[i],delimiter=',',usecols = [0])['frame_time'])
    word_values = np.zeros((len(frame_times),max_len_setting))
    check_next_word_array = np.zeros((len(frame_times),))
    e = xml.etree.ElementTree.parse(path_to_annotations+files_annotation_list[i]).getroot()
    annotation_data = []
    for atype in e.findall('tu'):
        word_frame_list = []
        target_word = atype.text
        target_word = target_word.strip()
        if '--' in target_word:
            word_frame_list =['--disfluency_token--']
        else:
            word_frame_list = nltk.word_tokenize(target_word)

        curr_words = [ word_to_ix[wrd] for wrd in word_frame_list]

        # delete this stuff
        lengths_list.append(len(curr_words))
        if len(curr_words) > 1:
            longer_than_one_list.append(curr_words)

        if len(curr_words)> max_len:
            max_len = len(curr_words)
            curr_words = curr_words[:max_len_setting]

        #%% problem here too!!!
        end_indx_advanced = find_nearest(frame_times,float(atype.get('end'))) + frame_delay
        if end_indx_advanced < len(word_values):
            # word_values[end_indx_advanced] = curr_words
            if (np.min(np.where(word_values[end_indx_advanced]==0)[0]) > 0):
                added_to_end_list += 1

            arr_strt_indx = np.min(np.where(word_values[end_indx_advanced]==0)[0])
            arr_end_indx = arr_strt_indx + len(curr_words)
            if arr_end_indx < max_len_setting:
                word_values[end_indx_advanced][arr_strt_indx:arr_end_indx] = np.array(curr_words)

    # output = pd.DataFrame([frame_times,word_values])
    output = pd.DataFrame(np.concatenate([np.expand_dims(frame_times,1),word_values],1).transpose())
    output=np.transpose(output)
    output.columns = ['frameTimes'] + [str(n) for n in range(max_len_setting)]
    output.to_csv(path_to_extracted_annotations+files_output_list[i], float_format = '%.6f', sep=',', index=False,header=True)
        
print('total_time: '+str(t.time()-t_1))
