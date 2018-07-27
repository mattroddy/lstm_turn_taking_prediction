import xml.etree.ElementTree
import os
import numpy as np
import pandas as pd
import time as t
import pickle

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
t_1 = t.time()

frame_delay = 2 # word should only be output 100 ms after it is said
path_to_features='./data/features/gemaps_features_processed/znormalized_pooled/'
path_to_annotations='./data/maptaskv2-1/Data/timed-units/'
path_to_extracted_annotations='./data/extracted_annotations/words_advanced_100/'
if not(os.path.exists(path_to_extracted_annotations)):
    os.makedirs(path_to_extracted_annotations)
files_feature_list = os.listdir(path_to_features)
files_annotation_list = list()
files_output_list = list()
for file in files_feature_list:
    base_name = os.path.basename(file)
    files_annotation_list.append(os.path.splitext(base_name)[0]+'.timed-units.xml')
    files_output_list.append(os.path.splitext(base_name)[0]+'.csv')

#%% First get vocabulary
no_change = 0
 
words_from_annotations = []
for i in range(0,len(files_feature_list)):
    e = xml.etree.ElementTree.parse(path_to_annotations+files_annotation_list[i]).getroot()
    for atype in e.findall('tu'):
        words_from_annotations.append(atype.text)
vocab = set(words_from_annotations)
word_to_ix = {word: i+1 for i, word in enumerate(vocab)} # +1 is because 0 represents no change
pickle.dump(word_to_ix,open('./data/extracted_annotations/word_to_ix.p','wb'))
#%% Create delayed frame annotations
for i in range(0,len(files_feature_list)):
    frame_times=np.array(pd.read_csv(path_to_features+files_feature_list[i],delimiter=',',usecols = [0])['frame_time'])
    word_values = np.zeros((len(frame_times),))  
    e = xml.etree.ElementTree.parse(path_to_annotations+files_annotation_list[i]).getroot()
    annotation_data = []
    for atype in e.findall('tu'):
        curr_word = word_to_ix[atype.text]
        end_indx_advanced = find_nearest(frame_times,float(atype.get('end'))) + frame_delay
        if end_indx_advanced < len(word_values):
            word_values[end_indx_advanced] = curr_word
    
    output = pd.DataFrame([frame_times,word_values])
    output=np.transpose(output)
    output.columns = ['frame_time','word']
    output.to_csv(path_to_extracted_annotations+files_output_list[i], float_format = '%.6f', sep=',', index=False,header=True)
        
print('total_time: '+str(t.time()-t_1))
