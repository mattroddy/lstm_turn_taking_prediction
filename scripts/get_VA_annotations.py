import xml.etree.ElementTree
import os
import numpy as np
import pandas as pd
import time as t

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
t_1 = t.time()
path_to_features='./data/signals/gemaps_features_50ms/'
path_to_annotations='./data/maptaskv2-1/Data/timed-units/'
path_to_extracted_annotations='./data/extracted_annotations/voice_activity/'
files_feature_list = os.listdir(path_to_features)
files_annotation_list = list()
files_output_list = list()
for file in files_feature_list:
    base_name = os.path.basename(file)
    files_annotation_list.append(os.path.splitext(base_name)[0]+'.timed-units.xml')
#    files_output_list.append(os.path.splitext(base_name)[0]+'.voice_activity.csv')
    files_output_list.append(os.path.splitext(base_name)[0]+'.csv')

#for i in range(0,len(files_feature_list[1])):
for i in range(0,len(files_feature_list)):
    frame_times=np.array(pd.read_csv(path_to_features+files_feature_list[i],delimiter=',',usecols = [1])['frameTime'])
    voice_activity = np.zeros((len(frame_times),))
#    voice_activity[:] = np.nan    
    e = xml.etree.ElementTree.parse(path_to_annotations+files_annotation_list[i]).getroot()
#    for atype in e.findall('tu'):
#        start_indx =find_nearest(frame_times,float(atype.get('start')))
#        end_indx = find_nearest(frame_times,float(atype.get('end')))
#        voice_activity[start_indx:end_indx+1]=1
    annotation_data = []
    for atype in e.findall('tu'):
        annotation_data.append((float(atype.get('start')), float(atype.get('end'))))
        
#    indx = 1
#    merge_count = 0
#    while indx < len(annotation_data):
#        if annotation_data[indx][0] - annotation_data[indx-1][1] < 0.180:
#            annotation_data[indx-1] = (annotation_data[indx-1][0],annotation_data[indx][1])
#            annotation_data.pop(indx)
#            merge_count += 1
#        else:
#            indx += 1
    
    # Then remove any detections less than 90ms as per ref above
    indx = 1
    less_than_25 = 0
    while indx < len(annotation_data):
        if annotation_data[indx][1]-annotation_data[indx][0] < 0.025:
            annotation_data.pop(indx)
            less_than_25 += 1
        else:
            indx += 1
    
    # find frames that contain voice activity for at least 50% of their duration (25ms)
    for strt_f,end_f in annotation_data:
        start_indx = find_nearest(frame_times,strt_f)
        end_indx = find_nearest(frame_times,end_f) - 1
        voice_activity[start_indx:end_indx+1]=1    
    
    output = pd.DataFrame([frame_times,voice_activity])
    output=np.transpose(output)
    output.columns = ['frameTimes','val']
    # uncomment this!!
    output.to_csv(path_to_extracted_annotations+files_output_list[i], float_format = '%.6f', sep=',', index=False,header=True)
        
print('total_time: '+str(t.time()-t_1))
#print('merge count: '+str(merge_count))
print('less than 25 count:'+ str(less_than_25))