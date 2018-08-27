# -*- coding: utf-8 -*-
# takes about 30 mins
import os
import time as t
import sys

if len(sys.argv)==2:
    speed_setting = int(sys.argv[1])
else:
    speed_setting = 0 # 0 for 50ms, 1 for 10ms

if speed_setting == 0:
    output_files_dir = './data/signals/gemaps_features_50ms/'
    smile_command = 'SMILExtract -C ./opensmile-2.3.0/config/gemaps_50ms/eGeMAPSv01a.conf'
else:
    output_files_dir = './data/signals/gemaps_features_10ms/'
    smile_command = 'SMILExtract -C ./opensmile-2.3.0/config/gemaps_10ms/eGeMAPSv01a.conf'

audio_files_dir = './data/signals/dialogues_mono/'
audio_files=os.listdir(audio_files_dir)
csv_file_list = [ file.split('.')[0]+'.'+file.split('.')[1]+'.csv' for file in audio_files]

if not(os.path.exists(output_files_dir)):
    os.mkdir(output_files_dir)

t_1=t.time()
total_num_files = len(audio_files)
file_indx = 0
for input_file,output_file in zip(audio_files,csv_file_list):
    #os.system('ls')
    file_indx +=1
    t_2 = t.time()
    print('processing file '+str(file_indx)+' out of '+str(total_num_files))
#    subprocess.check_output(smile_command + ' -I '+audio_files_dir+input_file+' -D '+output_files_dir+output_file)
    os.system(smile_command + ' -I '+audio_files_dir+input_file+' -D '+output_files_dir+output_file)
    print('time taken for file: '+str(t.time()-t_2))

print('total time taken: '+str(t.time()-t_1))
