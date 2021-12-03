# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 19:16:03 2021

@author: User
"""
import os
from pydub import AudioSegment

path = 'C:/Users/User/Downloads/Audios/mein_Datenset/Training/Silence/'
new_path = 'C:/Users/User/Downloads/Audios/mein_Datenset/Training/Real_Silence/'
file_list = os.listdir(path)

for f in file_list:
    current_file = path + f
    song = AudioSegment.from_wav(current_file)
    song = song - 36
    song.export(new_path+f, "wav")
    