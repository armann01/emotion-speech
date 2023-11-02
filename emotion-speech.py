import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

from itertools import cycle

import random 
from pydub import AudioSegment
from IPython.display import Audio, display

ravdess_path = "ravdess_audios"

ravdess_dir = os.listdir(ravdess_path)

gender_arr = []
path_arr = []
emotion_arr = []

emotions_convert = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

for actor in ravdess_dir:
    actor_files = os.listdir(os.path.join(ravdess_path, actor))
    for audio_file in actor_files:
        file_name = audio_file.split('.')[0]
        #print(audio_file)
        emotion_key = file_name.split('-')[2]
        gender_identifier = int(file_name.split('-')[6])
        
        path_arr.append(f"{ravdess_path}/{actor}/{audio_file}")
        
        if gender_identifier % 2 == 0:
            gender_arr.append("female")
        else:
            gender_arr.append("male")
            
        emotion_arr.append(emotions_convert[emotion_key])
        
df = pd.concat([
    pd.DataFrame(path_arr, columns = ['path']),
    pd.DataFrame(gender_arr, columns = ['gender']),
    pd.DataFrame(emotion_arr, columns = ['emotion'])
], axis = 1)


df['emotion'].unique()
noNeutral = df[df['emotion'] != 'neutral'].copy(deep = True)

def classify_emotion(row):
    file_path = row['path']
    emotion = row['emotion']
    
    # Load and play the audio
    display(Audio(file_path))
    
    # Get user input for emotion classification
    user_classification = input("Enter emotion classification for the audio: ")
    
    # Store user classification in the dataframe
    noNeutral.at[row.name, 'user_classification'] = user_classification

# Select random rows from the dataframe
random_rows = noNeutral.sample(n=100)  # You can adjust the number of samples as needed

# Add a new column for user classifications
noNeutral['user_classification'] = ''

# Iterate through the random rows, play the audio, and collect classifications

humanClassified = noNeutral[noNeutral['user_classification'] != '']
#for index, row in random_rows.iterrows():
#    classify_emotion(row)

labels = humanClassified['emotion']
humanPreds = humanClassified['user_classification']

sum(labels == humanPreds)/100