#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:46:01 2020
SUBGCLASSIFIER_SAMPLEMP3
@author: joemccoy
"""


import os, json
import shutil, os, glob
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression



# =============================================================================
# Enter path name to YOUR desktop here OR wherever you have saved 'Sub_g_classifier' folder.
# =============================================================================

path_name_to_desktop = '/Users/joemccoy/Desktop'


# =============================================================================
# FUNCTIONs
# =============================================================================
def remove_space(spaced_folder, filetype='mp3'):
    os.chdir(spaced_folder)
    filenames=[]
    for file in os.listdir(spaced_folder):
        if file[-3:] == filetype:
            count = file.count(".") - 1
            count3 = file.count("3") - 1
            r = file.replace("[","_").replace("]","_").replace("-","_").replace(".","_", count).replace("'","_").replace("`","_").replace("(","_").replace(")","_").replace("´","_").replace("&","and").replace("0","zero").replace("1","one").replace("2","two").replace("3","three", count3).replace("4","four").replace("5","five").replace("6","six").replace("7","seven").replace("8","eight").replace("9","nine").replace("!","_").replace(",","_").replace(" ","_").replace("__","_").replace("__","_").replace("__","_").replace("__","_").replace("__","_").replace("__","_")
            if( r != file):
                os.rename(file,r)

def tag_file(folder, filename, tag):
    os.chdir(folder)
    src=filename
    dst=tag+filename
    os.rename(src,dst)

def new_dir(dirName):
    try:
        os.makedirs(dirName)    
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")  
        
def extract_features(path_to_music_folder, output_path, filetype='mp3'):
    music_extractor_dir = path_name_to_desktop+'/Sub_g_classifier_example/Feature_extractor/essentia_streaming_extractor_music'
    # load filenames from path    
    os.chdir(output_path)    
    filenames=[]
    for file in os.listdir(path_to_music_folder):
        if file[-3:] == filetype:
            filenames.append(file)
            
    for i in range(len(filenames)):
        command = music_extractor_dir+' '+path_to_music_folder+filenames[i]+' '+filenames[i][:-4]+'_all_features.json'
        os.system(command)
        # executes feature extraction through command line (terminal). features for each mp3 are stored as a json file.
    
def move_files(source, destination):
    files=glob.glob(source+'/*')
    for f in files:
        shutil.move(f, destination)

def expand_list(df, column):
    e = df[column].apply(pd.Series)
    e = e.rename(columns = lambda x : column + 'n'+str(x))
    e = pd.concat([df[:], e[:]], axis=1)
    e = e.drop([column], axis = 1)
    return e    

def mapping(df,feature):
    featureMap=dict()
    count=0
    for i in sorted(df[feature].unique(),reverse=True):
        featureMap[i]=count
        count=count+1
    df[feature]=df[feature].map(featureMap)
    return df


# =============================================================================
# remove space
# =============================================================================

#file loc strings
RAW_DSET = [path_name_to_desktop+'/Sub_g_classifier_example/Samplemp3/Acid_Techno',path_name_to_desktop+'/Sub_g_classifier_example/Samplemp3/Ghetto_House', path_name_to_desktop+'/Sub_g_classifier_example/Samplemp3/Tech_House']
length=len(RAW_DSET)


for i in range (length):
    remove_space(RAW_DSET[i], filetype='mp3')
    
# =============================================================================
# add tags
# =============================================================================

acid = [pos_mp3 for pos_mp3 in os.listdir(RAW_DSET[0]) if pos_mp3.endswith('.mp3')]
ghet = [pos_mp3 for pos_mp3 in os.listdir(RAW_DSET[1]) if pos_mp3.endswith('.mp3')]
tech = [pos_mp3 for pos_mp3 in os.listdir(RAW_DSET[2]) if pos_mp3.endswith('.mp3')]

acid_l = len(acid)
ghet_l = len(ghet)
tech_l = len(tech)

for i in range (acid_l):
    tag_file(RAW_DSET[0], acid[i], "AT_")
for i in range (ghet_l):
    tag_file(RAW_DSET[1], ghet[i], "GH_")    
for i in range (tech_l):
    tag_file(RAW_DSET[2], tech[i], "TH_")
    
    
# =============================================================================
# feature extraction
# =============================================================================

new_dir(path_name_to_desktop+'/Sub_g_classifier_example/audio_features/')
audio_features = path_name_to_desktop+'/Sub_g_classifier_example/audio_features'

# =============================================================================
# extracting features from multiple audio files
# =============================================================================
  
  
path_to_music_folder = path_name_to_desktop+'/Sub_g_classifier_example/Samplemp3/Acid_Techno/'
extract_features(path_to_music_folder, audio_features, filetype='mp3')

path_to_music_folder = path_name_to_desktop+'/Sub_g_classifier_example/Samplemp3/Ghetto_House/'
extract_features(path_to_music_folder, audio_features, filetype='mp3')

path_to_music_folder = path_name_to_desktop+'/Sub_g_classifier_example/Samplemp3/Tech_House/'
extract_features(path_to_music_folder, audio_features, filetype='mp3')


# =============================================================================
# building pandas df
# =============================================================================
features = [
            'sc_dmean','sc_dmeanN','sc_dvar',
            'sc_dvarN','sc_max','sc_mean',
            'sc_med','sc_min','sc_stdev',
            'sc_var',
            'scomp_dmean','scomp_dmeanN','scomp_dvar',
            'scomp_dvarN','scomp_max','scomp_mean',
            'scomp_med','scomp_min','scomp_stdev',
            'scomp_var',
            'sd_dmean','sd_dmeanN','sd_dvar',
            'sd_dvarN','sd_max','sd_mean',
            'sd_med','sd_min','sd_stdev',
            'sd_var',
            'se_dmean','se_dmeanN','se_dvar',
            'se_dvarN','se_max','se_mean',
            'se_med','se_min','se_stdev',
            'se_var',
            'seh_dmean','seh_dmeanN','seh_dvar',
            'seh_dvarN','seh_max','seh_mean',
            'seh_med','seh_min','seh_stdev',
            'seh_var',
            'sel_dmean','sel_dmeanN','sel_dvar',
            'sel_dvarN','sel_max','sel_mean',
            'sel_med','sel_min','sel_stdev',
            'sel_var',
            'semh_dmean','semh_dmeanN','semh_dvar',
            'semh_dvarN','semh_max','semh_mean',
            'semh_med','semh_min','semh_stdev',
            'semh_var',
            'seml_dmean','seml_dmeanN','seml_dvar',
            'seml_dvarN','seml_max','seml_mean',
            'seml_med','seml_min','seml_stdev',
            'seml_var',
            'sent_dmean','sent_dmeanN','sent_dvar',
            'sent_dvarN','sent_max','sent_mean',
            'sent_med','sent_min','sent_stdev',
            'sent_var',
            'sflux_dmean','sflux_dmeanN','sflux_dvar',
            'sflux_dvarN','sflux_max','sflux_mean',
            'sflux_med','sflux_min','sflux_stdev',
            'sflux_var',
            'skur_dmean','skur_dmeanN','skur_dvar',
            'skur_dvarN','skur_max','skur_mean',
            'skur_med','skur_min','skur_stdev',
            'skur_var',
            'srms_dmean','srms_dmeanN','srms_dvar',
            'srms_dvarN','srms_max','srms_mean',
            'srms_med','srms_min','srms_stdev',
            'srms_var',
            'sroll_dmean','sroll_dmeanN','sroll_dvar',
            'sroll_dvarN','sroll_max','sroll_mean',
            'sroll_med','sroll_min','sroll_stdev',
            'sroll_var',
            'sskew_dmean','sskew_dmeanN','sskew_dvar',
            'sskew_dvarN','sskew_max','sskew_mean',
            'sskew_med','sskew_min','sskew_stdev',
            'sskew_var',
            'sspr_dmean','sspr_dmeanN','sspr_dvar',
            'sspr_dvarN','sspr_max','sspr_mean',
            'sspr_med','sspr_min','sspr_stdev',
            'sspr_var',
            'sstp_dmean','sstp_dmeanN','sstp_dvar',
            'sstp_dvarN','sstp_max','sstp_mean',
            'sstp_med','sstp_min','sstp_stdev',
            'sstp_var',
            'ZCR_dmean','ZCR_dmeanN','ZCR_dvar',
            'ZCR_dvarN','ZCR_max','ZCR_mean',
            'ZCR_med','ZCR_min','ZCR_stdev',
            'ZCR_var',  
            
            'bc',
            'bloud_mean','bloud_dmeanN','bloud_dvar',
            'bloud_dvarN','bloud_max','bloud_mean',
            'bloud_med','bloud_min','bloud_stdev',
            'bloud_var',
            'bpm',
            'bpm_hfp', 'bpm_fpw',
            'bpm_spb', 'bpm_sps', 'spw',
            'danceability', 'OR',    
            
            'ccr', 'cnr', 
            'cstr_mean','cstr_dmeanN','cstr_dvar',
            'cstr_dvarN','cstr_max','cstr_mean',
            'cstr_med','cstr_min','cstr_stdev',
            'cstr_var',
            'hpcp_c_mean','hpcp_c_dmeanN','hpcp_c_dvar',
            'hpcp_c_dvarN','hpcp_c_max','hpcp_c_mean',
            'hpcp_c_med','hpcp_c_min','hpcp_c_stdev',
            'hpcp_c_var',
            'hpcp_e_mean','hpcp_e_dmeanN','hpcp_e_dvar',
            'hpcp_e_dvarN','hpcp_e_max','hpcp_e_mean',
            'hpcp_e_med','hpcp_e_min','hpcp_e_stdev',
            'hpcp_e_var',           
            'tds', 'tetd', 'tf', 'tner',
            ]

#get jsons
all_jsons=[pos_json for pos_json in os.listdir(audio_features) if pos_json.endswith('.json')]


#create duplicate all jsons. this will be turned into genre tag string
jsons_tags=[pos_json for pos_json in os.listdir(audio_features) if pos_json.endswith('.json')]
jsons_tags_length=len(jsons_tags)


#jsons tags will be list of strings containing first 2 characters of all jsons files (the subgenre tag)
for i in range(jsons_tags_length):
    jsons_tags[i] = jsons_tags[i][0:2:]


#set columns to contain strings from feature list
jsons_data = pd.DataFrame(columns=features)

# =============================================================================
# index access jsons data. enumerate through all_jsons
# =============================================================================

for index, js in enumerate(all_jsons):
    with open(os.path.join(audio_features, js)) as json_file:
        json_text = json.load(json_file)
        
        
        spectral_centroid0  = json_text['lowlevel']['spectral_centroid']['dmean']
        spectral_centroid1  = json_text['lowlevel']['spectral_centroid']['dmean2']
        spectral_centroid2  = json_text['lowlevel']['spectral_centroid']['dvar']
        spectral_centroid3  = json_text['lowlevel']['spectral_centroid']['dvar2']
        spectral_centroid4  = json_text['lowlevel']['spectral_centroid']['max']
        spectral_centroid5  = json_text['lowlevel']['spectral_centroid']['mean']
        spectral_centroid6  = json_text['lowlevel']['spectral_centroid']['median']
        spectral_centroid7  = json_text['lowlevel']['spectral_centroid']['min']
        spectral_centroid8  = json_text['lowlevel']['spectral_centroid']['stdev']
        spectral_centroid9  = json_text['lowlevel']['spectral_centroid']['var']
        
        spectral_complexity0  = json_text['lowlevel']['spectral_complexity']['dmean']
        spectral_complexity1  = json_text['lowlevel']['spectral_complexity']['dmean2']
        spectral_complexity2  = json_text['lowlevel']['spectral_complexity']['dvar']
        spectral_complexity3  = json_text['lowlevel']['spectral_complexity']['dvar2']
        spectral_complexity4  = json_text['lowlevel']['spectral_complexity']['max']
        spectral_complexity5  = json_text['lowlevel']['spectral_complexity']['mean']
        spectral_complexity6  = json_text['lowlevel']['spectral_complexity']['median']
        spectral_complexity7  = json_text['lowlevel']['spectral_complexity']['min']
        spectral_complexity8  = json_text['lowlevel']['spectral_complexity']['stdev']
        spectral_complexity9  = json_text['lowlevel']['spectral_complexity']['var']
        
        spectral_decrease0  = json_text['lowlevel']['spectral_decrease']['dmean']
        spectral_decrease1  = json_text['lowlevel']['spectral_decrease']['dmean2']
        spectral_decrease2  = json_text['lowlevel']['spectral_decrease']['dvar']
        spectral_decrease3  = json_text['lowlevel']['spectral_decrease']['dvar2']
        spectral_decrease4  = json_text['lowlevel']['spectral_decrease']['max']
        spectral_decrease5  = json_text['lowlevel']['spectral_decrease']['mean']
        spectral_decrease6  = json_text['lowlevel']['spectral_decrease']['median']
        spectral_decrease7  = json_text['lowlevel']['spectral_decrease']['min']
        spectral_decrease8  = json_text['lowlevel']['spectral_decrease']['stdev']
        spectral_decrease9  = json_text['lowlevel']['spectral_decrease']['var']
        
        spectral_energy0  = json_text['lowlevel']['spectral_energy']['dmean']
        spectral_energy1  = json_text['lowlevel']['spectral_energy']['dmean2']
        spectral_energy2  = json_text['lowlevel']['spectral_energy']['dvar']
        spectral_energy3  = json_text['lowlevel']['spectral_energy']['dvar2']
        spectral_energy4  = json_text['lowlevel']['spectral_energy']['max']
        spectral_energy5  = json_text['lowlevel']['spectral_energy']['mean']
        spectral_energy6  = json_text['lowlevel']['spectral_energy']['median']
        spectral_energy7  = json_text['lowlevel']['spectral_energy']['min']
        spectral_energy8  = json_text['lowlevel']['spectral_energy']['stdev']
        spectral_energy9  = json_text['lowlevel']['spectral_energy']['var']
        
        spectral_energyband_high0  = json_text['lowlevel']['spectral_energyband_high']['dmean']
        spectral_energyband_high1  = json_text['lowlevel']['spectral_energyband_high']['dmean2']
        spectral_energyband_high2  = json_text['lowlevel']['spectral_energyband_high']['dvar']
        spectral_energyband_high3  = json_text['lowlevel']['spectral_energyband_high']['dvar2']
        spectral_energyband_high4  = json_text['lowlevel']['spectral_energyband_high']['max']
        spectral_energyband_high5  = json_text['lowlevel']['spectral_energyband_high']['mean']
        spectral_energyband_high6  = json_text['lowlevel']['spectral_energyband_high']['median']
        spectral_energyband_high7  = json_text['lowlevel']['spectral_energyband_high']['min']
        spectral_energyband_high8  = json_text['lowlevel']['spectral_energyband_high']['stdev']
        spectral_energyband_high9  = json_text['lowlevel']['spectral_energyband_high']['var']
        
        spectral_energyband_low0  = json_text['lowlevel']['spectral_energyband_low']['dmean']
        spectral_energyband_low1  = json_text['lowlevel']['spectral_energyband_low']['dmean2']
        spectral_energyband_low2  = json_text['lowlevel']['spectral_energyband_low']['dvar']
        spectral_energyband_low3  = json_text['lowlevel']['spectral_energyband_low']['dvar2']
        spectral_energyband_low4  = json_text['lowlevel']['spectral_energyband_low']['max']
        spectral_energyband_low5  = json_text['lowlevel']['spectral_energyband_low']['mean']
        spectral_energyband_low6  = json_text['lowlevel']['spectral_energyband_low']['median']
        spectral_energyband_low7  = json_text['lowlevel']['spectral_energyband_low']['min']
        spectral_energyband_low8  = json_text['lowlevel']['spectral_energyband_low']['stdev']
        spectral_energyband_low9  = json_text['lowlevel']['spectral_energyband_low']['var']
        
        spectral_energyband_middle_high0  = json_text['lowlevel']['spectral_energyband_middle_high']['dmean']
        spectral_energyband_middle_high1  = json_text['lowlevel']['spectral_energyband_middle_high']['dmean2']
        spectral_energyband_middle_high2  = json_text['lowlevel']['spectral_energyband_middle_high']['dvar']
        spectral_energyband_middle_high3  = json_text['lowlevel']['spectral_energyband_middle_high']['dvar2']
        spectral_energyband_middle_high4  = json_text['lowlevel']['spectral_energyband_middle_high']['max']
        spectral_energyband_middle_high5  = json_text['lowlevel']['spectral_energyband_middle_high']['mean']
        spectral_energyband_middle_high6  = json_text['lowlevel']['spectral_energyband_middle_high']['median']
        spectral_energyband_middle_high7  = json_text['lowlevel']['spectral_energyband_middle_high']['min']
        spectral_energyband_middle_high8  = json_text['lowlevel']['spectral_energyband_middle_high']['stdev']
        spectral_energyband_middle_high9  = json_text['lowlevel']['spectral_energyband_middle_high']['var']
        
        spectral_energyband_middle_low0  = json_text['lowlevel']['spectral_energyband_middle_low']['dmean']
        spectral_energyband_middle_low1  = json_text['lowlevel']['spectral_energyband_middle_low']['dmean2']
        spectral_energyband_middle_low2  = json_text['lowlevel']['spectral_energyband_middle_low']['dvar']
        spectral_energyband_middle_low3  = json_text['lowlevel']['spectral_energyband_middle_low']['dvar2']
        spectral_energyband_middle_low4  = json_text['lowlevel']['spectral_energyband_middle_low']['max']
        spectral_energyband_middle_low5  = json_text['lowlevel']['spectral_energyband_middle_low']['mean']
        spectral_energyband_middle_low6  = json_text['lowlevel']['spectral_energyband_middle_low']['median']
        spectral_energyband_middle_low7  = json_text['lowlevel']['spectral_energyband_middle_low']['min']
        spectral_energyband_middle_low8  = json_text['lowlevel']['spectral_energyband_middle_low']['stdev']
        spectral_energyband_middle_low9  = json_text['lowlevel']['spectral_energyband_middle_low']['var']
        
        spectral_entropy0  = json_text['lowlevel']['spectral_entropy']['dmean']
        spectral_entropy1  = json_text['lowlevel']['spectral_entropy']['dmean2']
        spectral_entropy2  = json_text['lowlevel']['spectral_entropy']['dvar']
        spectral_entropy3  = json_text['lowlevel']['spectral_entropy']['dvar2']
        spectral_entropy4  = json_text['lowlevel']['spectral_entropy']['max']
        spectral_entropy5  = json_text['lowlevel']['spectral_entropy']['mean']
        spectral_entropy6  = json_text['lowlevel']['spectral_entropy']['median']
        spectral_entropy7  = json_text['lowlevel']['spectral_entropy']['min']
        spectral_entropy8  = json_text['lowlevel']['spectral_entropy']['stdev']
        spectral_entropy9  = json_text['lowlevel']['spectral_entropy']['var']
        
        spectral_flux0  = json_text['lowlevel']['spectral_flux']['dmean']
        spectral_flux1  = json_text['lowlevel']['spectral_flux']['dmean2']
        spectral_flux2  = json_text['lowlevel']['spectral_flux']['dvar']
        spectral_flux3  = json_text['lowlevel']['spectral_flux']['dvar2']
        spectral_flux4  = json_text['lowlevel']['spectral_flux']['max']
        spectral_flux5  = json_text['lowlevel']['spectral_flux']['mean']
        spectral_flux6  = json_text['lowlevel']['spectral_flux']['median']
        spectral_flux7  = json_text['lowlevel']['spectral_flux']['min']
        spectral_flux8  = json_text['lowlevel']['spectral_flux']['stdev']
        spectral_flux9  = json_text['lowlevel']['spectral_flux']['var']
        
        spectral_kurtosis0  = json_text['lowlevel']['spectral_kurtosis']['dmean']
        spectral_kurtosis1  = json_text['lowlevel']['spectral_kurtosis']['dmean2']
        spectral_kurtosis2  = json_text['lowlevel']['spectral_kurtosis']['dvar']
        spectral_kurtosis3  = json_text['lowlevel']['spectral_kurtosis']['dvar2']
        spectral_kurtosis4  = json_text['lowlevel']['spectral_kurtosis']['max']
        spectral_kurtosis5  = json_text['lowlevel']['spectral_kurtosis']['mean']
        spectral_kurtosis6  = json_text['lowlevel']['spectral_kurtosis']['median']
        spectral_kurtosis7  = json_text['lowlevel']['spectral_kurtosis']['min']
        spectral_kurtosis8  = json_text['lowlevel']['spectral_kurtosis']['stdev']
        spectral_kurtosis9  = json_text['lowlevel']['spectral_kurtosis']['var']
        
        spectral_rms0  = json_text['lowlevel']['spectral_rms']['dmean']
        spectral_rms1  = json_text['lowlevel']['spectral_rms']['dmean2']
        spectral_rms2  = json_text['lowlevel']['spectral_rms']['dvar']
        spectral_rms3  = json_text['lowlevel']['spectral_rms']['dvar2']
        spectral_rms4  = json_text['lowlevel']['spectral_rms']['max']
        spectral_rms5  = json_text['lowlevel']['spectral_rms']['mean']
        spectral_rms6  = json_text['lowlevel']['spectral_rms']['median']
        spectral_rms7  = json_text['lowlevel']['spectral_rms']['min']
        spectral_rms8  = json_text['lowlevel']['spectral_rms']['stdev']
        spectral_rms9  = json_text['lowlevel']['spectral_rms']['var']
        
        spectral_rolloff0  = json_text['lowlevel']['spectral_rolloff']['dmean']
        spectral_rolloff1  = json_text['lowlevel']['spectral_rolloff']['dmean2']
        spectral_rolloff2  = json_text['lowlevel']['spectral_rolloff']['dvar']
        spectral_rolloff3  = json_text['lowlevel']['spectral_rolloff']['dvar2']
        spectral_rolloff4  = json_text['lowlevel']['spectral_rolloff']['max']
        spectral_rolloff5  = json_text['lowlevel']['spectral_rolloff']['mean']
        spectral_rolloff6  = json_text['lowlevel']['spectral_rolloff']['median']
        spectral_rolloff7  = json_text['lowlevel']['spectral_rolloff']['min']
        spectral_rolloff8  = json_text['lowlevel']['spectral_rolloff']['stdev']
        spectral_rolloff9  = json_text['lowlevel']['spectral_rolloff']['var']
        
        spectral_skewness0  = json_text['lowlevel']['spectral_skewness']['dmean']
        spectral_skewness1  = json_text['lowlevel']['spectral_skewness']['dmean2']
        spectral_skewness2  = json_text['lowlevel']['spectral_skewness']['dvar']
        spectral_skewness3  = json_text['lowlevel']['spectral_skewness']['dvar2']
        spectral_skewness4  = json_text['lowlevel']['spectral_skewness']['max']
        spectral_skewness5  = json_text['lowlevel']['spectral_skewness']['mean']
        spectral_skewness6  = json_text['lowlevel']['spectral_skewness']['median']
        spectral_skewness7  = json_text['lowlevel']['spectral_skewness']['min']
        spectral_skewness8  = json_text['lowlevel']['spectral_skewness']['stdev']
        spectral_skewness9  = json_text['lowlevel']['spectral_skewness']['var']
        
        spectral_spread0  = json_text['lowlevel']['spectral_spread']['dmean']
        spectral_spread1  = json_text['lowlevel']['spectral_spread']['dmean2']
        spectral_spread2  = json_text['lowlevel']['spectral_spread']['dvar']
        spectral_spread3  = json_text['lowlevel']['spectral_spread']['dvar2']
        spectral_spread4  = json_text['lowlevel']['spectral_spread']['max']
        spectral_spread5  = json_text['lowlevel']['spectral_spread']['mean']
        spectral_spread6  = json_text['lowlevel']['spectral_spread']['median']
        spectral_spread7  = json_text['lowlevel']['spectral_spread']['min']
        spectral_spread8  = json_text['lowlevel']['spectral_spread']['stdev']
        spectral_spread9  = json_text['lowlevel']['spectral_spread']['var']
        
        spectral_strongpeak0  = json_text['lowlevel']['spectral_strongpeak']['dmean']
        spectral_strongpeak1  = json_text['lowlevel']['spectral_strongpeak']['dmean2']
        spectral_strongpeak2  = json_text['lowlevel']['spectral_strongpeak']['dvar']
        spectral_strongpeak3  = json_text['lowlevel']['spectral_strongpeak']['dvar2']
        spectral_strongpeak4  = json_text['lowlevel']['spectral_strongpeak']['max']
        spectral_strongpeak5  = json_text['lowlevel']['spectral_strongpeak']['mean']
        spectral_strongpeak6  = json_text['lowlevel']['spectral_strongpeak']['median']
        spectral_strongpeak7  = json_text['lowlevel']['spectral_strongpeak']['min']
        spectral_strongpeak8  = json_text['lowlevel']['spectral_strongpeak']['stdev']
        spectral_strongpeak9  = json_text['lowlevel']['spectral_strongpeak']['var']
        

        
        zerocrossingrate0  = json_text['lowlevel']['zerocrossingrate']['dmean']
        zerocrossingrate1  = json_text['lowlevel']['zerocrossingrate']['dmean2']
        zerocrossingrate2  = json_text['lowlevel']['zerocrossingrate']['dvar']
        zerocrossingrate3  = json_text['lowlevel']['zerocrossingrate']['dvar2']
        zerocrossingrate4  = json_text['lowlevel']['zerocrossingrate']['max']
        zerocrossingrate5  = json_text['lowlevel']['zerocrossingrate']['mean']
        zerocrossingrate6  = json_text['lowlevel']['zerocrossingrate']['median']
        zerocrossingrate7  = json_text['lowlevel']['zerocrossingrate']['min']
        zerocrossingrate8  = json_text['lowlevel']['zerocrossingrate']['stdev']
        zerocrossingrate9  = json_text['lowlevel']['zerocrossingrate']['var']

        # RHYTHM
        
        beats_count = json_text['rhythm']['beats_count']        
        beats_loudness0  = json_text['rhythm']['beats_loudness']['dmean']
        beats_loudness1  = json_text['rhythm']['beats_loudness']['dmean2']
        beats_loudness2  = json_text['rhythm']['beats_loudness']['dvar']
        beats_loudness3  = json_text['rhythm']['beats_loudness']['dvar2']
        beats_loudness4  = json_text['rhythm']['beats_loudness']['max']
        beats_loudness5  = json_text['rhythm']['beats_loudness']['mean']
        beats_loudness6  = json_text['rhythm']['beats_loudness']['median']
        beats_loudness7  = json_text['rhythm']['beats_loudness']['min']
        beats_loudness8  = json_text['rhythm']['beats_loudness']['stdev']
        beats_loudness9  = json_text['rhythm']['beats_loudness']['var']
        
        bpm   = json_text['rhythm']['bpm']
        
        bpm_histogram_first_peak_bpm      = json_text['rhythm']['bpm_histogram_first_peak_bpm']
        bpm_histogram_first_peak_weight   = json_text['rhythm']['bpm_histogram_first_peak_weight']
        
        bpm_histogram_second_peak_bpm     = json_text['rhythm']['bpm_histogram_second_peak_bpm']
        bpm_histogram_second_peak_spread  = json_text['rhythm']['bpm_histogram_second_peak_spread']
        bpm_histogram_second_peak_weight  = json_text['rhythm']['bpm_histogram_second_peak_weight']
        
        danceability  = json_text['rhythm']['danceability']
        
        onset_rate    = json_text['rhythm']['onset_rate']
        
        # TONAL
        
        chords_changes_rate = json_text['tonal']['chords_changes_rate']
        chords_number_rate  = json_text['tonal']['chords_number_rate']
        
        chords_strength0  = json_text['tonal']['chords_strength']['dmean']
        chords_strength1  = json_text['tonal']['chords_strength']['dmean2']
        chords_strength2  = json_text['tonal']['chords_strength']['dvar']
        chords_strength3  = json_text['tonal']['chords_strength']['dvar2']
        chords_strength4  = json_text['tonal']['chords_strength']['max']
        chords_strength5  = json_text['tonal']['chords_strength']['mean']
        chords_strength6  = json_text['tonal']['chords_strength']['median']
        chords_strength7  = json_text['tonal']['chords_strength']['min']
        chords_strength8  = json_text['tonal']['chords_strength']['stdev']
        chords_strength9  = json_text['tonal']['chords_strength']['var']
        
        hpcp_crest0  = json_text['tonal']['hpcp_crest']['dmean']
        hpcp_crest1  = json_text['tonal']['hpcp_crest']['dmean2']
        hpcp_crest2  = json_text['tonal']['hpcp_crest']['dvar']
        hpcp_crest3  = json_text['tonal']['hpcp_crest']['dvar2']
        hpcp_crest4  = json_text['tonal']['hpcp_crest']['max']
        hpcp_crest5  = json_text['tonal']['hpcp_crest']['mean']
        hpcp_crest6  = json_text['tonal']['hpcp_crest']['median']
        hpcp_crest7  = json_text['tonal']['hpcp_crest']['min']
        hpcp_crest8  = json_text['tonal']['hpcp_crest']['stdev']
        hpcp_crest9  = json_text['tonal']['hpcp_crest']['var']
        
        hpcp_entropy0  = json_text['tonal']['hpcp_entropy']['dmean']
        hpcp_entropy1  = json_text['tonal']['hpcp_entropy']['dmean2']
        hpcp_entropy2  = json_text['tonal']['hpcp_entropy']['dvar']
        hpcp_entropy3  = json_text['tonal']['hpcp_entropy']['dvar2']
        hpcp_entropy4  = json_text['tonal']['hpcp_entropy']['max']
        hpcp_entropy5  = json_text['tonal']['hpcp_entropy']['mean']
        hpcp_entropy6  = json_text['tonal']['hpcp_entropy']['median']
        hpcp_entropy7  = json_text['tonal']['hpcp_entropy']['min']
        hpcp_entropy8  = json_text['tonal']['hpcp_entropy']['stdev']
        hpcp_entropy9  = json_text['tonal']['hpcp_entropy']['var']
        

        
        
        

        
        
        
        tuning_diatonic_strength          = json_text['tonal']['tuning_diatonic_strength']
        tuning_equal_tempered_deviation   = json_text['tonal']['tuning_equal_tempered_deviation']
        tuning_frequency                  = json_text['tonal']['tuning_frequency']
        tuning_nontempered_energy_ratio   = json_text['tonal']['tuning_nontempered_energy_ratio']
        

                              
                                                       
                                                       
        jsons_data.loc[index] = [

                                
                                spectral_centroid0, spectral_centroid1, spectral_centroid2, 
                                spectral_centroid3, spectral_centroid4, spectral_centroid5, 
                                spectral_centroid6, spectral_centroid7, spectral_centroid8, 
                                spectral_centroid9, 
                                spectral_complexity0, spectral_complexity1, spectral_complexity2, 
                                spectral_complexity3, spectral_complexity4, spectral_complexity5, 
                                spectral_complexity6, spectral_complexity7, spectral_complexity8, 
                                spectral_complexity9, 
                                spectral_decrease0, spectral_decrease1, spectral_decrease2, 
                                spectral_decrease3, spectral_decrease4, spectral_decrease5, 
                                spectral_decrease6, spectral_decrease7, spectral_decrease8, 
                                spectral_decrease9, 
                                spectral_energy0, spectral_energy1, spectral_energy2, 
                                spectral_energy3, spectral_energy4, spectral_energy5, 
                                spectral_energy6, spectral_energy7, spectral_energy8, 
                                spectral_energy9, 
                                spectral_energyband_high0, spectral_energyband_high1, spectral_energyband_high2, 
                                spectral_energyband_high3, spectral_energyband_high4, spectral_energyband_high5, 
                                spectral_energyband_high6, spectral_energyband_high7, spectral_energyband_high8, 
                                spectral_energyband_high9, 
                                spectral_energyband_low0, spectral_energyband_low1, spectral_energyband_low2, 
                                spectral_energyband_low3, spectral_energyband_low4, spectral_energyband_low5, 
                                spectral_energyband_low6, spectral_energyband_low7, spectral_energyband_low8, 
                                spectral_energyband_low9, 
                                spectral_energyband_middle_high0, spectral_energyband_middle_high1, spectral_energyband_middle_high2, 
                                spectral_energyband_middle_high3, spectral_energyband_middle_high4, spectral_energyband_middle_high5, 
                                spectral_energyband_middle_high6, spectral_energyband_middle_high7, spectral_energyband_middle_high8, 
                                spectral_energyband_middle_high9, 
                                spectral_energyband_middle_low0, spectral_energyband_middle_low1, spectral_energyband_middle_low2, 
                                spectral_energyband_middle_low3, spectral_energyband_middle_low4, spectral_energyband_middle_low5, 
                                spectral_energyband_middle_low6, spectral_energyband_middle_low7, spectral_energyband_middle_low8, 
                                spectral_energyband_middle_low9, 
                                spectral_entropy0, spectral_entropy1, spectral_entropy2, 
                                spectral_entropy3, spectral_entropy4, spectral_entropy5, 
                                spectral_entropy6, spectral_entropy7, spectral_entropy8, 
                                spectral_entropy9, 
                                spectral_flux0, spectral_flux1, spectral_flux2, 
                                spectral_flux3, spectral_flux4, spectral_flux5, 
                                spectral_flux6, spectral_flux7, spectral_flux8, 
                                spectral_flux9, 
                                spectral_kurtosis0, spectral_kurtosis1, spectral_kurtosis2, 
                                spectral_kurtosis3, spectral_kurtosis4, spectral_kurtosis5, 
                                spectral_kurtosis6, spectral_kurtosis7, spectral_kurtosis8, 
                                spectral_kurtosis9, 
                                spectral_rms0, spectral_rms1, spectral_rms2, 
                                spectral_rms3, spectral_rms4, spectral_rms5, 
                                spectral_rms6, spectral_rms7, spectral_rms8, 
                                spectral_rms9, 
                                spectral_rolloff0, spectral_rolloff1, spectral_rolloff2, 
                                spectral_rolloff3, spectral_rolloff4, spectral_rolloff5, 
                                spectral_rolloff6, spectral_rolloff7, spectral_rolloff8, 
                                spectral_rolloff9, 
                                spectral_skewness0, spectral_skewness1, spectral_skewness2, 
                                spectral_skewness3, spectral_skewness4, spectral_skewness5, 
                                spectral_skewness6, spectral_skewness7, spectral_skewness8, 
                                spectral_skewness9, 
                                spectral_spread0, spectral_spread1, spectral_spread2, 
                                spectral_spread3, spectral_spread4, spectral_spread5, 
                                spectral_spread6, spectral_spread7, spectral_spread8, 
                                spectral_spread9, 
                                spectral_strongpeak0, spectral_strongpeak1, spectral_strongpeak2, 
                                spectral_strongpeak3, spectral_strongpeak4, spectral_strongpeak5, 
                                spectral_strongpeak6, spectral_strongpeak7, spectral_strongpeak8, 
                                spectral_strongpeak9, 
                                zerocrossingrate0, zerocrossingrate1, zerocrossingrate2, 
                                zerocrossingrate3, zerocrossingrate4, zerocrossingrate5, 
                                zerocrossingrate6, zerocrossingrate7, zerocrossingrate8, 
                                zerocrossingrate9,
 
                                beats_count,
                                beats_loudness0, beats_loudness1, beats_loudness2, 
                                beats_loudness3, beats_loudness4, beats_loudness5, 
                                beats_loudness6, beats_loudness7, beats_loudness8, 
                                beats_loudness9,
                                bpm, 
                                bpm_histogram_first_peak_bpm, bpm_histogram_first_peak_weight,
                                bpm_histogram_second_peak_bpm, bpm_histogram_second_peak_spread, bpm_histogram_second_peak_weight,
                                danceability, onset_rate,

                                chords_changes_rate, chords_number_rate,
                                chords_strength0, chords_strength1, chords_strength2, 
                                chords_strength3, chords_strength4, chords_strength5, 
                                chords_strength6, chords_strength7, chords_strength8, 
                                chords_strength9,
                                hpcp_crest0, hpcp_crest1, hpcp_crest2, 
                                hpcp_crest3, hpcp_crest4, hpcp_crest5, 
                                hpcp_crest6, hpcp_crest7, hpcp_crest8, 
                                hpcp_crest9,
                                hpcp_entropy0, hpcp_entropy1, hpcp_entropy2, 
                                hpcp_entropy3, hpcp_entropy4, hpcp_entropy5, 
                                hpcp_entropy6, hpcp_entropy7, hpcp_entropy8, 
                                hpcp_entropy9,
                                tuning_diatonic_strength, tuning_equal_tempered_deviation, tuning_frequency, tuning_nontempered_energy_ratio,]
                                

    





# =============================================================================
# tags & data
# =============================================================================
#add sub_genre column from jsons tags variable created earlier
jsons_data['sub_genre'] = jsons_tags

# pandas df w no tag column
X=(jsons_data.loc[:, jsons_data.columns != 'sub_genre'])
#convert out of pandas format
X = X.loc[:, X.columns].values



#for only tag
y=jsons_data['sub_genre']
#array of tags
y = y.values


# =============================================================================
# Pipelines
# =============================================================================
knn_5 = Pipeline([('standardize', StandardScaler()),('pca', PCA(n_components=.95)),
                ('kNN', KNeighborsClassifier(n_neighbors = 5))])

svm_l = Pipeline([('standardize', StandardScaler()),('pca', PCA(n_components=.95)),
                ('svm_l', SVC(kernel='linear'))])

svm_p = Pipeline([('standardize', StandardScaler()),('pca', PCA(n_components=.95)),
                ('svm_p', SVC(kernel='poly', degree=1))])

svm_rbf = Pipeline([('standardize', StandardScaler()),('pca', PCA(n_components=.95)),
                ('svm_rbf', SVC(kernel='rbf'))])

svm_s = Pipeline([('standardize', StandardScaler()),('pca', PCA(n_components=.95)),
                ('svm_s', SVC(kernel='sigmoid'))])

lr_lbfgs = Pipeline([('standardize', StandardScaler()),('pca', PCA(n_components=.95)),
                ('lr_lbfgs', LogisticRegression(solver = 'lbfgs', max_iter=2000))])


# =============================================================================
# kNN
# =============================================================================
# confusion matrix
y_pred = cross_val_predict(knn_5, X, y, cv=5)
conf_mat = confusion_matrix(y, y_pred)
print("knn_5 confusion matrix:")
print(conf_mat)

# overall accuracy
kfold = KFold(n_splits=10)
results = cross_val_score(knn_5, X, y, cv=kfold)
print ("knn_5 overall average acc:",results.mean())


# =============================================================================
# SVM
# =============================================================================
# confusion matrix
y_pred = cross_val_predict(svm_p, X, y, cv=5)
conf_mat = confusion_matrix(y, y_pred)
print('')
print("svm_p confusion matrix:")
print(conf_mat)

# overall accuracy
kfold = KFold(n_splits=10)
results = cross_val_score(svm_p, X, y, cv=kfold)
print ("svm_p overall average acc:",results.mean())


# =============================================================================
# Logistics Regression
# =============================================================================
# confusion matrix
y_pred = cross_val_predict(lr_lbfgs, X, y, cv=5)
conf_mat = confusion_matrix(y, y_pred)
print('')
print("lr_lbfgs confusion matrix:")
print(conf_mat)

# evaluate pipeline
kfold = KFold(n_splits=10)
results = cross_val_score(lr_lbfgs, X, y, cv=kfold)
print ("lr_lbfgs average acc:",results.mean())

