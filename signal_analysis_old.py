"""
Author: Samarth Tandon
Copyright (c) 2018, Docturnal Private Limited.All rights reserved.
Docturnal Private Limited and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from Docturnal Private Limited is strictly prohibited
"""
import numpy as np
import xlsxwriter
import argparse
import logging
from scipy import stats
from librosa.core import resample as libresample
from scipy.io import wavfile
from scipy.stats import mstats,skew,kurtosis
from python_speech_features import mfcc
import os
import math
import time
import sys
from sympy import factorint
import wavio
import clip_wav
import spectrograph as spec
import utils
import ntpath
import warnings
warnings.filterwarnings("ignore")
score_label = [
    'upto200HzLSUM','200-500HzLSUM','500-1KHzLSUM','1K-1.5KHzLSUM','1.5-2KHzLSUM','2K-2.5KHzLSUM','2.5-3KHzLSUM','3K-3.5KHzLSUM','3.5K-4KHzLSUM','4K-4.5KHzLSUM','4.5K-5KHzLSUM',
    'upto200HzLmean','200-500HzLmean','500-1KHzLmean','1K-1.5KHzLmean','1.5-2KHzLmean','2K-2.5KHzLmean','2.5-3KHzLmean','3K-3.5KHzLmean','3.5K-4KHzLmean','4K-4.5KHzLmean','4.5K-5KHzLmean',
    'upto200HzLsigma','200-500HzLsigma','500-1KHzLsigma','1K-1.5KHzLsigma','1.5-2KHzLsigma','2K-2.5KHzLsigma','2.5-3KHzLsigma','3K-3.5KHzLsigma','3.5K-4KHzLsigma','4K-4.5KHzLsigma','4.5K-5KHzLsigma',
    'upto200HzLvar','200-500HzLvar','500-1KHzLvar','1K-1.5KHzLvar','1.5-2KHzLvar','2K-2.5KHzLvar','2.5-3KHzLvar','3K-3.5KHzLvar','3.5K-4KHzLvar','4K-4.5KHzLvar','4.5K-5KHzLvar',
    'upto200HzLcofvar','200-500HzLcofvar','500-1KHzcofLvar','1K-1.5KHzcofLvar','1.5-2KHzcofLvar','2K-2.5KHzcofLvar','2.5-3KHzcofLvar','3K-3.5KHzcofLvar','3.5K-4KHzcofLvar','4K-4.5KHzcofLvar','4.5K-5KHzcofLvar',
    'upto200HzLtoptenamp','200-500HzLtoptenamp','500-1KHzLtoptenamp','1K-1.5KHzLtoptenamp','1.5-2KHzLtoptenamp','2K-2.5KHzLtoptenamp','2.5-3KHzLtoptenamp','3K-3.5KHzLtoptenamp','3.5K-4KHztoptenamp','4K-4.5KHzLtoptenamp','4.5K-5KHzLtoptenamp' ,
    'upto200HzLcentroid','200-500HzLcentroid','500-1KHzLcentroid','1K-1.5KHzLcentroid','1.5-2KHzLcentroid','2K-2.5KHzLcentroid','2.5-3KHzLcentroid','3K-3.5KHzLcentroid','3.5K-4KHzLcentroid','4K-4.5KHzLcentroid','4.5K-5KHzLcentroid',
    'upto200HzLflatness','200-500HzLflatness','500-1KHzLflatness','1K-1.5KHzLflatness','1.5-2KHzLflatness','2K-2.5KHzLflatness','2.5-3KHzLflatness','3K-3.5KHzLflatness','3.5K-4KHzLflatness','4K-4.5KHzLflatness','4.5K-5KHzLflatness',
    'upto200HzLskewness','200-500HzLskewness','500-1KHzLskewness','1K-1.5KHzLskewness','1.5-2KHzLskewness','2K-2.5KHzLskewness','2.5-3KHzLskewness','3K-3.5KHzLskewness','3.5K-4KHzLskewness','4K-4.5KHzLskewness','4.5K-5KHzLskewness',
    'upto200HzLkurtosis','200-500HzLkurtosis','500-1KHzLkurtosis','1K-1.5KHzLkurtosis','1.5-2KHzLkurtosis','2K-2.5KHzLskewness','2.5-3KHzLkurtosis','3K-3.5KHzLkurtosis','3.5K-4KHzLkurtosis','4K-4.5KHzLkurtosis','4.5K-5KHzLkurtosis',
    'upto200HzLmfcc','200-500HzLmfcc','500-1KHzLmfcc','1K-1.5KHzLmfcc','1.5-2KHzLmfcc','2K-2.5KHzLskewness','2.5-3KHzLmfcc','3K-3.5KHzLmfcc','3.5K-4KHzLmfcc','4K-4.5KHzLmfcc','4.5K-5KHzLmfcc',
    'upto200HzLenergy','200-500HzLenergy','500-1KHzLenergy','1K-1.5KHzLenergy','1.5-2KHzLenergy','2K-2.5KHzLskewness','2.5-3KHzLenergy','3K-3.5KHzLenergy','3.5K-4KHzLenergy','4K-4.5KHzLenergy','4.5K-5KHzLenergy',

    'upto200HzRSUM','200-500HzRSUM','500-1KHzRSUM','1K-1.5KHzRSUM','1.5-2KHzRSUM','2K-2.5KHzRSUM','2.5-3KHzRSUM','3K-3.5KHzRSUM','3.5K-4KHzRSUM','4K-4.5KHzRSUM','4.5K-5KHzRSUM',
    'upto200HzRmean','200-500HzRmean','500-1KHzRmean','1K-1.5KHzRmean','1.5-2KHzRmean','2K-2.5KHzRmean','2.5-3KHzRmean','3K-3.5KHzRmean','3.5K-4KHzRmean','4K-4.5KHzRmean','4.5K-5KHzRmean',
    'upto200HzRsigma','200-500HzRsigma','500-1KHzRsigma','1K-1.5KHzRsigma','1.5-2KHzRsigma','2K-2.5KHzRsigma','2.5-3KHzRsigma','3K-3.5KHzRsigma','3.5K-4KHzRsigma','4K-4.5KHzRsigma','4.5K-5KHzRsigma',
    'upto200HzRvariance','200-500HzRvariance','500-1KHzRvariance','1K-1.5KHzRvariance','1.5-2KHzRvariance','2K-2.5KHzRvariance','2.5-3KHzRvariance','3K-3.5KHzRvariance','3.5K-4KHzRvariance','4K-4.5KHzRvariance','4.5K-5KHzRvariance',
    'upto200HzRcofvar','200-500HzRcofvar','500-1KHzcofRvar','1K-1.5KHzcofRvar','1.5-2KHzcofRvar','2K-2.5KHzcofRvar','2.5-3KHzcofRvar','3K-3.5KHzcofRvar','3.5K-4KHzcofRvar','4K-4.5KHzcofRvar','4.5K-5KHzcofRvar',
    'upto200HzRtoptenamp','200-500HzRtoptenamp','500-1KHzRtoptenamp','1K-1.5KHzRtoptenamp','1.5-2KHzRtoptenamp','2K-2.5KHzRtoptenamp','2.5-3KHzRtoptenamp','3K-3.5KHzRtoptenamp','3.5K-4KHzRtoptenamp','4K-4.5KHzRtoptenamp','4.5K-5KHzRtoptenamp',
    'upto200HzRcentroid','200-500HzRcentroid','500-1KHzRcentroid','1K-1.5KHzRcentroid','1.5-2KHzRcentroid','2K-2.5KHzRcentroid','2.5-3KHzRcentroid','3K-3.5KHzRcentroid','3.5K-4KHzRcentroid','4K-4.5KHzRcentroid','4.5K-5KHzRcentroid',
    'upto200HzRflatness','200-500HzRflatness','500-1KHzRflatness','1K-1.5KHzRflatness','1.5-2KHzRflatness','2K-2.5KHzRflatness','2.5-3KHzRflatness','3K-3.5KHzRflatness','3.5K-4KHzRflatness','4K-4.5KHzRflatness','4.5K-5KHzRflatness',
    'upto200HzRskewness','200-500HzRskewness','500-1KHzRskewness','1K-1.5KHzRskewness','1.5-2KHzRskewness','2K-2.5KHzRskewness','2.5-3KHzRskewness','3K-3.5KHzRskewness','3.5K-4KHzRskewness','4K-4.5KHzRskewness','4.5K-5KHzRskewness',
    'upto200HzRkurtosis','200-500HzRkurtosis','500-1KHzRkurtosis','1K-1.5KHzRkurtosis','1.5-2KHzRkurtosis','2K-2.5KHzRskewness','2.5-3KHzRkurtosis','3K-3.5KHzRkurtosis','3.5K-4KHzRkurtosis','4K-4.5KHzRkurtosis','4.5K-5KHzRkurtosis',
    'upto200HzRmfcc','200-500HzRmfcc','500-1KHzRmfcc','1K-1.5KHzRmfcc','1.5-2KHzRmfcc','2K-2.5KHzRskewness','2.5-3KHzRmfcc','3K-3.5KHzRmfcc','3.5K-4KHzRmfcc','4K-4.5KHzRmfcc','4.5K-5KHzRmfcc',
    'upto200HzRenergy','200-500HzRenergy','500-1KHzRenergy','1K-1.5KHzRenergy','1.5-2KHzRenergy','2K-2.5KHzRskewness','2.5-3KHzRenergy','3K-3.5KHzRenergy','3.5K-4KHzRenergy','4K-4.5KHzRenergy','4.5K-5KHzRenergy']

summary_var_list = ['sum','mean', 'sigma', 'var', 'cof_var','top_ten_amp','spectral_centroid',
                    'spectral_flatness','spectral_skewness','kurtosis', 'mfcc','energy'
                    ]
batch_index = ['upto200Hz','200-500Hz','500-1000Hz','1000-1500Hz','1500-2000Hz','2000-2500Hz'
               ,'2500-3000Hz','3000-3500Hz','3500-4000Hz','4000-4500Hz','4500-5000Hz']


def spectral_flatness(channel):
    """
     Spectral Flatness Geometric mean Power Spectrum / airthematic mean power spectrum
    :param channel: Band Channel
    :return: spectral Flatness
    """
    Power_spectrum_ch =channel**2
    Gmean_Power_spectrum_Lch = mstats.gmean(Power_spectrum_ch)
    Amean_Power_spectrum_Lch = np.mean(Power_spectrum_ch)
    spectral_flatness_ch = Gmean_Power_spectrum_Lch/Amean_Power_spectrum_Lch
    return spectral_flatness_ch


def write_bands_in_xls(band,worksheet):
    """
    Function to write bands in the excel worksheet
    :param band: Split bands
    :param worksheet: worksheet object
    :return: NA
    """
    row =0
    col =0
    for freq,ch_1,ch_2 in (band):
        worksheet.write(row,col,freq)
        worksheet.write(row,col+1,ch_1)
        worksheet.write(row,col+2,ch_2)
        row=row+1


def write_sumVar_in_xls(varlist,worksheet):
    """
    Update summary varibales in the worksheet
    :param varlist: list of variables
    :param worksheet: worksheet object
    :return: NA
    """
    row =1
    for var in varlist:
        for col in range(len(var)):
            worksheet.write(row,col+1,var[col])
            col=col+1
        row=row+1
     

def write_in_xlsx(filename, bands_list ,sumvar_lch,sumvar_rch , dwt_cof_list):
    
    filename=filename+'.xlsx'
    workbook = xlsxwriter.Workbook(filename,{'nan_inf_to_errors': True})
    worksheet_1_0 = workbook.add_worksheet('upto200Hz');worksheet_1_1 = workbook.add_worksheet('200-500Hz')
    worksheet_1_2 = workbook.add_worksheet('500-1000Hz');worksheet_1_3 = workbook.add_worksheet('1000-1500Hz')
    worksheet_1_4 = workbook.add_worksheet('1500-2000Hz');worksheet_1_5 = workbook.add_worksheet('2000-2500Hz')
    worksheet_1_6 = workbook.add_worksheet('2500-3000Hz');worksheet_1_7 = workbook.add_worksheet('3000-3500Hz')
    worksheet_1_8 = workbook.add_worksheet('3500-4000Hz');worksheet_1_9 = workbook.add_worksheet('4000-4500Hz')
    worksheet_1_10 = workbook.add_worksheet('4500-5000Hz')
    worksheet_list =[worksheet_1_0,worksheet_1_1,worksheet_1_2,worksheet_1_3,worksheet_1_4,worksheet_1_5,
                      worksheet_1_6,worksheet_1_7,worksheet_1_8,
                      worksheet_1_9,worksheet_1_10]
    index=0
    for band in bands_list:
        write_bands_in_xls(band, worksheet_list[index])
        index=index+1
    
    worksheet_2_1 = workbook.add_worksheet('SummaryVar_Lchannel')
    write_sumVar_in_xls(sumvar_lch, worksheet_2_1)
    worksheet_2_2 = workbook.add_worksheet('SummaryVar_Rchannel')
    write_sumVar_in_xls(sumvar_rch, worksheet_2_2)
    #Write Labels
    row = 1
    for var in summary_var_list:
        worksheet_2_1.write(row,0,var)
        worksheet_2_2.write(row,0,var)
        row =row+1
    col =1
    for var in batch_index:
        worksheet_2_1.write(0,col,var)
        worksheet_2_2.write(0,col,var)
        col =col +1
    worksheet_3_0 = workbook.add_worksheet('Train_var')
    col =1
    for lable_name in score_label:
        worksheet_3_0.write(0,col,lable_name)
        col+=1
    # Generate Train varibles in concatenated manner 
    row =1
    col = 1
    for var in sumvar_lch:
        for value in range(len(var)):
            worksheet_3_0.write(row,col,var[value])
            col=col+1

    for var in sumvar_rch:
        for value in range(len(var)):
            worksheet_3_0.write(row,col,var[value])
            col=col+1
    logging.debug("Output file : {}".format(filename))
    workbook.close()


def zero_crossing_rate(m_array):
    """
    Function to get zero crossing rate

    """
    return ((m_array[:-1] * m_array[1:]) < 0).sum()


def split_output_band(ch_output):
    """
    Function to split channel bands
    :param ch_output: array of channel outputs with frequency and dual channel data
    :return: Bands of frequency
    """
    bandone =[];subband_one =[]; bandTwo  =[];bandThree =[];bandFour =[] ;bandFive =[];
    bandSix =[];bandSeven   =[];bandEight =[];bandNine  =[];bandTen  =[];
    
    for i in range(len(ch_output)):
        if ch_output[i,0] <= 200:
            bandone.append( ch_output[i,:] )
        elif ch_output[i,0] >200 and ch_output[i,0] <=   500:
            subband_one.append( ch_output[i,:] )
        elif ch_output[i,0] >500 and ch_output[i,0] <=  1000:
            bandTwo.append(ch_output[i,:])
        elif ch_output[i,0] >1000 and ch_output[i,0]<= 1500:
            bandThree.append ( ch_output[i,:] )
        elif ch_output[i,0] >1500 and ch_output[i,0]<= 2000:
            bandFour.append(ch_output[i,:])
        elif ch_output[i,0] >2000 and ch_output[i,0]<= 2500:
            bandFive.append(ch_output[i,:])
        elif ch_output[i,0] >2500 and ch_output[i,0]<= 3000:
            bandSix.append(ch_output[i,:])
        elif ch_output[i,0] >3000 and ch_output[i,0]<= 3500:
            bandSeven.append(ch_output[i,:])
        elif ch_output[i,0] >3500 and ch_output[i,0]<= 4000:
            bandEight.append(ch_output[i,:])
        elif ch_output[i,0] >4000 and ch_output[i,0]<= 4500:
            bandNine.append(ch_output[i,:])
        elif ch_output[i,0] >4500 and ch_output[i,0]<= 5000:
            bandTen.append(ch_output[i,:])
            
    # Convert list to nd-array    
    band_1 = np.asarray(bandone)
    sub_band_1 = np.asarray(subband_one)
    band_2 = np.asarray(bandTwo)
    band_3 = np.asarray(bandThree)
    band_4 = np.asarray(bandFour)
    band_5 = np.asarray(bandFive)
    band_6 = np.asarray(bandSix)
    band_7 = np.asarray(bandSeven)
    band_8 = np.asarray(bandEight)
    band_9 = np.asarray(bandNine)
    band_10 = np.asarray(bandTen)
    band =[band_1, sub_band_1, band_2,band_3 ,band_4 ,band_5,band_6 , band_7 ,band_8,band_9,band_10]
    return band


def bestFFTlength(n):
    """
    Calculate best FFT length
    :param n: value of the max FFT length
    :return:
    """
    while max(factorint(n))>=n:
        n -= 1
    return n


def extract_wav_fet(wav_file,spectogrpahs=False,output_dir='xlxdirectory'):
    """
    This is the main function to be executed which generates the spectral values of the
    given wav file .
    :param wav_file: Absolute path to the wav file
    :param spectogrpahs: Parameter to generate spectrographs default False
    :param output_dir: Output directory
    :return:
    """
    directory, filename = ntpath.split(wav_file)
    xlxdirectory = os.path.join(directory, output_dir)
    if not os.path.isdir(xlxdirectory):
        os.mkdir(xlxdirectory)
    pathtoXl = os.path.join(xlxdirectory, filename[:-4])
    try:
        sampFreq , data = wavfile.read(wav_file)
        duration = len(data)/float(sampFreq)
        data_type = data.dtype
    except:
        wav_param = wavio.read(wav_file)
        sampFreq = wav_param.rate
        data = wav_param.data
        data_type = wav_param.data.dtype
        duration = (wav_param.data.shape[0]) / float(wav_param.rate)
    if int(duration)==0:
        loggig.log("Error in Processing File Duration {}s".format(duration))
        return

    logging.debug("Attributes before Processing")
    logging.debug("File Data Type:{}".format(data_type))
    logging.debug("File Duration :{}s".format(duration))
    logging.debug("Sampling Frequency: {}".format(sampFreq))
    logging.debug("Channel Length: {}".format(len(data[:,0])))
    logging.debug("Channel shape :{}".format(data.shape))

    start_time = time.time()
    # Determine pitch of signal
    maxFrequencyIndex = np.argmax(data[:,1])
    maxFrequency = maxFrequencyIndex * (sampFreq/2.0)/len(data)

    logging.debug("Normalizing data ..")

    if data_type =='int16':
        data = data / (2. ** 15)  # to normalize the values
    elif data_type =='int32':
        data = wav_param.data / (2. **31)
    else:
        sys.exit("NO rule to handle {}".format(data_type))
    
    # Separation of Channels
    left_channel = data[:,0]
    right_channel = data[:,1]

    logging.warn("Downsampling to 16 KHz . Will be removed in future")
    # Down Sample to 16kHz
    targetsampFreq = 16000
    left_channel = libresample(left_channel,sampFreq,targetsampFreq)
    right_channel = libresample(right_channel,sampFreq,targetsampFreq)

    logging.debug("Channel Length after downsampling:{}".format(len(left_channel)))

    ch_length = len(left_channel)  # calculate the channel length
    data = np.asarray([left_channel,right_channel])

    logging.debug("computing Fourier...")
    # Compute Fast Fourier Transform 
    fourier = np.fft.fft(data,bestFFTlength(ch_length),axis = 1)
    real_fourier = np.absolute(fourier).T

    left_channel_fourier = real_fourier[:,0]
    right_channel_fourier = real_fourier[:,1]
    
    # Get single side band spectrum
    left_channel_fourier_sss = left_channel_fourier[0:(ch_length//2)+1]
    right_channel_fourier_sss = right_channel_fourier[0:(ch_length//2)+1]
    
    # Rationalize for real Power for both channel
    left_channel_fourier_sss[1:-1] = np.asarray(2*left_channel_fourier_sss[1:-1])
    right_channel_fourier_sss[1:-1] = np.asarray(2*right_channel_fourier_sss[1:-1])

    # frequency vector
    f = (np.arange(0,ch_length/2+1)/float(ch_length))*float(targetsampFreq)
    freq = np.asarray(f[:len(right_channel_fourier_sss)])

    # Band splitting
    logging.debug('Spliiting Bands into multiple frequencies')
    ch_output = np.transpose([freq,np.asarray(right_channel_fourier_sss),np.asarray(left_channel_fourier_sss)])
    bands_list = split_output_band(ch_output)
    
    #---------------------------------Summary Variables---------------------------------------------------
    logging.debug("Generating Spectral Features")
    logging.debug("Spectral Centroid")
    logging.debug("Spectral Fatness")
    logging.debug("signal skewness")
    logging.debug("signal Kurtosis")
    logging.debug(" MFCC cooeficients")
    logging.debug("Signal energy")
    # -----------------------------------------------------------------------------------------------------

    spectral_centroid_Lch_list = []
    spectral_centroid_Rch_list = []
    for band in bands_list:
        spectral_centroid_Lch = np.sum(band[:,0]*band[:,1])/np.sum(band[:,1])
        spectral_centroid_Lch_list.append(spectral_centroid_Lch)
    
    for band in bands_list:
        spectral_centroid_Rch = np.sum(band[:,0]*band[:,2])/np.sum(band[:,2])
        spectral_centroid_Rch_list.append(spectral_centroid_Rch)

    spectral_flatness_lch_list =[]
    spectral_flatness_rch_list =[]
    
    for band in bands_list:
        spectral_flatness_lch= spectral_flatness(band[:,1])
        spectral_flatness_lch_list.append(spectral_flatness_lch)

    for band in bands_list:
        spectral_flatness_rch= spectral_flatness(band[:,2])
        spectral_flatness_rch_list.append(spectral_flatness_rch)

    # skewness of a spectrum is the third central moment of this spectrum,
    # divided by the 1.5 power of the second central moment

    spectral_skewness_lch_list = []
    spectral_skewness_rch_list = []
    for band in bands_list:
        spectral_skewness_lch = skew(band[:,1])
        spectral_skewness_lch_list.append(spectral_skewness_lch)
        
    for band in bands_list:
        spectral_skewness_rch = skew(band[:,2])
        spectral_skewness_rch_list.append(spectral_skewness_rch)

    # spectral kurtosis (fourth spectral moment)

    lch_kurtosis_list=[]
    rch_kurtosis_list=[]
    
    for band in bands_list:
        lch_kurtosis = kurtosis(band[:,1])
        lch_kurtosis_list.append(lch_kurtosis)
    
    for band in bands_list:
        rch_kurtosis = kurtosis(band[:,2])
        rch_kurtosis_list.append(rch_kurtosis)   
    
    # MFCC parameters
    mfcc_var_lch_list =[]
    mfcc_var_rch_list=[]
    
    for band in bands_list:
        mfcc_var_lch = np.mean(mfcc(band[:,1],targetsampFreq,nfft=1024))
        mfcc_var_lch_list.append(mfcc_var_lch)

    for band in bands_list:
        mfcc_var_rch = np.mean(mfcc(band[:,2],targetsampFreq,nfft=1024))
        mfcc_var_rch_list.append(mfcc_var_rch)

    energy_lchBsand_list = []
    energy_rchBand_list =[]

    for band in bands_list:
        energy_perBand = np.sum( band[:,1]**2) / len(band[:,1])
        energy_lchBsand_list.append(np.mean(energy_perBand))
        
    for band in bands_list:
        energy_perBand = np.sum( band[:,2]**2) / len(band[:,2])
        energy_rchBand_list.append(energy_perBand)

    # Temporal Variables
    logging.debug ("Temporal information")
    logging.debug ("Computing Signal energy")
    logging.debug ("computing signal Sum")
    logging.debug ("computing signal mean")
    logging.debug ("computing signal standard deviation")
    logging.debug ("computing variation and coeficients of variation")
    logging.debug ("computing top ten Amplitude ")

    Lsum_list = []; Rsum_list =[]; Lmean_list=[];Rmean_list=[];Lsigma_list =[];Rsigma_list=[]
    Lvar_list =[]; Rvar_list=[]; LZCR_list=[];RZCR_list =[];Lcof_var_list=[];Rcof_var_list=[]
    
    for band in bands_list:
        Lsum = np.sum(band[:,1])
        Lmean = np.mean(band[:,1])
        Lsigma= np.std(band[:,1])
        Lvariation= np.var(band[:,1])
        Lcof_var = stats.variation(band[:,1])
        #LZCR = zero_crossing_rate(band[:,1])
        
        Rsum =np.sum(band[:,2])
        Rmean = np.mean(band[:,2])
        Rsigma= np.std(band[:,2])
        Rvariation=np.var(band[:,2])
        Rcof_var = stats.variation(band[:,2])
        #RZCR = zero_crossing_rate(band[:,2])
        
        Lsum_list.append(Lsum)
        Rsum_list.append(Rsum)
        
        Lmean_list.append(Lmean)
        Rmean_list.append(Rmean)
        
        Lsigma_list.append(Lsigma)
        Rsigma_list.append(Rsigma)
        
        Lvar_list.append(Lvariation)
        Rvar_list.append(Rvariation)
        
        Lcof_var_list.append(Lcof_var)
        Rcof_var_list.append(Rcof_var)
    
    # Top ten amplitudes
    top_ten_amp_LC_list =[];top_ten_amp_RC_list=[]
    
    for band in bands_list:
        temp = np.partition(band[:,1] ,len(band[:,1])-1)
        top_ten_amp_LC = np.sum( temp[-10:].reshape(1, -1)[0] )
        top_ten_amp_LC_list.append(top_ten_amp_LC)
    
        temp= np.partition(band[:,2] ,len(band[:,2])-1)
        top_ten_amp_RC = np.sum(temp[-10:].reshape(1, -1)[0])
        top_ten_amp_RC_list.append(top_ten_amp_RC)
    
    summary_var_lch = [Lsum_list,Lmean_list,Lsigma_list,Lvar_list, Lcof_var_list,top_ten_amp_LC_list,
                       spectral_centroid_Lch_list,spectral_flatness_lch_list,spectral_skewness_lch_list,
                       lch_kurtosis_list, mfcc_var_lch_list ,energy_lchBsand_list]
    
    summary_var_rch = [Rsum_list,Rmean_list, Rsigma_list, Rvar_list, Lcof_var_list,top_ten_amp_RC_list,
                       spectral_centroid_Rch_list,spectral_flatness_rch_list,spectral_skewness_rch_list,
                       rch_kurtosis_list, mfcc_var_rch_list,energy_rchBand_list]

    dwt_cof_list = []
    # Write in xls 
    logging.debug ("Feeding Data to the Excel ")
    write_in_xlsx(pathtoXl, bands_list, summary_var_lch, summary_var_rch,dwt_cof_list)
    logging.debug ("Signal_analysis.py elaspsed {} seconds\n".format(time.time()-start_time))

    # Delete the temporary file
    if spectogrpahs:
        logging.debug("Generating Spectographs {}".format(output_file))
        data_directory = utils.makedir(os.path.split(output_file)[0], os.path.split(output_file)[1][:-4])
        spec.Plot_stft(output_file, data_directory)

def main(args):
    wav_file = args.file
    if os.path.isfile(wav_file):
        extract_wav_fet(wav_file,spectogrpahs=args.graphs,output_dir=args.output)
    else:
        sys.exit("File Not found ")





if __name__ == '__main__':
    logfile = debug.log
    logging.basicConfig(filename=logfile,
                        level=logging.DEBUG,
                        filemode='w',
                        format='%(asctime)s : %(message)s',
                        datefmt='%m/%d/%Y :%I:%M: %p')
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    log.addHandler(console)

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Extract features from wav file')
    # Required positional argument
    parser.add_argument('-f,','--file',type=str,default=None, help='wav file path')
    parser.add_argument('-g','--graphs',action='store_true',help='generate Spectographs')
    parser.add_argument('-o','--output',type=str,default='xlxdirectory',help='Name of output dir')
    args = parser.parse_args()
    main(args)
