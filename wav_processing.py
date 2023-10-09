"""
Author: Samarth Tandon
Copyright (c) 2018, Docturnal Private Limited.All rights reserved.
Docturnal Private Limited and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from Docturnal Private Limited is strictly prohibited
"""
import os
import signal_analysis
from os.path import expanduser
import argparse
import logging
import warnings
from accumulate_logs import get_accumulated_logs
warnings.filterwarnings("ignore")
home = expanduser("~")

def get_wav(path):
    wav_list = []
    wav_name_lst = []
    for dirname, dirnames, files in os.walk(path):
        for file in files:
            if file.endswith((".WAV",".wav")):
                file_path = os.path.join(dirname,file)
                wav_list.append(file_path)
    return wav_list


def get_xls(xlspath):
    excel_list = []
    excel_name_lst=[]
    for dirname, dirnames, files in os.walk(xlspath):
        for file in files:
            if file.endswith((".xlsx" ,".xls")):
                excel_list.append(os.path.join(dirname,file))
    for excel_name in excel_list:
        if excel_name.endswith('.xlsx'):
            excel_name_lst.append(os.path.split(excel_name)[1][:-5])
    return excel_list,excel_name_lst


def main(args):
    logging.info(home)
    try :
        logging.info(os.uname())
    except :
        logging.debug("Running on windows machine")

    if os.path.isdir(args.path):
        logging.info("Reading wav files from:{}".format(args.path))
        wavfiles_list = get_wav(args.path)

        if os.path.isdir(os.path.join(args.path,args.output)):
            xl_lst, xls_names = get_xls(os.path.join(args.path,args.output))
            unprocessed_files = [wav_file for wav_file in wavfiles_list
                                 if not os.path.split(wav_file)[1][:-4] in xls_names]
        else:
            unprocessed_files = wavfiles_list

        logging.info("Number of files un-processed {}".format(len(unprocessed_files)))
        if len(unprocessed_files) > 0:
            unprocessed_files.sort()
            for wav_file in unprocessed_files:
                signal_analysis.extract_wav_fet(wav_file,output_dir=args.output,debug=args.debug,graphs=args.spectogrpahs)

            logging.info("Accumulating training features in csv")
            get_accumulated_logs(os.path.join(args.path,args.output),debug=args.debug,save_csv=args.debug)
    else:
        logging.error("Path not found {}".format(args.path))
        raise Exception("Unable to reach path provided")
    return 0


if __name__ == '__main__':
    logfile = 'debug.log'
    logging.basicConfig(filename=logfile,
                        level=logging.DEBUG,
                        filemode='a',
                        format='%(asctime)s : %(message)s',
                        datefmt='%m/%d/%Y :%I:%M: %p')
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    # Instantiate the parser

    parser = argparse.ArgumentParser(description='Generate spectograph of the wav files and augument them')
    # Required argument
    parser.add_argument('-p','--path',type=str,required=True, help='Path of root folder containing all the wav files')
    parser.add_argument('-g','--spectogrpahs',action='store_true',help='Generate spectogrpahs')
    parser.add_argument('-o','--output',type=str,default='xlxdirectory',help='output directory default=xlxdirectory')
    parser.add_argument('-d','--debug',action='store_true',help='Debug wav files')
    args = parser.parse_args()
    main(args)
