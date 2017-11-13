# -*- coding: utf-8 -*-
"""
ete011 Enrico Tedeschi @ UiT - Universitet i Tromsø, Faculty of Computer Science.
Longitudinal study on Bitcoin blockchain with BAS - Blockchain Analytics System.
BAS retrieves portion of the Bitcoin blockchain, it saves it into a pandas data frame "D" and it gives you a detailed
insights about fees, tolls, performance and scalability trends of the system.
Usage: observ.py -d number
    -h | --help : usage
    -i          : gives info of the blockchain stored in /info
    -p          : plot data
    -t number   : get the amount of unconfirmed transactions for <number> minutes
    -d j        : retrieve information about transactions and save them in a Panda DataSet, having a
                jump of j blocks with a default number of blocks retrieved b = 10

"""
import sys, getopt
import numpy as np
import string
import re
import os.path
import datetime
import time
import statsmodels.api as sm
import matplotlib.lines as mlines
import matplotlib.axis as ax
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import urllib2
import json
import ast
import urllib
import math
import matplotlib as mpl
import matplotlib.ticker
import calendar
import bisect
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import csv
import javabridge
import subprocess as sub
import requests
import warnings

import retrieval as basretrieve
import plotting as basplot
import data_manipulation as basmanipulation

from lxml import html
from lxml import etree
from scipy.stats import spearmanr
from matplotlib.colors import LogNorm
from blockchain import blockexplorer
from time import sleep
from forex_python.converter import CurrencyRates
from forex_python.bitcoin import BtcConverter
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from scipy import stats
from scipy.stats import norm
from docopt import docopt
from matplotlib.ticker import FormatStrFormatter
from shutil import copyfile
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.metrics import *
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import andrews_curves
from timeout import timeout

# GLOBAL VARIABLES:
global temporary_blocks # file that saves temporary blocks #%todo: replace file_name with temporary_blocks
temporary_blocks = "../dataframe/blocks.txt"
global temporary_transactions  # file for save the temporary transactions
temporary_transactions = "../dataframe/temporary_transactions.txt"  # todo: rename file_tx with temporary_transactions
global latest_block_url
latest_block_url = "https://blockchain.info/latestblock"
global unconfirmed_txs_url
unconfirmed_txs_url = "https://blockchain.info/unconfirmed-transactions?format=json"
global block_hash_url
block_hash_url = "https://blockchain.info/rawblock/"
global dataframe    #todo: rename from txs_dataset to dataframe
dataframe = "../dataframe/D.tsv"
global info_file
info_file = "../info/info.txt"
global plot_directory
plot_directory = "../plot/"

__author__ = "Enrico Tedeschi"
__copyright__ = "Copyright (C) 2017 Enrico Tedeschi"
__license__ = "Public Domain"
__version__ = "0.9 beta"


def main(argv):
    try:
        warnings.filterwarnings("ignore")
        # global plot_number
        # plot_number = 0
        # args_list = sys.argv
        # args_size = len(sys.argv)
        # earliest_hash = get_earliest_hash()
        # start_v = None
        # end_v = None

        opts, args = getopt.getopt(argv, "hipt:d:")
        valid_args = False

        for opt, arg in opts:
            if opt == "-t":   # retrieve unconfirmed transactions
                print basretrieve.fetch_unconfirmed_txs(int(arg))
                valid_args = True
            if opt == "-i":   # blockchain info
                print basmanipulation.get_dataframe_info()
                valid_args = True
            if opt == "-p":
                basplot.plot()
                valid_args = True
            if opt == "-d":
                # fetch transactions
                jump = basretrieve.fetch_txs(int(arg))
                # create dataset
                basretrieve.read_txs_file()
                if os.path.isfile(info_file):
                    os.remove(info_file)
                print basmanipulation.get_dataframe_info(jump)
                # plot dataset
                valid_args = True
            if opt == "-h":  # usage
                print (__doc__)
                valid_args = True
        if valid_args is False:
            print (__doc__)
    except getopt.GetoptError:
        print (__doc__)
        sys.exit(2)


def printProgress (iteration, total, prefix='', suffix='', decimals=1, barLength = 100):
    """
    call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percents = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def progressBar(index_progress_bar, prefix, max_val):
    """
    :param index_progress_bar   :   number which tells where the progress are in the bar
    :param prefix               :   prefix of the progress bar
    :param max_val              :   value to reach to complete the progress bar
    """
    if index_progress_bar == 0:
        printProgress(index_progress_bar, max_val, prefix=prefix, suffix='Complete', barLength=50)
    else:
        sleep(0.01)
        printProgress(index_progress_bar, max_val, prefix=prefix, suffix='Complete',
                      barLength=50)

if __name__ == "__main__":
    main(sys.argv[1:])
