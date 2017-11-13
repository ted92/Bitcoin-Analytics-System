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

import retrieval as basretrieve
import plotting as basplot
import main as basmain

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


def get_dataframe_info(jump=0):
    """
    get all the info regarding D
    """
    # check first if info file already exists
    if os.path.isfile(info_file):
        with io.FileIO(info_file, "r") as file:
            file.seek(0)
            return_string = file.read()
    else:
        number_txs, number_blocks, start_date, end_date, jump = get_info(jump)
        return_string = bcolors.HEADER + "=====================================================\n" \
                        + "Dataframe D:" + bcolors.ENDC + "\n\nEvaluation in between:\n" \
                        + bcolors.BOLD + bcolors.OKGREEN + str(start_date) + "\n" + str(end_date) + "\n\n" \
                        + bcolors.ENDC + "Transaction evaluated:\n" + bcolors.BOLD + bcolors.OKGREEN + str(number_txs) \
                        + bcolors.ENDC + "\n\n" "Blocks evaluated:\n" + bcolors.BOLD + bcolors.OKGREEN \
                        + str(number_blocks) + bcolors.ENDC + "\n\n" + "jump J:" \
                        + bcolors.BOLD + bcolors.OKGREEN + "\n" \
                        + str(jump) + "\n" + bcolors.ENDC + bcolors.HEADER \
                        + "=====================================================\n" + bcolors.ENDC

    with io.FileIO(info_file, "w") as file:
        file.write(return_string)
    return return_string


def calculate_profit(fee, creation_time, price, bitcoin_hash_rate, miner, number_of_miners=1):
    """
    return the profit of a single block, knowing the creation time and the fee paid from the transactions.
    We consider 2 type of miners
    1 - Hashing power of 100 MH/s and consumption of 6.8 Watt
    2 - Hashing power of 25200 MH/s and consumption of 1250 Watt
    :param fee: M money from transactions, in BTC
    :param creation_time: time to mine a block in seconds
    :param bitcoin_hash_rate: hashing rate of Bitcoin
    :param miner: type of miner for the evaluation
    :param number_of_miners: how many miners need to be considered
    :param price: current BTC price
    :return: profit in BTC
    """

    profit = 0
    reward = 12.5
    propagation_time = 15.7

    if miner == 1:  # AntMiner U3
        label = "AntMiner U3"
        btc_x_hour = 0.063 * 0.04 / price
        btc_x_sec = btc_x_hour / (60 * 60)
        hashing_rate = 1000000000.0
        cost_x_hash = btc_x_sec / hashing_rate
    elif miner == 3:  # AntPool S9- Biggest miners
        label = "AntMiner S9"
        btc_x_hour = 1.375 * 0.04 / price
        btc_x_sec = btc_x_hour /(60*60)
        hashing_rate = 14000000000000.0
        cost_x_hash = btc_x_sec/hashing_rate

    # cost_x_hash = cost_x_hash / price   # in BTC
    p_orphan = 1 - math.exp(-(propagation_time/creation_time))
    cost = cost_x_hash * hashing_rate * number_of_miners * creation_time
    revenue = (reward + fee) * (hashing_rate*number_of_miners/bitcoin_hash_rate) * (1 - p_orphan)

    profit = revenue - cost

    return profit, revenue, cost, label


def get_info(jump=0):
    """
    gets info about the number of blocks, transactions and the date of retrieval considering the Jump used
    :return: number of transactions, number of blocks, start date, end date
    :return:
    """
    number_txs = ''
    number_blocks = ''
    start_date = ''
    end_date = ''
    if os.path.isfile(dataframe):
        df = get_dataframe()
        df = df[['B_ep', 't_f']]
        number_txs = len(df['B_ep'].values)
        df = df.groupby('B_ep').mean().reset_index()
        number_blocks = len(df['B_ep'].values)
        epoch_list = df['B_ep'].values
        start_date = epoch_datetime(epoch_list[0])
        end_date = epoch_datetime(epoch_list[-1])
    return number_txs, number_blocks, start_date, end_date, jump


def get_dataframe():
    """
    read from all the dataframe# files and merge them to have an unique structure
    :return: new_df  dataframe
    """
    i = 0
    old_df = None
    new_df = None
    while True:
        # if the file exists
        df_name = "../dataframe/D"+str(i)+".tsv"
        if os.path.isfile(df_name):
            df = pd.DataFrame.from_csv(df_name, sep='\t')
            new_df = pd.concat([old_df, df])
            old_df = new_df
        elif i == 0:
            # dataframe without any number as suffix
            new_df = pd.DataFrame.from_csv(dataframe, sep='\t')
            break
        else:
            break
        i += 1
    return new_df


def polynomial_interpolation(description, x, y, degree=2):
    """
    given two lists of data it generates two new lists containing the functions interpolated
    :param  description :   description of the function
    :param  x           :   x values of the data to interpolate
    :param  y           :   y values of the data to interpolate
    :param  degree      : degree of the function to get
    :return             : x and y values to be plotted. Interpolated values. f is the function to write in the plot.
    """
    # order lists
    together = zip(x, y)
    sorted_together = sorted(together)

    x_vals = [el[0] for el in sorted_together]
    y_vals = [el[1] for el in sorted_together]

    # calculate polynomial
    z = np.polyfit(x_vals, y_vals, degree)
    f = np.poly1d(z)

    print description + ": "
    print f
    print "\n"

    x_new = np.linspace(x_vals[0], x_vals[-1], len(x_vals))
    y_new = f(x_new)

    return x_new, y_new, f


def satoshi_bitcoin(sat):
    """
    get a value in satoshi, convert it in BTC
    :param sat:
    :return:
    """
    bitcoin = float(sat) / 100000000
    return bitcoin


def byte_megabyte(b):
    """
    get a value in byte, convert in megabyte
    :param b:
    :return:
    """

    megabyte = float(b) / 1000000
    return megabyte


def sec_minutes(s):
    """
    get a value in seconds convert it in minutes
    :param s:
    :return:
    """

    min = float(s) / 60
    return min


def sec_hours(s):
    """
    get a value in seconds convert it in hours
    :param s:
    :return:
    """

    h = float(s) / (60 * 60)
    return h


def epoch_date_dd(df):
    """
    get a df with a column of epoch 'B_ep', returns
    another column with the date yyyy-mm-dd so it
    orders the date by day
    :param df:  dataframe in input
    :return:    new dataframe containing the 'date' attribute
    """
    df['date'] = df['B_ep'].apply(epoch_datetime)
    df['date'] = df['date'].apply(revert_date_time)

    df['date'] = df['date'].str.slice(start=0, stop=10)
    return df


def revert_date_time(t):
    """
    given a format dd-mm-yyyy return yyyy:mm:dd
    :param t:
    :return:
    """

    return datetime.datetime.strptime(t, '%d-%m-%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')


def epoch_date_mm(df):
    """
    get a df with a column of epoch, returns another column with the date yyyy-mm so it orders the date by month
    :param df:
    :return:
    """
    df['date'] = df['B_ep'].apply(epoch_datetime)
    df['date'] = df['date'].apply(revert_date_time)

    df['date'] = df['date'].str.slice(start=0, stop=7)

    return df


def epoch_date_yy(df):
    """
    get a df with a column of epoch, returns another column with the date yyyy so it orders the date by year
    :param df:
    :return:
    """
    df['date'] = df['B_ep'].apply(epoch_datetime)
    df['date'] = df['date'].apply(revert_date_time)

    df['date'] = df['date'].str.slice(start=0, stop=4)

    return df


def reward_intervals(h):
    """

    :param h: height of a blovk
    :return: r reward of the block at height h
    """
    r = 0
    if int(h) < 210000:
        r = 50
    elif(int(h) >= 210000) and (int(h) < 420000):
        r = 25
    elif(int(h) >= 420000) and (int(h) < 840000):
        r = 12.5
    return r


def block_creation_time_intervals(t):
    """

    :param t: block creation time in minutes
    :return:
    """
    if t <= 0:
        t = "0"
    elif(t > 0) and (t < 2):
        t = "<02"
    elif(t >= 2) and (t < 4):
        t = "<04"
    elif(t >= 4) and (t < 5):
        t = "<05"
    elif(t >= 5) and (t < 8):
        t = "<08"
    elif(t >= 8) and (t < 10):
        t = "<10"
    elif(t >= 10) and (t < 12):
        t = "<12"
    elif(t >= 12) and (t < 15):
        t = "<15"
    elif(t >= 15) and (t < 20):
        t = "<20"
    elif(t >= 20) and (t < 40):
        t = "<40"
    else:
        t = ">40"
    return t


def block_size_intervals(q):
    """

    :param q: block size
    :return: string with its interval
    """
    if q <= 0:
        q = "0"
    elif (q > 0.0) and (q < 0.003):
        q = "<0.003"
    elif (q >= 0.003) and (q < 0.05):
        q = "<0.05"
    elif (q >= 0.05) and (q < 0.1):
        q = "<0.1"
    elif (q >= 0.1) and (q < 0.3):
        q = "<0.3"
    elif (q >= 0.3) and (q < 0.5):
        q = "<0.5"
    elif (q >= 0.5) and (q < 0.8):
        q = "<0.8"
    elif (q >= 0.8) and (q < 1):
        q = "<1"
    else:
        q = ">1"
    return q


def fee_more_intervals(fee):
    """

    :param fee:
    :return:
    """
    if fee <= 0.0001:
        # category 1 --> 0
        fee = "0"
    elif (fee > 0.0001) and (fee < 0.0002):
        # category 2 --> 0.0001
        fee = "<0.0002"
    elif (fee >= 0.0002) and (fee < 0.0004):
        # category 3 --> 0.0002
        fee = "<0.0004"
    elif (fee >= 0.0004) and (fee < 0.0006):
        # category 4 --> 0.001
        fee = "<0.0006"
    elif (fee >= 0.0006) and (fee < 0.0008):
        # category 5 --> 0.01
        fee = "<0.0008"
    elif (fee >= 0.0008) and (fee < 0.001):
        # category 5 --> 0.01
        fee = "<0.001"
    elif (fee >= 0.001) and (fee < 0.01):
        # category 5 --> 0.01
        fee = "<0.01"
    elif (fee >= 0.01) and (fee < 0.1):
        # category 5 --> 0.01
        fee = "<0.1"
    else:
        # category 6 --> >0.01
        fee = ">0.1"

    return fee


def fee_intervals(fee):
    """

    :param fee:
    :return:    the fee to be inserted in a df, this fee is a category in which the previous numerical
    fee is in, e.g. 0.00023 will be in 0.0005 category
    """

    if fee < 0.0001:
        # category 1 --> 0
        fee = "0"
    elif(fee >= 0.0001) and (fee < 0.0002):
        # category 2 --> 0.0001
        fee = ">0.0001"
    elif(fee >= 0.0002) and (fee < 0.0005):
        # category 3 --> 0.0002
        fee = ">0.0002"
    elif(fee >= 0.0005) and (fee < 0.001):
        # category 4 --> 0.001
        fee = ">0.0005"
    elif(fee >= 0.001) and (fee < 0.01):
        # category 5 --> 0.01
        fee = ">0.001"
    else:
        # category 6 --> >0.01
        fee = ">0.01"

    return fee


def fee_density_intervals(fee):
    """

    :param fee:
    :return:    the fee density to be inserted in a df, this fee is a category in which the previous numerical fee
    is in, e.g. 0.00023 will be in >0.0002 category
    """

    if fee <= 0:
        # category 1 --> 0
        fee = "0"
    elif (fee > 0) and (fee < 50):
        # category 2 --> 0.0001
        fee = "<50"
    elif (fee >= 50) and (fee < 100):
        # category 3 --> 0.0002
        fee = "<100"
    elif(fee >= 100) and (fee < 200):
        # category 4 --> 0.001
        fee = "<200"
    elif(fee >= 200) and (fee < 300):
        # category 5 --> 0.01
        fee = "<300"
    else:
        # category 6 --> >0.01
        fee = ">300"

    return fee


def calculate_percentage_txs_fee(df):
    """

    :return: array with the % of fee paid on the net transaction
    """
    # calculate % of fee considering the net output:

    output = df['t_ou'].values
    fee = df['t_f'].values

    percentage_output = []

    for o, f in zip(output, fee):
        if o == 0:
            o = 0.000001
        p = (float(f) * 100) / float(o)
        percentage_output.append(p)

    return percentage_output


def remove_minor_miners(df, number=7):
    """

    :param  df: dataframe containing miners
    :param number: number of miners to consider
    :return: new dataframe without the rows mined from these minor miners
    """
    miners = df['B_mi'].value_counts()
    miners = miners.head(number)

    # remove all the other miners
    df = df.loc[df['B_mi'].isin(miners.index)]

    # remove miners

    return df


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


def epoch_datetime(epoch):
    """
    convert epoch to datetime %Y-%m-%d %H:%M:%S
    :param epoch: time in epoch
    :return: time in datetime with %Y-%m-%d %H:%M:%S format
    """
    datetime = time.strftime('%d-%m-%Y %H:%M:%S', time.localtime(float(epoch)))
    return datetime


def miner_profit_function(x):
    a = (1.896/(10**8))
    b = (5.997/(10**7))
    c = (3.831/(10**5))

    y = -a*(x**2) + b*x + c
    return y


def feedensity_latency_function(x):
    a = (5.416 / (10 ** 8))
    b = (2.215 / (10 ** 3))
    c = (1.598)

    y = a*(x**2) - b*x + c
    return y


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
