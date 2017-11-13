"""
Retrieval file containing all the methods for the blockchain data retrieval.
Get informations about blocks, transactions, miners and BTC price.
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

import main as basmain
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


def fetch_txs(jump):
    """
    Organize the files and prepare the retrieval of transaction
    :return:
    """
    blocks_to_retrieve = 10
    # remove the blockchain file
    if os.path.isfile(temporary_blocks):
        os.remove(temporary_blocks)

    # remove transactions file
    if os.path.isfile(temporary_transactions):
        os.remove(temporary_transactions)

    if os.path.isfile(dataframe):
        # file already exists add data to the dataset
        # get the latest block hash
        # find jump in the info file
        found = False
        with open(info_file, "r") as fileinfo:
            for line in fileinfo:
                if found is False:
                    if line.startswith('jump'):
                        found = True
                else:
                    jump = re.findall('\d+', line)
                    break
        jump = int(jump[0])

        df = pd.DataFrame.from_csv(dataframe, sep='\t')
        height_list = df['B_he'].values
        last_block = height_list[-1]
        last_block = int(last_block) - jump

        b_array = get_json_request("https://blockchain.info/block-height/" + str(last_block) + "?format=json")
        blocks = b_array['blocks']
        b = blocks[0]
        block_hash = b['hash']
        get_blockchain(blocks_to_retrieve, block_hash)

    else:
        # file doesn't exist
        # retrieve the last block hash
        latest_block = get_json_request(latest_block_url)
        block_hash = latest_block['hash']
        get_blockchain(blocks_to_retrieve, block_hash)

    return jump


@timeout(360)
def get_blockchain(number_of_blocks, hash):
    """
    it retreives number_of_blocks blocks from the Bitcoin blockchain, given an hash where to start.

    :param number_of_blocks: int, blocks to retrieve
    :param hash: str, hash of the block from where start the retrieval
    :return: None
    """
    fetch_time_list = []        # time to fetch each block
    epoch_list = []             # block epoch
    creation_time_list = []     # block creation time
    fee_list = []               # block total revenue
    hash_list = []              # block hash
    size_list = []              # block size Q
    height_list = []            # block height
    bandwidth_list = []         # bandwidth for block retrieval
    avg_transaction_list = []   # average transaction latency in a block
    list_transactions = []      # number of transactions in the block
    list_miners = []            # block miner

    # -------- PROGRESS BAR -----------
    index_progress_bar = 0
    prefix = 'Saving Blockchain:'
    basmain.progressBar(index_progress_bar, prefix, number_of_blocks)
    # ---------------------------------

    # ================== RETRIEVE BLOCKS ==================
    # retrieve blocks using json data from blockchain.info API

    current_block = get_json_request(block_hash_url + hash)

    hash = current_block['prev_block']
    start_time = datetime.datetime.now()
    current_block = get_json_request(block_hash_url + hash)
    end_time = datetime.datetime.now()

    for i in range(number_of_blocks):
        # ---------- PROGRESS BAR -----------
        index_progress_bar += 1
        basmain.progressBar(index_progress_bar, prefix, number_of_blocks)
        # -----------------------------------

        time_to_fetch = end_time - start_time
        time_in_seconds = get_time_in_seconds(time_to_fetch)
        fetch_time_list.append(time_in_seconds)

        epoch = current_block['time']
        epoch_list.append(epoch)

        hash = current_block['hash']
        hash_list.append(hash)

        fee = current_block['fee']
        fee_list.append(fee)

        size = current_block['size']
        size_list.append(size)

        height = current_block['height']
        height_list.append(height)

        avg_tr = get_avg_transaction_time(current_block)
        avg_transaction_list.append(avg_tr)

        block_size = float(size) / 1000000  # -------> calculate read Bandwidth with MB/s
        bandwidth = block_size / time_in_seconds
        bandwidth_list.append(bandwidth)

        transactions = len(current_block['tx'])
        list_transactions.append(transactions)

        # transaction writes
        txs = current_block['tx']
        # write transactions in file transactions.txt
        with io.FileIO(temporary_transactions, "a+") as file:
            file.write(str(txs))
            file.write("\n" + str(current_block['time']) + "\n")

        hash_prev_block = current_block['prev_block']

        start_time = datetime.datetime.now()  # ------------------------------------------------------------------------
        prev_block = get_json_request("https://blockchain.info/block-index/" + str(hash_prev_block) + "?format=json")
        end_time = datetime.datetime.now()  # --------------------------------------------------------------------------

        prev_epoch_time = prev_block['time']
        current_creation_time = int(current_block['time']) - int(prev_epoch_time)
        creation_time_list.append(current_creation_time)

        miner = web_crawler(hash)
        list_miners.append(miner)

        current_block = prev_block

    to_write_list = [hash_list, epoch_list, creation_time_list, size_list, fee_list, height_list, bandwidth_list,
                     list_transactions, avg_transaction_list, list_miners]

    # writing all the data retrieved in the file
    write_blockchain(to_write_list)


def write_blockchain(to_write_list):
    """
    :param to_write_list      - Required: list to_write_list: it contains all the lists that need to be written:
        [0] hash: hash list
        [1] epoch: epoch list
        [2] creation_time: creation time list
        [3] size: size list
        [4] fee: fee list
        [5] height: height list
        [6] bandwidth: bandwidth list
        [7] transactions: number of transactions in every block list
        [8] avg_tr_list: list with the average time that a transaction need to be visible in the blockchain in a certain block
        [9] list_miner: list with all the miners for each block
    """
    n = len(to_write_list[0])
    # ---------- PROGRESS BAR -----------
    index_progress_bar = 0
    prefix = 'Writing .txt file:'
    basmain.progressBar(index_progress_bar, prefix, n)
    # -----------------------------------

    with io.FileIO(temporary_blocks, "a+") as file:
        for i in range(n):
            # --- WRITE IN FILE ---
            write_file(to_write_list, file, i)
            # ---------------------

            # -------- PROGRESS BAR -----------
            index_progress_bar += 1
            basmain.progressBar(index_progress_bar, prefix, n)
            # ---------------------------------


def write_file(list_to_write, file, index):
    """
    write the list_to_write in the file
    :param list_to_write: list containing all the other list that need to be written in the blockchain file
    :param file: open file used in write_blockchain() method
    :param index: index of which element needs to be written
    :return: none
    """
    file.write("hash: " + str(list_to_write[0][index]) + "\nepoch: " + str(list_to_write[1][index]) + "\ncreation_time: " + str(
        list_to_write[2][index]) + "\nsize: " + str(list_to_write[3][index]) + "\nfee: " + str(
        list_to_write[4][index]) + "\nheight: " + str(list_to_write[5][index]) + "\nbandwidth: " + str(
        list_to_write[6][index]) + "\ntransactions: " + str(list_to_write[7][index]) + "\navgttime: " + str(
        list_to_write[8][index]) + "\nmined_by: " + str(list_to_write[9][index]) + "\n\n")


def read_txs_file():
    """
    read the txs file and generates the dataset D
    :return:
    """
    epoch_list = []
    fee_list = []
    size_list = []
    approval_time_list = []
    input = []
    output = []
    hash_tx = []

    # the file transactions.txt contains the transactions in each block and the epoch for that block at the end
    if os.path.isfile(temporary_transactions):
        with io.FileIO(temporary_transactions, "r") as file:
            file.seek(0)
            txs = file.read()

        list_txs = txs.split("\n")
        list_txs.pop()

        # -------- PROGRESS BAR -----------
        index_progress_bar = 0
        prefix = 'Reading ' + temporary_transactions + ':'
        basmain.progressBar(index_progress_bar, 'Reading ' + temporary_transactions + ':', (2 * len(list_txs)) + (len(epoch_list)))
        # ---------------------------------

        # delete the epoch from the list just retrieved
        i = 0
        for el in list_txs:
            epoch_list.append(list_txs[i + 1])
            list_txs.remove(list_txs[i + 1])
            i += 1
            # ---------- PROGRESS BAR -----------
            index_progress_bar += 1
            basmain.progressBar(index_progress_bar, prefix, (2 * len(list_txs)) + (len(epoch_list)))
            # -----------------------------------

        i = 0
        for t in list_txs:
            list_txs[i] = ast.literal_eval(t)
            i += 1
            # ---------- PROGRESS BAR -----------
            index_progress_bar += 1
            basmain.progressBar(index_progress_bar, prefix, (2 * len(list_txs)) + (len(epoch_list)))
            # -----------------------------------

        for i in range(len(epoch_list)):
            # ---------- PROGRESS BAR -----------
            index_progress_bar += 1
            basmain.progressBar(index_progress_bar, prefix, (2 * len(list_txs)) + (len(epoch_list)))
            # -----------------------------------

            list_txs[i].pop(0)  # remove the first transaction of each block since it is only the reward
            temp_input, temp_output, temp_fee_list, temp_size_list, temp_approval_time_list, temp_hash_tx = \
                calculate_transactions_fee(list_txs[i], int(epoch_list[i]))
            input.extend(temp_input)
            output.extend(temp_output)
            fee_list.extend(temp_fee_list)
            size_list.extend(temp_size_list)
            approval_time_list.extend(temp_approval_time_list)
            hash_tx.extend(temp_hash_tx)

        # ---- CALCULATE % OF FEE ----
        f_percentile = []
        for f_in, f_ou in zip(input, output):
            if float(f_in)!= 0:
                percentile = 100 - (float(f_ou * 100) / float(f_in))
            else:
                percentile = 0
            f_percentile.append(percentile)
            # ----------------------------

        transactions_list = get_list_from_file('transactions')
        transactions_list[:] = [int(x) for x in transactions_list]
        transactions_list[:] = [x - 1 for x in transactions_list]

        indexes_list = []
        val = 0
        for x in transactions_list:
            val = x + val
            indexes_list.append(val)

        block_size = get_list_from_file('size')
        block_creation_time = get_list_from_file('creation_time')
        block_height = get_list_from_file('height')
        block_epoch = get_list_from_file('epoch')
        block_txs = get_list_from_file('transactions')
        block_hash = get_list_from_file('hash')
        block_relayedby = get_list_from_file('mined_by')

        b_s = []
        b_ct = []
        b_h = []
        b_ep = []
        b_t = []
        b_hash = []
        b_rel = []

        i = 0
        counter = 0
        for tx in input:
            if i < indexes_list[counter]:
                b_s.append(block_size[counter])
                b_ct.append(block_creation_time[counter])
                b_h.append(block_height[counter])
                b_ep.append(block_epoch[counter])
                b_t.append(block_txs[counter])
                b_hash.append(block_hash[counter])
                b_rel.append(block_relayedby[counter])
                i += 1
            else:
                counter += 1
                b_s.append(block_size[counter])
                b_ct.append(block_creation_time[counter])
                b_h.append(block_height[counter])
                b_ep.append(block_epoch[counter])
                b_t.append(block_txs[counter])
                b_hash.append(block_hash[counter])
                b_rel.append(block_relayedby[counter])
                i += 1

        if os.path.isfile(dataframe):
            # file exists
            # read the old file:
            old_df = pd.DataFrame.from_csv(dataframe, sep='\t')

            # create the new one
            new_df = pd.DataFrame.from_items(
                [('t_ha', hash_tx), ('t_in', input), ('t_ou', output), ('t_f', fee_list), ('t_q', size_list),
                 ('t_%', f_percentile), ('t_l', approval_time_list),
                 ('Q', b_s), ('B_T', b_ct), ('B_he', b_h), ('B_ep', b_ep), ('B_t', b_t),
                 ('B_h', b_hash), ('B_mi', b_rel)])

            # merge old and new
            new_df = pd.concat([old_df, new_df])

        else:
            # file doesn't exist
            new_df = pd.DataFrame.from_items(
                [('t_ha', hash_tx), ('t_in', input), ('t_ou', output), ('t_f', fee_list), ('t_q', size_list),
                 ('t_%', f_percentile), ('t_l', approval_time_list),
                 ('Q', b_s), ('B_T', b_ct), ('B_he', b_h), ('B_ep', b_ep), ('B_t', b_t),
                 ('B_h', b_hash), ('B_mi', b_rel)])

        new_df.to_csv(dataframe, sep='\t')


def fetch_unconfirmed_txs(m):
    """
    for m minutes observe the trnasactions coming in the Bitcoin network and calculate
    the requested throughput.
    :param m: minutes
    :return: string to print with the requested throughput
    """
    print "Analyzing new transactions for " + str(m) + " minutes."
    t_end = time.time() + 60 * m
    # -------- PROGRESS BAR -----------
    total = 60 * m
    prefix = 'Fetching unconfirmed txs:'
    basmain.progressBar(0, prefix, total)
    # ---------------------------------
    list_unconfirmed_txs = []
    while time.time() < t_end:
        # -------- PROGRESS BAR -----------
        index = t_end - time.time()
        index = total - index
        basmain.progressBar(index, prefix, total)
        # ---------------------------------
        unconfirmed_txs = get_json_request(unconfirmed_txs_url)
        for el in unconfirmed_txs[u'txs']:
            if not el['hash'] in list_unconfirmed_txs:
                list_unconfirmed_txs.append(el['hash'])

    str_return = "\n" + str(len(list_unconfirmed_txs)) + " transactions requested approval in " + str(m) +\
                 " minutes.\nRequested throughput: " + str(float(len(list_unconfirmed_txs)) / float((m*60)))\
                 + " txs/sec"
    return str_return


def calculate_transactions_fee(txs, epoch=None):
    """
    given a json list of transactions, it produces the input and output fee list for each transaction in txs, plus the
    size list of each transation
    :param txs: list of transactions in json format
    :param epoch: Optional if the transaction has been approved it represents the epoch of the block in which
    this transaction is
    :return: input, output, fee, size, approval time list, txs hash list
    """
    # calculate total fee for each unconfirmed transaction
    input_fee = 0
    output_fee = 0

    in_list = []
    out_list = []
    fees_list = []
    sizes_list = []

    list_hashes_checked = []
    approval_time_list = []

    i = 0
    for tx in txs:
        try:
            sizes_list.append(tx['size'])
            # consider a transaction only one time
            if tx['hash'] in list_hashes_checked:
                pass
            else:
                list_hashes_checked.append(tx['hash'])
                # ===================================== GET THE TOTAL INPUT FEE ==============
                for input in tx['inputs']:
                    prev_out = input[u'prev_out']
                    input_fee += int(prev_out[u'value'])

                    # print "INPUT: " + str(prev_out[u'value'])
                in_list.append(input_fee)
                # ============================================================================

                # ===================================== GET THE TOTAL OUTPUT FEE ==============
                for output in tx['out']:
                    # print "OUTPUT: " + str(output[u'value'])

                    output_fee += int(output[u'value'])
                out_list.append(output_fee)
                # ============================================================================
                fees_list.append(float(input_fee) - float(output_fee))
                # print "FEE: " + str(float(input_fee) - float(output_fee))
                # print "APPROVAL TIME: " + str(approval_time) + "\n"
            input_fee = 0
            output_fee = 0
            # if the transactions are already approved -- calculate the approval time
            if epoch != None:
                epoch_tx = tx['time']

                approval_time = float(epoch) - float(epoch_tx)
                approval_time_list.append(approval_time)

        except KeyError as e:
            print e
            pass
    return in_list, out_list, fees_list, sizes_list, approval_time_list, list_hashes_checked


def web_crawler(hash):
    """
    parse blockchain.info to get information about miners
    :param hash: hash of the block
    :return: str miner
    """
    # ----- get the miner from the parsing of the webpage
    page = requests.get('https://blockchain.info/block/' + hash)
    tree = html.fromstring(page.content)

    string = page.content
    index_start = string.find("Relayed By")

    string = string[index_start:(index_start + 150)]
    miner = find_between(string, '">', '</a>')
    return miner


def get_list_from_file(attribute):
    """
        return a list of "attribute" values for all the blocks in blockchain_new.txt

        :param str attribute: it could be every attribute of a block such as "size", "epoch", "hash" ...
        :return: a list containing the attribute for all the blocks

     """

    list_to_return = []

    if os.path.isfile(temporary_blocks):
        # open the file and read in it
        with open(temporary_blocks, "r") as blockchain_file:
            for line in blockchain_file:
                # regular expression that puts in a list the line just read: ['hash', '<value>']
                list = line.split(": ")
                list[-1] = list[-1].strip()
                # list = re.findall(r"[\w']+", line) # old regex
                if list and (list[0] == attribute):
                    list_to_return.append(list[1])
                    # print list[0] + " " + list[1]
        return list_to_return
    else:
        return False


def get_avg_transaction_time(block):
    """
    get the average time, per block, of the time that a transaction
    take to be visible in the blockchain after it has been requested.

    :param block: the block to be analized
    :return: int: return the average of the time of all the transactions in the block
    """

    block_time = float(block['time'])
    tx = block['tx']

    t_sum = 0
    for t in tx:
        approval_time = block_time - int(t['time'])
        t_sum = t_sum + approval_time

    average_per_block = t_sum / len(tx)
    return average_per_block


def get_time_in_seconds(time_to_fetch):
    """
    from time with format %H%M%S given in input return time in seconds
    :param time: time with format %H%M%S
    :return: time in seconds
    """
    # -------- TIME CONVERSION IN SECONDS ---------
    x = time.strptime(str(time_to_fetch).split('.')[0], '%H:%M:%S')
    # get the milliseconds to add at the time in second
    millisec = str(time_to_fetch).split('.')[1]
    millisec = "0." + millisec
    # get the time in seconds
    time_to_return = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
    time_to_return = float(time_to_return) + float(millisec)
    return time_to_return


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def get_json_request(url):
    """
    Read the url and load it with json data.
    :param url: str, url where to get the json data
    :return: str, data requested in json format
    """
    json_req = urllib2.urlopen(url).read()
    request = json.loads(json_req)

    return request

