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
import main as basmain
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
global dataframe    # todo: rename from txs_dataset to dataframe
dataframe = "../dataframe/D.tsv"
global info_file
info_file = "../info/info.txt"
global plot_directory
plot_directory = "../plot/"

__author__ = "Enrico Tedeschi"
__copyright__ = "Copyright (C) 2017 Enrico Tedeschi"
__license__ = "Public Domain"
__version__ = "0.9 beta"


def initialize_plt():
    """
    initialize the plt environment
    """
    plt.clf()
    plt.figure(0)
    axes = plt.gca()
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')
    return axes


def plot():
    df = basmanipulation.get_dataframe()
    axes = initialize_plt()
    print plot_reward_fee.__doc__
    plot_reward_fee(df, axes)

    axes = initialize_plt()
    print plot_profit_multiple_miners.__doc__
    plot_profit_multiple_miners(df, axes)

    axes = initialize_plt()
    print plot_profit_creation_time.__doc__
    plot_profit_creation_time(df, axes)

    axes = initialize_plt()
    print plot_total_btc.__doc__
    plot_total_btc(df, axes)

    axes = initialize_plt()
    print plot_fee_input_miners.__doc__
    plot_fee_input_miners(df, axes)

    axes = initialize_plt()
    print plot_fee_latency.__doc__
    plot_fee_latency(df, axes)

    # axes = initialize_plt()
    # print plot_txs_fee_distribution.__doc__
    # plot_txs_fee_distribution(df, axes)

    # axes = initialize_plt()
    # print plot_txs_feedensity_distribution.__doc__
    # plot_txs_feedensity_distribution(df, axes)

    axes = initialize_plt()
    print plot_fee_latency_years.__doc__
    plot_fee_latency_years(df, axes)

    axes = initialize_plt()
    print plot_blocksize_latency.__doc__
    plot_blocksize_latency(df, axes)

    axes = initialize_plt()
    print plot_throughput.__doc__
    plot_throughput(df, axes)

    axes = initialize_plt()
    print plot_creation_time_miners.__doc__
    plot_creation_time_miners(df, axes)

    axes = initialize_plt()
    print plot_block_size.__doc__
    plot_block_size(df, axes)

    # axes = initialize_plt()
    # print plot_top_miners_monthly.__doc__
    # plot_top_miners_monthly(df, axes)

    axes = initialize_plt()
    print plot_trendy_miners.__doc__
    plot_trendy_miners(df, axes)

    axes = initialize_plt()
    print plot_number_of_miners.__doc__
    plot_number_of_miners(df, axes)


def plot_reward_fee(df, axes):
    """
    Plot the revenue from the block reward R compared to the fee from users M
    """
    info = "reward_fee"
    new_df = df[['B_he', 't_f', 'B_ep']]
    new_df = new_df.groupby(['B_ep', 'B_he']).sum().reset_index()
    new_df = basmanipulation.epoch_date_dd(new_df)
    del new_df["B_ep"]
    new_df['reward'] = new_df['B_he'].apply(basmanipulation.reward_intervals)
    del new_df['B_he']
    new_df = new_df.groupby('date').sum().reset_index()
    new_df['t_f'] = new_df['t_f'].apply(basmanipulation.satoshi_bitcoin)

    new_df['t_f'] = new_df['t_f'].apply(lambda x: x * 2)
    new_df['reward'] = new_df['reward'].apply(lambda x: x * 2)

    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.plot_date(new_df['date'].values, new_df['t_f'], "-b", label="$M$", lw=2)
    plt.plot_date(new_df['date'].values, new_df['reward'].values, "-g", label="$R$", lw=2)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 18}
    # axes.set_ylim([0, 26])
    matplotlib.rc('font', **font)
    plt.xticks(rotation=45)
    plt.legend(loc="best")
    plt.xlabel("date")
    plt.ylabel("BTC")
    plt.legend(loc="best")
    plt.savefig(plot_directory+info, bbox_inches='tight', dpi=500)
    print(plot_directory+info + ".png created")


def plot_profit_multiple_miners(df, axes):
    """
    Plot the profit using AntMinerS9 having 1, 50, 100, 500 miners in the mining pool.
    """
    info = "profit_multiple miners"
    miner = 3
    new_df = df[['B_ep', 't_f', 'B_T']]
    new_df = new_df.groupby(['B_ep', 'B_T']).sum().reset_index()
    new_df['t_f'] = new_df['t_f'].apply(basmanipulation.satoshi_bitcoin)

    epoch = new_df['B_ep'].values

    # ----- create arrays to determine the BTC price for each epoch
    btc_usd = basretrieve.get_json_request("https://api.blockchain.info/charts/market-price?timespan=all&format=json")
    values = btc_usd['values']
    y = []  # BTC price in USD
    x = []  # epoch
    for el in values:
        y.append(el['y'])
        x.append(el['x'])
    # create array containing x and y coordinates of the BTC-USD price
    y = np.asarray(y)
    x = np.asarray(x)
    epoch[:] = [int(el) for el in epoch]
    usd_array = []  # array with value in USD according to epoch
    for ep in epoch:
        index = basmanipulation.find_nearest(x, ep)
        usd_array.append(y[index])
    new_df['btc_price'] = usd_array

    hashing_rate = basretrieve.get_json_request("https://api.blockchain.info/charts/hash-rate?timespan=all&format=json")
    values = hashing_rate['values']
    y = []  # BTC price in USD
    x = []  # epoch
    for el in values:
        y.append(el['y'])
        x.append(el['x'])
    # create array containing x and y coordinates of the BTC-USD price
    y = np.asarray(y)
    x = np.asarray(x)
    epoch[:] = [int(el) for el in epoch]
    hashing_rate_array = []  # array with value in USD according to epoch
    for ep in epoch:
        index = basmanipulation.find_nearest(x, ep)
        hashing_rate_array.append(y[index])

    hashing_rate_array[:] = [int(el) for el in hashing_rate_array]
    hashing_rate_array[:] = [el * 1000000000000 for el in hashing_rate_array]
    new_df['hashing_rate'] = hashing_rate_array

    creation_time = new_df['B_T'].values
    fee = new_df['t_f'].values
    prices = new_df['btc_price'].values
    hashing_rate = new_df['hashing_rate'].values

    # calculate the profit considering more miners working together
    number_of_miners = [1, 50, 100, 500]

    for el in number_of_miners:
        profits = []
        revenues = []
        costs = []

        # calculate profits revenue and costs
        for f, t, p, h in zip(fee, creation_time, prices, hashing_rate):
            if t <= 0:
                t = 0.001
            profit, revenue, cost, label = basmanipulation.calculate_profit(f, t, p, h, miner, el)
            profits.append(profit)
            revenues.append(revenue)
            costs.append(cost)

        new_df['profit' + str(el)] = profits
        # new_df['revenue' + str(el)] = revenues
        # new_df['cost'+ str(el)] = costs

    new_df['B_T'] = new_df['B_T'].apply(basmanipulation.sec_minutes)

    new_df['B_T'] = new_df['B_T'].apply(basmanipulation.block_creation_time_intervals)

    new_df = new_df.groupby('B_T').median().reset_index()
    del new_df['B_ep']
    del new_df['btc_price']
    # del new_df['hashing_rate']
    del new_df['t_f']

    profit_array = []
    new_creation_time_categories = []
    numb_miners = []
    creation_time_categories = new_df['B_T'].values
    for el in number_of_miners:
        profit_array.extend(new_df['profit'+str(el)].values)
        new_creation_time_categories.extend(creation_time_categories)
        del new_df['profit'+str(el)]
        for cat in creation_time_categories:
            numb_miners.append(el)

    new_df = pd.DataFrame.from_items(
        [('B_T', new_creation_time_categories), ('profit', profit_array), ('number of miners', numb_miners)])

    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 0.9})
    g = sns.pointplot(x="B_T", y="profit", data=new_df, hue='number of miners')
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    g.set(ylabel=r'$\langle \Pi \rangle$ (BTC)')
    sns.plt.ylim(0, )
    plt.legend(loc="best")
    plt.savefig(plot_directory + info, bbox_inches='tight', dpi=500)
    print(plot_directory + info + ".png created")


def plot_profit_creation_time(df, axes):
    """
    Plot the revenue, costs and profit for miners according the creation time.
    """
    info = "profit_creation_time"
    info1 = "cost_creation_time"
    info2 = "revenue_creation_time"

    new_df = df[['B_ep', 't_f', 'B_T']]
    new_df = new_df.groupby(['B_ep', 'B_T']).sum().reset_index()
    new_df['t_f'] = new_df['t_f'].apply(basmanipulation.satoshi_bitcoin)

    new_df = basmanipulation.epoch_date_yy(new_df)
    new_df = new_df[new_df.date != '2013']
    new_df = new_df[new_df.date != '2014']

    epoch = new_df['B_ep'].values

    # ----- create arrays to determine the BTC price for each epoch
    btc_usd = basretrieve.get_json_request("https://api.blockchain.info/charts/market-price?timespan=all&format=json")
    values = btc_usd['values']
    y = []  # BTC price in USD
    x = []  # epoch
    for el in values:
        y.append(el['y'])
        x.append(el['x'])
    # create array containing x and y coordinates of the BTC-USD price
    y = np.asarray(y)
    x = np.asarray(x)
    epoch[:] = [int(el) for el in epoch]
    usd_array = []  # array with value in USD according to epoch
    for ep in epoch:
        index = basmanipulation.find_nearest(x, ep)
        usd_array.append(y[index])
    new_df['btc_price'] = usd_array

    hashing_rate = basretrieve.get_json_request("https://api.blockchain.info/charts/hash-rate?timespan=all&format=json")
    values = hashing_rate['values']
    y = []  # hashing rate of bitcoin network
    x = []  # epoch
    for el in values:
        y.append(el['y'])
        x.append(el['x'])
    y = np.asarray(y)
    x = np.asarray(x)
    epoch[:] = [int(el) for el in epoch]
    hashing_rate_array = []  # array with value in USD according to epoch
    for ep in epoch:
        index = basmanipulation.find_nearest(x, ep)
        hashing_rate_array.append(y[index])

    hashing_rate_array[:] = [int(el) for el in hashing_rate_array]
    hashing_rate_array[:] = [el * 1000000000000 for el in hashing_rate_array]
    new_df['hashing_rate'] = hashing_rate_array

    creation_time = new_df['B_T'].values
    fee = new_df['t_f'].values
    prices = new_df['btc_price'].values
    hashing_rate = new_df['hashing_rate'].values

    profits = []
    revenues = []
    costs = []

    # calculate profits revenue and costs
    for f, t, p, h in zip(fee, creation_time, prices, hashing_rate):
        if t <= 0:
            t = 0.01
        profit, revenue, cost, miner = basmanipulation.calculate_profit(f, t, p, h, miner=3)
        profits.append(profit)
        revenues.append(revenue)
        costs.append(cost)

    new_df['profit'] = profits
    new_df['revenue'] = revenues
    new_df['cost'] = costs

    new_df['B_T'] = new_df['B_T'].apply(basmanipulation.sec_minutes)
    ax1 = new_df.plot(x='B_T', y='cost', linestyle=' ', marker='o', color='red',
                      label=r'$\langle C\rangle$ with ' + miner, markersize=2)
    ax1.set_xlabel("$\mathcal{T}(min)$")
    ax1.set_ylabel(r"Cost $\langle C\rangle$ (BTC)")
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 0.0001)
    plt.savefig(plot_directory+info1, bbox_inches='tight', dpi=500)

    ax2 = new_df.plot(x='B_T', y='revenue', linestyle=' ', marker='o', color='green',
                      label=r'$\langle V\rangle$ with ' + miner, markersize=2)
    ax2.set_xlabel("$\mathcal{T}(min)$")
    ax2.set_ylabel(r"Revenue $\langle V\rangle$ (BTC)")
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 0.0001)
    plt.savefig(plot_directory+info2, bbox_inches='tight', dpi=500)

    ax = new_df.plot(x='B_T', y='profit',linestyle=' ',marker='o', color='orange',
                     label=r'$\langle \Pi \rangle$ with ' + miner, markersize=2)
    ax.set_xlabel("$\mathcal{T}(min)$")
    ax.set_ylabel(r"Profit $\langle \Pi \rangle$ (BTC)")
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 0.00009)
    # ax.set_ylim(-0.00001, 0.0000005)

    # -- REGRESSION
    regression = []
    regression.append(r"$f_{\langle \Pi \rangle}(\mathcal{T})$")
    regression.append(39)
    regression.append(2)

    new_x, new_y, f = basmanipulation.polynomial_interpolation(r"$f^{"
        + str(regression[2]) + r"}_{\langle \Pi \rangle}(\mathcal{T})$",
            new_df['B_T'].values, new_df['profit'].values, regression[2])
    plt.plot(new_x, new_y, "r-", label=r"$f^{"
                                       + str(regression[2]) + r"}_{\langle \Pi \rangle}(\mathcal{T})$", lw=2)

    new_x, new_y, f = basmanipulation.polynomial_interpolation(r"$f^{" + str(regression[1])
                                                               + r"}_{\langle \Pi \rangle}(\mathcal{T})$",
                                                               new_df['B_T'].values,
                                               new_df['profit'].values, regression[1])
    plt.plot(new_x, new_y, "g-", label=r"$f^{"
                                       + str(regression[1]) + r"}_{\langle \Pi \rangle}(\mathcal{T})$", lw=2)

    # --- calculating MAE
    predicted = []
    real = new_df['profit'].values
    for el in new_df['B_T'].values:
        x = float(el)
        # pred_y = f(x)
        pred_y = basmanipulation.miner_profit_function(x)
        predicted.append(pred_y)
    predicted = np.asarray(predicted)
    print "MAE: " + str(mean_absolute_error(real, predicted)) + "\n"
    print "MSE: " + str(math.sqrt(mean_squared_error(real, predicted))) + "\n"
    plt.legend(loc="best")
    plt.savefig(plot_directory + info, bbox_inches='tight', dpi=500)
    print(plot_directory + info + ".png created")


def plot_total_btc(df, axes):
    """
    Plot the total bitcoin in circulation
    """
    info = "total_btc"
    df_btc = df[['t_in', 'B_ep']]

    df_btc = basmanipulation.epoch_date_dd(df_btc)
    df_btc['t_in'] = df_btc['t_in'].apply(basmanipulation.satoshi_bitcoin)
    df_btc = df_btc.groupby('date').sum().reset_index()
    df_btc['t_in'] = df_btc['t_in'].apply(lambda x: x * 2)

    df_btc = df_btc.interpolate(method='cubic')
    ax = df_btc.plot(x='date', y='t_in', label='total BTC in circulation')
    ax.set_ylabel("money (BTC)")
    plt.legend(loc="best")
    plt.savefig(plot_directory + info, bbox_inches='tight', dpi=500)
    print(plot_directory + info + ".png created")


def plot_fee_input_miners(df, axes):
    """
    Plot the comparison of the percentage of fee paid between the 20 biggest mining pools
    """
    info = "fee_input_miners"
    df_fee_per = basmanipulation.remove_minor_miners(df, 20)
    df_fee_per = df_fee_per[['B_ep', 't_in', 't_f', 'B_mi', 't_ou']]
    df_fee_per = basmanipulation.epoch_date_yy(df_fee_per)
    del df_fee_per['B_ep']

    df_fee_per['t_per'] = basmanipulation.calculate_percentage_txs_fee(df_fee_per)

    df_inputtxs = df_fee_per[['t_in', 't_f', 'B_mi', 't_per', 'date']]
    df_inputtxs = df_inputtxs.groupby(['date', 'B_mi']).median().reset_index()
    df_inputtxs['t_in'] = df_inputtxs['t_in'].apply(basmanipulation.satoshi_bitcoin)
    df_inputtxs['t_f'] = df_inputtxs['t_f'].apply(basmanipulation.satoshi_bitcoin)
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 0.9})
    g = sns.pointplot(x="B_mi", y="t_per", data=df_inputtxs, hue='date')
    g.set(xlabel='major miners', ylabel='$t_f$ %')
    g.set_xticklabels(g.get_xticklabels(), rotation=65)
    plt.legend(loc="best")
    plt.savefig(plot_directory + info, bbox_inches='tight', dpi=500)
    print(plot_directory + info + ".png created")


def plot_fee_latency(df, axes):
    """
    Plot the transaction fee in relation with the fee latency.
    """
    info = "fee_latency"
    df_fl = df[['B_ep', 't_l', 't_f', 't_q']]
    df_fl = df_fl.groupby('B_ep').median().reset_index()
    df_fl = basmanipulation.epoch_date_yy(df_fl)
    df_fl = df_fl[df_fl.date != '2013']
    df_fl = df_fl[df_fl.date != '2014']
    df_fl = df_fl[df_fl.date != '2015']
    df_fl = df_fl[df_fl.date != '2016']
    del df_fl['B_ep']

    df_fl['fee_density'] = df_fl['t_f'] / df_fl['t_q']
    df_fl = df_fl[df_fl.fee_density < 450]
    df_fl = df_fl[df_fl.fee_density > 0.0]
    df_fl['t_l'] = df_fl['t_l'].apply(basmanipulation.sec_hours)

    ax = df_fl.plot(x='t_f', y='t_l', linestyle=' ', marker='o', color='orange',
                                      label='$t_l$', markersize=2 ,rot=45)
    ax.set_xlabel("$t_f$ (BTC)")
    # ax.set_xlabel(r"$\rho$ (sat/byte)")
    ax.set_ylabel("$t_l$ (h)")
    # ax.set_ylim(0, 0.0001)
    ax.set_ylim(0, 4)
    # ax.set_xlim(0, 0.003)
    ax.set_xlim(0, 0.003)
    # ax.set_xlim(0, 700)

    # -- REGRESSION
    regression = []
    regression.append(r"$f_{t_l}(t_f)$")
    regression.append(2)
    regression.append(39)
    new_x, new_y, f = basmanipulation.polynomial_interpolation(r"$f^{" + str(regression[2]) +
                                                               r"}_{t_l}(t_f)$", df_fl['t_f'].values,
                                               df_fl['t_l'].values, regression[2])
    plt.plot(new_x, new_y, "g-", label=r"$f^{" + str(regression[2]) + r"}_{t_l}(t_f)$", lw=2)

    new_x, new_y, f = basmanipulation.polynomial_interpolation(r"$f^{" + str(regression[1]) +
                                                               r"}_{t_l}(t_f)$", df_fl['t_f'].values,
                                               df_fl['t_l'].values, regression[1])
    plt.plot(new_x, new_y, "r-", label=r"$f^{" + str(regression[1]) + r"}_{t_l}(t_f)$", lw=2)

    # --- calculating MAE
    predicted = []
    real = df_fl['t_l'].values
    for el in df_fl['fee_density'].values:
        x = float(el)
        pred_y = basmanipulation.feedensity_latency_function(x)
        predicted.append(pred_y)
    predicted = np.asarray(predicted)
    print "MAE: " + str(mean_absolute_error(real, predicted)) + "\n"
    print "RMSE: " + str(math.sqrt(mean_squared_error(real, predicted))) + "\n"
    plt.legend(loc="best")
    plt.savefig(plot_directory + info, bbox_inches='tight', dpi=500)
    print(plot_directory + info + ".png created")


def plot_txs_fee_distribution(df, axes):
    """
    Plot the transaction fee distribution, divided in category.
    """
    info = "txs_fee_distribution"
    df_distr = df[['t_f', 't_q', 't_%', 't_l', 'Q', 'B_T', 'B_ep']]
    # split date to have yyyy-mm
    df_distr = basmanipulation.epoch_date_mm(df_distr)
    df_distr['t_f'] = df_distr['t_f'].apply(basmanipulation.satoshi_bitcoin)
    # get a category for the fee
    df_distr['t_f'] = df_distr['t_f'].apply(basmanipulation.fee_intervals)
    # groub by date and then miners, count how many transactions a miner approved in a certain month
    df_distr = df_distr.groupby(['t_f', 'date']).size().to_frame('size').reset_index()
    # df_grouped.plot(data=df_grouped, x ='date', y='size', kind='area')

    df_0 = df_distr[df_distr.t_f == "0"]
    df_1 = df_distr[df_distr.t_f == ">0.0001"]
    df_2 = df_distr[df_distr.t_f == ">0.0002"]
    df_3 = df_distr[df_distr.t_f == ">0.0005"]
    df_4 = df_distr[df_distr.t_f == ">0.001"]
    df_5 = df_distr[df_distr.t_f == ">0.01"]

    # create a new dataframe having as columns the different t_f
    new_df = pd.DataFrame.from_items(
        [('0 (BTC)', df_0['size'].values), ('>0.0001', df_1['size'].values), ('>0.0002', df_2['size'].values),
         ('>0.0005', df_3['size'].values), ('>0.001', df_4['size'].values),
         ('>0.01', df_5['size'].values), ('date', df_0['date'].values)])

    dates = new_df['date'].values

    col_list = list(new_df)
    col_list.remove('date')
    new_df['total'] = new_df[col_list].sum(axis=1)
    df1 = new_df.drop(['date'], axis=1)

    percent = df1.div(df1.total, axis='index') * 100

    percent = percent.drop(['total'], axis=1)
    percent['date'] = dates

    ax = percent.plot.area(x='date')
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    plt.legend(loc="best")
    plt.savefig(plot_directory + info, bbox_inches='tight', dpi=500)
    print(plot_directory + info + ".png created")


def plot_txs_feedensity_distribution(df, axes):
    """
    Plot the transaction fee density distribution, divided in category.
    """
    info = "txs_feedensity_distribution"
    df_fdens = df[['t_f', 't_q', 'B_ep']]
    df_fdens = basmanipulation.epoch_date_mm(df_fdens)
    df_fdens['f_dens'] = df_fdens['t_f'] / df_fdens['t_q']

    # get a category for the fee
    df_fdens['f_dens'] = df_fdens['f_dens'].apply(basmanipulation.fee_density_intervals)
    # groub by date and then miners, count how many transactions a miner approved in a certain month
    df_fdens = df_fdens.groupby(['f_dens', 'date']).size().to_frame('size').reset_index()

    df_0 = df_fdens[df_fdens.f_dens == "0"]
    df_1 = df_fdens[df_fdens.f_dens == "<50"]
    df_2 = df_fdens[df_fdens.f_dens == "<100"]
    df_3 = df_fdens[df_fdens.f_dens == "<200"]
    df_4 = df_fdens[df_fdens.f_dens == "<300"]
    df_5 = df_fdens[df_fdens.f_dens == ">300"]

    # create a new dataframe having as columns the different t_f
    new_df = pd.DataFrame.from_items(
        [('0 (sat/byte)', df_0['size'].values), ('<50', df_1['size'].values), ('<100', df_2['size'].values),
         ('<200', df_3['size'].values), ('<300', df_4['size'].values),
         ('>300', df_5['size'].values), ('date', df_0['date'].values)])

    dates = new_df['date'].values

    col_list = list(new_df)
    col_list.remove('date')
    new_df['total'] = new_df[col_list].sum(axis=1)
    df1 = new_df.drop(['date'], axis=1)

    percent = df1.div(df1.total, axis='index') * 100

    percent = percent.drop(['total'], axis=1)
    percent['date'] = dates

    ax = percent.plot.area(x='date')
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    plt.legend(loc="best")
    plt.savefig(plot_directory + info, bbox_inches='tight', dpi=500)
    print(plot_directory + info + ".png created")


def plot_fee_latency_years(df, axes):
    """
    Plot the relation between the transaction fee and the latency, distributed during years
    """
    info = "fee_latency_years"
    # plot the top miners according their fee and latency in transactions to see which miners are more picky
    # while choosing transactions
    new_df = df[['t_l', 't_f', 'B_ep']]

    # select by year:
    new_df = basmanipulation.epoch_date_yy(new_df)
    del new_df['B_ep']

    new_df['t_f'] = new_df['t_f'].apply(basmanipulation.satoshi_bitcoin)
    new_df['fee_category'] = new_df['t_f'].apply(basmanipulation.fee_more_intervals)
    del new_df['t_f']
    new_df['t_l'] = new_df['t_l'].apply(basmanipulation.sec_hours)
    new_df = new_df.groupby(['date', 'fee_category']).mean().reset_index()
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 0.9})
    g = sns.pointplot(x="fee_category", y="t_l", data=new_df, hue='date')
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    g.set(xlabel='$t_f$ (BTC)', ylabel='$t_l$ (h)')
    sns.plt.ylim(0, )
    plt.legend(loc="best")
    plt.savefig(plot_directory + info, bbox_inches='tight', dpi=500)
    print(plot_directory + info + ".png created")


def plot_blocksize_latency(df, axes):
    """
    Plot the block size Q in relation with the transaction latency.
    """
    info = "blocksize_latency"
    # print block size and transaction latency relation
    new_df = df[['B_ep', 't_l', 'Q']]
    new_df = basmanipulation.epoch_date_yy(new_df)
    del new_df['B_ep']
    new_df['Q'] = new_df['Q'].apply(basmanipulation.byte_megabyte)
    new_df['q_category'] = new_df['Q'].apply(basmanipulation.block_size_intervals)

    new_df = new_df.groupby(['date', 'q_category']).mean().reset_index()
    new_df = new_df[new_df.date != '2013']
    new_df['t_l'] = new_df['t_l'].apply(basmanipulation.sec_hours)

    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 0.9})
    g = sns.pointplot(x="q_category", y="t_l", data=new_df, hue='date')

    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    g.set(xlabel='$Q$ (MB)', ylabel='$t_l$ (h)')
    sns.plt.ylim(0, )
    plt.legend(loc="best")
    plt.savefig(plot_directory + info, bbox_inches='tight', dpi=500)
    print(plot_directory + info + ".png created")


def plot_throughput(df, axes):
    """
    Plot throughput during time
    """
    info = "throughput"
    df_thr = df[['B_t', 'B_T', 'B_ep']]
    df_thr = basmanipulation.epoch_date_dd(df_thr)

    df_thr = df_thr.groupby(['date']).size().to_frame('size').reset_index()
    size = df_thr['size'].values

    df_thr = df[['B_t', 'B_T', 'B_ep']]
    df_thr = basmanipulation.epoch_date_dd(df_thr)
    df_thr = df_thr.groupby(['date', 'B_ep']).median().reset_index()
    df_thr = df_thr.groupby(['date']).sum().reset_index()
    df_thr['size'] = size
    df_thr = df_thr[['date', 'size', 'B_T']]
    df_thr['thr'] = df_thr['size'] / df_thr['B_T']
    ax = df_thr.plot(x='date', y ='thr', color='orange', label='throughput $\gamma$', rot=45)
    ax.set_ylabel("$\gamma$ (txs/s)")
    plt.legend(loc="best")
    plt.savefig(plot_directory + info, bbox_inches='tight', dpi=500)
    print(plot_directory + info + ".png created")


def plot_creation_time_miners(df, axes):
    """
    Bar plot of occasional miners and mining pools about the creation time.
    """
    info = "creation_time_miners"
    df_new = df[['B_ep', 'B_mi', 'B_T']]
    df_new = df_new.groupby(['B_ep', 'B_mi']).mean().reset_index()
    df_new = basmanipulation.epoch_date_yy(df_new)
    df_new = df_new[df_new['date'] > '2015']
    del df_new['date']
    del df_new['B_ep']
    miners = df_new['B_mi'].values
    df_new['B_T'] = df_new['B_T'].apply(basmanipulation.sec_minutes)
    # divide miners in IP and mining pool
    # true / false list for adding the ip address or mining pool
    truefalse_list = []
    # match the string with re
    for mi in miners:
        pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
        if pattern.match(str(mi)):
            truefalse_list.append('Occasional miner')
        else:
            truefalse_list.append('Mining pool')

    df_new['Miner'] = truefalse_list

    del df_new['B_mi']

    breaks = [0, 8, 15, 20, 50, pd.np.inf]
    diff = np.diff(breaks).tolist()
    # make tuples of *breaks* and length of intervals
    joint = list(zip(breaks, diff))
    # format label
    s1 = "{left:,.0f} to {right:,.0f}"
    labels = [s1.format(left=yr[0], right=yr[0] + yr[1] - 1) for yr in joint]
    df_new['minutes'] = pd.cut(df_new['B_T'], breaks, labels=labels, right=False)
    df_new.loc[df_new['minutes'] == 'NaN', 'minutes'] = '0 to 7'
    del df_new['B_T']
    df_new = df_new.groupby(['minutes', 'Miner']).size().to_frame('blocks').reset_index()

    g = sns.factorplot(x='minutes', y='blocks', hue='Miner', data=df_new, kind="bar", legend_out=False, color='orange')
    g.fig.suptitle('From 2016 to 2017')
    plt.legend(loc="best")
    plt.savefig(plot_directory + info, bbox_inches='tight', dpi=500)
    print(plot_directory + info + ".png created")


def plot_block_size(df, axes):
    """
    Plot the block size during time.
    """
    info = "block_size"
    df_block_size = df[['Q', 'B_ep']]
    df_block_size = basmanipulation.epoch_date_mm(df_block_size)

    df_block_size = df_block_size.groupby(['date']).mean().reset_index()
    df_block_size['Q'] = df_block_size['Q'].apply(basmanipulation.byte_megabyte)

    ax = df_block_size.plot(x='date', y='Q')
    ax.set_xlabel("date")
    ax.set_ylabel("$Q$ (MB)")

    plt.legend(loc="best")
    plt.savefig(plot_directory + info, bbox_inches='tight', dpi=500)
    print(plot_directory + info + ".png created")


def plot_top_miners_monthly(df, axes):
    """
    Plot the occasional miners and the mining pools every months according to how many blocks they mine.
    """
    info = "top_miners_monthly"
    df_topminers = df[['B_ep', 'B_mi']]
    df_grouped1 = df_topminers.groupby(['B_ep', 'B_mi']).size().to_frame('size').reset_index()
    df_grouped1 = basmanipulation.epoch_date_mm(df_grouped1)
    del df_grouped1['size']
    df_grouped = df_grouped1.groupby(['date', 'B_mi']).size().to_frame('size').reset_index()

    # for each month put the date as column and the miner as row
    dates = df_grouped['date'].values
    miners = df_grouped['B_mi'].values
    txs = df_grouped['size'].values

    prev_date = dates[0]

    # new_df = pd.DataFrame()
    new_dates = []
    new_dates.append(prev_date)
    for d in dates:
        if d == prev_date:
            # do not add a new column
            pass
        else:
            new_dates.append(d)
            prev_date = d

    new_df = pd.DataFrame(np.nan, index=miners, columns=new_dates)
    # for every miner create a list containing the miner and the size
    miners_per_date = []
    miners_complete_list = []

    prev_date = dates[0]
    for d, m, t in zip(dates, miners, txs):
        pair_miner_size = []
        if d == prev_date:
            # column must stay the same one
            # add miners
            pair_miner_size.append(m)
            pair_miner_size.append(t)
            miners_per_date.append(pair_miner_size)
        else:
            miners_complete_list.append(miners_per_date)
            miners_per_date = []
            pair_miner_size.append(m)
            pair_miner_size.append(t)
            miners_per_date.append(pair_miner_size)
            prev_date = d
            # column change
    miners_complete_list.append(miners_per_date)
    # insert in the new df
    i = 0
    for d in range(len(miners_complete_list)):
        for m in range(len(miners_complete_list[d])):
            # add in column d
            new_df.iloc[i, new_df.columns.get_loc(list(new_df.columns.values)[d])] = miners_complete_list[d][m][1]
            i += 1

    new_df = new_df.groupby(new_df.index).sum().reset_index()
    new_df = new_df.fillna(0)

    miners = new_df['index'].values

    # true / false list for adding the ip address or mining pool
    truefalse_list = []
    # match the string with re
    for mi in miners:
        pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
        if pattern.match(mi):
            truefalse_list.append(True)
        else:
            truefalse_list.append(False)

    new_df['is_ip'] = truefalse_list

    df_mining_pool = new_df[new_df['is_ip'] == False]
    df_occ_miners = new_df[new_df['is_ip'] == True]

    number_occ_miners = len(df_occ_miners.index)
    number_mining_pools = len(df_mining_pool.index)

    df_mining_pool = df_mining_pool.groupby('is_ip').sum().reset_index()
    df_occ_miners = df_occ_miners.groupby('is_ip').sum().reset_index()
    x = list(df_occ_miners.columns.values)
    x.pop(0)

    y_ip = []
    for row in df_occ_miners.iterrows():
        index, data = row
        y_ip.append(data.tolist())

    y_ip = y_ip[0]
    y_ip.pop(0)

    y_noip = []
    for row in df_mining_pool.iterrows():
        index, data = row
        y_noip.append(data.tolist())
    y_noip = y_noip[0]
    y_noip.pop(0)

    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.plot_date(x, y_ip, "-b", label="Occasional miners " + str(number_occ_miners), lw=2)
    plt.plot_date(x, y_noip, "-g", label="Mining pools " + str(number_mining_pools), lw=2)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 18}

    # plotting
    matplotlib.rc('font', **font)
    plt.legend(loc="best")
    plt.xlabel("date")
    plt.ylabel("blocks approved")
    plt.legend(loc="best")
    plt.savefig(plot_directory + info, bbox_inches='tight', dpi=500)
    print(plot_directory + info + ".png created")


def plot_trendy_miners(df, axes):
    """
    Plot the transactions approved by the 15 major miners during the years.
    """
    info = "trendy_miners"
    # add date to df from epoch
    df = basmanipulation.remove_minor_miners(df, 15)
    df = df[['B_ep', 'B_mi']]
    df = basmanipulation.epoch_date_mm(df)

    # groub by date and then miners, count how many transactions a miner approved in a certain month
    df_grouped = df.groupby(['date', 'B_mi']).size().to_frame('size').reset_index()
    # calculate how many transactions were apprved by each miner in every year
    sns.set_context("notebook", font_scale=0.9, rc={"lines.linewidth": 0.7})
    g = sns.pointplot(x="date", y="size", hue="B_mi", data=df_grouped)
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    # reduce number of labels to show
    ticklabels = g.get_xticklabels()
    new_ticklabels = []
    i = 0
    for el in ticklabels:
        if i == 0:
            i = 8
        else:
            el = ""
            i -= 1
        new_ticklabels.append(el)
    g.set(xlabel='date', ylabel='transactions approved', xticklabels=new_ticklabels)
    sns.plt.ylim(0, )
    plt.legend(loc="best")
    plt.savefig(plot_directory + info, bbox_inches='tight', dpi=500)
    print(plot_directory + info + ".png created")


def plot_number_of_miners(df, axes):
    """
    Plot the number of active miners in the network.
    """
    info = "number_of_miners"
    df_miners = df[['B_ep', 'B_mi']]
    df_miners = basmanipulation.epoch_date_mm(df_miners)
    del df_miners['B_ep']
    df_miners = df_miners.groupby(['date', 'B_mi']).size().to_frame('size').reset_index()
    del df_miners['size']
    df_miners = df_miners.groupby('date').size().to_frame('miners').reset_index()

    ax = df_miners.plot(x='date', y='miners', color='orange')
    ax.set_xlabel("date")
    ax.set_ylabel("number of miners")
    plt.legend(loc="best")
    plt.savefig(plot_directory + info, bbox_inches='tight', dpi=500)
    print(plot_directory + info + ".png created")
