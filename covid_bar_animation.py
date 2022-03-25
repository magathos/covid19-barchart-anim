#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
from iso3166 import Country
import flag
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from tqdm import tqdm

matplotlib.rcParams['animation.embed_limit'] = 2**10


datacvs = 'owid-covid-data.csv'

# Load useful data from CVS file as pandas dataframe
df = pd.read_csv(datacvs, usecols=['continent','location','date','new_cases_smoothed_per_million','total_deaths','new_deaths','total_deaths_per_million','new_deaths_per_million','new_deaths_smoothed_per_million','reproduction_rate','new_tests','new_tests_per_thousand','tests_per_case'], parse_dates=[2])

# Shorten dataframe to minimal set of columns and pick out single day
dfshort=df[['location','date','total_deaths_per_million']]
df_now = df[df["date"]=='20220106'].sort_values(by="total_deaths_per_million", ascending=False).reset_index(drop=True)

# List locations (countries) and map colours
exclude_loc = set(['World', 'Africa', 'Asia', 'Oceania', 'Europe', 'European Union', 'South America', 'High income', 'International', 'Upper middle income', 'Low income', 'Lower middle income', 'North America'])
locations=set(df["location"].to_dict().values())
loclist = list(locations)
loclist.sort()

cols = plt.cm.Pastel1.colors + plt.cm.Pastel2.colors + plt.cm.Paired.colors + plt.cm.Accent.colors + plt.cm.Set1.colors + plt.cm.Set2.colors + plt.cm.Dark2.colors + plt.cm.tab20c.colors + plt.cm.Set3.colors + plt.cm.tab20.colors + plt.cm.tab20b.colors
color_dict = dict(zip(loclist, [cols[k] for k in np.arange(len(loclist)) % len(cols)]))

focus_country = Country.gr

# Transition functions

def initData(initFrame=None, N=len(loclist)):
    # Initialize bar positions and velocities
    global Y_NOW, V_NOW, RANK_NOW
    if initFrame is None:
        Y_NOW = N - 1.0*np.arange(N)
    else:
        sdf = df_interp_bydate.get_group(datelist[initFrame]).sort_values(by='total_deaths_per_million', ascending=False).reset_index(drop=True)
        target_dict = dict(zip(sdf['location'].to_numpy(), np.arange(len(sdf))))

        # Read off target locations in loclist order and update
        Y_NOW = np.array([1.0*target_dict[k] for k in loclist])

    V_NOW = 1.0*np.zeros(N)
    RANK_NOW = Y_NOW.copy()

    return

def V(delta_y, sec_per_transition=0.7):
    '''Vertical velocity of bar coordinate as a function of distance to true location'''
    # [TODO: needed in vertical index units per seconds]
    c = 2./sec_per_transition
    v = c*delta_y

    # Slow down
    #   if (abs(delta_y) < 0.2):
    #       v = 0.05*delta_y/abs(delta_y)

    return v



def update_positions(y_target, v_mask):
    ''' Calculates the interpolated coordinate.
        All arrays are sorted as loclist.
     Input:
      - y_target :   array with target coordinate values
      - v_mask :     mask array where ones indicate a desired change in V
      # - y_now    :   the current coordinate value
     Output (in place):
      - global Y_NOW    :   array with current coordinates [y units]
      - global V_NOW    :   array with current velocities [y units/sec]
    '''

    global Y_NOW, V_NOW

    dt = 1.0/fps
    dy = y_target - Y_NOW

    # Change velocity where necessary
    V_NOW += v_mask*(V(dy) - V_NOW)
    Y_NOW += V_NOW*dt

    reached = np.where(abs(V_NOW*dt) > abs(dy))[0]
    if reached.size > 0:
        Y_NOW[reached] = y_target[reached]
        V_NOW[reached] = 0.0

    return


def data_interp(datelist, df):
    '''Interpolates data for new list of timestamps'''

    # Create DataFrame object
    df_interp = pd.DataFrame(columns=['location','date','total_deaths_per_million'])

    # group by country
    dfl = df.groupby('location')

    # iterate through countries
    for l in dfl.groups.keys():
        dg = dfl.get_group(l)
        x = [k.timestamp() for k in dg['date'].to_list()]
        y = dg['total_deaths_per_million']

        l_interp = np.interp([k.timestamp() for k in datelist.to_list()], x, y)
        l_df = pd.DataFrame({'location':[l]*len(datelist), 'date':datelist, 'total_deaths_per_million':l_interp})
        df_interp = pd.concat([df_interp, l_df])

    return df_interp

def focus_color_shift(focus_idx):
    '''Return color shift of focus bar'''
    if not focus_idx:
        return None

    if V_NOW[focus_idx] > 0:
        col = (0.0, 1.0, 0.0, 0.9)
    elif V_NOW[focus_idx] < 0:
        col = (1.0, 0.0, 0.0, 0.9)
    else:
        col = None

    return col


def draw_barchart(date, focus=focus_country.english_short_name, nplot=11, magnify=False):
    ''' Draw the bar chart
    In:
        - date  : date entry in datelist
        - focus : Focus country to follow
        - nplot : Number of positions to plot from top or around focus country
    Out:
        - ax    : Plot axis filled with the barchart
    '''

    # The following global variables as well as Y_target and update_mask are all sorted as loclist
    global Y_NOW, V_NOW, RANK_NOW

    # Select data on date
    dff = df_interp_bydate.get_group(date)
    # sort by column of interest
    dff = dff.sort_values(by='total_deaths_per_million', ascending=False).copy(deep=True)
    dff.reset_index(drop=True, inplace=True)

    # Dictionary of target ranks over locations
    target_dict = dict(zip(dff['location'].to_numpy(), np.arange(len(dff))))

    # Read off target ranks in loclist order and compare with previous rank list
    Y_target = np.array([target_dict[k] for k in loclist])
    update_mask = (RANK_NOW != Y_target).astype(float)

    # Move bars based
    update_positions(Y_target, update_mask)
    RANK_NOW = 1.0*Y_target.copy()

    # add new positions
    y_dict = dict(zip(loclist, Y_NOW))
    dff['y_new'] = dff['location'].map(y_dict)

    # Select range of indices to plot
    focus_indices = []

    # if no focus country is fiven, plot top N
    if focus not in locations:
        focus = None
        print("No focus!")
        imin = 0
        imaxpp = nplot
    else:
        focus_index = dff[dff['location']==focus].index.to_numpy()[0]
        focus_indices.append(focus_index)
        imin = np.max([focus_index - (nplot//2),0])
        imaxpp = imin + nplot

    #     idx_plot = dff.iloc[list(dff['total_deaths_per_million'] <= bar_num)].iloc[i]
    idx_plot = np.arange(imin,imaxpp)
    dfs = dff.iloc[idx_plot]
    #     dfrest = dff.iloc[np.array(set(np.arange(len(dff))) - set(idx_plot))]

    # Plot horizontal bars
    ax.clear()
    y_pos = nplot - np.arange(nplot)

    fcol = focus_color_shift(loclist.index(focus))

    bars = ax.barh(y_pos, dfs['total_deaths_per_million'], color=[color_dict[k] for k in dfs['location'].to_numpy()], alpha=0.9, tick_label=[str(k) for k in list(dfs.index.to_numpy())])
    #     ax.set_yticks(y_pos, labels=list(dfs.index.to_numpy()))
    dx = dfs['total_deaths_per_million'].max() / 200

    #     bars = ax.barh(imin - idx_rest, dfs['total_deaths_per_million'], color=[color_dict[k] for k in dfs['location'].to_numpy()], alpha=0.9, tick_label=[str(k) for k in list(dfrest.index.to_numpy())])



    y_pos = np.array([y_dict[k] for k in dfs['location'].to_numpy()])
    foc_pos = y_dict[focus]

    # update positions for moving bars
    for i, (bar, location) in enumerate(zip(bars, dfs['location'])):
        bar.set_y(imaxpp - y_pos[i] - bar.get_height()/2)

        width = bar.get_width()

        fw = 'light'
        if location == focus:
            foc_y = bar.get_y()
            fw = 'bold'
            if fcol:
                focfc = 0.7*np.array(bar.get_fc())
                focfc += 0.3*np.array(fcol)
                bar.set_fc(focfc)

        ax.text(width + dx, bar.get_y() + bar.get_height() / 2, location, size=14, weight=fw, ha='left', va='center')
        ax.annotate(f'{width:.0F}',
                    xy = (width , bar.get_y() + bar.get_height() / 2),
                    xytext = (-25, 0),
                    textcoords = "offset points",
                    fontsize = 'x-large',
                    fontweight = fw,
                    ha = 'right',
                    va = 'center')

    if focus and magnify:
        for bar in bars:
            magfactor = (0.8 + 0.4*np.exp(-0.5/0.5*(bar.get_y()-foc_y)**2))
            bar.set_height(bar.get_height()*magfactor)
            bar.set_y(bar.get_y() - (magfactor-1)*bar.get_height()/2.0)
#             bar.set_width(bar.get_width()*magfactor)


    ax.text(1, 0.2, date.strftime("%d %B, %Y"), transform=ax.transAxes, color='#AAAAAA', size=24, ha='right', weight=800)
    ax.text(0, 1.06, 'Deaths per million (total)', transform=ax.transAxes, size=12, color='#AAAAAA')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', color='#777777', labelsize=12)
    ax.tick_params(axis='y', color='#777777', labelsize=16)

    #     ax.tick_params(labelsize = 'medium')
    #     ax.set_yticks([str(k) for k in list(dfs.index.to_numpy())])
    ax.grid(True, axis = 'x')
    if focus:
        ax.set_xlim(0, 2.0*dff.iloc[focus_index]['total_deaths_per_million'])
        dy = foc_pos - focus_index
        ax.set_ylim(ax.get_ylim()[0]-dy, ax.get_ylim()[1]-dy)

    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0, 1.15, 'World ranking in COVID-19 deaths per million population', transform=ax.transAxes, size=24, weight=600, ha='left', va='top')
    ax.text(1, 0, '@magathos', transform=ax.transAxes, color='#777777', ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    ax.set_frame_on(False)

    # -- WIP --
def update_plot(date, focus=focus_country.english_short_name, nplot=None):

    update_positions(date)
    update_velocities(date)
    draw_barchart(date, focus, nplot)
    return

if __name__ == '__main__':

    duration = 200.0                # animation duration in seconds

    sec_per_transition = 1.0        # UNUSED how many seconds does a transition last
    fps = 24                      # frames per second
    nframes = int(fps*duration)        # total number of frames
    fpt = fps/sec_per_transition  # UNUSED frames per transition

    startdate='03/24/2020'        # start animation from date
    enddate='03/20/2022'          # end animation at date

    # create array of timestamps by dividing date range with uniform steps
    datelist = pd.date_range(start=startdate, end=enddate, periods=nframes)

    # interpolate data for datelist
    print("Interpolating data at", len(datelist), "times.")
    df_interp = data_interp(datelist, dfshort)
    df_interp_bydate = df_interp.groupby('date')

    #plt.style.available
    plt.style.use('dark_background')
    N_plot = 15

    print('Initializing data')
    initData(initFrame=10)

    # print('Plotting...')
    # fig = plt.figure(figsize=(14, (N_plot+3)/2.))
    # ax = fig.add_subplot(111)

    # draw_barchart(datelist[11], focus=focus_country.english_short_name, nplot=N_plot)
    # fig.savefig('testfig.png')
    # print('DONE')
    # sys.exit()

    initData(0)
    N_plot=11 # USE update_plot() wrapper as animation callable

    fig, ax = plt.subplots(figsize=(15, 8))
    animator = animation.FuncAnimation(fig, draw_barchart, frames=tqdm(datelist, file=sys.stdout), interval=1000./fps)
    f_anim = os.path.join(os.getcwd(), 'animation.mp4')
    f_gif = os.path.join(os.getcwd(), 'animation.gif')
    writervideo = animation.FFMpegWriter(fps=fps)
    imgkvideo = animation.ImageMagickWriter(fps=fps)
    animator.save(f_anim, writer=writervideo)

    print("DONE")
