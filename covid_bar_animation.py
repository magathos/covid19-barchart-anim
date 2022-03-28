#!/usr/bin/env python3

import os, sys
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
from IPython.display import HTML
import tqdm

matplotlib.rcParams['animation.embed_limit'] = 2**10
matplotlib.rcParams['markers.fillstyle'] = 'full'
matplotlib.rcParams['hatch.linewidth'] = 3

datacvs = 'covid-19-data/public/data/owid-covid-data.csv'

# Load useful data from CVS file as pandas dataframe
df = pd.read_csv(datacvs, usecols=['continent','location','date','new_cases_smoothed_per_million','total_deaths','new_deaths','total_deaths_per_million','new_deaths_per_million','new_deaths_smoothed_per_million','reproduction_rate','new_tests','new_tests_per_thousand','tests_per_case'], parse_dates=[2])

# Shorten dataframe to minimal set of columns and pick out single day
dfshort=df[['continent','location','date','total_deaths_per_million','new_deaths_smoothed_per_million']]
df_now = df[df["date"]=='20220321'].sort_values(by="total_deaths_per_million", ascending=False).reset_index(drop=True)

# List locations (countries) and map colours
nocountry_mask = pd.isna(df.continent.values)
exclude_locs = set(df[nocountry_mask]['location'].values)
# exclude_loc = set(['World', 'Africa', 'Asia', 'Oceania', 'Europe', 'European Union', 'South America', 'High income', 'International', 'Upper middle income', 'Low income', 'Lower middle income', 'North America'])
locations=set(df["location"].to_dict().values()) - exclude_locs
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
        sdf = df_interp.groupby('date').get_group(datelist[initFrame]).sort_values(by='total_deaths_per_million', ascending=False).reset_index(drop=True)
        target_dict = dict(zip(sdf['location'].to_numpy(), np.arange(len(sdf))))

        # Read off target locations in loclist order and update 
        Y_NOW = np.array([1.0*target_dict[k] for k in loclist])
        
    V_NOW = 1.0*np.zeros(N)
    RANK_NOW = Y_NOW.copy()
    
    return

def V(delta_y, sec_per_transition=0.7):
    '''Vertical velocity of bar coordinate as a function of distance to true location'''
    # [TODO: needed in vertical index units per second]
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
    df_interp = pd.DataFrame(columns=['continent','location','date','total_deaths_per_million','new_deaths_smoothed_per_million'])
    
    # group by country
    dfl = df.groupby('location')
    
    # iterate through countries
    for l in dfl.groups.keys():
        dg = dfl.get_group(l)
        x = [k.timestamp() for k in dg['date'].to_list()]
        y1 = dg['total_deaths_per_million']
        y2 = dg['new_deaths_smoothed_per_million']

        l_interp_1 = np.interp([k.timestamp() for k in datelist.to_list()], x, y1)
        l_interp_2 = np.interp([k.timestamp() for k in datelist.to_list()], x, y2)
        l_df = pd.DataFrame({'continent':[dg['continent'].values[0]]*len(datelist), 'location':[l]*len(datelist), 'date':datelist, 'total_deaths_per_million':l_interp_1, 'new_deaths_smoothed_per_million':l_interp_2})
        df_interp = pd.concat([df_interp, l_df])
    
    return df_interp

def rank_countries(df_daily, col='total_deaths_per_million'):
    '''Adds column with rank of countries to daily dataframe'''
    
    nocountry_idx = np.where(pd.isna(df_daily.continent.values))[0]
    df_daily['rank'][nocountry_idx] = df_daily[nocountry_idx][col].rank(ascending=False, pct=True)
    
    return 

def focus_color_shift(focus_idx):
    '''Return color shift of focus bar'''
    if not focus_idx:
        return None
    
    v = V_NOW[focus_idx]
    k = min(np.abs(v)/10.0,1.0)
    if v > 0:
        col = (1-k, 1.0, 1-k, 0.9)
    elif v < 0:
        col = (1.0, 1-k, 1-k, 0.9)
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

def draw_barchart_padded(date, ax, focus=focus_country.english_short_name, nplot=11, magnify=False):
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
    # Remove unwanted location entries (continents, income, etc.)
    dff = dff[~pd.isna(dff.continent.values)]
    # sort by column of interest
    dff = dff.sort_values(by='total_deaths_per_million', ascending=False).copy(deep=True)
    dff.reset_index(drop=True, inplace=True)
    
    # Dictionary of target ranks over locations
    target_dict = dict(zip(dff['location'].to_numpy(), np.arange(len(dff))+1))
    
    # Read off target ranks in loclist order and compare with previous rank list
    Y_target = np.array([1.0*target_dict[k] for k in loclist])
    update_mask = (RANK_NOW != Y_target).astype(float)
    
    # Move bars based 
    update_positions(Y_target, update_mask)
    RANK_NOW = 1.0*Y_target.copy()
    
    # put new positions in location dict
    y_dict = dict(zip(loclist, Y_NOW))
    # dff['y_new'] = dff['location'].map(y_dict)


    # Select range of indices to plot
    nplotpp = nplot + 10 

    # if no focus country is fiven, plot top N
    if focus not in locations:
        focus = None
        print("No focus!")
        imin = 0
        imaxpp = nplot
    else: 
        focus_index = dff[dff['location']==focus].index.to_numpy()[0]
        # focus_indices.append(focus_index)
        imin = np.max([focus_index - (nplotpp//2),0])
        imaxpp = np.min([imin + nplotpp, len(loclist)])

    #     idx_plot = dff.iloc[list(dff['total_deaths_per_million'] <= bar_num)].iloc[i]
    idx_plot = np.arange(imin,imaxpp)
    dfs = dff.iloc[idx_plot]
    #     dfrest = dff.iloc[np.array(set(np.arange(len(dff))) - set(idx_plot))]
    
    # Plot horizontal bars
    ax.clear()
    y_pos = nplotpp - np.arange(nplotpp) 
    
    fcol = focus_color_shift(loclist.index(focus))
    bar_colors = [color_dict[k] for k in dfs['location'].to_numpy()]
    tick_labels = [str(k) for k in list(dfs.index.to_numpy())]

    bars = ax.barh(y_pos, dfs['total_deaths_per_million'], color=bar_colors, alpha=0.9, tick_label=tick_labels)

    #     ax.set_yticks(y_pos, labels=list(dfs.index.to_numpy()))
    dx = dfs['total_deaths_per_million'].max() / 200
    
    #     bars = ax.barh(imin - idx_rest, dfs['total_deaths_per_million'], color=[color_dict[k] for k in dfs['location'].to_numpy()], alpha=0.9, tick_label=[str(k) for k in list(dfrest.index.to_numpy())])
    
    
    y_pos = np.array([y_dict[k] for k in dfs['location'].to_numpy()])
    foc_pos = y_dict[focus]

    if focus:
        dy = foc_pos - focus_index
        bottom = ax.get_ylim()[0] - dy + (nplotpp - nplot)/2 
        top = ax.get_ylim()[1] - dy - (nplotpp - nplot)/2
    
    # update positions for moving bars
    for i, (bar, location) in enumerate(zip(bars, dfs['location'])):
        bar.set_y(imaxpp - y_pos[i] - bar.get_height()/2)

        width = bar.get_width()

        fw = 'light'
        fc = 'white'
        if location == focus:
            foc_y = bar.get_y()
            fw = 'bold'
            if fcol:
                focfc = 0.7*np.array(bar.get_fc())
                focfc += 0.3*np.array(fcol)
                fc = fcol
                # bar.set_fc(focfc)
        if bottom < bar.get_y() < top:
            ax.text(width + 3*dx, bar.get_y() + bar.get_height() / 2, location, size=14, weight=fw, color=fc, ha='left', va='center')
            ax.annotate(f'{width:.0F}',
                        xy = (width , bar.get_y() + bar.get_height() / 2),
                        xytext = (-25, 0),
                        textcoords = "offset points",
                        fontsize = 'x-large',
                        fontweight = fw,
                        color = fc,
                        ha = 'right',
                        va = 'center')    
        
    if focus and magnify:
        for bar in bars:
            magfactor = (0.8 + 0.4*np.exp(-0.5/0.5*(bar.get_y()-foc_y)**2))
            bar.set_height(bar.get_height()*magfactor)
            bar.set_y(bar.get_y() - (magfactor-1)*bar.get_height()/2.0)
#             bar.set_width(bar.get_width()*magfactor)
                
        
    ax.text(1, 0.2, date.strftime("%d %B, %Y"), transform=ax.transAxes, color='#AAAAAA', size=24, ha='right', weight=800)
    ax.text(0, 1.06, 'Deaths per million (total)', transform=ax.transAxes, size=12, color='#BBBBBB')
    
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', color='#777777', labelsize=12)
    ax.tick_params(axis='y', color='#777777', labelsize=16, width=3, length=5)

    #     ax.tick_params(labelsize = 'medium')    
    #     ax.set_yticks([str(k) for k in list(dfs.index.to_numpy())])
    ax.grid(True, axis = 'x')
    
    if focus:
        ax.set_xlim(0, 2.0*dff.iloc[focus_index]['total_deaths_per_million'])
        ax.set_ylim(bottom, top)

   
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    
    ax.set_axisbelow(True)
    
    ax.text(0, 1.15, 'World ranking in COVID-19 deaths per million population', transform=ax.transAxes, size=24, weight=600, ha='left', va='top')
#     ax.text(1, 0, '@magathos', transform=ax.transAxes, color='#777777', ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    ax.text(1, 0, '@magathos\ndata source: OurWorldInData.org', transform=ax.transAxes, color='#CCCCCC', ha='right', va='bottom')

    ax.set_frame_on(False)
    
    return ax

### Stem & bar plot for continent

def initStemplot(date_idx, focus=None):
    
    x = df_interp_bydate.get_group(datelist[date_idx]).groupby('continent').get_group(mycontinent)
    ax_stems = ax_new.twinx()
    
    bbars = ax_new.bar(np.arange(len(x)), list(x['new_deaths_smoothed_per_million']), color='darkred', alpha=0.5)
    stems = ax_stems.stem(x['location'].replace({'Bosnia and Herzegovina': 'Bosnia & Herzegovina'}), x['total_deaths_per_million'], basefmt='None', use_line_collection=False) #  

    ax_new.set_xlabel(mycontinent, fontsize=16)
    ax_new.xaxis.set_label_coords(0.5,-.75)
    ax_new.set_ylim(0,20)
    ax_new.set_ylabel('Daily deaths/million', fontsize=12)
    ax_new.spines['top'].set_visible(False)
    ax_new.spines['right'].set_visible(False)

    stemcolor = stems.markerline.get_color()
    ax_stems.set_ylim(0, max(df_now.groupby('continent').get_group(mycontinent).total_deaths_per_million.values))
    ax_stems.spines['top'].set_visible(False)
    ax_stems.spines['bottom'].set_visible(False)
    ax_stems.spines['left'].set_visible(False)
#     ax_stems.spines['right'].set_bounds((0, max(df_now.total_deaths_per_million.values)))
    ax_stems.spines['right'].set_bounds((0, 5000))
    ax_stems.spines['right'].set_position(('outward', -20))
    ax_stems.tick_params(axis='y', color=stemcolor, labelcolor=stemcolor)
    ax_stems.spines['right'].set_color(stemcolor)
    ax_stems.set_ylabel('Total', color=stemcolor, fontsize=12)

    hlines = [ax_stems.axhline(0, lw=2, ls='-', color='lightgrey', alpha=0.4, label=mycontinent)]
    if not focus is None:
        # yfoc = x.groupby('location').get_group(focus)['total_deaths_per_million']
        fline = ax_stems.axhline(0.0, lw=1, ls='--', color=stemcolor, alpha=0.7)
        hlines += [fline]
    ax_stems.legend(frameon=False, loc=(0.85, 0.85))


    ax_new.spines['left'].set_bounds((0, ax_new.get_ylim()[1]))
    ax_new.spines['left'].set_position(('outward', -20))
    ax_new.spines['bottom'].set_bounds((0, len(x)-1))
    ax_new.spines['bottom'].set_position(('outward', 10))

    ppp = plt.setp(ax_new.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    ppp = plt.setp(stems.stemlines, 'linewidth', 1)
    ppp = plt.setp(stems.markerline, 'markersize', 3)    
        
#     ax_new.text(1, -1, 'data source: OurWorldInData.org', transform=ax_new.transAxes, color='#CCCCCC', ha='right', va='bottom')
#     ax_new.text(1, -1, '@magathos', transform=ax_new.transAxes, color='#777777', ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
#     ax_new.text(1, -1, '@magathos', transform=ax_new.transAxes, color='#AAAAAA', ha='right', va='bottom')

    return bbars, stems, hlines

def draw_stemgraph(date, stems, bbars, hlines, focus):
    
    dff = df_interp_bydate.get_group(date)
    continent = list(dff.groupby('location').get_group(focus)['continent'])[0]
    dfcont = dff.groupby('continent').get_group(continent).sort_values(by='location')
    dfcont.reset_index(drop=True, inplace=True)
    y1 = dfcont['new_deaths_smoothed_per_million']
    y2 = dfcont['total_deaths_per_million']
    for i,line in enumerate(stems.stemlines):
        line.set_ydata((0, y2[i]))
        if not focus is None:
            if dfcont['location'][i] == focus:
                hlines[1].set_data(([0.05, 0.95], [y2[i], y2[i]]))
    ycont = dff.groupby('location').get_group(continent)['total_deaths_per_million']
    hlines[0].set_data(([0.05, 0.95], [ycont, ycont]))
    

    for i, b in enumerate(bbars):
        b.set(height=y1[i])
        if y1[i]>20:
            b.set_hatch('x')
            # b.set_alpha(0.9)
        else:
            b.set_hatch('')
            # b.set_alpha(baralpha)
    stems.markerline.set_ydata(y2)
    
    
    return stems.stemlines, stems.markerline, bbars


import matplotlib.dates as mdates
from datetime import datetime

def init_timeline(ax_time, event_dates=None):

    # Convert date strings (e.g. 2014-10-18) to datetime
    dates = [datetime.strptime(datetime.fromtimestamp(k.timestamp()).isoformat(timespec='minutes'), "%Y-%m-%dT%H:%M") for k in datelist]
    y1 = df_interp.groupby('location').get_group(focus_country.english_short_name).sort_values(by='date')['total_deaths_per_million']
    y2 = df_interp.groupby('location').get_group(focus_country.english_short_name).sort_values(by='date')['new_deaths_smoothed_per_million']
#     for d in dates:
#         y3 = df_interp_bydate.get_group(d).total_deaths_per_million.rank()

    l1, = ax_time.plot(dates[0], [0], alpha=0.8)
    a1 = ax_time.fill_between(dates[0], [0], [0], alpha=0.4)    
    l2, = ax_time.plot(dates[0], [0], color='darkred', alpha=0.8, marker='o', mec='w', mfc='None', markevery=[-1])
#     p2, = ax_time.plot(dates[i], y2[i], 'o', mfc='none')
    a2 = ax_time.fill_between(dates[0], [0], [0], color='darkred', alpha=0.4)

    ax_time.set_xlim(dates[0],dates[-1])
    ax_time.set_ylim(0,np.max(y2))
    ax_time.autoscale(False)
    # ax_time.set_axis_off()
    # ax_time.barh([0],datelist[1])

    if not event_dates is None:
        ax_time.plot(event_dates, np.zeros_like(event_dates), "-o",
                color="k", markerfacecolor="w")  # Baseline and markers on it.

    # ax.vlines(dates, 0, levels, color="tab:red")  # The vertical stems.
    # annotate lines
    # for d, l, r in zip(dates, levels, names):
    #     ax.annotate(r, xy=(d, l),
    #                 xytext=(-3, np.sign(l)*3), textcoords="offset points",
    #                 horizontalalignment="right",
    #                 verticalalignment="bottom" if l > 0 else "top")


    # format xaxis with 4 month intervals
    ax_time.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    papapa = plt.setp(ax_time.get_xticklabels(), ha="right")
    # remove y axis and spines
    ax_time.yaxis.set_visible(False)
    ax_time.xaxis.set_visible(True)
    # ax_time.spines["bottom"].setp
    ax_time.spines["left"].set_visible(False)
    ax_time.spines["right"].set_visible(False)
    ax_time.spines["top"].set_visible(False)

    ax_time.margins(y=0.0)
    
    return dates, y1, y2

def update_timeline(i, ax_time, dates, y1, y2):
        
    # rescaling factor
    rescale = np.max(y2)/np.max(y1)
    for li in ax_time.lines:
        li.remove()
    for co in ax_time.collections:
        co.remove()
    for tx in ax_time.get_children():
        if (isinstance(tx, matplotlib.text.Text) and hasattr(tx, 'arrowprops')):
            tx.remove()

    l1, = ax_time.plot(dates[:i], rescale*y1[:i], alpha=0.8, antialiased=True)
    a1 = ax_time.fill_between(dates[:i], rescale*y1[:i], [0]*i, alpha=0.4)    
    l2, = ax_time.plot(dates[:i], y2[:i], color='darkred', alpha=0.8, marker='o', mec='w', mfc='None', markevery=[-1], antialiased=True)
    #     p2, = ax_time.plot(dates[i], y2[i], 'o', mfc='none')
    a2 = ax_time.fill_between(dates[:i], y2[:i], [0]*i, color='darkred', alpha=0.4)
    
    #     dxy = np.array([dx, dy])
    t2 = ax_time.annotate("", xy=(dates[i], 0), xycoords='data', xytext=(dates[i], -2), arrowprops=dict(arrowstyle='-|>, head_width=0.6, head_length=0.6', color='lightgreen', alpha=0.6, connectionstyle="arc3"), annotation_clip=False) 

    #     t1 = ax_time.get_annotations().set_xy(p2.get_xy() + dxy)

    # update filled plots instead of redrawing 
    # la.set_xdata(np.append(la.get_xdata(),datetime(2020, 9, 23, 10, 6)))
    # la.set_ydata(np.append(la.get_ydata(),0.5))
    # lb.set_xdata(np.append(la.get_xdata(),datetime(2020, 9, 23, 10, 6)))
    # lb.set_ydata(np.append(la.get_ydata(),0.5))    
    
    return l1, a1, l2, a2 #, t1, t2


def draw_joint_plot(i, ax, stems, bbars, hlines, focus=None):
    
    date = datelist[i]
    ax_hbar = draw_barchart_padded(date, ax)
    lines, markers, bars = draw_stemgraph(date, stems, bbars, hlines, focus)
    
    # return hbars.patches lines + [markers] + bars.patches
    return lines + [markers] + bars.patches

def draw_joint_time_plot(i, ax, stems, bbars, hlines, ax_time, dates, y1, y2, focus=None):
    
    date = datelist[i]
    ax_hbar = draw_barchart_padded(date, ax)
    lines, markers, bars = draw_stemgraph(date, stems, bbars, hlines, focus)
    l1, a1, l2, a2 = update_timeline(i, ax_time, dates, y1, y2)
    
    # return hbars.patches lines + [markers] + bars.patches
    return lines + [markers] + bars.patches + [l1, a1, l2, a2]

if __name__ == '__main__':

    duration = 200.0                # animation duration in seconds
    sec_per_transition = 1.0        # UNUSED how many seconds does a transition last
    fps = 24                      # frames per second
    nframes = int(fps*duration)        # total number of frames
    fpt = fps/sec_per_transition  # UNUSED frames per transition

    focus_country = Country.gr
    color_dict[focus_country.english_short_name] = (0.2,0.4,0.7)

    startdate='03/24/2020'        # start animation from date
    enddate='03/26/2022'          # end animation at date

    # create array of timestamps by dividing date range with uniform steps
    datelist = pd.date_range(start=startdate, end=enddate, periods=nframes)

    # interpolate data for datelist
    print("Interpolating data at", len(datelist), "times.")
    df_interp = data_interp(datelist, dfshort)
    df_interp_bydate = df_interp.groupby('date')

    #plt.style.available
    plt.style.use('dark_background')

# ------------------------------ BEGIN OLD
#     N_plot = 15

#     print('Initializing data')
#     initData(0)
#     N_ploto=11 # USE update_plot() wrapper as animation callable

#     figo, axo = plt.subplots(figsize=(15, 8))
#     animatoro = animation.FuncAnimation(figo, draw_barchart, frames=tqdm(datelist, file=sys.stdout), interval=1000./fps)
#     f_animo = os.path.join(os.getcwd(), 'animationo.mp4')
#     f_gifo = os.path.join(os.getcwd(), 'animationo.gif')
#     writerovideo = animation.FFMpegWriter(fps=fps)
#     imgkovideo = animation.ImageMagickWriter(fps=fps)
#     animatoro.save(f_animo, writer=writerovideo)

# ------------------------------ END OLD
    
    N_plot=11 # USE update_plot() wrapper as animation callable
    mycontinent='Europe'

    print('Initializing data')
    initData(0)

    fig, (ax, ax_time, ax_new) = plt.subplots(nrows=3, figsize=(14,12), gridspec_kw={'height_ratios':[12, 1, 3]}) #, constrained_layout=True)
    bbars, stems, hlines = initStemplot(0, focus=focus_country.english_short_name)
    dates, y1, y2 = init_timeline(ax_time)
    
    print('Generating Animation')
    # ars = draw_joint_time_plot(1, ax, stems, bbars, hlines, ax_time, dates, y1, y2, focus_country.english_short_name)
    animator_joint_time = animation.FuncAnimation(fig, draw_joint_time_plot, frames=len(datelist), fargs=(ax, stems, bbars, hlines, ax_time, dates, y1, y2, focus_country.english_short_name), blit=True, interval=1000./fps)

    # Save video to file using ffmpeg
    print('Saving video to file...')
    f_anim = os.path.join(os.getcwd(), 'animation.mp4')
    f_gif = os.path.join(os.getcwd(), 'animation.gif')
    writervideo = animation.FFMpegWriter(fps=fps) 
    animator_joint_time.save(f_anim, writer=writervideo)
    print("DONE")
