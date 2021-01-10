import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from statsmodels.tsa.stattools import adfuller,kpss,grangercausalitytests
from numpy import log

pd.options.mode.chained_assignment = None

def time_interval_tagger(csv_string,interval):
    """
    Interval (Monthly): 1 means monthly, 3 means quarterly, 6 means half-yearly
    Purpose: Tag a year slice for each document for subsequent time series analysis 
    """
    df = pd.read_csv(csv_string)
    for i in tqdm(range(len(df))):
        date = df.loc[i,'comment_timestamp'].split()[0]
        if interval == 1:
            df.loc[i,'Year Slice'] = int(date[:4]) +  (interval/12) * ((int(date[5:7])//interval)-1) 
        elif interval == 3:
            if int(date[5:7])%3 == 0:
                df.loc[i,'Year Slice'] = int(date[:4]) +  (interval/12) * ((int(date[5:7])//interval)-1) 
            else:
                df.loc[i,'Year Slice'] = int(date[:4]) + (interval/12) * (int(date[5:7])//interval)
        elif interval == 4:
            if int(date[5:7])%4 == 0:
                df.loc[i,'Year Slice'] = int(date[:4]) +  (interval/12) * ((int(date[5:7])//interval)-1) 
            else:
                df.loc[i,'Year Slice'] = int(date[:4]) + (interval/12) * (int(date[5:7])//interval)
        else:
            df.loc[i,'Year Slice'] = int(date[:4]) + (interval/12) * (int(date[5:7])//interval)
    return df

def derive_facet_grid_with_adfuller(df,interval):
    year_count_df = df.groupby('Year Slice').count()
    year_count_df.reset_index(inplace=True)
    year_count_df=year_count_df[['Year Slice','title']]
    year_count_df.rename(columns={'title':'count'},inplace=True)
    year_count_dict = {}
    for i in range(len(year_count_df)):
        year_count_dict[year_count_df.loc[i,'Year Slice']] = year_count_df.loc[i,'count']
    reddit_grouped_year = df.groupby(['Year Slice','Dominant_Topic']).count()
    reddit_grouped_year.reset_index(inplace=True)
    reddit_grouped_year= reddit_grouped_year[['Year Slice','Dominant_Topic','title']]
    reddit_grouped_year.rename(columns={'title':'count'},inplace=True)
    year_slice_lst = []
    for x in [2015,2016,2017,2018,2019,2020]:
        factor = int(12/interval)
        for i in range(0,factor+1):
            value = x + i * (interval/12)
            year_slice_lst.append(value)
    year_slice_list = list(dict.fromkeys(year_slice_lst))
    year = pd.DataFrame(year_slice_list,columns=['Year Slice'])
    dominant_topic_list = [int(x) for x in sorted(list(df['Dominant_Topic'].unique()))]
    dominant_topic= pd.DataFrame(dominant_topic_list,columns=['Dominant_Topic'])
    # Form cartesian product of tables (cross join)
    year['count'] = 0
    dominant_topic['count'] = 0
    new_reddit_grouped = year.merge(dominant_topic,how='outer')
    new_reddit_grouped.drop(columns=['count'],inplace=True)
    new_reddit = new_reddit_grouped.merge(reddit_grouped_year,how='left',on=['Year Slice','Dominant_Topic'])
    new_reddit = new_reddit.fillna(0)
    for i in range(len(new_reddit)):
        if new_reddit.loc[i,'Year Slice'] in year_count_dict.keys():
            new_reddit.loc[i,'Year_Slice_Count'] = year_count_dict[new_reddit.loc[i,'Year Slice']]
        new_reddit.loc[i,'Perc_Topic_by_year'] = (new_reddit.loc[i,'count'] / new_reddit.loc[i,'Year_Slice_Count'])*100
    new_reddit = new_reddit.fillna(0)
    new_reddit.drop(columns=['count','Year_Slice_Count'],inplace=True)
    new_reddit_pivot = pd.pivot_table(new_reddit,columns='Dominant_Topic',index='Year Slice')
    new_reddit_pivot.columns = new_reddit_pivot.columns.get_level_values(1)
    topic_dict = {}
    for x in dominant_topic_list:
        topic_dict[x] = 'Topic {}'.format(x)
    new_reddit_pivot.rename(columns=topic_dict,inplace=True)
    new_reddit_pivot.reset_index(inplace=True)
    new_reddit_pivot['index'] = new_reddit_pivot.index
    long_reddit = pd.wide_to_long(new_reddit_pivot,stubnames= 'Topic ',i='index',j='topic')
    long_reddit.columns = long_reddit.columns.get_level_values(0)
    long_reddit.reset_index(inplace=True)
    long_reddit.rename(columns={'Topic ':'Topic Contribution'},inplace=True)
    # Drop all November and December values as they screw up the time series
    long_reddit = long_reddit[long_reddit['Year Slice']<=2020.75000]
    long_reddit.reset_index(inplace=True,drop=True)
    # Create the facet grid for the plots
    sns.set(font_scale=3)
    g = sns.FacetGrid(long_reddit,col="topic",height=9,aspect=3.5,sharex=False,sharey=False,col_wrap=2)
    g.map_dataframe(sns.lineplot, x="Year Slice",y='Topic Contribution')
    g.set(ylabel='Percentage Contribution (%)',xlabel='Year')
    g.fig.subplots_adjust(wspace=0.3,hspace=0.2)
    g.savefig('facetgrid.png')
    plt.show()
    # Enacting the adfuller test + KPSS test to ascertain stationarity and printing out to a dataframe 
    data = []
    for i in range(min(dominant_topic_list),max(dominant_topic_list)+1):
        df_list = []
        sliced_df = long_reddit[long_reddit.topic == i]
        X = sliced_df['Topic Contribution']
        result = adfuller(X,autolag='AIC')
        # Start with the ADF Tests
        print('Enacting ADF Tests:\n')
        print('Topic {}:\n'.format(i))
        df_list.append(i)
        print('ADF Statistic: %f' % result[0])
        df_list.append(result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        df_list.append(result[4]['1%']) 
        df_list.append(result[4]['5%']) 
        df_list.append(result[4]['10%']) 
        result_text = ''
        if result[1] < 0.05:
            result_text = 'Time Series is Stationary'
            df_list.append(result_text)
            print('Time Series is Stationary')
        else:
            result_text = 'Time Series is Non-Stationary'
            df_list.append(result_text)
            print('Time Series is Non-Stationary')
        # Move onto the KPSS Tests 
        print('Enacting KPSS Tests:\n')
        kpss_result = kpss(X,regression='c',nlags='auto')
        print('Topic {}:\n'.format(i))
        print('KPSS Statistic: %f' % kpss_result[0])
        df_list.append(kpss_result[0])
        print('p_value: %f' % kpss_result[1])
        for key, value in kpss_result[3].items():
            print('\t%s: %.3f' % (key, value))
        df_list.append(kpss_result[3]['1%']) 
        df_list.append(kpss_result[3]['5%']) 
        df_list.append(kpss_result[3]['10%']) 
        if kpss_result[0] < kpss_result[3]['5%']:
            result_text = 'Time Series is Stationary'
            df_list.append(result_text)
            print('Time Series is Stationary')
        else:
            result_text = 'Time Series is Non-Stationary'
            df_list.append(result_text)
            print('Time Series is Non-Stationary')
        data.append(df_list)
    stationarity_df = pd.DataFrame(data,columns=['Topic Number','ADF Statistic','Critical Value (ADF 1%)','Critical Value (ADF 5%)','Critical Value (ADF 10%)','ADF Result','KPSS Statistic','Critical Value (KPSS 1%)','Critical Value (KPSS 5%)','Critical Value (KPSS 10%)','KPSS Result'])
    for i in range(len(stationarity_df)):
        if stationarity_df.loc[i,'ADF Result'] == 'Time Series is Stationary' and stationarity_df.loc[i,'KPSS Result'] == 'Time Series is Stationary':
            stationarity_df.loc[i,'Final Stationarity Result'] = 'Definite Stationary'
        elif stationarity_df.loc[i,'ADF Result'] == 'Time Series is Non-Stationary' and stationarity_df.loc[i,'KPSS Result'] == 'Time Series is Non-Stationary':
            stationarity_df.loc[i,'Final Stationarity Result'] = 'Non-Stationary'
        else: 
            stationarity_df.loc[i,'Final Stationarity Result'] = 'Non-Conclusive'
    stationarity_df.to_csv('stat_df.csv',index=False)
    return stationarity_df, g

if __name__ == "__main__":
    csv_string = 'reddit_lda.csv'
    tagged_reddit = time_interval_tagger(csv_string,3)
    stat_df, facet_grid = derive_facet_grid_with_adfuller(tagged_reddit,3)
