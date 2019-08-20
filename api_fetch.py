from requests import get
import pandas as pd
from boto.s3.connection import S3Connection
from datetime import datetime, timedelta
# import os

def harvest(which):
    # Gets secret api key from quandl config vars
    # quandl = S3Connection(os.environ['use_key'])
    quandl = "SigGdWFysHuacAVCDsCd"

    # Group 1, keep Change, Settle, Volume, Previous Day Open Interest
    # The 1 futures of the metals are also harvested for only the daily
    CMEgold_futures = f"https://www.quandl.com/api/v3/datasets/CHRIS/CME_GC1.json?api_key={quandl}"
    CMEgold_futures2 = f"https://www.quandl.com/api/v3/datasets/CHRIS/CME_GC2.json?api_key={quandl}"
    CMEsilver_futures = f"https://www.quandl.com/api/v3/datasets/CHRIS/CME_SI1.json?api_key={quandl}"
    CMEsilver_futures2 = f"https://www.quandl.com/api/v3/datasets/CHRIS/CME_SI2.json?api_key={quandl}"
    CMEpalladium_futures = f"https://www.quandl.com/api/v3/datasets/CHRIS/CME_PA1.json?api_key={quandl}"
    CMEpalladium_futures2 = f"https://www.quandl.com/api/v3/datasets/CHRIS/CME_PA2.json?api_key={quandl}"
    CMEplatinum_futures = f"https://www.quandl.com/api/v3/datasets/CHRIS/CME_PL1.json?api_key={quandl}"
    CMEplatinum_futures2 = f"https://www.quandl.com/api/v3/datasets/CHRIS/CME_PL2.json?api_key={quandl}"
    CME_TenYear_futures = f"https://www.quandl.com/api/v3/datasets/CHRIS/CME_TY1.json?api_key={quandl}"
    CME_TenYear_futures2 = f"https://www.quandl.com/api/v3/datasets/CHRIS/CME_TY2.json?api_key={quandl}"

    #Group 2, keep Change, Settle, Wave, Volume, Prev. Day Open Interest
    ICE_USD_futures = f"https://www.quandl.com/api/v3/datasets/CHRIS/ICE_DX1.json?api_key={quandl}"
    ICE_USD_futures2 = f"https://www.quandl.com/api/v3/datasets/CHRIS/ICE_DX2.json?api_key={quandl}"
    ICE_ZAR_futures = f"https://www.quandl.com/api/v3/datasets/CHRIS/ICE_ZR1.json?api_key={quandl}"
    ICE_ZAR_futures2 = f"https://www.quandl.com/api/v3/datasets/CHRIS/ICE_ZR2.json?api_key={quandl}"
    # SGX_EuroD_futures = f"https://www.quandl.com/api/v3/datasets/CHRIS/SGX_ED1.json?api_key={quandl}"

    #Group 3, keep Open Interest, last one on list also decides cut off date.
    # Last one needs to be shortest weekly separation
    Gold_OpenInt = f"https://www.quandl.com/api/v3/datasets/CFTC/088691_FO_ALL.json?api_key={quandl}"
    Silver_OpenInt = f"https://www.quandl.com/api/v3/datasets/CFTC/084691_FO_ALL.json?api_key={quandl}"
    Palladium_OpenInt = f"https://www.quandl.com/api/v3/datasets/CFTC/075651_FO_ALL.json?api_key={quandl}"
    Platinum_OpenInt = f"https://www.quandl.com/api/v3/datasets/CFTC/076651_FO_ALL.json?api_key={quandl}"

    # Sets the URLs in iterable list
    url_list = [CMEgold_futures,CMEgold_futures2,CMEsilver_futures,CMEsilver_futures2,CMEpalladium_futures,CMEpalladium_futures2,CMEplatinum_futures,CMEplatinum_futures2,CME_TenYear_futures,CME_TenYear_futures2,ICE_USD_futures,ICE_USD_futures2,ICE_ZAR_futures,ICE_ZAR_futures2,Gold_OpenInt,Silver_OpenInt,Palladium_OpenInt,Platinum_OpenInt]
    
    # This is here to avoid the timeout
    how_many = {"a":[0,1,2,3,4,5],"b":[6,7,8,9,10,11],"c":[12,13,14,15,16,17]}
    info_list = [get(url_list[i]).json() for i in how_many[which]]
    df_list = [pd.DataFrame(info["dataset"]["data"],columns=info["dataset"]["column_names"]) for info in info_list]
    for i in range(len(df_list)):
        df_list[i].to_csv(f"data/csv/Data_Raw_{how_many[which][i]}.csv")
    # # Gets the info in iterable list
    # info_list = [get(url).json() for url in url_list]
    # # Gets info in pd dataframes in iterable list
    # df_list = [pd.DataFrame(info["dataset"]["data"],columns=info["dataset"]["column_names"]) for info in info_list]
    # for i in range(len(df_list)):
    #     df_list[i].to_csv(f"data/csv/Data_Raw_{i}.csv")
    return 1

def manipulate():
    # This is broken up to avoid a timeout
    # Condenses and saves some relevant parts of df_list
    df_list = [pd.read_csv(f"data/csv/Data_Raw_{i}.csv") for i in range(18)]
    daily_metal = {0:"Gold",2:"Silver",4:"Palladium",6:"Platinum"}
    for i in range(0,7,2):
        df_list[i].to_csv(f"data/csv/{daily_metal[i]}Prices.csv")
    # Removes 
    for i in range(1,6):
        df_list[i-1]["Previous Day Open Interest"] += df_list[i]["Previous Day Open Interest"]
        df_list.pop(i)
    for i in range(6,8):
        df_list[i-1]["Prev. Day Open Interest"] += df_list[i]["Prev. Day Open Interest"]
        df_list.pop(i)
    
    # Creates a list that functions well in testing based off manual assignment from domain knowledge
    pruned_list = [None]*11
    # prunes group 1
    metal_namer = {0:"Gold",1:"Silver",2:"Palladium",3:"Platinum",4:"Ten_Yr_Futures"}
    for i in range(5):
        pruned_list[i] = df_list[i][["Date","Change","Settle","Volume","Previous Day Open Interest"]]
        pruned_list[i].columns = ["Date",f"{metal_namer[i]}_Change",f"{metal_namer[i]}_Settle",f"{metal_namer[i]}_Volume",f"{metal_namer[i]}_Prev. Day Open Interest"]
    money_namer = {5:"USD",6:"ZAR"}
    # prunes group 2
    for i in range(5,7):
        pruned_list[i] = df_list[i][["Date","Change","Settle","Volume","Wave","Prev. Day Open Interest"]]
        pruned_list[i].columns = ["Date",f"{money_namer[i]}_Change",f"{money_namer[i]}_Settle",f"{money_namer[i]}_Volume",f"{money_namer[i]}_Wave",f"{money_namer[i]}_Prev. Day Open Interest"]
    # prunes the last group
    for i in range(7,11):
        pruned_list[i] = df_list[i][["Date","Open Interest","Money Manager Shorts","Money Manager Longs","Producer/Merchant/Processor/User Longs","Producer/Merchant/Processor/User Shorts"]]
    
    # Gets cut off date before which records are too old to be compatible with other more modern records
    cut_off_date = df_list[len(df_list)-1]["Date"][len(df_list[len(df_list)-1].index)-1]
    cut_off_date = (datetime.strptime(cut_off_date,"%Y-%m-%d") - timedelta(days=6))
    cut_off_date = cut_off_date.strftime("%Y-%m-%d")

    # prunes all groups that have old records by the date
    for i in range(7):
        pruned_list[i] = pruned_list[i].loc[pruned_list[i]["Date"]>=cut_off_date]
    
    # Creates a list of merged dataframes that cover the same subject but with different records
    # Groups the daily records to weekly to make this possible
    grouped_list = [None]*7
    #   Creates the weekly bins that data will be placed in 
    tuple_list = [((datetime.strptime(cut_off_date,"%Y-%m-%d")+timedelta(days=(w*7))),(datetime.strptime(cut_off_date,"%Y-%m-%d")+timedelta(days=(w*7 + 6)))) for w in range(len(df_list[len(df_list)-1].index))]
    cut_bins = pd.IntervalIndex.from_tuples(tuple_list,closed="both")
    #   Step that fills the grouped list
    for bin_candidate in range(7):
        # Determines the method by which a column should be aggregated
        agg_dic = {column:("sum" if "change" in column.lower() else "mean") for column in list(pruned_list[bin_candidate].columns)}
        agg_dic.pop("Date")
        bins = pd.cut(pruned_list[bin_candidate]["Date"],cut_bins)
        grouped_list[bin_candidate] = pruned_list[bin_candidate].groupby(bins).agg(agg_dic)
        # Adds back the date column to the dataframes so that they can be merged
        grouped_list[bin_candidate] = grouped_list[bin_candidate].assign(Date=list(pruned_list[len(pruned_list)-1]["Date"].iloc[::-1]))
        grouped_list[bin_candidate].index.names = ["Business_Week"]
    
    # Creates an iterable list of merged dataframes considering dublicate info relevant to each metal
    # This could be better stored in a sql database to prevent redundancy.
    # This is faster and easier and we didn't have the time, so...
    merged_list = [grouped_list[i].\
        merge(pruned_list[7+i], on = "Date", how = "inner").\
        merge(grouped_list[4], on = "Date", how = "inner").\
        merge(grouped_list[5], on = "Date", how = "inner").\
        merge(grouped_list[6], on = "Date", how = "inner").\
        set_index(grouped_list[i].index) for i in range(4)]
    
    # Saves the 4 merged dataframes of info to csv
    for i in range(4):
        merged_list[i].to_csv(f"data/csv/{metal_namer[i]}_Data.csv")
    
    return 1