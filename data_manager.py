import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
import random

#常數設定
VIEW_DAYS = 250         
MIN_SIMULATION_DAYS = 720
MA_PERIODS = [5, 10, 20, 60, 120]

#計算RSI指標
def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

#主要數據抓取
@st.cache_data(ttl=3600, show_spinner="📈 正在載入並計算指標 (MA, RSI)...")
def fetch_historical_data(ticker: str = "TSLA") -> pd.DataFrame | None:
    period = 'max'  # 抓取所有可用歷史數據

    try:
        data = yf.download(ticker.upper(), period=period, interval='1d', progress=False)
        
        if data.empty:
            return None
            
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data['Date'] = pd.to_datetime(data['Date'])
        
        # 計算指標
        for p in MA_PERIODS:
            data[f'MA{p}'] = data['Close'].rolling(window=p).mean()
            
        data['RSI'] = calculate_rsi(data, window=14)
        
        # 移除 NaN 並重設索引
        data.dropna(inplace=True) 
        data = data.reset_index(drop=True)
        
        return data

    except Exception as e:
        return None
    
#隨機選取起始點
def select_random_start_index(data: pd.DataFrame) -> tuple[int, int] | None:
    total_days = len(data)
    required_days = VIEW_DAYS + MIN_SIMULATION_DAYS
    
    if total_days < VIEW_DAYS:
         return None
         
    if total_days < required_days:
        max_start_index = total_days - VIEW_DAYS
        start_view_index = 0
        sim_start_index = start_view_index + VIEW_DAYS
        
        return start_view_index, sim_start_index
    
    max_start_index = total_days - required_days
    
    start_view_index = random.randint(0, max_start_index)
    sim_start_index = start_view_index + VIEW_DAYS
    
    return start_view_index, sim_start_index

#根據索引取得價格資訊，並強制將日期轉換為 Python 原生 datetime 物件。
def get_price_info_by_index(data: pd.DataFrame, index: int) -> tuple[datetime, float, float]:
    if data is not None and index < len(data):
        current_row = data.iloc[index]
        
        # 1. 取得日期物件
        date_timestamp = current_row['Date']
        
        # 2. 如果是 Series，先用 .iloc[0] 提取單個 Timestamp (針對不同 Pandas 版本防呆)
        if isinstance(date_timestamp, pd.Series):
             date_timestamp = date_timestamp.iloc[0]
        
        # 3. 強制轉換為 Python 原生的 datetime 物件 
        date = date_timestamp.to_pydatetime() 
        
        # 價格使用 .item() 提取單個 float，這是必要的
        open_price = current_row['Open'].item()
        close_price = current_row['Close'].item()
        
        return date, open_price, close_price
    return datetime.now(), 0.0, 0.0

