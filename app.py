import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np 
import uuid 
from datetime import datetime
import random 

#導入data_manager
# 假設 data_manager.py 檔已存在且內容如預期
from data_manager import (
    fetch_historical_data, 
    select_random_start_index, 
    get_price_info_by_index, 
    VIEW_DAYS,             
    MIN_SIMULATION_DAYS, 
    MA_PERIODS
)

#初始化狀態與常數
DEFAULT_TICKER = "TSLA" 
INITIAL_CAPITAL = 100000.0
MA_COLORS = {5: 'lightgray', 10: 'gray', 20: 'red', 60: 'blue', 120: 'white'}

# --- 交易/槓桿常數 (Req 3: 修正手續費率) ---
FEE_RATE = 0.005
LEVERAGE_FEE_RATE = 0.01 
MIN_MARGIN_RATE = 0.05 # 最小保證金比例 5% (用於計算強制平倉價，即最大槓桿 20倍)

# --- 資產類型與單位映射 ---
ASSET_CONFIGS = {
    'Stock': {'unit': '股', 'mode_long': '現貨買', 'mode_short': '現貨空', 'mode_margin_long': '融資多', 'mode_margin_short': '融券空', 'default_qty': 1000.0, 'min_qty': 1.0}, 
    # Req: 匯率調整為 100 點
    'Forex': {'unit': '點', 'mode_long': '現貨買', 'mode_short': '現貨空', 'mode_margin_long': '保證金多', 'mode_margin_short': '保證金空', 'default_qty': 100.0, 'min_qty': 100.0}, 
    'Crypto': {'unit': '顆', 'mode_long': '現貨買', 'mode_short': '現貨空', 'mode_margin_long': '合約多', 'mode_margin_short': '合約空', 'default_qty': 1.0, 'min_qty': 0.001}
}
# --- 交易模式映射 ---
TRADE_MODE_MAP = {
    'Spot_Buy': {'mode_type': 'Spot', 'position_type': '多頭', 'trans_type': '現貨買入開倉', 'pos_mode': '現貨'},
    'Margin_Long': {'mode_type': 'Margin', 'position_type': '多頭', 'trans_type': '槓桿買入開倉', 'pos_mode': '融資'},
    'Margin_Short': {'mode_type': 'Margin', 'position_type': '空頭', 'trans_type': '槓桿賣出開倉', 'pos_mode': '融券'},
}

#Session State 初始化
st.session_state.setdefault('ticker', DEFAULT_TICKER)
st.session_state.setdefault('asset_type', 'Stock') 
st.session_state.setdefault('initialized', False)
st.session_state.setdefault('core_data', None)
st.session_state.setdefault('start_view_index', 0)
st.session_state.setdefault('current_sim_index', 0)
st.session_state.setdefault('max_sim_index', 0)
st.session_state.setdefault('sim_active', True)
st.session_state.setdefault('end_sim_index_on_settle', None) 
st.session_state.setdefault('balance', INITIAL_CAPITAL)
st.session_state.setdefault('plot_layout', None) # 用於保存 Plotly 佈局/縮放狀態 (Req 2)

st.session_state.setdefault('positions', []) 
st.session_state.setdefault('transactions', [])
st.session_state.setdefault('start_date', None) 

#計算當前總資產(現金+所有倉位的未實現市值/淨值)
def get_current_asset_value(core_data, current_idx):
    if st.session_state.core_data is None or st.session_state.core_data.empty:
         return st.session_state.balance
         
    if st.session_state.sim_active and current_idx < len(core_data):
        price = core_data['Open'].iloc[current_idx].item() if 'Open' in core_data.columns else 0.0
    else:
        # 模擬結束後，使用最後的現金餘額作為總資產
        return st.session_state.balance
    
    # 總部位淨值計算
    total_position_net_value = 0.0
    
    for pos in st.session_state.positions:
        qty = pos['qty']
        cost = pos['cost']
        pos_mode = pos['pos_mode']
        
        # 現貨 (Spot): 市值 (Value)
        if pos_mode == '現貨':
             # 現貨部位的本金已從 balance 扣除，所以這裡計算市值來加入總資產
             total_position_net_value += (qty * price)
             
        # 融資/融券 (Margin/Leveraged): 原始保證金 + 未實現損益
        elif pos_mode in ['融資', '融券']:
             # 原始保證金
             initial_cost = pos['initial_cost'] 
             leverage = pos['leverage']
             margin_required = initial_cost / leverage
             
             # 未實現損益 (PnL)
             if pos_mode == '融資':
                 unrealized_pnl = (qty * price) - (qty * cost)
             else: # 融券/合約空
                 unrealized_pnl = (qty * cost) - (qty * price)
                 
             # 淨值 = 保證金 + 未實現損益
             total_position_net_value += (margin_required + unrealized_pnl)
            
    # 總資產 = 可用現金(餘額) + 所有部位的淨值
    return st.session_state.balance + total_position_net_value

#計算所有倉位的總未實現損益 (包含現貨與槓桿)
def get_total_unrealized_pnl(price):
    total_pnl = 0.0
    for pos in st.session_state.positions:
        qty = pos['qty']
        cost = pos['cost']
        
        # 多頭 (現貨/融資)
        if pos['pos_mode'] in ['現貨', '融資']:
            total_pnl += (qty * price) - (qty * cost)
        # 空頭 (融券)
        elif pos['pos_mode'] in ['融券']:
            total_pnl += (qty * cost) - (qty * price)
            
    return total_pnl

# --- 現貨部位彙總 ---
def get_spot_summary(core_data, current_idx):
    if not st.session_state.sim_active or core_data is None or current_idx >= len(core_data):
        return {'qty': 0.0, 'avg_cost': 0.0, 'unrealized_pnl': 0.0}

    price = core_data['Open'].iloc[current_idx].item()
    
    spot_positions = [pos for pos in st.session_state.positions if pos['pos_mode'] == '現貨']
    
    if not spot_positions:
        return {'qty': 0.0, 'avg_cost': 0.0, 'unrealized_pnl': 0.0}

    # Aggregate quantities and total cost for average calculation
    total_qty = sum(pos['qty'] for pos in spot_positions)
    total_cost = sum(pos['qty'] * pos['cost'] for pos in spot_positions)
    
    avg_cost = total_cost / total_qty if total_qty > 0 else 0.0
    
    # Calculate unrealized PnL
    unrealized_pnl = sum((pos['qty'] * price) - (pos['qty'] * pos['cost']) for pos in spot_positions)
    
    return {
        'qty': total_qty, 
        'avg_cost': avg_cost, 
        'unrealized_pnl': unrealized_pnl
    }

#資產歸零或為負時，結束模擬
def check_and_end_simulation(asset_value):
    if asset_value <= 0:
        # 如果已經在結束狀態，就不重複報錯
        if st.session_state.sim_active: 
            st.session_state.sim_active = False
            st.error("🚨風險控制警告！總資產已歸零或為負，模擬強制結束！")
        return True
    return False

# --- 結算所有倉位 ---
def settle_portfolio(force_end=False):
    """
    結算所有持倉部位。
    如果 force_end=True (提早結算)，則結束模擬並使用收盤價結算。
    如果 force_end=False (平倉所有倉位按鈕)，則繼續模擬並使用開盤價結算。
    """
    if not st.session_state.sim_active and not force_end:
        return st.warning("模擬已結束。")

    # 1. 決定結算價格
    current_idx = st.session_state.current_sim_index
    core_data = st.session_state.core_data

    if core_data is None or core_data.empty:
        return st.warning("無數據可供結算。")

    if current_idx >= len(core_data):
        # 處理索引超出範圍的情況 (例如 next_ten_days 跑到最後一天)
        settle_price = core_data['Close'].iloc[-1].item() if not core_data.empty else 0.0
    elif force_end:
        # 提早結算，使用收盤價
        settle_price = core_data['Close'].iloc[current_idx].item()
    else:
        # 手動平倉所有，使用開盤價
        settle_price = core_data['Open'].iloc[current_idx].item()

    if settle_price <= 0:
        st.error("結算失敗：無法取得有效的結算價格。")
        if force_end:
             st.session_state.sim_active = False # 強制結束
             st.session_state.end_sim_index_on_settle = current_idx
        return

    positions_to_close = list(st.session_state.positions) # 複製列表以迭代

    if not positions_to_close:
        if force_end:
            st.info("模擬結束，沒有持倉部位需要結算。")
    else:
        if force_end:
            st.info(f"開始結算 {len(positions_to_close)} 個持倉部位 (強制結束)，結算價格: ${settle_price:,.2f}")
        else:
             st.info(f"開始平倉 {len(positions_to_close)} 個持倉部位 (繼續模擬)，平倉價格: ${settle_price:,.2f}")
             
        for pos in positions_to_close:
            # 必須檢查 pos 是否仍在 session_state.positions 內，
            # 避免在迭代過程中被 close_position_lot 移除
            if pos in st.session_state.positions: 
                trade_type = '自動結算賣出平倉' if pos['pos_mode'] in ['現貨', '融資'] else '自動結算買回平倉'
                
                # close_position_lot 會更新 positions 列表
                close_position_lot(pos['id'], pos['qty'], settle_price, trade_type, pos['pos_mode'], mode='自動結算')

    # 2. 決定是否結束模擬狀態
    if force_end:
        st.session_state.sim_active = False
        st.session_state.end_sim_index_on_settle = current_idx
        
        final_asset = get_current_asset_value(core_data, current_idx)
        
        # 避免重複顯示 "總資產已歸零" 的錯誤
        if final_asset > 0:
            st.success(f"所有部位結算完成！最終總資產: ${final_asset:,.2f}")
    
#重新開始回測前初始化
def reset_state():
    st.session_state.initialized = False
    st.session_state.core_data = None
    st.session_state.start_view_index = 0
    st.session_state.current_sim_index = 0
    st.session_state.max_sim_index = 0
    st.session_state.sim_active = True
    st.session_state.balance = INITIAL_CAPITAL
    st.session_state.transactions = []
    st.session_state.start_date = None
    st.session_state.end_sim_index_on_settle = None 
    st.session_state.positions = []
    st.session_state.plot_layout = None # 重置圖表布局狀態

#設定回測起始點 
def initialize_data_and_simulation(asset_type):
    # Req 4: 使用輸入的 ticker 抓數據，並使用選取的 asset_type 來定義交易規則。
    ticker = st.session_state.ticker.upper()
    
    data = fetch_historical_data(ticker) 

    if data is None: 
        st.error(f"無法載入 {st.session_state.ticker} 的數據，請確認代碼是否正確。")
        return
        
    st.session_state.core_data = data
    
    total_days = len(data)
    required_days = VIEW_DAYS + MIN_SIMULATION_DAYS
    
    if total_days < required_days:
        st.warning(f"注意：{st.session_state.ticker} 有效數據 ({total_days} 天) 少於回測所需最低天數 ({required_days} 天)。回測將從最早數據開始，且長度不足 720 根。")
            
    st.success(f"{st.session_state.ticker} 數據載入成功！共 {total_days} 筆有效數據。")

    start_indices = select_random_start_index(st.session_state.core_data)
    if start_indices is not None:
        start_view_idx, _ = start_indices
        
        data_end_idx = start_view_idx + required_days
        truncated_data = st.session_state.core_data.iloc[start_view_idx:data_end_idx].reset_index(drop=True)

        st.session_state.core_data = truncated_data
        
        st.session_state.start_view_index = 0
        st.session_state.current_sim_index = VIEW_DAYS
        st.session_state.max_sim_index = len(truncated_data) - 1
        
        st.session_state.initialized = True
        st.session_state.sim_active = True
        st.session_state.asset_type = asset_type
        
        date_ts = st.session_state.core_data['Date'].iloc[st.session_state.current_sim_index]
        st.session_state.start_date = date_ts.to_pydatetime()

        unit = ASSET_CONFIGS[asset_type]['unit']
        st.success(f"回測已初始化！**{st.session_state.ticker}** 的日線模擬 ({unit}為單位)。")
        st.info(f"💡 規則依據您選擇的 **{asset_type}** 類型執行。")


#平倉記錄 
def close_position_lot(pos_id: str, settle_qty: float, settle_price: float, trade_type: str, pos_mode: str, mode: str = '自動'):
    pos_index = next((i for i, pos in enumerate(st.session_state.positions) if pos['id'] == pos_id), -1)
    
    if pos_index == -1: 
        return False
    
    pos = st.session_state.positions[pos_index]
    
    # 數量檢查 (現在所有 qty 都是 float，直接比較)
    if settle_qty <= 0 or settle_qty > pos['qty']: 
        min_qty = ASSET_CONFIGS[st.session_state.asset_type]['min_qty']
        st.error(f"平倉失敗：平倉股數 {settle_qty:,.3f} 無效或超過持有股數 {pos['qty']:,.3f}。")
        return False

    current_datetime, _, _ = get_price_info_by_index(st.session_state.core_data, st.session_state.current_sim_index)
    
    # --- 1. 計算手續費並扣除 (依照模式區分手續費率) ---
    is_leverage = pos_mode in ['融資', '融券']
    fee_rate_used = LEVERAGE_FEE_RATE if is_leverage else FEE_RATE
    
    close_amount = settle_qty * settle_price
    close_fee = close_amount * fee_rate_used
    
    # 2. 扣除平倉手續費
    st.session_state.balance -= close_fee
    
    # 3. 處理平倉邏輯
    is_fully_closed = (settle_qty == pos['qty'])
    
    # 計算應歸還的保證金比例
    original_qty = pos['qty']
    original_initial_cost = pos['initial_cost'] 
    leverage = pos.get('leverage', 1.0)
    original_margin = original_initial_cost / leverage 
    
    # 按比例歸還保證金或現貨成本
    if pos_mode == '現貨':
        return_margin_or_cost = settle_qty * settle_price # 現貨是直接回流資金 (成本+損益)
        realized_pnl = settle_qty * (settle_price - pos['cost'])
    
    # 槓桿部位 (融資/融券)
    elif pos_mode in ['融資', '融券']: 
        
        is_long = (pos_mode == '融資')
        
        # PnL 計算
        if is_long:
            realized_pnl = settle_qty * (settle_price - pos['cost'])
        else: # 融券/合約空
            realized_pnl = settle_qty * (pos['cost'] - settle_price)
            
        # 歸還的保證金 (只有槓桿部位需要)
        return_margin_or_cost = original_margin * (settle_qty / original_qty)
    
    else:
        return False

    # 4. 將 PnL + 歸還的保證金/現貨成本 存入現金
    if pos_mode == '現貨':
        # 現貨: 現金回流 = 平倉總額 (包含損益)
        st.session_state.balance += return_margin_or_cost
    else:
        # 槓桿: 現金回流 = 歸還的保證金 + 實現損益
        st.session_state.balance += (return_margin_or_cost + realized_pnl)
    
    # 5. 記錄交易紀錄
    transactions_entry = {
        '模式': pos_mode, 
        '類型': trade_type, 
        '股數': -settle_qty, # 平倉股數永遠是負的
        '價格': settle_price, 
        '金額': return_margin_or_cost, 
        '損益': realized_pnl,
        '開倉總值': settle_qty * pos['cost'], 
        '手續費': close_fee,
        '日期': current_datetime,
        'leverage': leverage 
    }
    st.session_state.transactions.append(transactions_entry)
    
    # 6. 更新倉位或移除
    if is_fully_closed:
        st.session_state.positions.pop(pos_index)
        st.info(f"倉位 ID {pos_id[-4:]} 已完全平倉 ({trade_type}) (實現損益: ${realized_pnl:,.2f})。")
    else: 
        new_qty = pos['qty'] - settle_qty
        
        # 按比例調整 pos 的 'initial_cost'，以計算剩餘部位的保證金
        pos['initial_cost'] = pos['initial_cost'] * (new_qty / pos['qty'])
        st.session_state.positions[pos_index]['qty'] = new_qty
        
        st.info(f"倉位 ID {pos_id[-4:]} 已部分平倉 {settle_qty:,.3f} {ASSET_CONFIGS[st.session_state.asset_type]['unit']} (剩餘 {new_qty:,.3f} {ASSET_CONFIGS[st.session_state.asset_type]['unit']})。")

    # 7. 平倉後檢查風控
    total_asset_new = get_current_asset_value(st.session_state.core_data, st.session_state.current_sim_index)
    check_and_end_simulation(total_asset_new)
    
    return True
    
#檢查所有獨立倉位的止損/止盈/強制平倉觸發
def check_sl_tp_trigger(core_data, current_idx):
    if not st.session_state.sim_active: return
    if current_idx >= len(core_data): return

    high = core_data['High'].iloc[current_idx].item()
    low = core_data['Low'].iloc[current_idx].item()
    
    positions_to_close_info = [] 
    
    for pos in st.session_state.positions:
        sl = pos['sl']
        tp = pos['tp']
        triggered = False
        settle_price = 0.0
        close_type = ''
        
        # --- 強制平倉檢查 (Liquidation Check) ---
        liq_price = pos.get('liquidation_price', 0.0)
        is_margin = pos['pos_mode'] in ['融資', '融券']

        if is_margin and liq_price > 0:
            if pos['pos_mode'] == '融資': 
                if low <= liq_price:
                    settle_price = liq_price 
                    triggered = True
                    close_type = '強制平倉多頭'
            
            elif pos['pos_mode'] == '融券': 
                 if high >= liq_price:
                    settle_price = liq_price 
                    triggered = True
                    close_type = '強制平倉空頭'
        
        # --- SL/TP 檢查 (如果尚未觸發強制平倉) ---
        if not triggered:
            # 多頭 (現貨/融資)
            if pos['pos_mode'] in ['現貨', '融資'] and pos['qty'] > 0:
                if sl > 0 and low <= sl: 
                    settle_price = sl 
                    triggered = True
                    close_type = 'SL/TP 賣出平倉'
                    st.warning(f"🛑 倉位 {pos['id'][-4:]} **多頭停損觸發** 於 ${settle_price:,.2f}！")
                    
                elif tp > 0 and high >= tp: 
                    settle_price = tp 
                    triggered = True
                    close_type = 'SL/TP 賣出平倉'
                    st.success(f"✅ 倉位 {pos['id'][-4:]} **多頭停利觸發** 於 ${settle_price:,.2f}！")

            # 空頭 (融券)
            elif pos['pos_mode'] in ['融券'] and pos['qty'] > 0:
                if sl > 0 and high >= sl: 
                    settle_price = sl 
                    triggered = True
                    close_type = 'SL/TP 買回平倉'
                    st.error(f"❌ 倉位 {pos['id'][-4:]} **空頭停損觸發** 於 ${settle_price:,.2f}！")
                    
                elif tp > 0 and low <= tp: 
                    settle_price = tp 
                    triggered = True
                    close_type = 'SL/TP 買回平倉'
                    st.success(f"✅ 倉位 {pos['id'][-4:]} **空頭停利觸發** 於 ${settle_price:,.2f}！")
        
        
        if triggered and settle_price > 0:
            positions_to_close_info.append({
                'id': pos['id'], 
                'qty': pos['qty'], 
                'price': settle_price,
                'type': close_type,
                'pos_mode': pos['pos_mode']
            })

    #處理所有觸發的平倉
    for close_info in positions_to_close_info:
        close_position_lot(close_info['id'], close_info['qty'], close_info['price'], close_info['type'], close_info['pos_mode'], mode='自動')

#執行單一交易日的模擬推進邏輯
def _advance_one_day():
    if not st.session_state.sim_active: return False

    if st.session_state.current_sim_index < st.session_state.max_sim_index:
        st.session_state.current_sim_index += 1

        #檢查SL/TP/Liq觸發
        check_sl_tp_trigger(st.session_state.core_data, st.session_state.current_sim_index)
        
        # 檢查風控
        total_asset_new = get_current_asset_value(st.session_state.core_data, st.session_state.current_sim_index)
        return not check_and_end_simulation(total_asset_new)
    else:
        # 如果是最後一天，且沒有手動結束，則自動結算
        settle_portfolio(force_end=True)
        return False

#模擬進入下一天
def next_day():
    if not st.session_state.sim_active: 
        return st.warning("模擬已結束。")
    
    total_asset = get_current_asset_value(st.session_state.core_data, st.session_state.current_sim_index)
    if check_and_end_simulation(total_asset): 
        return
    
    _advance_one_day()

#模擬進入下十天
def next_ten_days():
    if not st.session_state.sim_active: 
        return st.warning("模擬已結束。")
    
    total_asset = get_current_asset_value(st.session_state.core_data, st.session_state.current_sim_index)
    if check_and_end_simulation(total_asset): 
        return

    days_to_advance = min(10, st.session_state.max_sim_index - st.session_state.current_sim_index)
    
    if days_to_advance <= 0:
        settle_portfolio(force_end=True)
        st.warning("回測結束：已到達最大模擬日數，已自動平倉。")
        return

    for _ in range(days_to_advance):
        if not _advance_one_day():
            break

    if st.session_state.sim_active and st.session_state.current_sim_index >= st.session_state.max_sim_index:
        settle_portfolio(force_end=True)
        actual_sim_days = MIN_SIMULATION_DAYS 
        st.warning(f"回測結束：已到達最大模擬日數 (共 {actual_sim_days} 根 K 棒)，已自動平倉。")

#買入、賣出、做空功能 
def execute_trade(trade_mode_key, quantity, price, leverage=1.0):
    if not st.session_state.sim_active: return st.error("模擬已結束，無法執行交易。")
    if quantity <= 0: 
        min_qty = ASSET_CONFIGS[st.session_state.asset_type]['min_qty']
        return st.error(f"交易數量必須大於或等於最小數量 {min_qty:,.3f}。")
    if price <= 0: return st.error("價格必須大於0")

    config = TRADE_MODE_MAP.get(trade_mode_key)
    if not config: return st.error("無效的交易模式。")
    
    pos_mode_label = config['pos_mode']
    trans_type_label = config['trans_type']
    
    cost_amount = quantity * price
    
    # 判斷是否為槓桿交易
    is_leverage = trade_mode_key in ['Margin_Long', 'Margin_Short']
    
    # --- 1. 槓桿交易單向單倉位檢查 ---
    if is_leverage:
        # 檢查是否有同方向的槓桿倉位存在
        existing_leverage_pos = [p for p in st.session_state.positions if p['pos_mode'] == pos_mode_label]
        if existing_leverage_pos:
            return st.error(f"🚨 槓桿交易限制：您已持有一個 {pos_mode_label} 的倉位 (ID: {existing_leverage_pos[0]['id'][-4:]})，請先平倉後再開新倉。")

    # --- 2. 計算手續費並扣除 (依照模式區分手續費率) ---
    fee_rate_used = LEVERAGE_FEE_RATE if is_leverage else FEE_RATE
    fee = cost_amount * fee_rate_used
    
    st.session_state.balance -= fee
    
    if check_and_end_simulation(get_current_asset_value(st.session_state.core_data, st.session_state.current_sim_index)):
        return

    current_datetime, _, _ = get_price_info_by_index(st.session_state.core_data, st.session_state.current_sim_index)
    
    
    # 現貨買入 & 槓桿買入 (多頭部位，資金流出)
    if trade_mode_key in ['Spot_Buy', 'Margin_Long']:
        
        if trade_mode_key == 'Spot_Buy':
            leverage = 1.0
            margin_required = cost_amount 
            liquidation_price = 0.0 
        else: # Margin_Long
            margin_required = cost_amount / leverage
            
            # 強制平倉價 (Long: Liq Price = Open Price * (1 - (1 / Leverage)))
            liquidation_price = price * (1.0 - (1.0 / leverage))
            
        # 保證金檢查: 現金餘額必須覆蓋所需保證金
        if st.session_state.balance < margin_required:
             # 回補手續費，因為交易失敗
             st.session_state.balance += fee
             return st.error(f"[{pos_mode_label}]買入：現金餘額 (${st.session_state.balance:,.2f}) 不足支付所需的保證金/成本 (${margin_required:,.2f})！(已退還手續費)")
        
        unique_id = str(uuid.uuid4())[:8] 
        
        new_position = {
            'id': unique_id,
            'open_date': current_datetime,
            'pos_mode': pos_mode_label, 
            'qty': quantity, # float
            'cost': price,
            'initial_cost': cost_amount, 
            'leverage': leverage,        
            'liquidation_price': liquidation_price, 
            'sl': 0.0,
            'tp': 0.0
        }
        
        # 資金扣除: 扣除保證金/現貨成本
        st.session_state.balance -= margin_required
        
        st.success(f"[{pos_mode_label}] 成功開多 {quantity:,.3f} {ASSET_CONFIGS[st.session_state.asset_type]['unit']} @ ${price:,.2f} (槓桿: {leverage}x, 保證金: ${margin_required:,.2f})。")
            
        st.session_state.transactions.append({
            '日期': current_datetime,
            '模式': pos_mode_label, 
            '類型': trans_type_label, 
            '股數': quantity, 
            '價格': price, 
            '金額': -margin_required, 
            '損益': np.nan,
            '開倉總值': cost_amount, 
            '手續費': fee,
            'leverage': leverage 
        })
        
        st.session_state.positions.append(new_position)
        
    # 槓桿賣出 (空頭部位)
    elif trade_mode_key == 'Margin_Short':
        
        margin_required = cost_amount / leverage
        
        # 強制平倉價 (Short: Liq Price = Open Price * (1 + (1 / Leverage)))
        liquidation_price = price * (1.0 + (1.0 / leverage))

        # 保證金檢查
        if st.session_state.balance < margin_required:
             # 回補手續費
             st.session_state.balance += fee
             return st.error(f"[{pos_mode_label}]賣出：現金餘額 (${st.session_state.balance:,.2f}) 不足支付所需的保證金 (${margin_required:,.2f})！(已退還手續費)")

        unique_id = str(uuid.uuid4())[:8] 
        
        new_position = {
            'id': unique_id,
            'open_date': current_datetime,
            'pos_mode': pos_mode_label, 
            'qty': quantity, # float
            'cost': price,
            'initial_cost': cost_amount, 
            'leverage': leverage,        
            'liquidation_price': liquidation_price, 
            'sl': 0.0,
            'tp': 0.0
        }
        
        # 資金處理: 扣除保證金
        st.session_state.balance -= margin_required
        
        st.success(f"[{pos_mode_label}] 成功開空 {quantity:,.3f} {ASSET_CONFIGS[st.session_state.asset_type]['unit']} @ ${price:,.2f} (槓桿: {leverage}x, 保證金: ${margin_required:,.2f})。")

        st.session_state.transactions.append({ 
            '日期': current_datetime,
            '模式': pos_mode_label, 
            '類型': trans_type_label, 
            '股數': -quantity, 
            '價格': price, 
            '金額': -margin_required, 
            '損益': np.nan,
            '開倉總值': cost_amount, 
            '手續費': fee,
            'leverage': leverage 
        })
        
        st.session_state.positions.append(new_position)
    
    #交易後檢查風控
    total_asset_new = get_current_asset_value(st.session_state.core_data, st.session_state.current_sim_index)
    check_and_end_simulation(total_asset_new)


#GUI
st.set_page_config(layout="wide")

if not st.session_state.initialized:
    #初始化介面 
    with st.sidebar:
        st.header("Ksim V2 (多資產回測)")
        
        # Req 4: 選擇回測資產類型 (定義交易規則)
        selected_asset_type = st.radio(
            "選擇回測資產類型 (定義交易規則)",
            ('Stock', 'Forex', 'Crypto'),
            format_func=lambda x: {'Stock': '📈 股票', 'Forex': '💱 匯率', 'Crypto': '₿ 加密貨幣'}[x]
        )
        
        st.session_state.ticker = st.text_input(
            "請輸入代碼 (e.g. TSLA, JPY=X, BTC-USD)",
            value=st.session_state.ticker 
        ).strip().upper() 
        
        if st.button("🚀點擊開始回測"):
            if st.session_state.ticker:
                reset_state()
                initialize_data_and_simulation(selected_asset_type)
            else:
                st.error("請輸入有效的代碼！")
    
    st.info(f"請在左側欄選擇資產類型 (定義規則)，輸入代碼 (抓取數據)，並點擊 '🚀點擊開始回測'。目前預設代碼: {st.session_state.ticker}")
    st.stop()
    
#獲取當前數據
core_data = st.session_state.core_data
current_idx = st.session_state.current_sim_index
asset_type = st.session_state.asset_type
asset_config = ASSET_CONFIGS[asset_type]
unit_name = asset_config['unit']
min_qty = asset_config['min_qty']
default_qty = asset_config['default_qty']

current_datetime, open_price, close_price = get_price_info_by_index(core_data, current_idx) 

#側邊欄的控制、交易面板、狀態顯示
with st.sidebar:
    st.subheader(f"📈 {st.session_state.ticker} ({unit_name}回測)")
    st.markdown("---")
    
    #回測進度 
    days_passed_sim = current_idx - VIEW_DAYS + 1
    days_remaining = st.session_state.max_sim_index - current_idx
    
    st.markdown(f"**回測進度**")
    st.markdown(f"**已模擬 K 棒:** **{max(0, days_passed_sim)}** 根")
    st.markdown(f"**剩餘 K 棒:** **{max(0, days_remaining)}** 根")
    st.markdown("---")
    
    #控制按鈕 
    if st.session_state.sim_active:
        st.button("➡️ 下一天", on_click=next_day, use_container_width=True) 
        st.button("⏭️ 下十天", on_click=next_ten_days, use_container_width=True) 
        st.markdown("---")
        # Req 5: 簡化按鈕名稱
        st.button("🛑 **提早結算**", on_click=lambda: settle_portfolio(force_end=True), help="結束模擬並以當日收盤價平倉所有部位。", use_container_width=True)
    else:
        st.button("重新開始回測", on_click=reset_state, use_container_width=True)
    
    st.markdown("---")
    
    #交易面板(開倉功能)
    st.subheader("🛒 開倉交易")
    
    if st.session_state.sim_active:
        
        # 動態顯示交易模式 (Req 4)
        trade_mode_option = st.radio(
             "交易模式",
             ('Spot_Buy', 'Margin_Long', 'Margin_Short'), 
             format_func=lambda x: {
                 'Spot_Buy': asset_config['mode_long'], 
                 'Margin_Long': asset_config['mode_margin_long'], 
                 'Margin_Short': asset_config['mode_margin_short']
             }[x],
             horizontal=True, 
             key='trade_mode_new'
        )

        is_margin_trade = trade_mode_option in ['Margin_Long', 'Margin_Short']
        leverage = 1.0
        
        if is_margin_trade:
            # 槓桿滑桿
            leverage = st.slider("槓桿倍數 (Leverage)", min_value=1.0, max_value=20.0, value=2.0, step=0.5, format='%.1fx', key='leverage_slider')
        
        #數量模式選擇 (Req 4)
        quantity_mode = st.radio(f"數量模式 ({unit_name} / %)", ('Absolute', 'Percentage'), format_func=lambda x: unit_name if x == 'Absolute' else '百分比 (%)', horizontal=True, key='qty_mode_open')
        
        final_quantity = 0.0
        
        # 決定數量輸入的格式和步驟
        is_integer_qty = min_qty >= 1.0 and min_qty == int(min_qty)
        qty_format = '%i' if is_integer_qty else '%.3f'
        step_val = min_qty if min_qty < 1.0 else (1.0 if is_integer_qty else min_qty)
        
        # FIX: 確保 number_input 的所有數值參數 (min_value, value, step) 型別一致
        num_type_cast = int if is_integer_qty else float 

        
        if quantity_mode == 'Absolute':
            
            value_float = float(default_qty)
            
            quantity = st.number_input(f"{unit_name} (Quantity)", 
                                       min_value=num_type_cast(min_qty), 
                                       value=num_type_cast(value_float), 
                                       step=num_type_cast(step_val),     
                                       format=qty_format,
                                       key='abs_qty_input')
            final_quantity = float(quantity)
        else:
            # 百分比開倉加入滑桿
            percentage = st.slider("開倉比例 (%)", min_value=1.0, max_value=100.0, value=50.0, step=1.0, key='percent_qty_open_slider')
        
            # 以現金餘額計算最大可購買數量 (已乘槓桿)
            asset_to_use = st.session_state.balance * (percentage / 100.0)
            
            max_shares_leveraged = (asset_to_use / open_price * leverage) if open_price > 0 else 0.0
            
            # 確保數量是 min_qty 的倍數 
            if is_integer_qty:
                 final_quantity = float(int(max_shares_leveraged / min_qty) * min_qty)
            else:
                 # Crypto/小數：四捨五入到 min_qty 的精度
                 precision = len(str(min_qty).split('.')[-1])
                 final_quantity = round(max_shares_leveraged / min_qty) * min_qty
                 final_quantity = round(final_quantity, precision) 

            
            if final_quantity < min_qty:
                 st.info(f"⚠️ 百分比計算的 {unit_name} 不足 {min_qty:,.3f} {unit_name}。")
                 final_quantity = 0.0

            st.markdown(f"**換算數量:** **{final_quantity:,.3f}** {unit_name}")

        estimated_cost = final_quantity * open_price
        estimated_margin = estimated_cost / leverage
        
        # 使用正確的費率計算預估手續費
        fee_rate_used_display = LEVERAGE_FEE_RATE if is_margin_trade else FEE_RATE
        estimated_fee = estimated_cost * fee_rate_used_display
        
        # 預估強制平倉數值
        estimated_liq_price = 0.0
        liq_display = "N/A"
        if is_margin_trade:
             if trade_mode_option == 'Margin_Long':
                 estimated_liq_price = open_price * (1.0 - (1.0 / leverage))
             elif trade_mode_option == 'Margin_Short':
                 estimated_liq_price = open_price * (1.0 + (1.0 / leverage))
                 
             if estimated_liq_price > 0:
                  liq_display = f"${estimated_liq_price:,.2f}"

        st.info(f"交易參考價 (開盤價): **${open_price:,.2f}**")
        st.markdown(f"**開倉總值:** **${estimated_cost:,.2f}**")
        st.markdown(f"**預估手續費 ({fee_rate_used_display*100:.2f}%):** **${estimated_fee:,.2f}**")
        if is_margin_trade:
             st.markdown(f"**預估保證金:** **${estimated_margin:,.2f}**")
             st.markdown(f"**預估強制平倉價:** **{liq_display}**")


        if st.button(f"執行開倉 ({TRADE_MODE_MAP[trade_mode_option]['position_type']})", use_container_width=True, key='execute_trade_open'):
            if final_quantity >= min_qty and open_price > 0:
                execute_trade(trade_mode_option, final_quantity, open_price, leverage)
            else:
                st.error(f"{unit_name}數量無效或價格無效，無法執行交易！")
    else:
        st.info("模擬已結束。請點擊 '重新開始回測'。")

    st.markdown("---")

    #資金與部位狀態
    st.subheader("📈 資金與部位狀態")
    
    current_open_price = open_price if open_price > 0 else 0.0
    
    unrealized_pnl = get_total_unrealized_pnl(current_open_price)
    total_asset = get_current_asset_value(core_data, current_idx)
    spot_summary = get_spot_summary(core_data, current_idx) 
    
    st.metric("總資產 (含未實現)", f"${total_asset:,.2f}")
    st.metric("現金餘額 (可用)", f"${st.session_state.balance:,.2f}")
    st.metric("當日未實現損益 (開盤價)", f"${unrealized_pnl:,.2f}")

    st.markdown("---")
    st.markdown("**現貨部位彙總** (現貨模式)")
    st.metric(f"總 {unit_name} 數", f"{spot_summary['qty']:,.3f} {unit_name}") # 使用 .3f 確保小數顯示
    st.metric("現貨均價", f"${spot_summary['avg_cost']:,.2f}")
    st.metric("現貨未實現損益", f"${spot_summary['unrealized_pnl']:,.2f}")


#K線圖
display_start_idx = 0 
display_end_idx = current_idx + 1

data_to_display = core_data.iloc[display_start_idx : display_end_idx].copy()
x_axis_date = data_to_display['Date'] 

fig = make_subplots(
    rows=3, cols=1, 
    row_heights=[0.6, 0.2, 0.2], 
    shared_xaxes=True,
    vertical_spacing=0.02,
    subplot_titles=(f"{st.session_state.ticker} 日線 K 棒 (MA $5, 10, 20, 60, 120$)", "成交量", "RSI(14)") 
)

# K線 
fig.add_trace(go.Candlestick(x=x_axis_date, open=data_to_display['Open'], high=data_to_display['High'],
                             low=data_to_display['Low'], close=data_to_display['Close'], name='K-Line',
                             customdata=data_to_display[['Open', 'High', 'Low', 'Close']].values,
                             hovertemplate = '<b>開盤</b>: $%{customdata[0]:.2f}<br>' +
                                             '<b>最高</b>: $%{customdata[1]:.2f}<br>' +
                                             '<b>最低</b>: $%{customdata[2]:.2f}<br>' +
                                             '<b>收盤</b>: $%{customdata[3]:.2f}<extra>K 線</extra>'), row=1, col=1)

# MA均線
for p_ma in MA_PERIODS:
    fig.add_trace(go.Scatter(x=x_axis_date, y=data_to_display[f'MA{p_ma}'], mode='lines', 
                             name=f'MA{p_ma}', line=dict(color=MA_COLORS[p_ma], width=1),
                             hovertemplate=f'MA{p_ma}: %{{y:.2f}}<extra></extra>'), row=1, col=1) 
    
# --- 🎯 繪製倉位關鍵線 (開倉價, 強制平倉價, SL, TP) 並貼齊價格刻度 (Req 1) ---
for pos in st.session_state.positions:
    # 價格資訊 (開倉價, 強制平倉價, SL, TP)
    lines_to_plot = {
        '開倉價': {'price': pos['cost'], 'color': 'yellow', 'dash': 'dot'},
    }
    
    # 判斷方向
    is_long_pos = pos['pos_mode'] in ['現貨', '融資']
    pos_direction = '多' if is_long_pos else '空'
    
    # 只有槓桿部位才會有強制平倉價
    if pos['pos_mode'] in ['融資', '融券']: 
         lines_to_plot['強制平倉'] = {'price': pos.get('liquidation_price', 0.0), 'color': 'red', 'dash': 'dash'}
    
    # 止損/止盈 (如果設定了)
    if pos['sl'] > 0:
        lines_to_plot['止損價 (SL)'] = {'price': pos['sl'], 'color': 'red', 'dash': 'dot'}
    if pos['tp'] > 0:
        lines_to_plot['止盈價 (TP)'] = {'price': pos['tp'], 'color': 'green', 'dash': 'dot'}

    for name, line_info in lines_to_plot.items():
        if line_info['price'] > 0:
            
            # 簡化標籤名稱
            short_name = ''
            if name == '開倉價': short_name = '開'
            elif '止損' in name: short_name = 'SL'
            elif '止盈' in name: short_name = 'TP'
            elif '強制平倉' in name: short_name = 'Liq'

            # Req 1: 標籤格式：[多/空][開/SL/TP] @ $價格
            annotation_label = f"{pos_direction}{short_name} @ ${line_info['price']:,.2f}"
            
            fig.add_hline(
                y=line_info['price'], 
                line_width=1, 
                line_dash=line_info['dash'], 
                line_color=line_info['color'], 
                row=1, 
                col=1,
                name=f"{name} ({pos['id'][-4:]})",
                annotation_text=annotation_label, 
                # 關鍵設定：將標籤貼在右側 Y 軸上
                annotation_position="right", 
                annotation_x=1.01,         # 標註的 X 位置 (使用 paper 座標)
                annotation_xref="paper",   # 使用 paper 座標系統
                annotation_font_color=line_info['color'],
                # Req 1: 透明背景
                annotation_bgcolor='rgba(0,0,0,0)',
                annotation_bordercolor='rgba(0,0,0,0)',
            )

# 成交量 
fig.add_trace(go.Bar(x=x_axis_date, y=data_to_display['Volume'], name='Volume', marker_color='grey',
                     hovertemplate = '<b>成交量</b>: %{y:,.0f}<extra></extra>'), row=2, col=1)

# RSI 
fig.add_trace(go.Scatter(x=x_axis_date, y=data_to_display['RSI'], mode='lines', name='RSI(14)', 
                         line=dict(color='orange', width=2),
                         hovertemplate = '<b>RSI(14)</b>: %{y:.2f}<extra></extra>'), row=3, col=1)

# RSI 70/30臨界線 
fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=3, col=1, name='Overbought')
fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=3, col=1, name='Oversold')

# 圖表顯示風格 
fig.update_xaxes(showticklabels=False, row=1, col=1, type='category')
fig.update_xaxes(showticklabels=False, row=2, col=1, type='category')
fig.update_xaxes(showticklabels=False, row=3, col=1, type='category')

# ... VLINE logic ... (保持不變)
if not st.session_state.sim_active and st.session_state.end_sim_index_on_settle is not None:
    start_sim_relative_index = VIEW_DAYS 
    if start_sim_relative_index >= 0: 
        for r in [1, 2, 3]:
            fig.add_vline(
                x=start_sim_relative_index, 
                line_width=2, 
                line_dash="dot", 
                line_color="green", 
                row=r, 
                col=1,
                annotation_text="回測開始日",
                annotation_position="top left"
            )
        
    end_sim_relative_index = st.session_state.end_sim_index_on_settle - display_start_idx
    
    for r in [1, 2, 3]:
        fig.add_vline(
            x=end_sim_relative_index, 
            line_width=2, 
            line_dash="dot", 
            line_color="white", 
            row=r, 
            col=1,
            annotation_text="回測結束日",
            annotation_position="top right"
        )

# Req 2: 應用上一次儲存的縮放狀態 (在基礎佈局設定之後)
if st.session_state.plot_layout:
    try:
        if 'xaxis.range' in st.session_state.plot_layout:
             # 僅應用 x 軸的範圍設定
             fig.update_layout({
                 'xaxis': {'range': st.session_state.plot_layout['xaxis.range']},
                 'xaxis2': {'range': st.session_state.plot_layout['xaxis2.range']},
                 'xaxis3': {'range': st.session_state.plot_layout['xaxis3.range']},
             })
    except Exception as e:
         # 如果應用失敗，重置狀態
         st.session_state.plot_layout = None
         # print(f"Failed to apply previous layout: {e}") 
        
fig.update_layout(
    xaxis_rangeslider_visible=False, 
    template="plotly_dark", 
    height=800, 
    showlegend=True, 
    dragmode='pan', 
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode='x unified',
    hoverlabel=dict(bgcolor="rgba(128, 128, 128, 0.7)", font_size=12, font_color="white"),
    margin=dict(t=50, b=50, l=50, r=100), 
    
    xaxis=dict(showspikes=True, spikemode='across', spikesnap='data', spikedash='dot', spikethickness=1, unifiedhovertitle=dict(text='\u200b')),
    xaxis2=dict(unifiedhovertitle=dict(text='\u200b')), 
    xaxis3=dict(unifiedhovertitle=dict(text='\u200b')),
    
    yaxis=dict(showspikes=True, spikemode='across', spikesnap='data', spikedash='dot', spikethickness=1, side='right', type='log'), 
    yaxis2=dict(showspikes=True, spikemode='across', spikesnap='data', spikedash='dot', spikethickness=1, side='right'),
    yaxis3=dict(showspikes=True, spikemode='across', spikesnap='data', spikedash='dot', spikethickness=1, side='right')
)

plotly_config = {
    'displayModeBar': True,  
    'scrollZoom': True,      
    'modeBarButtonsToRemove': [
        'select2d', 
        'lasso2d', 
        'zoom2d', 
        'hoverClosestCartesian', 
        'hoverCompareCartesian'
    ],
    'modeBarButtonsToAdd': ['pan2d', 'zoomIn2d', 'zoomOut2d', 'resetScale2d'] 
}

chart_event = st.plotly_chart(
    fig, 
    use_container_width=True, 
    config=plotly_config,
    # 新增 key，讓 Streamlit 自動追蹤圖表狀態
    key="main_candlestick_chart" 
)

# 捕捉並儲存新的佈局狀態
# 儲存使用者對 x 軸的縮放和平移 (即 rangeslider.range 和 range)
if "main_candlestick_chart" in st.session_state and st.session_state.main_candlestick_chart:
    current_layout = st.session_state.main_candlestick_chart.get('layout', {})
    
    if current_layout:
        saved_layout = {}
        # 尋找所有 x 軸的 range 資訊
        for i in [None, 2, 3]:
            xaxis_key = f'xaxis{i}' if i else 'xaxis'
            range_key = f'{xaxis_key}.range'
            
            # 必須檢查 key 是否存在，避免使用者還沒縮放就報錯
            if xaxis_key in current_layout and 'range' in current_layout[xaxis_key]:
                 saved_layout[range_key] = current_layout[xaxis_key]['range']
                 
        if saved_layout:
             st.session_state.plot_layout = saved_layout


#交易倉位GUI
st.markdown("---")
st.header("🎯 交易倉位 (Position Lots)")

# 修正後的 SL/TP 儲存函數 (依靠按鈕觸發)
def save_edited_positions(edited_df: pd.DataFrame):
    if edited_df is None: return

    # 使用 Index (ID) 來對應變動
    edited_positions_dict = edited_df.to_dict('index')

    changes_made = False

    for pos in st.session_state.positions:
        pos_id = pos['id']
        
        if pos_id in edited_positions_dict:
            edited_row = edited_positions_dict[pos_id]
            
            # 嘗試讀取並處理 SL/TP
            new_sl = edited_row.get('SL', pos['sl'])
            new_tp = edited_row.get('TP', pos['tp'])

            try:
                 new_sl = float(new_sl or 0.0) 
            except:
                 new_sl = pos['sl'] # 設回原值
                 
            try:
                 new_tp = float(new_tp or 0.0)
            except:
                 new_tp = pos['tp'] # 設回原值

            # 檢查並處理負值輸入 (防呆)
            if new_sl < 0:
                 new_sl = pos['sl']
                 st.warning(f"ID {pos_id[-4:]}: 止損價 (SL) 價格不能為負值。")
            if new_tp < 0:
                 new_tp = pos['tp']
                 st.warning(f"ID {pos_id[-4:]}: 止盈價 (TP) 價格不能為負值。")

            # 檢查是否有實際變動
            if pos['sl'] != new_sl or pos['tp'] != new_tp:
                 pos['sl'] = new_sl
                 pos['tp'] = new_tp
                 changes_made = True
    
    return changes_made # 回傳是否有變動

if st.session_state.positions:
    
    #建立DataFrame顯示倉位
    df_positions_data = []
    
    current_open_price = open_price
    
    for pos in st.session_state.positions:
        qty = pos['qty']
        cost = pos['cost']
        unrealized_pnl = 0.0
        
        # PnL 計算
        if pos['pos_mode'] in ['現貨', '融資']:
             unrealized_pnl = (qty * current_open_price) - (qty * cost)
        elif pos['pos_mode'] in ['融券']:
             unrealized_pnl = (qty * cost) - (qty * current_open_price)
             
        # 模式名稱客製化 (Req 4: 根據 asset_config 顯示)
        mode_label_display = pos['pos_mode']
        if mode_label_display == '現貨':
             mode_label_display = asset_config['mode_long'] 
        elif mode_label_display == '融資':
             mode_label_display = asset_config['mode_margin_long']
        elif mode_label_display == '融券':
             mode_label_display = asset_config['mode_margin_short']
             
        df_positions_data.append({
            'ID': pos['id'],
            '模式': mode_label_display, 
            '槓桿': f"{pos.get('leverage', 1.0):.1f}x" if pos.get('leverage', 1.0) > 1.0 else '現貨',
            '數量': pos['qty'],
            '開倉價': pos['cost'],
            '強制平倉價': pos.get('liquidation_price', np.nan),
            '未實現損益': unrealized_pnl,
            'SL': pos['sl'],
            'TP': pos['tp'],
        })
        
    df_positions = pd.DataFrame(df_positions_data)
    
    #倉位GUI
    edited_df_from_state = st.data_editor(
        df_positions.set_index('ID'),
        column_config={
            "模式": st.column_config.TextColumn("模式", disabled=True),
            "槓桿": st.column_config.TextColumn("槓桿", disabled=True),
            "數量": st.column_config.NumberColumn(f"數量 ({unit_name})", format="%.3f", disabled=True), # 允許小數顯示
            "開倉價": st.column_config.NumberColumn("開倉價", format="$%.2f", disabled=True),
            "強制平倉價": st.column_config.NumberColumn("強制平倉價", format="$%.2f", disabled=True),
            "未實現損益": st.column_config.NumberColumn("未實現損益", format="$%+.2f", help="以當日開盤價計算", disabled=True),
            "SL": st.column_config.NumberColumn("止損價 (SL)", format="$%.2f", step=0.01),
            "TP": st.column_config.NumberColumn("止盈價 (TP)", format="$%.2f", step=0.01),
        },
        hide_index=False,
        use_container_width=True,
        key='positions_table_edit' # 確保 key 唯一
    )

    # 新增儲存按鈕，防止卡頓
    if st.button("💾 儲存 SL/TP 設定", key='save_sltp_button', use_container_width=True):
        changes_made = save_edited_positions(edited_df_from_state)
        if changes_made:
            st.success("SL/TP 設定已儲存！")
        else:
            st.info("沒有偵測到 SL/TP 變動。")
        # 重新執行以確保所有組件狀態更新
        st.rerun()

    st.info("💡 提醒：請點擊表格中的 **止損價 (SL)** 和 **止盈價 (TP)** 欄位即可直接輸入價格。")
    
    
    #平倉操作GUI 
    st.markdown("---")
    st.subheader("手動平倉操作")
    
    if st.session_state.sim_active:
         
         # 平倉所有倉位按鈕 (Req 5: 簡化按鈕名稱)
         st.button("🔴 **平倉所有倉位**", 
              key='manual_settle_all', 
              use_container_width=True, 
              help="手動結算所有持倉部位，並以當日開盤價結算。回測不會停止。",
              on_click=settle_portfolio) # on_click=settle_portfolio (force_end=False by default)
         
         st.markdown("---")
         st.subheader("手動平倉單一倉位/部分平倉")
         
         pos_options = {pos['id']: f"ID: {pos['id'][-4:]} ({pos['pos_mode']} {pos['qty']:,.3f} {unit_name} @ {pos['cost']:,.2f})" for pos in st.session_state.positions}
         
         if pos_options:
            selected_pos_id = st.selectbox("選擇要平倉的倉位", options=list(pos_options.keys()), format_func=lambda x: pos_options[x], key='close_pos_select')
            
            st.markdown(f"**當前選擇倉位:** {selected_pos_id[-4:]}")
            
            pos_to_close = next((pos for pos in st.session_state.positions if pos['id'] == selected_pos_id), None)
            
            # 修正: max_qty 必須是 float 
            max_qty = pos_to_close['qty'] if pos_to_close else 0.0
            
            close_qty_mode = st.radio("平倉數量模式", ('Absolute_close', 'Percentage_close'), format_func=lambda x: unit_name if x == 'Absolute_close' else '百分比 (%)', horizontal=True, key='close_qty_mode')

            qty_to_close = 0.0
            
            # 決定數量輸入的格式和步驟 (與上方開倉邏輯同步)
            is_integer_qty_close = min_qty >= 1.0 and min_qty == int(min_qty)
            close_qty_format = '%i' if is_integer_qty_close else '%.3f'
            close_step_val = min_qty if min_qty < 1.0 else (1.0 if is_integer_qty_close else min_qty)
            
            # FIX: 確保 number_input 的所有數值參數 (min_value, value, max_value, step) 型別一致
            close_num_type_cast = int if is_integer_qty_close else float

            if close_qty_mode == 'Absolute_close':
                 # 修正: max_qty 和 value_float 必須確保可以被正確轉型
                 value_float = min(min_qty, max_qty) if max_qty > 0 else 0.0
                 
                 close_qty_input = st.number_input(f"平倉數量 (Max {max_qty:,.3f} {unit_name})", 
                                                  min_value=close_num_type_cast(min_qty),        
                                                  max_value=close_num_type_cast(max_qty),       
                                                  value=close_num_type_cast(value_float),       
                                                  step=close_num_type_cast(close_step_val),     
                                                  format=close_qty_format,
                                                  key='abs_qty_close')
                 qty_to_close = float(close_qty_input)
            else:
                percentage_close = st.number_input("平倉比例 (%)", min_value=0.01, max_value=100.0, value=100.0, step=0.01, key='percent_qty_close')
                
                temp_qty_to_close = max_qty * (percentage_close / 100.0)
                
                # 確保數量是 min_qty 的倍數
                if is_integer_qty_close:
                    qty_to_close = float(int(temp_qty_to_close / min_qty) * min_qty)
                else:
                    # Crypto/小數：四捨五入到 min_qty 的精度
                    precision = len(str(min_qty).split('.')[-1])
                    qty_to_close = round(temp_qty_to_close / min_qty) * min_qty
                    qty_to_close = round(qty_to_close, precision)
                    
                st.markdown(f"**換算數量:** **{qty_to_close:,.3f}** {unit_name}")
            
            if qty_to_close < min_qty and max_qty >= min_qty:
                 st.error(f"平倉數量必須大於或等於最小數量 {min_qty:,.3f}。")
                 qty_to_close = 0.0

            st.info(f"平倉參考價 (開盤價): **${current_open_price:,.2f}**")

            #平倉按鈕
            if st.button("🔴 **執行平倉** (按當日開盤價結算)", key='manual_close', use_container_width=True):  
                 if qty_to_close >= min_qty and pos_to_close:
                     close_type = '手動賣出平倉' if pos_to_close['pos_mode'] in ['現貨', '融資'] else '手動買回平倉'
                     success = close_position_lot(selected_pos_id, qty_to_close, current_open_price, close_type, pos_to_close['pos_mode'], mode='手動')
                     
                     if success:
                        st.rerun()
                 else:
                    st.error("請確認平倉數量和倉位選擇是否正確。")    
         else:
             st.info("沒有可以平倉的倉位。")
else:
    st.info("目前沒有任何開倉倉位。")


#交易紀錄GUI
st.markdown("---")
st.header("📝 交易紀錄 (開/平倉紀錄)")

if st.session_state.transactions:
    df_tx = pd.DataFrame(st.session_state.transactions)
    
    # 模式名稱客製化 (Req 4: 根據 asset_config 顯示)
    df_tx['模式'] = df_tx['模式'].replace({
        '現貨': asset_config['mode_long'], 
        '融資': asset_config['mode_margin_long'], 
        '融券': asset_config['mode_margin_short']
    })
    
    # 計算損益百分比 (槓桿交易以保證金為計算基礎)
    df_tx['損益 (%)'] = np.nan
    closed_tx = df_tx['損益'].notna()
    
    initial_cost = df_tx['開倉總值'].fillna(0)
    pnl = df_tx['損益'].fillna(0)
    leverage = df_tx['leverage'].fillna(1.0)
    
    # 計算實際保證金/現貨成本
    margin_required = initial_cost / leverage 

    # PnL % = PnL / Margin_Required
    valid_calc = (margin_required != 0) & closed_tx
    df_tx.loc[valid_calc, '損益 (%)'] = (pnl[valid_calc] / margin_required[valid_calc]) * 100

    
    def format_trade_table(s):
        """格式化欄位顏色和數字顯示"""
        
        #金額欄位 
        is_buy_cover = (df_tx['金額'] < 0)
        is_sell_short = (df_tx['金額'] > 0)
        amount_styles = np.select(
            [is_buy_cover, is_sell_short],
            ['color: red', 'color: green'],
            default=''
        )
        
        #損益欄位
        is_profit = (df_tx['損益'] > 0)
        is_loss = (df_tx['損益'] < 0)

        pnl_styles = np.select(
            [is_profit, is_loss],
            ['color: green', 'color: red'], 
            default=''
        )
        
        #損益 (%) 欄位
        is_profit_pct = (df_tx['損益 (%)'] > 0)
        is_loss_pct = (df_tx['損益 (%)'] < 0)
        pnl_pct_styles = np.select(
            [is_profit_pct, is_loss_pct],
            ['color: green', 'color: red'], 
            default=''
        )
        
        if s.name == '金額':
            return [f'{style}' for style in amount_styles]
        elif s.name == '損益':
            return [f'{style}' for style in pnl_styles]
        elif s.name == '損益 (%)':
            return [f'{style}' for style in pnl_pct_styles]
        
        return [''] * len(s) 
        
    #數字格式化
    format_mapping = {
        '股數': '{:,.3f}', # 允許小數顯示
        '價格': '${:,.2f}',
        '金額': '${:,.2f}',
        '損益': '{:+.2f}',
        '損益 (%)': '{:+.2f}%', 
        '手續費': '-${:,.2f}' 
    }
    
    # 不顯示日期項目
    display_columns = ['模式', '類型', '股數', '價格', '金額', '損益', '損益 (%)', '手續費']
    
    df_tx_display = df_tx.reindex(columns=display_columns)

    styler = df_tx_display.style.apply(format_trade_table, axis=0).format(format_mapping, subset=['股數', '價格', '金額', '損益', '損益 (%)', '手續費'])

    st.dataframe(styler, use_container_width=True)
else:
    st.info("尚無交易紀錄。")
