# main_darcy.py
# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
import math
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- IMPORT UTILS ---
# We now rely primarily on the 'ud' alias to access constants and functions
import utils_darcy as ud

# --- 1. GLOBAL DATA LOADING & UTILITIES ---

DATA_KEYS_PARQUET = ud.get_parquet_config()

# --- 2. APP MODULES ---

def run_database_app(df):
    st.title("📂 Database")
    max_data_date = ud.get_max_trade_date(df)
    
    # Initialize State
    ud.initialize_database_state(max_data_date)

    def save_db_state(key, saved_key):
        if key in st.session_state:
            st.session_state[saved_key] = st.session_state[key]
    
    # UI Inputs
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        db_ticker = st.text_input("Ticker (blank=all)", value=st.session_state.saved_db_ticker, key="db_ticker_input", on_change=save_db_state, args=("db_ticker_input", "saved_db_ticker")).strip().upper()
    with c2: 
        start_date = st.date_input("Trade Start Date", value=st.session_state.saved_db_start, key="db_start", on_change=save_db_state, args=("db_start", "saved_db_start"))
    with c3: 
        end_date = st.date_input("Trade End Date", value=st.session_state.saved_db_end, key="db_end", on_change=save_db_state, args=("db_end", "saved_db_end"))
    with c4:
        # UPDATED: Use Constant
        db_exp_end = st.date_input("Expiration Range (end)", value=st.session_state.saved_db_exp, key="db_exp", on_change=save_db_state, args=("db_exp", "saved_db_exp"))
    
    ot1, ot2, ot3, _ = st.columns([1.5, 1.5, 1.5, 5.5])
    with ot1: 
        inc_cb = st.checkbox("Calls Bought", value=st.session_state.saved_db_inc_cb, key="db_inc_cb", on_change=save_db_state, args=("db_inc_cb", "saved_db_inc_cb"))
    with ot2: 
        inc_ps = st.checkbox("Puts Sold", value=st.session_state.saved_db_inc_ps, key="db_inc_ps", on_change=save_db_state, args=("db_inc_ps", "saved_db_inc_ps"))
    with ot3: 
        inc_pb = st.checkbox("Puts Bought", value=st.session_state.saved_db_inc_pb, key="db_inc_pb", on_change=save_db_state, args=("db_inc_pb", "saved_db_inc_pb"))
    
    # Filter Data
    filtered_df = ud.filter_database_trades(
        df, db_ticker, start_date, end_date, db_exp_end, inc_cb, inc_ps, inc_pb
    )
    
    if filtered_df.empty:
        st.warning("No data found matching these filters.")
        return
        
    # Display Styled Dataframe
    st.subheader("Non-Expired Trades")
    st.caption("⚠️ User should check OI to confirm trades are still open")
    
    styled_df = ud.get_database_styled_view(filtered_df)
    
    # UPDATED: Use Constant for Max Rows
    st.dataframe(
        styled_df, 
        use_container_width=True, 
        hide_index=True, 
        height=ud.get_table_height(filtered_df, max_rows=ud.DB_TABLE_MAX_ROWS)
    )
    st.markdown("<br><br><br>", unsafe_allow_html=True)

def run_rankings_app(df):
    st.title("🏆 Rankings")
    max_data_date = ud.get_max_trade_date(df)
    
    # Use Constant for Lookback
    start_default = max_data_date - timedelta(days=ud.RANK_LOOKBACK_DAYS)
    
    ud.initialize_rankings_state(start_default, max_data_date)

    def save_rank_state(key, saved_key):
        if key in st.session_state:
            st.session_state[saved_key] = st.session_state[key]
    
    # UI Inputs
    # UPDATED: Adjusted columns to fit 5 items on one row
    c1, c2, c3, c4, c5 = st.columns([1, 1, 0.6, 1, 1], gap="small")
    
    with c1: 
        rank_start = st.date_input("Trade Start Date", value=st.session_state.saved_rank_start, key="rank_start", on_change=save_rank_state, args=("rank_start", "saved_rank_start"))
    with c2: 
        rank_end = st.date_input("Trade End Date", value=st.session_state.saved_rank_end, key="rank_end", on_change=save_rank_state, args=("rank_end", "saved_rank_end"))
    with c3: 
        limit = st.number_input("Limit", value=st.session_state.saved_rank_limit, min_value=1, max_value=200, key="rank_limit", on_change=save_rank_state, args=("rank_limit", "saved_rank_limit"))
    with c4: 
        # UPDATED: Use Keys from Utils
        mc_options = list(ud.RANK_MC_THRESHOLDS.keys())
        default_mc_index = mc_options.index("10B") if "10B" in mc_options else 0
        min_mkt_cap_rank = st.selectbox("Min Market Cap", mc_options, index=default_mc_index, key="rank_mc", on_change=save_rank_state, args=("rank_mc", "saved_rank_mc"))
    with c5:
        # UPDATED: Converted Checkbox to Selectbox
        saved_ema = st.session_state.saved_rank_ema
        # Handle both boolean (legacy) and string (new) state
        ema_idx = 1 if (saved_ema is True or saved_ema == "Yes") else 0
        
        ema_str = st.selectbox("Hide < 8 EMA", ["No", "Yes"], index=ema_idx, key="rank_ema", on_change=save_rank_state, args=("rank_ema", "saved_rank_ema"))
        filter_ema = (ema_str == "Yes")

    # --- INSERT USER GUIDE HERE ---
    with st.expander("ℹ️ Page User Guide"):
        st.markdown("### 🏆 Methodology Breakdown")
        ug1, ug2 = st.columns(2, gap="medium")
        with ug1:
            st.markdown("**1. 🧠 Smart Money Score (0-100)**")
            st.markdown("""
            A weighted multi-factor model designed to identify high-conviction institutional flow, normalized on a 0-100 scale.
            * **Sentiment (35%):** Net Premium `(Calls Bought + Puts Sold) - Puts Bought`.
            * **Impact (30%):** Sentiment relative to Market Cap (detects "outsized" bets).
            * **Momentum (35%):** Flow velocity over the last 3 active trading days.
            """)
        with ug2:
            st.markdown("**2. 🤡 Legacy Volume Score**")
            st.markdown("""
            A raw counting metric for determining simple directionality.
            * **Formula:** `(Calls Bought + Puts Sold) - Puts Bought`.
            * **Sorting:** Ranked by this raw Score first, then by total Trade Count.
            """)
        st.markdown("---")
        st.caption("💡 **Top 3 Ideas:** This tab takes the top **Smart Money** candidates and runs them through a technical filter (Trend Alignment, RSI, Whale Support levels) to find the best confluence.")
    # ------------------------------

    f_filtered = ud.filter_rankings_data(df, rank_start, rank_end)
    
    if f_filtered.empty:
        st.warning("No trades found matching these dates.")
        return

    tab_rank, tab_ideas, tab_vol = st.tabs(["🧠 Smart Money", "💡 Top 3", "🤡 Bulltard"])
    mc_thresh = ud.RANK_MC_THRESHOLDS.get(min_mkt_cap_rank, 1e10)
    with st.spinner("Crunching numbers..."):
        top_bulls, top_bears, valid_data = ud.calculate_smart_money_score(df, rank_start, rank_end, mc_thresh, filter_ema, limit)

    with tab_rank:
        if valid_data.empty:
            st.warning("Not enough data for Smart Money scores.")
        else:
            sm_config = {
                "Symbol": st.column_config.TextColumn("Ticker", width=60),
                "Score": st.column_config.ProgressColumn("Score", format="%d", min_value=0, max_value=100),
                "Trade_Count": st.column_config.NumberColumn("Qty", width=50),
                "Last Trade": st.column_config.TextColumn("Last", width=70)
            }
            cols_to_show = ["Symbol", "Score", "Trade_Count", "Last Trade"]
            sm1, sm2 = st.columns(2, gap="large")
            with sm1:
                st.markdown(f"<div style='color: #71d28a; font-weight:bold;'>Top Bullish Scores</div>", unsafe_allow_html=True)
                if not top_bulls.empty:
                    st.dataframe(top_bulls[cols_to_show], use_container_width=True, hide_index=True, column_config=sm_config, height=ud.get_table_height(top_bulls, max_rows=100))
            with sm2:
                st.markdown(f"<div style='color: #f29ca0; font-weight:bold;'>Top Bearish Scores</div>", unsafe_allow_html=True)
                if not top_bears.empty:
                    st.dataframe(top_bears[cols_to_show], use_container_width=True, hide_index=True, column_config=sm_config, height=ud.get_table_height(top_bears, max_rows=100))
        st.markdown("<br><br>", unsafe_allow_html=True)

    with tab_ideas:
        if top_bulls.empty: st.info("No Bullish candidates found to analyze.")
        else:
            st.caption(f"ℹ️ Analyzing the Top {len(top_bulls)} 'Smart Money' tickers for confluence...")
            best_ideas = ud.generate_top_ideas(top_bulls, df)
            cols = st.columns(3)
            for i, cand in enumerate(best_ideas):
                with cols[i]:
                    with st.container(border=True):
                        st.markdown(f"### #{i+1} {cand['Ticker']}")
                        st.metric("Conviction", f"{cand['Score']:.1f}/10", f"${cand['Price']:.2f}")
                        st.markdown("**Strategy:**")
                        if cand['Suggestions']['Sell Puts']: st.success(f"🛡️ **Sell Put:** {cand['Suggestions']['Sell Puts']}")
                        if cand['Suggestions']['Buy Calls']: st.info(f"🟢 **Buy Call:** {cand['Suggestions']['Buy Calls']}")
                        st.markdown("---")
                        for r in cand['Reasons']: st.caption(f"• {r}")
        st.markdown("<br><br>", unsafe_allow_html=True)

    with tab_vol:
        st.caption("ℹ️ Legacy Methodology: Score = (Calls + Puts Sold) - (Puts Bought).")
        bull_df, bear_df = ud.calculate_volume_rankings(f_filtered, mc_thresh, filter_ema, limit)
        rank_col_config = {
            "Symbol": st.column_config.TextColumn("Symbol", width=60),
            "Trade Count": st.column_config.NumberColumn("#", width=50),
            "Last Trade": st.column_config.TextColumn("Last Trade", width=90),
            "Score": st.column_config.NumberColumn("Score", width=50),
        }
        cols_final = ["Symbol", "Trade Count", "Last Trade", "Score"]
        v1, v2 = st.columns(2)
        with v1:
            st.markdown(f"<div style='color: #71d28a; font-weight:bold;'>Bullish Volume</div>", unsafe_allow_html=True)
            if not bull_df.empty: st.dataframe(bull_df[cols_final], use_container_width=True, hide_index=True, column_config=rank_col_config, height=ud.get_table_height(bull_df, max_rows=100))
        with v2:
            st.markdown(f"<div style='color: #f29ca0; font-weight:bold;'>Bearish Volume</div>", unsafe_allow_html=True)
            if not bear_df.empty: st.dataframe(bear_df[cols_final], use_container_width=True, hide_index=True, column_config=rank_col_config, height=ud.get_table_height(bear_df, max_rows=100))
        st.markdown("<br><br>", unsafe_allow_html=True)

def run_pivot_tables_app(df):
    st.title("🎯 Pivot Tables")
    max_data_date = ud.get_max_trade_date(df)
    ud.initialize_pivot_state(max_data_date, max_data_date)

    def save_pv_state(key, saved_key):
        if key in st.session_state: st.session_state[saved_key] = st.session_state[key]

    col_filters, col_calculator = st.columns([1, 1], gap="medium")
    with col_filters:
        st.markdown("<h4 style='font-size: 1rem; margin-top: 0; margin-bottom: 10px;'>🔍 Filters</h4>", unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        with fc1: td_start = st.date_input("Trade Start Date", value=st.session_state.saved_pv_start, key="pv_start", on_change=save_pv_state, args=("pv_start", "saved_pv_start"))
        with fc2: td_end = st.date_input("Trade End Date", value=st.session_state.saved_pv_end, key="pv_end", on_change=save_pv_state, args=("pv_end", "saved_pv_end"))
        with fc3: ticker_filter = st.text_input("Ticker (blank=all)", value=st.session_state.saved_pv_ticker, key="pv_ticker", on_change=save_pv_state, args=("pv_ticker", "saved_pv_ticker")).strip().upper()
        
        fc4, fc5, fc6 = st.columns(3)
        with fc4: 
            opts_not = list(ud.PIVOT_NOTIONAL_MAP.keys())
            curr_not = st.session_state.saved_pv_notional
            idx_not = opts_not.index(curr_not) if curr_not in opts_not else 0
            sel_not = st.selectbox("Min Dollars", options=opts_not, index=idx_not, key="pv_notional", on_change=save_pv_state, args=("pv_notional", "saved_pv_notional"))
            min_notional = ud.PIVOT_NOTIONAL_MAP[sel_not]
        with fc5: 
            opts_mc = list(ud.PIVOT_MC_MAP.keys())
            curr_mc = st.session_state.saved_pv_mkt_cap
            idx_mc = opts_mc.index(curr_mc) if curr_mc in opts_mc else 0
            sel_mc = st.selectbox("Mkt Cap Min", options=opts_mc, index=idx_mc, key="pv_mkt_cap", on_change=save_pv_state, args=("pv_mkt_cap", "saved_pv_mkt_cap"))
            min_mkt_cap = ud.PIVOT_MC_MAP[sel_mc]
        with fc6: 
            opts_ema = ["All", "Yes"]
            curr_ema = st.session_state.saved_pv_ema
            idx_ema = opts_ema.index(curr_ema) if curr_ema in opts_ema else 0
            ema_filter = st.selectbox("Over 21 Day EMA", options=opts_ema, index=idx_ema, key="pv_ema_filter", on_change=save_pv_state, args=("pv_ema_filter", "saved_pv_ema"))

    with col_calculator:
        st.markdown("<h4 style='font-size: 1rem; margin-top: 0; margin-bottom: 10px;'>💰 Puts Sold Calculator</h4>", unsafe_allow_html=True)
        cc1, cc2, cc3 = st.columns(3)
        with cc1: c_premium = st.number_input("Premium", min_value=0.00, value=st.session_state.saved_calc_premium, step=0.05, format="%.2f", key="calc_premium", on_change=save_pv_state, args=("calc_premium", "saved_calc_premium"))
        with cc2: c_strike = st.number_input("Strike Price", min_value=0.01, value=st.session_state.saved_calc_strike, step=1.0, format="%.2f", key="calc_strike", on_change=save_pv_state, args=("calc_strike", "saved_calc_strike"))
        with cc3: c_expiry = st.date_input("Expiration", value=st.session_state.saved_calc_expiry, key="calc_expiry", on_change=save_pv_state, args=("calc_expiry", "saved_calc_expiry"))
        
        dte = (c_expiry - date.today()).days + 1
        coc_ret = (c_premium / c_strike) * 100 if c_strike > 0 else 0.0
        annual_ret = (coc_ret / dte) * 365 if dte > 0 else 0.0

        st.session_state["calc_out_ann"] = f"{annual_ret:.1f}%"
        st.session_state["calc_out_coc"] = f"{coc_ret:.1f}%"
        st.session_state["calc_out_dte"] = str(max(0, dte))

        cc4, cc5, cc6 = st.columns(3)
        with cc4: st.text_input("Annualised Return", key="calc_out_ann")
        with cc5: st.text_input("Cash on Cash Return", key="calc_out_coc")
        with cc6: st.text_input("Days to Expiration", key="calc_out_dte")

    st.markdown("""<div style="display: flex; gap: 20px; font-size: 14px; margin-top: 10px; margin-bottom: 20px; align-items: center;">
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 3px; background:#b7e1cd"></div> This Friday</div>
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 3px; background:#fce8b2"></div> Next Friday</div>
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 3px; background:#f4c7c3"></div> Two Fridays</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="light-note" style="margin-top: 5px;">ℹ️ Market Cap filtering can be buggy. If empty, reset \'Mkt Cap Min\' to 0B.</div>', unsafe_allow_html=True)
    st.markdown('<div class="light-note" style="margin-top: 5px;">ℹ️ Scroll down to see the Risk Reversals table.</div>', unsafe_allow_html=True)

    d_range = df[(df["Trade Date"].dt.date >= td_start) & (df["Trade Date"].dt.date <= td_end)].copy()
    if d_range.empty: return

    cb_pool, ps_pool, pb_pool, df_rr = ud.generate_pivot_pools(d_range)
    df_cb_f = ud.filter_pivot_dataframe(cb_pool, ticker_filter, min_notional, min_mkt_cap, ema_filter)
    df_ps_f = ud.filter_pivot_dataframe(ps_pool, ticker_filter, min_notional, min_mkt_cap, ema_filter)
    df_pb_f = ud.filter_pivot_dataframe(pb_pool, ticker_filter, min_notional, min_mkt_cap, ema_filter)
    df_rr_f = ud.filter_pivot_dataframe(df_rr, ticker_filter, min_notional, min_mkt_cap, ema_filter)

    row1_c1, row1_c2, row1_c3 = st.columns(3)
    with row1_c1:
        st.subheader("Calls Bought")
        tbl = ud.get_pivot_styled_view(df_cb_f)
        if not tbl.empty: st.dataframe(tbl.style.format(ud.PIVOT_TABLE_FMT).map(ud.highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=ud.get_table_height(tbl, max_rows=50), column_config=ud.COLUMN_CONFIG_PIVOT)
    with row1_c2:
        st.subheader("Puts Sold")
        tbl = ud.get_pivot_styled_view(df_ps_f)
        if not tbl.empty: st.dataframe(tbl.style.format(ud.PIVOT_TABLE_FMT).map(ud.highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=ud.get_table_height(tbl, max_rows=50), column_config=ud.COLUMN_CONFIG_PIVOT)
    with row1_c3:
        st.subheader("Puts Bought")
        tbl = ud.get_pivot_styled_view(df_pb_f)
        if not tbl.empty: st.dataframe(tbl.style.format(ud.PIVOT_TABLE_FMT).map(ud.highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=ud.get_table_height(tbl, max_rows=50), column_config=ud.COLUMN_CONFIG_PIVOT)
    
    st.subheader("Risk Reversals")
    tbl_rr = ud.get_pivot_styled_view(df_rr_f, is_rr=True)
    if not tbl_rr.empty: 
        st.dataframe(tbl_rr.style.format(ud.PIVOT_TABLE_FMT).map(ud.highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=ud.get_table_height(tbl_rr, max_rows=50), column_config=ud.COLUMN_CONFIG_PIVOT)
        st.markdown("<br><br>", unsafe_allow_html=True)
    else: st.caption("No matched RR pairs found.")

def run_strike_zones_app(df):
    st.title("📊 Strike Zones")
    
    exp_range_default = (date.today() + timedelta(days=ud.SZ_DEFAULT_EXP_OFFSET))
    ud.initialize_strike_zone_state(exp_range_default)

    def save_sz_state(key, saved_key):
        if key in st.session_state: st.session_state[saved_key] = st.session_state[key]
    
    col_settings, col_visuals = st.columns([1, 2.5], gap="large")
    with col_settings:
        ticker = st.text_input("Ticker", value=st.session_state.saved_sz_ticker, key="sz_ticker", on_change=save_sz_state, args=("sz_ticker", "saved_sz_ticker")).strip().upper()
        td_start = st.date_input("Trade Date (start)", value=st.session_state.saved_sz_start, key="sz_start", on_change=save_sz_state, args=("sz_start", "saved_sz_start"))
        td_end = st.date_input("Trade Date (end)", value=st.session_state.saved_sz_end, key="sz_end", on_change=save_sz_state, args=("sz_end", "saved_sz_end"))
        exp_end = st.date_input("Exp. Range (end)", value=st.session_state.saved_sz_exp, key="sz_exp", on_change=save_sz_state, args=("sz_exp", "saved_sz_exp"))
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            st.markdown("**View Mode**")
            view_mode = st.radio("Select View", ["Price Zones", "Expiry Buckets"], index=0 if st.session_state.saved_sz_view == "Price Zones" else 1, label_visibility="collapsed", key="sz_view", on_change=save_sz_state, args=("sz_view", "saved_sz_view"))
            st.markdown("**Zone Width**")
            width_mode = st.radio("Select Sizing", ["Auto", "Fixed"], index=0 if st.session_state.saved_sz_width_mode == "Auto" else 1, label_visibility="collapsed", key="sz_width_mode", on_change=save_sz_state, args=("sz_width_mode", "saved_sz_width_mode"))
            if width_mode == "Fixed": 
                # UPDATED: Use Constant for Default
                fixed_size_choice = st.select_slider("Fixed bucket size ($)", options=[1, 5, 10, 25, 50, 100], value=st.session_state.saved_sz_fixed, key="sz_fixed", on_change=save_sz_state, args=("sz_fixed", "saved_sz_fixed"))
            else: fixed_size_choice = ud.SZ_DEFAULT_FIXED_SIZE
        with c_sub2:
            st.markdown("**Include**")
            inc_cb = st.checkbox("Calls Bought", value=st.session_state.saved_sz_inc_cb, key="sz_inc_cb", on_change=save_sz_state, args=("sz_inc_cb", "saved_sz_inc_cb"))
            inc_ps = st.checkbox("Puts Sold", value=st.session_state.saved_sz_inc_ps, key="sz_inc_ps", on_change=save_sz_state, args=("sz_inc_ps", "saved_sz_inc_ps"))
            inc_pb = st.checkbox("Puts Bought", value=st.session_state.saved_sz_inc_pb, key="sz_inc_pb", on_change=save_sz_state, args=("sz_inc_pb", "saved_sz_inc_pb"))
            
    with col_visuals: chart_container = st.container()

    edit_pool_raw = ud.filter_strike_zone_data(df, ticker, td_start, td_end, exp_end, inc_cb, inc_ps, inc_pb)
    if edit_pool_raw.empty:
        with col_visuals: st.warning("No trades match current filters.")
        return

    order_type_col = "Order Type" if "Order Type" in edit_pool_raw.columns else "Order type"
    editor_input = edit_pool_raw[["Include", "Trade Date", order_type_col, "Symbol", "Strike", "Expiry_DT", "Contracts", "Dollars"]].copy()
    editor_input["Dollars"] = pd.to_numeric(editor_input["Dollars"], errors='coerce').fillna(0)
    editor_input["Contracts"] = pd.to_numeric(editor_input["Contracts"], errors='coerce').fillna(0)

    column_configuration = {
        "Include": st.column_config.CheckboxColumn("Include", default=True),
        "Trade Date": st.column_config.DateColumn("Trade Date", format="DD MMM YY"),
        "Expiry_DT": st.column_config.DateColumn("Expiry", format="DD MMM YY"),
        "Dollars": st.column_config.NumberColumn("Dollars", format="$%d"),
        "Contracts": st.column_config.NumberColumn("Qty", format="%d"),
        order_type_col: st.column_config.TextColumn("Order Type"),
        "Symbol": st.column_config.TextColumn("Symbol"),
        "Strike": st.column_config.TextColumn("Strike"),
    }
    
    st.subheader("Data Table & Selection")
    edited_df = st.data_editor(editor_input, column_config=column_configuration, disabled=["Trade Date", order_type_col, "Symbol", "Strike", "Expiry_DT", "Contracts", "Dollars"], hide_index=True, use_container_width=True, key="sz_editor")
    f_final = edit_pool_raw[edited_df["Include"]].copy()
    st.markdown("<br><br>", unsafe_allow_html=True)

    with chart_container:
        if f_final.empty: st.info("No rows selected. Check the 'Include' boxes below.")
        else:
            spot, ema8, ema21, sma200 = ud.get_strike_zone_technicals(ticker)
            def pct_from_spot(x):
                if x is None or np.isnan(x): return "—"
                return f"{(x/spot-1)*100:+.1f}%"
            
            badges = [f'<span class="price-badge-header">Price: ${spot:,.2f}</span>']
            if ema8: badges.append(f'<span class="badge">EMA(8): ${ema8:,.2f} ({pct_from_spot(ema8)})</span>')
            if ema21: badges.append(f'<span class="badge">EMA(21): ${ema21:,.2f} ({pct_from_spot(ema21)})</span>')
            if sma200: badges.append(f'<span class="badge">SMA(200): ${sma200:,.2f} ({pct_from_spot(sma200)})</span>')
            st.markdown('<div class="metric-row">' + "".join(badges) + "</div>", unsafe_allow_html=True)

            if view_mode == "Price Zones":
                html_code = ud.generate_price_zones_html(f_final, spot, width_mode, fixed_size_choice)
                st.markdown(html_code, unsafe_allow_html=True)
            else:
                html_code = ud.generate_expiry_buckets_html(f_final)
                st.markdown(html_code, unsafe_allow_html=True)
            st.caption("ℹ️ You can exclude individual trades from the graphic by unchecking them in the Data Tables box below.")

def run_price_divergences_app(df_global):
    st.title("📉 Price Divergences")
    st.markdown("""<style>.top-note { color: #888888; font-size: 14px; margin-bottom: 2px; font-family: inherit; } [data-testid="stDataFrame"] th { font-weight: 900 !important; } </style>""", unsafe_allow_html=True)
    ud.initialize_divergence_state()

    def save_rsi_state(key, saved_key):
        st.session_state[saved_key] = st.session_state[key]
        
    dataset_map = ud.get_parquet_config()
    options = list(dataset_map.keys())
    tab_div, tab_hist = st.tabs(["📉 Active/Recent Divergences", "📜 Divergences History"])

    with tab_div:
        data_option_div = st.pills("Dataset", options=options, selection_mode="single", default=options[0] if options else None, label_visibility="collapsed", key="rsi_div_pills")
        with st.expander("ℹ️ Page User Guide"):
            c_guide1, c_guide2 = st.columns(2)
            with c_guide1:
                st.markdown("#### ⚙️ Settings & Inputs")
                st.markdown("""
                * **Dataset**: Selects the universe of stocks to scan (e.g., SP500, NASDAQ).
                * **Days Since Signal**: Filters the view to show only signals that were confirmed within this number of past trading days.
                * **Min RSI Delta**: The minimum required difference between the two RSI pivot points.
                * **Strict 50-Cross**: If "Yes", signal is invalid if RSI crossed 50 between pivots.
                """)
            with c_guide2:
                st.markdown("#### 📊 Table Columns")
                st.markdown("""
                * **RSI Δ**: RSI value at first pivot vs second pivot.
                * **Price Δ**: Price at first pivot vs second pivot.
                * **RSI %ile (New)**: The historical percentile rank of the **2nd Pivot (The Signal Candle)**.
                    * **Value**: Shows how rare the signal RSI is (e.g., **5** = Bottom 5% of all history).
                    * **Highlighting**: The cell turns YELLOW if **EITHER** pivot (1 or 2) was historically extreme (<10% or >90%).
                """)
        
        if data_option_div:
            try:
                key = dataset_map[data_option_div]
                master = ud.load_parquet_and_clean(key)
                if master is not None and not master.empty:
                    t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
                    with st.expander(f"View Scanned Tickers ({data_option_div})"):
                        if t_col:
                            unique_tickers = sorted(master[t_col].unique().tolist())
                            st.write(", ".join(unique_tickers))
                        else: st.caption("No ticker column found in dataset.")

                    target_highlight_daily = None
                    highlight_list_weekly = []
                    date_col_raw = next((c for c in master.columns if 'DATE' in c.upper()), None)
                    if date_col_raw:
                        master[date_col_raw] = pd.to_datetime(master[date_col_raw])
                        max_dt_obj = master[date_col_raw].max()
                        target_highlight_daily = max_dt_obj.strftime('%Y-%m-%d')
                        days_to_subtract = max_dt_obj.weekday()
                        current_week_monday = (max_dt_obj - timedelta(days=days_to_subtract))
                        prev_week_monday = current_week_monday - timedelta(days=7)
                        highlight_list_weekly = [current_week_monday.strftime('%Y-%m-%d'), prev_week_monday.strftime('%Y-%m-%d')]
                    
                    # UPDATED: Use Constants for Defaults
                    c_d1, c_d2, c_d3, c_d4, c_d5 = st.columns(5)
                    with c_d1: days_since = st.number_input("Days Since Signal", min_value=1, value=st.session_state.saved_rsi_div_days_since, step=1, key="rsi_div_days_since", on_change=save_rsi_state, args=("rsi_div_days_since", "saved_rsi_div_days_since"))
                    with c_d2: div_diff = st.number_input("Min RSI Delta", min_value=0.5, value=st.session_state.saved_rsi_div_diff, step=0.5, key="rsi_div_diff", on_change=save_rsi_state, args=("rsi_div_diff", "saved_rsi_div_diff"))
                    with c_d3: div_lookback = st.number_input("Max Candle Between Pivots", min_value=30, value=st.session_state.saved_rsi_div_lookback, step=5, key="rsi_div_lookback", on_change=save_rsi_state, args=("rsi_div_lookback", "saved_rsi_div_lookback"))
                    
                    with c_d4:
                         curr_strict = st.session_state.saved_rsi_div_strict
                         idx_strict = ud.DIV_STRICT_OPTS.index(curr_strict) if curr_strict in ud.DIV_STRICT_OPTS else 0
                         strict_div_str = st.selectbox("Strict 50-Cross Invalidation", ud.DIV_STRICT_OPTS, index=idx_strict, key="rsi_div_strict", on_change=save_rsi_state, args=("rsi_div_strict", "saved_rsi_div_strict"))
                         strict_div = (strict_div_str == "Yes")
                    with c_d5:
                         curr_source = st.session_state.saved_rsi_div_source
                         idx_source = ud.DIV_SOURCE_OPTS.index(curr_source) if curr_source in ud.DIV_SOURCE_OPTS else 0
                         div_source = st.selectbox("Candle Price Methodology", ud.DIV_SOURCE_OPTS, index=idx_source, key="rsi_div_source", on_change=save_rsi_state, args=("rsi_div_source", "saved_rsi_div_source"))
                    
                    raw_results_div = []
                    progress_bar = st.progress(0, text="Scanning Divergences...")
                    grouped = master.groupby(t_col)
                    grouped_list = list(grouped)
                    total_groups = len(grouped_list)
                    
                    for i, (ticker, group) in enumerate(grouped_list):
                        d_d, d_w = ud.prepare_data(group.copy())
                        
                        if d_d is not None:
                            daily_divs = ud.find_divergences(d_d, ticker, 'Daily', min_n=0, periods_input=ud.DIV_CSV_PERIODS_DAYS, optimize_for='PF', lookback_period=div_lookback, price_source=div_source, strict_validation=strict_div, recent_days_filter=days_since, rsi_diff_threshold=div_diff)
                            if daily_divs and 'RSI' in d_d.columns:
                                all_rsi = d_d['RSI'].dropna().values
                                if len(all_rsi) > 0:
                                    for div in daily_divs:
                                        p1 = (all_rsi < div['RSI1']).mean() * 100
                                        p2 = (all_rsi < div['RSI2']).mean() * 100
                                        div['RSI1_Pct'] = p1; div['RSI2_Pct'] = p2
                                        div['Extreme_Flag'] = (p1 < 10 or p2 < 10) if div['Type'] == 'Bullish' else (p1 > 90 or p2 > 90)
                            raw_results_div.extend(daily_divs)
                        
                        if d_w is not None: 
                            weekly_divs = ud.find_divergences(d_w, ticker, 'Weekly', min_n=0, periods_input=ud.DIV_CSV_PERIODS_WEEKS, optimize_for='PF', lookback_period=div_lookback, price_source=div_source, strict_validation=strict_div, recent_days_filter=days_since, rsi_diff_threshold=div_diff)
                            if weekly_divs and 'RSI' in d_w.columns:
                                all_rsi_w = d_w['RSI'].dropna().values
                                if len(all_rsi_w) > 0:
                                    for div in weekly_divs:
                                        p1 = (all_rsi_w < div['RSI1']).mean() * 100
                                        p2 = (all_rsi_w < div['RSI2']).mean() * 100
                                        div['RSI1_Pct'] = p1; div['RSI2_Pct'] = p2
                                        div['Extreme_Flag'] = (p1 < 10 or p2 < 10) if div['Type'] == 'Bullish' else (p1 > 90 or p2 > 90)
                            raw_results_div.extend(weekly_divs)
                        if i % 10 == 0 or i == total_groups - 1: progress_bar.progress((i + 1) / total_groups)
                    progress_bar.empty()
                    
                    if raw_results_div:
                        df_all_results = pd.DataFrame(raw_results_div)
                        res_div_df = df_all_results[df_all_results["Is_Recent"] == True].copy()
                        if res_div_df.empty: st.warning(f"No signals found in the last {days_since} days.")
                        else:
                            res_div_df = res_div_df.sort_values(by='Signal_Date_ISO', ascending=False)
                            consolidated = res_div_df.groupby(['Ticker', 'Type', 'Timeframe']).head(1)
                            for tf in ['Daily', 'Weekly']:
                                targets = highlight_list_weekly if tf == 'Weekly' else ([target_highlight_daily] if target_highlight_daily else [])
                                date_header = "Week Δ" if tf == 'Weekly' else "Day Δ"
                                for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                                    st.subheader(f"{emoji} {tf} {s_type} Signals")
                                    tbl_df = consolidated[(consolidated['Type']==s_type) & (consolidated['Timeframe']==tf)].copy()
                                    price_header = "Close Price Δ" if div_source == 'Close' else ("Low Price Δ" if s_type == 'Bullish' else "High Price Δ")
                                    pct_col_title = "RSI Low %ile" if s_type == 'Bullish' else "RSI High %ile"

                                    if not tbl_df.empty:
                                        if 'RSI2_Pct' not in tbl_df.columns: tbl_df['RSI2_Pct'] = 50
                                        if 'Extreme_Flag' not in tbl_df.columns: tbl_df['Extreme_Flag'] = False
                                        def style_div_df(df_in):
                                            def highlight_cells(row):
                                                styles = [''] * len(row)
                                                if row['Signal_Date_ISO'] in targets:
                                                    if 'Date_Display' in df_in.columns:
                                                        idx = df_in.columns.get_loc('Date_Display')
                                                        styles[idx] = 'background-color: rgba(255, 244, 229, 0.7); color: #e67e22; font-weight: bold;'
                                                if row.get('Extreme_Flag', False):
                                                    if 'RSI2_Pct' in df_in.columns:
                                                        idx_p = df_in.columns.get_loc('RSI2_Pct')
                                                        styles[idx_p] = 'background-color: rgba(255, 235, 59, 0.25); color: #f57f17; font-weight: bold;'
                                                return styles
                                            return df_in.style.apply(highlight_cells, axis=1)
                                        st.dataframe(style_div_df(tbl_df), column_config={"Ticker": st.column_config.TextColumn("Ticker"), "Tags": st.column_config.ListColumn("Tags"), "Date_Display": st.column_config.TextColumn(date_header), "RSI_Display": st.column_config.TextColumn("RSI Δ"), "RSI2_Pct": st.column_config.NumberColumn(pct_col_title, format="%d", help="Percentile rank of the signal RSI relative to ticker history."), "Price_Display": st.column_config.TextColumn(price_header), "Last_Close": st.column_config.TextColumn("Last Close")}, column_order=["Ticker", "Tags", "Date_Display", "RSI_Display", "Price_Display", "Last_Close", "RSI2_Pct"], hide_index=True, use_container_width=True, height=ud.get_table_height(tbl_df, max_rows=50))
                                    else: st.info("No signals.")
                    else: st.warning("No Divergence signals found.")
                else: st.error(f"Failed to load dataset.")
            except Exception as e: st.error(f"Analysis failed: {e}")

    with tab_hist:
        c_h1, c_h2, c_h3, c_h4 = st.columns(4)
        with c_h1:
            hist_ticker_in = st.text_input("Ticker", value=st.session_state.rsi_hist_ticker, key="rsi_hist_ticker_in").strip().upper()
            st.session_state.rsi_hist_ticker = hist_ticker_in
        # UPDATED: Use Constants for History Defaults
        with c_h2: h_lookback = st.number_input("Max Days Btwn Pivots", min_value=30, value=ud.DIV_LOOKBACK_DEFAULT, step=5, key="rsi_hist_lookback")
        with c_h3: h_diff = st.number_input("Min RSI Delta", min_value=0.5, value=ud.DIV_RSI_DIFF_DEFAULT, step=0.5, key="rsi_hist_diff")
        with c_h4:
            h_strict_str = st.selectbox("50-Cross Inval", ud.DIV_STRICT_OPTS, index=0, key="rsi_hist_strict")
            h_strict = (h_strict_str == "Yes")
        c_h5, c_h6, c_h7 = st.columns(3)
        with c_h5: h_source = st.selectbox("Candle Price Method", ud.DIV_SOURCE_OPTS, index=0, key="rsi_hist_source")
        with c_h6: h_per_days = st.text_input("Periods (Days)", value="5, 21, 63, 126, 252", key="rsi_hist_p_days")
        with c_h7: h_per_weeks = st.text_input("Periods (Weeks)", value="4, 13, 26, 52", key="rsi_hist_p_weeks")

        current_params = {"t": hist_ticker_in, "lb": h_lookback, "str": h_strict, "src": h_source, "pd": h_per_days, "pw": h_per_weeks, "diff": h_diff}
        run_hist = False
        if current_params != st.session_state.rsi_hist_last_run_params: run_hist = True
        
        if hist_ticker_in and run_hist:
            with st.spinner(f"Analyzing lifetime history for {hist_ticker_in}..."):
                try:
                    ticker_map = ud.load_ticker_map()
                    df_h = ud.get_ticker_technicals(hist_ticker_in, ticker_map)
                    if df_h is None or df_h.empty: df_h = ud.fetch_yahoo_data(hist_ticker_in)
                    
                    if df_h is not None and not df_h.empty:
                        d_d_h, d_w_h = ud.prepare_data(df_h.copy())
                        raw_results_hist = []
                        p_days_parsed = ud.parse_periods(h_per_days)
                        p_weeks_parsed = ud.parse_periods(h_per_weeks)
                        if d_d_h is not None: 
                            d_daily = ud.find_divergences(d_d_h, hist_ticker_in, 'Daily', min_n=0, periods_input=p_days_parsed, lookback_period=h_lookback, price_source=h_source, strict_validation=h_strict, recent_days_filter=99999, rsi_diff_threshold=h_diff)
                            d_daily = ud.inject_volume_data(d_daily, d_d_h)
                            raw_results_hist.extend(d_daily)
                        if d_w_h is not None: 
                            d_weekly = ud.find_divergences(d_w_h, hist_ticker_in, 'Weekly', min_n=0, periods_input=p_weeks_parsed, lookback_period=h_lookback, price_source=h_source, strict_validation=h_strict, recent_days_filter=99999, rsi_diff_threshold=h_diff)
                            d_weekly = ud.inject_volume_data(d_weekly, d_w_h)
                            raw_results_hist.extend(d_weekly)
                        st.session_state.rsi_hist_results = pd.DataFrame(raw_results_hist)
                        st.session_state.rsi_hist_last_run_params = current_params
                    else:
                        st.error(f"Could not load data for {hist_ticker_in}")
                        st.session_state.rsi_hist_results = pd.DataFrame()
                except Exception as e: st.error(f"Error: {e}")
        
        if st.session_state.rsi_hist_results is not None and not st.session_state.rsi_hist_results.empty:
            res_df_h = st.session_state.rsi_hist_results.copy().sort_values(by='Signal_Date_ISO', ascending=False)
            for tf in ['Daily', 'Weekly']:
                p_cols_to_show = []
                current_periods = ud.parse_periods(h_per_days if tf == 'Daily' else h_per_weeks)
                for p in current_periods:
                    col_key = f"Ret_{p}"
                    if col_key in res_df_h.columns: p_cols_to_show.append(col_key)

                for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                    st.subheader(f"{emoji} {tf} {s_type} History")
                    tbl_df = res_df_h[(res_df_h['Type']==s_type) & (res_df_h['Timeframe']==tf)].copy()
                    if not tbl_df.empty:
                        def style_ret(df_in):
                            def highlight_val(val):
                                if pd.isna(val): return ''
                                color = '#1e7e34' if val > 0 else '#c5221f'
                                return f'color: {color}; font-weight: bold;'
                            style_obj = df_in.style
                            for p_c in p_cols_to_show: style_obj = style_obj.map(highlight_val, subset=[p_c])
                            return style_obj
                        cfg = { "P1_Date_ISO": st.column_config.TextColumn("Date 1", width="medium"), "Signal_Date_ISO": st.column_config.TextColumn("Date 2", width="medium"), "RSI1": st.column_config.NumberColumn("RSI 1", format="%.0f"), "RSI2": st.column_config.NumberColumn("RSI 2", format="%.0f"), "Price1": st.column_config.NumberColumn("Price 1", format="$%.2f"), "Price2": st.column_config.NumberColumn("Price 2", format="$%.2f") }
                        for p_c in p_cols_to_show:
                            days = p_c.split('_')[1]
                            cfg[p_c] = st.column_config.NumberColumn(f"{days}{'d' if tf=='Daily' else 'w'} %", format="%+.2f%%")
                        cols_base = ["P1_Date_ISO", "Signal_Date_ISO", "RSI1", "RSI2", "Price1", "Price2"]
                        st.dataframe(style_ret(tbl_df[cols_base + p_cols_to_show]), column_config=cfg, hide_index=True, use_container_width=True, height=(min(len(tbl_df), 50) + 1) * 35 )
                    else: st.caption("No signals found.")
        
        st.divider()
        st.subheader("💾 Data Downloads")
        col_dl_1, col_dl_2 = st.columns(2)
        with col_dl_1:
            st.markdown(f"**Option 1: {hist_ticker_in} Complete History**")
            st.caption("Download all Daily/Weekly, Bullish/Bearish divergences for this specific ticker.")
            if st.session_state.rsi_hist_results is not None and not st.session_state.rsi_hist_results.empty:
                export_df = ud.process_divergence_export_columns(st.session_state.rsi_hist_results)
                csv_single = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(label=f"⬇️ Download {hist_ticker_in} History (CSV)", data=csv_single, file_name=f"{hist_ticker_in}_Divergence_History.csv", mime="text/csv", key="dl_single_ticker_hist")
            else: st.info("Input a ticker above to generate data.")
        with col_dl_2:
            st.markdown(f"**Option 2: Bulk Dataset History**")
            st.caption("Scan complete history for EVERY ticker in the selected dataset.")
            bulk_dataset_opt = st.selectbox("Select Dataset", options=options, index=0, key="rsi_hist_bulk_sel", label_visibility="collapsed")
            if st.button("🚀 Process Bulk History", key="btn_bulk_hist"):
                try:
                    key = dataset_map[bulk_dataset_opt]
                    master_bulk = ud.load_parquet_and_clean(key)
                    if master_bulk is not None and not master_bulk.empty:
                        t_col_b = next((c for c in master_bulk.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
                        if t_col_b:
                            bulk_results = []
                            grouped_bulk = master_bulk.groupby(t_col_b)
                            total_b = len(grouped_bulk)
                            prog_b = st.progress(0, text="Processing Bulk History...")
                            p_days_parsed = ud.parse_periods(h_per_days)
                            p_weeks_parsed = ud.parse_periods(h_per_weeks)
                            for idx, (tkr, grp) in enumerate(grouped_bulk):
                                d_d_b, d_w_b = ud.prepare_data(grp.copy())
                                if d_d_b is not None: 
                                    res_d = ud.find_divergences(d_d_b, tkr, 'Daily', min_n=0, periods_input=p_days_parsed, lookback_period=h_lookback, price_source=h_source, strict_validation=h_strict, recent_days_filter=99999, rsi_diff_threshold=h_diff)
                                    res_d = ud.inject_volume_data(res_d, d_d_b)
                                    bulk_results.extend(res_d)
                                if d_w_b is not None:
                                    res_w = ud.find_divergences(d_w_b, tkr, 'Weekly', min_n=0, periods_input=p_weeks_parsed, lookback_period=h_lookback, price_source=h_source, strict_validation=h_strict, recent_days_filter=99999, rsi_diff_threshold=h_diff)
                                    res_w = ud.inject_volume_data(res_w, d_w_b)
                                    bulk_results.extend(res_w)
                                if idx % 5 == 0: prog_b.progress((idx+1)/total_b)
                            prog_b.empty()
                            if bulk_results: st.session_state.rsi_hist_bulk_df = pd.DataFrame(bulk_results)
                            else: st.warning("No divergences found in dataset.")
                        else: st.error("Ticker column missing in dataset.")
                except Exception as e: st.error(f"Bulk Process Error: {e}")
            if st.session_state.rsi_hist_bulk_df is not None and not st.session_state.rsi_hist_bulk_df.empty:
                bulk_export = ud.process_divergence_export_columns(st.session_state.rsi_hist_bulk_df)
                csv_bulk = bulk_export.to_csv(index=False).encode('utf-8')
                st.success(f"Ready: {len(bulk_export)} rows generated.")
                st.download_button(label="⬇️ Download Full Dataset History (CSV)", data=csv_bulk, file_name=f"Bulk_Divergence_History_{date.today().strftime('%Y%m%d')}.csv", mime="text/csv", key="dl_bulk_ticker_hist")
        st.divider()

def run_rsi_scanner_app(df_global):
    st.title("🤖 RSI Scanner")

    # UPDATED: Use Constants for Defaults
    if 'saved_rsi_pct_low' not in st.session_state: st.session_state.saved_rsi_pct_low = ud.RSI_SCAN_DEFAULT_PCT_LOW
    if 'saved_rsi_pct_high' not in st.session_state: st.session_state.saved_rsi_pct_high = ud.RSI_SCAN_DEFAULT_PCT_HIGH
    if 'saved_rsi_pct_min_n' not in st.session_state: st.session_state.saved_rsi_pct_min_n = ud.RSI_SCAN_DEFAULT_MIN_N
    if 'saved_rsi_pct_periods' not in st.session_state: st.session_state.saved_rsi_pct_periods = ud.RSI_SCAN_DEFAULT_PERIODS

    def save_rsi_state(key, saved_key):
        st.session_state[saved_key] = st.session_state[key]

    dataset_map = DATA_KEYS_PARQUET
    options = list(dataset_map.keys())
    tab_bot, tab_pct = st.tabs(["🤖 Contextual Backtester", "🔢 RSI Percentiles"])

    with tab_bot:
        c_left, c_mid, c_right = st.columns([1, 0.2, 3])
        with c_left:
            st.markdown("#### 1. Asset & Scope")
            # UPDATED: Constants
            ticker = st.text_input("Ticker", value=ud.RSI_BOT_DEFAULT_TICKER, key="rsi_bt_ticker_input").strip().upper()
            lookback_years = st.number_input("Lookback Years", min_value=1, max_value=20, value=ud.RSI_BOT_DEFAULT_LOOKBACK)
            rsi_tol = st.number_input("RSI Tolerance", min_value=0.5, max_value=10.0, value=ud.RSI_BOT_DEFAULT_TOLERANCE, step=0.5, help="Search for RSI +/- this value.")
            
            use_hist_date = st.checkbox("Test from Past Date", value=False, help="Enable this to see what the tool would have shown on a specific day in the past.")
            if use_hist_date: ref_date = st.date_input("Select Reference Date", value=date.today() - timedelta(days=5))
            else: ref_date = date.today()

            dedupe_str = st.selectbox("De-dupe Signals", ["Yes", "No"], index=0, help="If Yes (Recommended): Simulates 'One Trade at a Time'.")
            dedupe_signals = (dedupe_str == "Yes")
            
        with c_right:
            st.markdown("#### 2. Contextual Filters")
            # UPDATED: Constants for Options
            f_sma200 = st.selectbox("Price vs 200 SMA", ud.RSI_BOT_FILTERS_OPTS, index=0, key="f_sma200")
            f_sma50 = st.selectbox("Price vs 50 SMA", ud.RSI_BOT_FILTERS_OPTS, index=0, key="f_sma50")

        st.divider()
        if ticker:
            ticker_map = ud.load_ticker_map()
            df = ud.load_backtest_data(ticker, ticker_map)
            
            if df is None or df.empty: st.error(f"Could not retrieve data for {ticker}.")
            else:
                with st.spinner(f"Analyzing {ticker} context..."):
                    bt_data = ud.run_contextual_backtest(df, ref_date, lookback_years, rsi_tol, f_sma200, f_sma50, dedupe_signals)

                if bt_data and "error" in bt_data: st.error(bt_data["error"])
                elif bt_data:
                    current_rsi = bt_data['current_rsi']
                    rsi_rank = bt_data['rsi_rank']
                    matches_count = bt_data['matches_count']
                    res_df = bt_data['results_df']
                    trade_log = bt_data['trade_log']
                    ref_date_str = bt_data['ref_date_str']

                    st.subheader(f"📊 Analysis: {ticker} on {ref_date_str}")
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric(f"Price ({ref_date_str})", f"${bt_data['close_price']:.2f}")
                    sc2.metric("Reference RSI", f"{current_rsi:.1f}")
                    sc3.metric("RSI Hist. Rank", f"{rsi_rank:.1f}%", help=f"Percentile Rank: Bottom {rsi_rank:.1f}%")
                    sc4.metric("Total Signals", f"{matches_count}", help="Raw count of days matching criteria (RSI + Filters) BEFORE De-duping is applied.")
                    
                    if not res_df.empty:
                        col_order = ["Days", "Count", "Min DD", "Avg DD", "Median DD", "Max DD", "Lump EV", "Lump WR", "Optimal Entry", "Optimal EV", "Optimal WR"]
                        st.dataframe(
                            ud.get_rsi_backtest_styled(res_df[col_order]),
                            column_config={
                                "Days": st.column_config.NumberColumn("Hold", help="Trading Days held"),
                                "Lump EV": st.column_config.NumberColumn("Lump EV", help="Avg Return (Lump Sum entry)."),
                                "Lump WR": st.column_config.NumberColumn("Lump WR", help="Win Rate (Lump Sum entry)."),
                                "Optimal Entry": st.column_config.TextColumn("Best Entry", help="Strategy (Lump Sum vs DCA) with highest historical EV."),
                                "Optimal EV": st.column_config.NumberColumn("Best EV", help="Avg Return of the Best Entry strategy."),
                                "Optimal WR": st.column_config.NumberColumn("Best WR", help="Win Rate of the Best Entry strategy."),
                                "Min DD": st.column_config.NumberColumn("Lump Min DD", help="Smallest (Best Case) Asset Drawdown."),
                                "Max DD": st.column_config.NumberColumn("Lump Max DD", help="Largest (Worst Case) Asset Drawdown."),
                                "Avg DD": st.column_config.NumberColumn("Lump Avg DD", help="Average Asset Drawdown."),
                                "Median DD": st.column_config.NumberColumn("Lump Med DD", help="Median Asset Drawdown."),
                            },
                            use_container_width=True, hide_index=True, height=ud.get_table_height(res_df, max_rows=10)
                        )
                        st.markdown("##### 🧠 Strategic Insights")
                        c_i1, c_i2 = st.columns(2)
                        best_row = res_df.loc[res_df['Optimal EV'].idxmax()]
                        with c_i1:
                            st.success(f"""**🏆 Best Historical Hold**\nIf you entered on {ref_date_str} (RSI {current_rsi:.1f}), the best historical strategy was holding for **{best_row['Days']} Days**.\n* **Avg Return:** +{best_row['Optimal EV']:.2f}%\n* **Win Rate:** {best_row['Optimal WR']:.1f}%\n* **Strategy:** {best_row['Optimal Entry']}""")
                        with c_i2:
                            if trade_log:
                                trade_log_df = pd.DataFrame(trade_log)
                                csv = trade_log_df.to_csv(index=False).encode('utf-8')
                                st.download_button(label="⬇️ Download Trade History (CSV)", data=csv, file_name=f"{ticker}_RSI_Backtest_{ref_date_str}.csv", mime="text/csv", help="Downloads log of all trades using Close Prices.")
                        
                        st.divider()
                        st.subheader("🔍 Trade Drill-Down")
                        if trade_log:
                            df_log = pd.DataFrame(trade_log)
                            unique_periods = sorted([p for p in df_log['Period'].unique() if isinstance(p, int)])
                            period_opts = [f"{p} Days" for p in unique_periods]
                            sel_period_str = st.selectbox("Select Holding Period to Inspect", period_opts, index=len(period_opts)-1) 
                            if sel_period_str:
                                sel_p_int = int(sel_period_str.split(" ")[0])
                                subset = df_log[df_log['Period'] == sel_p_int].copy()
                                st.dataframe(subset[["Entry Date", "Entry Price", "Exit Date", "Exit Price", "Return %", "Max Drawdown %"]].style.format({"Entry Price": "${:,.2f}", "Exit Price": "${:,.2f}", "Return %": "{:+.2f}%", "Max Drawdown %": "{:+.2f}%"}).map(lambda x: 'color: #c5221f; font-weight: bold;' if x < -15 else '', subset=['Max Drawdown %']).map(lambda x: 'color: #71d28a; font-weight: bold;' if x > 0 else 'color: #f29ca0;', subset=['Return %']), use_container_width=True, hide_index=True)
                    else: st.warning(f"No historical matches found for RSI {current_rsi:.1f} (+/- {rsi_tol}). Try widening tolerance.")

    with tab_pct:
        data_option_pct = st.pills("Dataset", options=options, selection_mode="single", default=options[0] if options else None, label_visibility="collapsed", key="rsi_pct_pills")
        if data_option_pct:
            try:
                key = dataset_map[data_option_pct]
                master = ud.load_parquet_and_clean(key)
                if master is not None and not master.empty:
                    t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
                    with st.expander(f"View Scanned Tickers ({data_option_pct})"):
                        if t_col:
                            unique_tickers = sorted(master[t_col].unique().tolist())
                            st.write(", ".join(unique_tickers))
                        else: st.caption("No ticker column found in dataset.")

                    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
                    with c_p1: pct_low = st.number_input("RSI Low (e.g. 10)", min_value=1, max_value=40, value=st.session_state.saved_rsi_pct_low, step=1, key="rsi_pct_low", on_change=save_rsi_state, args=("rsi_pct_low", "saved_rsi_pct_low"))
                    with c_p2: pct_high = st.number_input("RSI High (e.g. 90)", min_value=60, max_value=99, value=st.session_state.saved_rsi_pct_high, step=1, key="rsi_pct_high", on_change=save_rsi_state, args=("rsi_pct_high", "saved_rsi_pct_high"))
                    with c_p3: min_n_pct = st.number_input("Min N", min_value=1, value=st.session_state.saved_rsi_pct_min_n, step=1, key="rsi_pct_min_n", on_change=save_rsi_state, args=("rsi_pct_min_n", "saved_rsi_pct_min_n"))
                    with c_p4: periods_str_pct = st.text_input("Periods", value=st.session_state.saved_rsi_pct_periods, key="rsi_pct_periods", on_change=save_rsi_state, args=("rsi_pct_periods", "saved_rsi_pct_periods"))
            
                    periods_pct = ud.parse_periods(periods_str_pct)
                    raw_results_pct = []
                    prog_bar = st.progress(0, text="Scanning Percentiles...")
                    grouped = master.groupby(t_col)
                    total_groups = len(grouped)
                    for i, (ticker, group) in enumerate(grouped):
                        d_d, _ = ud.prepare_data(group.copy())
                        if d_d is not None:
                            raw_results_pct.extend(ud.find_rsi_percentile_signals(d_d, ticker, pct_low=pct_low/100.0, pct_high=pct_high/100.0, min_n=min_n_pct, timeframe='Daily', periods_input=periods_pct, optimize_for='SQN'))
                        if i % 10 == 0: prog_bar.progress((i+1)/total_groups)
                    prog_bar.empty()
                    
                    if raw_results_pct:
                        df_pct = pd.DataFrame(raw_results_pct)
                        df_pct = df_pct.sort_values(by=['Date_Obj', 'Ticker'], ascending=[False, True])
                        st.dataframe(df_pct, column_config={"Ticker": st.column_config.TextColumn("Ticker"), "Date": st.column_config.TextColumn("Date"), "Action": st.column_config.TextColumn("Signal"), "RSI_Display": st.column_config.TextColumn("RSI Transition"), "Signal_Price": st.column_config.TextColumn("Signal Price"), "Last_Close": st.column_config.TextColumn("Last Close"), "Profit Factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"), "Win Rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"), "EV": st.column_config.NumberColumn("EV", format="%.1f%%"), "EV Target": st.column_config.NumberColumn("EV Target", format="$%.2f"), "SQN": st.column_config.NumberColumn("SQN", format="%.2f")}, hide_index=True, use_container_width=True, height=ud.get_table_height(df_pct, max_rows=50))
                    else: st.info("No percentile signals found.")
                else: st.error("Failed to load data.")
            except Exception as e: st.error(f"Error: {e}")

import traceback


def run_seasonality_app(ticker_map, scan_date, scan_lookback, mc_thresh_val):
    all_tickers = [k for k in ticker_map.keys() if not k.upper().endswith('_PARQUET')]
    valid_tickers = []
    def check_mc(t):
        if get_market_cap(t) < mc_thresh_val: return None
        return t
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(check_mc, t): t for t in all_tickers}
        for future in as_completed(futures):
            res = future.result()
            if res: valid_tickers.append(res)
    
    results = []; all_csv_rows = {k: [] for k in SEAS_SCAN_PERIODS.keys()}
    def calc_forward_returns(ticker_sym):
        try:
            d_df = fetch_history_optimized(ticker_sym, ticker_map)
            if d_df is None or d_df.empty: return None, None
            d_df.columns = [c.strip().upper() for c in d_df.columns]
            date_c = next((c for c in d_df.columns if 'DATE' in c), None)
            close_c = next((c for c in d_df.columns if 'CLOSE' in c), None)
            if not date_c or not close_c: return None, None
            d_df[date_c] = pd.to_datetime(d_df[date_c])
            d_df = d_df.sort_values(date_c).reset_index(drop=True)
            cutoff = pd.to_datetime(date.today()) - timedelta(days=scan_lookback*365)
            d_df_hist = d_df[d_df[date_c] >= cutoff].copy().reset_index(drop=True)
            if len(d_df_hist) < 252: return None, None
            
            recent_perf = 0.0
            if len(d_df) > 21:
                last_p = d_df[close_c].iloc[-1]; prev_p = d_df[close_c].iloc[-22] 
                recent_perf = ((last_p - prev_p) / prev_p) * 100
            
            target_doy = scan_date.timetuple().tm_yday
            d_df_hist['DOY'] = d_df_hist[date_c].dt.dayofyear
            matches = d_df_hist[(d_df_hist['DOY'] >= target_doy - SEAS_SCAN_WINDOW_DAYS) & 
                                (d_df_hist['DOY'] <= target_doy + SEAS_SCAN_WINDOW_DAYS)].copy()
            matches['Year'] = matches[date_c].dt.year
            matches = matches.drop_duplicates(subset=['Year'])
            curr_y = date.today().year
            matches = matches[matches['Year'] < curr_y]
            if len(matches) < SEAS_SCAN_MIN_SAMPLES: return None, None
            
            stats_row = {'Ticker': ticker_sym, 'N': len(matches), 'Recent_21d': recent_perf}
            hist_lag_returns = []
            for idx in matches.index:
                if idx >= 21:
                    p_now = d_df_hist.loc[idx, close_c]; p_prev = d_df_hist.loc[idx - 21, close_c]
                    hist_lag_returns.append((p_now - p_prev) / p_prev)
            stats_row['Hist_Lag_21d'] = (np.mean(hist_lag_returns) * 100) if hist_lag_returns else 0.0
            
            ticker_csv_rows = {k: [] for k in SEAS_SCAN_PERIODS.keys()}
            for p_name, trading_days in SEAS_SCAN_PERIODS.items():
                returns = []
                for idx in matches.index:
                    entry_p = d_df_hist.loc[idx, close_c]
                    exit_idx = idx + trading_days
                    if exit_idx < len(d_df_hist):
                        exit_p = d_df_hist.loc[exit_idx, close_c]
                        ret = (exit_p - entry_p) / entry_p
                        returns.append(ret)
                        ticker_csv_rows[p_name].append({
                            "Ticker": ticker_sym, "Start Date": d_df_hist.loc[idx, date_c].date(),
                            "Entry Price": entry_p, "Exit Date": d_df_hist.loc[exit_idx, date_c].date(),
                            "Exit Price": exit_p, "Return (%)": ret * 100
                        })
                if returns:
                    returns_arr = np.array(returns)
                    avg_ret = np.mean(returns_arr) * 100; win_r = np.mean(returns_arr > 0) * 100
                    std_dev = np.std(returns_arr) * 100; sharpe = avg_ret / std_dev if std_dev > 0.1 else 0.0
                else: avg_ret = 0.0; win_r = 0.0; sharpe = 0.0
                stats_row[f"{p_name}_EV"] = avg_ret; stats_row[f"{p_name}_WR"] = win_r; stats_row[f"{p_name}_Sharpe"] = sharpe
            return stats_row, ticker_csv_rows
        except Exception: return None, None

    with ThreadPoolExecutor(max_workers=20) as executor: 
        futures = {executor.submit(calc_forward_returns, t): t for t in valid_tickers}
        for future in as_completed(futures):
            res_stats, res_details = future.result()
            if res_stats: results.append(res_stats)
            if res_details:
                for k in all_csv_rows.keys():
                    if res_details[k]: all_csv_rows[k].extend(res_details[k])
    return pd.DataFrame(results) if results else pd.DataFrame(), all_csv_rows

def run_ema_distance_app(df_global):
    st.title("📏 EMA Distance Analysis")
    col_in1, col_in2, col_in3 = st.columns([1, 1, 1])
    with col_in1:
        # UPDATED: Use Constant
        ticker = st.text_input("Ticker", value=ud.EMA_DIST_DEFAULT_TICKER).upper().strip()
    with col_in2:
        # UPDATED: Use Constant
        years_back = st.number_input("Years to Analyze", min_value=1, max_value=20, value=ud.EMA_DIST_DEFAULT_YEARS, step=1)
    with col_in3:
        direction = st.selectbox("Analysis Direction", ["Above (Overbought)", "Below (Oversold)"], index=0)
        is_below = "Below" in direction
    
    if not ticker:
        st.warning("Please enter a ticker.")
        return

    with st.spinner(f"Crunching data for {ticker}..."):
        df_clean = ud.calculate_ema_distance_data(ticker, years_back)
        if df_clean is None: return

    close_col = 'CLOSE' if 'CLOSE' in df_clean.columns else 'Close'
    low_col = 'LOW' if 'LOW' in df_clean.columns else 'Low'
    high_col = 'HIGH' if 'HIGH' in df_clean.columns else 'High'
    date_col = next((c for c in df_clean.columns if 'DATE' in c), "DATE")
    
    current_price = df_clean[close_col].iloc[-1]
    current_ema8 = df_clean['EMA_8'].iloc[-1]
    current_dist_50 = df_clean['Dist_50'].iloc[-1]

    st.subheader(f"{ticker} vs Moving Avgs & Percentiles")
    with st.expander("ℹ️ Table User Guide"):
        st.markdown(f"""**1. Key Metrics Tracked.**\nThe app calculates the percentage distance (the "Gap") between the current price and five different moving averages.\n\n**2. The "Rubber Band" Logic (Percentiles).**\nRather than just showing the current gap, the app looks at {years_back} years of history for that specific ticker to see how rare the current gap is.\n\n**3. Visual Highlighting System.**\nThe table uses a traffic-light system to categorize the current price action:\n* 🟢 **Reversion Zone (Green):** Gap is near the historical median. Suggests a safe entry or mean reversion.\n* 🔴 **Extreme Zone (Red):** Gap is historically extended (Top 10% or Bottom 10%). Price is statistically over-extended.""")

    stats_data = []
    thresholds = {} 
    metrics = [("Close vs 8-EMA", df_clean['EMA_8'], df_clean['Dist_8']), ("Close vs 21-EMA", df_clean['EMA_21'], df_clean['Dist_21']), ("Close vs 50-SMA", df_clean['SMA_50'], df_clean['Dist_50']), ("Close vs 100-SMA", df_clean['SMA_100'], df_clean['Dist_100']), ("Close vs 200-SMA", df_clean['SMA_200'], df_clean['Dist_200'])]
    
    # Dynamic Percentiles based on direction
    pct_labels = ['p50 (Med)', 'p30', 'p20', 'p10 (Ext)', 'p05'] if is_below else ['p50 (Med)', 'p70', 'p80', 'p90 (Ext)', 'p95']
    pct_values = [50, 30, 20, 10, 5] if is_below else [50, 70, 80, 90, 95]

    for label, ma_series, dist_series in metrics:
        p_vals = np.percentile(dist_series, pct_values)
        # Store thresholds for signals later
        thresholds[dist_series.name] = { 'ext': p_vals[3], 'mean': p_vals[0] } 
        
        row = {"Metric": label, "Price": current_price, "MA Level": ma_series.iloc[-1], "Gap": dist_series.iloc[-1], "Avg": dist_series.mean()}
        for i, pl in enumerate(pct_labels):
            row[pl] = p_vals[i]
        stats_data.append(row)

    df_stats = pd.DataFrame(stats_data)
    
    def color_combined(row):
        styles = [''] * len(row)
        gap = row['Gap']
        
        # Extract dynamic thresholds from row using column names
        # Note: We rely on the order or explicit naming. 
        # For simplicity, we grab the extreme and median values directly
        p_mean = row[pct_labels[0]]
        p_ext = row[pct_labels[3]]
        
        idx_gap = df_stats.columns.get_loc("Gap")
        
        if is_below:
            # Below Logic: Gaps are negative.
            # Extreme (Red) = Gap <= p10 (e.g. -20% <= -15%)
            # Safe (Green) = Gap >= p50 (e.g. -2% >= -5%)
            if gap <= p_ext: styles[idx_gap] = 'background-color: #fce8e6; color: #c5221f; font-weight: bold;'
            elif gap >= p_mean: styles[idx_gap] = 'background-color: #e6f4ea; color: #1e7e34; font-weight: bold;'
        else:
            # Above Logic: Gaps are positive.
            # Extreme (Red) = Gap >= p90
            # Safe (Green) = Gap <= p50
            if gap >= p_ext: styles[idx_gap] = 'background-color: #fce8e6; color: #c5221f; font-weight: bold;'
            elif gap <= p_mean and (current_price > current_ema8): styles[idx_gap] = 'background-color: #e6f4ea; color: #1e7e34; font-weight: bold;'
            
        return styles

    st.dataframe(df_stats.style.apply(color_combined, axis=1).format(ud.fmt_pct_display, subset=["Gap", "Avg"] + pct_labels), use_container_width=True, hide_index=True, column_config={"Price": st.column_config.NumberColumn("Price", format="$%.2f"), "MA Level": st.column_config.NumberColumn("MA Level", format="$%.2f")})
    
    st.subheader("Combo Over-Extension Signals")
    
    # Logic Switching for Signals
    t8_ext = thresholds['Dist_8']['ext']; t21_ext = thresholds['Dist_21']['ext']; t50_ext = thresholds['Dist_50']['ext']
    
    if is_below:
        # Check if gaps are LOWER than the extreme threshold (more negative)
        m_d = (df_clean['Dist_8'] <= t8_ext) & (df_clean['Dist_21'] <= t21_ext)
        m_fs = (df_clean['Dist_8'] <= t8_ext) & (df_clean['Dist_50'] <= t50_ext)
        m_t = (df_clean['Dist_8'] <= t8_ext) & (df_clean['Dist_21'] <= t21_ext) & (df_clean['Dist_50'] <= t50_ext)
        # Target: Price BOUNCE (positive return)
        backtest_target = abs(ud.EMA_DIST_BACKTEST_DD) # +0.08
        compare_col = df_clean[high_col] # Check Highs for bounce
        trig_desc = "Gap ≤ p10"
        
    else:
        # Check if gaps are HIGHER than extreme threshold
        m_d = (df_clean['Dist_8'] >= t8_ext) & (df_clean['Dist_21'] >= t21_ext)
        m_fs = (df_clean['Dist_8'] >= t8_ext) & (df_clean['Dist_50'] >= t50_ext)
        m_t = (df_clean['Dist_8'] >= t8_ext) & (df_clean['Dist_21'] >= t21_ext) & (df_clean['Dist_50'] >= t50_ext)
        # Target: Price DROP (negative return)
        backtest_target = -abs(ud.EMA_DIST_BACKTEST_DD) # -0.08
        compare_col = df_clean[low_col] # Check Lows for drop
        trig_desc = "Gap ≥ p90"
    
    res_d = ud.run_ema_backtest(m_d, df_clean[close_col], compare_col, target_pct=backtest_target)
    res_fs = ud.run_ema_backtest(m_fs, df_clean[close_col], compare_col, target_pct=backtest_target)
    res_t = ud.run_ema_backtest(m_t, df_clean[close_col], compare_col, target_pct=backtest_target)

    d_active = "✅" if bool(m_d.iloc[-1]) else "❌"
    fs_active = "✅" if bool(m_fs.iloc[-1]) else "❌"
    t_active = "✅" if bool(m_t.iloc[-1]) else "❌"

    combo_rows = [
        {"Combo Rule": "Double EMA", "Triggers": f"(8-EMA {trig_desc}), (21-EMA {trig_desc})", "Occurrences": res_d[0], f"Hit Rate (>={abs(backtest_target)*100:.0f}% Move)": res_d[1], "Median Days": f"{int(res_d[2])} days", "Active Today?": d_active, "raw_status": bool(m_d.iloc[-1])},
        {"Combo Rule": "Fast vs Swing", "Triggers": f"(8-EMA {trig_desc}), (50-SMA {trig_desc})", "Occurrences": res_fs[0], f"Hit Rate (>={abs(backtest_target)*100:.0f}% Move)": res_fs[1], "Median Days": f"{int(res_fs[2])} days", "Active Today?": fs_active, "raw_status": bool(m_fs.iloc[-1])},
        {"Combo Rule": "Triple Stack", "Triggers": f"(8-EMA {trig_desc}), (50-SMA {trig_desc}), (21-EMA {trig_desc})", "Occurrences": res_t[0], f"Hit Rate (>={abs(backtest_target)*100:.0f}% Move)": res_t[1], "Median Days": f"{int(res_t[2])} days", "Active Today?": t_active, "raw_status": bool(m_t.iloc[-1])}
    ]
    
    df_combo = pd.DataFrame(combo_rows)
    def style_combo(row): return ['font-weight: bold; color: #c5221f;' if row['raw_status'] else ''] * len(row)
    st.dataframe(df_combo.style.apply(style_combo, axis=1).format({f"Hit Rate (>={abs(backtest_target)*100:.0f}% Move)": "{:.1f}%"}), use_container_width=True, hide_index=True, column_config={"Triggers": st.column_config.TextColumn("Trigger Conditions", width="large"), "Active Today?": st.column_config.TextColumn("Active Today?", width="small")})

    st.subheader("Visualizing the % Distance from 50 SMA")
    chart_data = pd.DataFrame({'Date': pd.to_datetime(df_clean[date_col]), 'Distance (%)': df_clean['Dist_50']})
    # UPDATED: Use Constant
    chart_data = chart_data[chart_data['Date'] >= (chart_data['Date'].max() - timedelta(days=ud.EMA_DIST_CHART_LOOKBACK))]

    bars = alt.Chart(chart_data).mark_bar().encode(x=alt.X('Date:T', title=None), y=alt.Y('Distance (%)', title='% Dist from 50 SMA'), color=alt.condition(alt.datum['Distance (%)'] > 0, alt.value("#71d28a"), alt.value("#f29ca0")), tooltip=['Date', 'Distance (%)'])
    rule = alt.Chart(pd.DataFrame({'y': [current_dist_50]})).mark_rule(color='#333', strokeDash=[5, 5], strokeWidth=2).encode(y='y:Q')
    st.altair_chart((bars + rule).properties(height=300).interactive(), use_container_width=True)


