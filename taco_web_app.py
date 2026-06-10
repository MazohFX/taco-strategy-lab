    fig.update_layout(height=430, margin=dict(l=20, r=20, t=30, b=20), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    osc_fig = go.Figure()
    osc_fig.add_trace(go.Scatter(x=df.index, y=df["osc"], mode="lines", name="TACO Oscillator", line=dict(color="#bd37dc", width=2)))
    osc_fig.add_hline(y=settings.upper, line_dash="dash", line_color="rgba(255,0,0,.45)")
    osc_fig.add_hline(y=0, line_dash="dash", line_color="rgba(100,100,100,.45)")
    osc_fig.add_hline(y=settings.lower, line_dash="dash", line_color="rgba(0,200,90,.45)")
    osc_fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(range=[-110, 110]))
    st.plotly_chart(osc_fig, use_container_width=True)

    if not equity.empty:
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(x=equity.index, y=equity["equity"], mode="lines", name="Equity", line=dict(color="#2aa889", width=2)))
        equity_fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(equity_fig, use_container_width=True)


st.title("TACO Strategy Lab")
st.caption("Python-Backtester und visuelle Website fuer den TACO Asset Comparison Oscillator.")

CORE_METRICS = ["Trades", "Winrate", "Profit Factor", "Net Profit", "Max DD", "Expectancy R", "Max Loss Streak"]
PRACTICE_METRICS = ["Avg Realized Win", "Avg Realized Loss", "Avg R", "Intratrade MAE", "Avg MFE", "Stop Breach Count", "Stop Breach Avg"]

test_mode = st.sidebar.radio("Modus", ["Manual Backtest", "Cycle Scanner", "SL Scanner", "TACO Radar", "Walk Forward Analysis", "Seasonality Lab"], horizontal=False)

if test_mode == "Seasonality Lab":
    render_seasonality_lab()
    st.stop()

with st.sidebar:
    auto_match_cot = st.checkbox("Auto-match COT market to selected asset", True)

    st.header("Daten")
    data_mode = st.radio("Datenquelle", ["Demo", "CSV Upload", "Yahoo Symbol"], horizontal=True)
    asset_df = comp_df = None
    if data_mode == "CSV Upload":
        asset_file = st.file_uploader("Chart Asset CSV", type=["csv"])
        comp_file = st.file_uploader("Comparison Asset CSV", type=["csv"])
        if asset_file and comp_file:
            asset_df = normalize_ohlc(pd.read_csv(asset_file))
            comp_df = normalize_ohlc(pd.read_csv(comp_file))
    elif data_mode == "Yahoo Symbol":
        asset_preset = st.selectbox("Chart Asset Preset", list(ASSET_PRESETS.keys()))
        comp_preset = st.selectbox("Comparison Asset Preset", list(COMPARISON_PRESETS.keys()))
        asset_symbol = st.text_input("Chart Asset Symbol", ASSET_PRESETS[asset_preset])
        comp_symbol = st.text_input("Comparison Asset Symbol", COMPARISON_PRESETS[comp_preset])
        st.caption("Hinweis: Yahoo liefert freie Index-/Futures-Proxies, nicht zwingend deinen exakten CFD-Brokerkurs.")
        if st.button("Daten laden"):
            asset_df = load_yahoo(asset_symbol)
            comp_df = load_yahoo(comp_symbol)
            if asset_df is None or comp_df is None:
                st.warning("Yahoo-Daten konnten nicht geladen werden. Nutze CSV oder Demo.")
    else:
        asset_preset = "Demo"
        asset_symbol = "Demo"
        asset_df, comp_df = make_demo_data()

    st.header("Einstellungen")
    enable_take_profit = st.checkbox("Enable Take Profit", True)
    selected_tp_mode = st.selectbox("Take Profit Mode", ["Risk Reward", "Fixed %"]) if enable_take_profit else "None"
    settings = Settings(
        cycle_length=st.number_input("Cycle Length", 2, 100, 10),
        smoothing=st.number_input("Glaettung", 1, 50, 5),
        softness=st.number_input("Normalization Softness", 0.25, 5.0, 1.35, step=0.05),
        mode=st.selectbox("Mode", ["Ratio Z-Score", "Return Spread"]),
        trade_direction=st.selectbox("Trade Direction", ["Long & Short", "Long Only", "Short Only"]),
        start_year=st.number_input("Start Year", 1900, 2100, 2015),
        end_year=st.number_input("End Year", 1900, 2100, 2026),
        upper=st.number_input("Upper Bound", value=75.0),
        lower=st.number_input("Lower Bound", value=-75.0),
        risk_pct=st.number_input("Risk Per Trade %", 0.1, 10.0, 1.0, step=0.5),
        stop_pct=st.number_input("Fixed Stop Loss %", 0.05, 20.0, 0.65, step=0.05),
        tp_mode=selected_tp_mode,
        rr=st.number_input("Take Profit R Multiple", 0.1, 20.0, 2.0, step=0.1),
        fixed_tp_pct=st.number_input("Fixed Take Profit %", 0.05, 50.0, 1.3, step=0.05),
        exit_on_zero=st.checkbox("Exit When Oscillator Returns To Zero", False),
        time_exit=st.checkbox("Enable Exit After X Bars", False),
        exit_after_bars=st.number_input("Exit After X Bars", 1, 500, 20),
        initial_capital=st.number_input("Initial Capital", 100.0, 1_000_000.0, 10_000.0, step=100.0),
        commission_pct=st.number_input("Commission %", 0.0, 2.0, 0.05, step=0.01),
        slippage_pct=st.number_input("Slippage %", 0.0, 2.0, 0.02, step=0.01),
    )
    if not enable_take_profit and not settings.exit_on_zero and not settings.time_exit:
        st.warning(
            "Take Profit ist deaktiviert und es ist kein Zero-Line- oder Time-Exit aktiv. "
            "Dann hat die Strategie nur den Stop Loss als echten Exit."
        )

    if test_mode == "Walk Forward Analysis":
        scan_assets = []
        scan_comps = []
        scan_directions = []
        scan_cycle_from = 5
        scan_cycle_to = 30
        scan_cycle_step = 1
        top_curve_min_trades = 50
        top_curve_max_loss_streak = 5
        run_scan = False
        sl_asset_preset = list(ASSET_PRESETS.keys())[0]
        sl_comp_preset = list(COMPARISON_PRESETS.keys())[0]
        sl_directions = []
        sl_from = 0.25
        sl_to = 2.0
        sl_step = 0.05
        run_sl_scan = False
        run_radar = False

        st.header("Walk Forward Analysis")
        wf_asset_preset = st.selectbox("WF Asset", list(ASSET_PRESETS.keys()))
        wf_comp_preset = st.selectbox("WF Comparison Asset", list(COMPARISON_PRESETS.keys()))
        wf_start_year = st.number_input("Walk Forward Start Year", 1900, 2100, 2015)
        wf_end_year = st.number_input("Walk Forward End Year", 1900, 2100, 2026)
        wf_in_sample_years = st.number_input("In Sample Window Years", 1, 50, 20)
        wf_cycle_from = st.number_input("WF Cycle From", 2, 100, 5)
        wf_cycle_to = st.number_input("WF Cycle To", 2, 100, 30)
        wf_cycle_step = st.number_input("WF Cycle Step", 1, 20, 1)
        wf_sl_from = st.number_input("WF Stop Loss From %", 0.05, 20.0, 0.25, step=0.05)
        wf_sl_to = st.number_input("WF Stop Loss To %", 0.05, 20.0, 2.00, step=0.05)
        wf_sl_step = st.number_input("WF Stop Loss Step %", 0.05, 5.0, 0.05, step=0.05)
        wf_min_trades = st.number_input("WF Min In-Sample Trades", 1, 1000, 30)
        wf_max_loss_streak = st.number_input("WF Max In-Sample Loss Streak", 0, 100, 5)
        run_wf = st.button("Run Walk Forward", type="primary")
    elif test_mode == "TACO Radar":
        scan_assets = []
        scan_comps = []
        scan_directions = []
        scan_cycle_from = 5
        scan_cycle_to = 30
        scan_cycle_step = 1
        top_curve_min_trades = 50
        top_curve_max_loss_streak = 5
        run_scan = False
        sl_asset_preset = list(ASSET_PRESETS.keys())[0]
        sl_comp_preset = list(COMPARISON_PRESETS.keys())[0]
        sl_directions = []
        sl_from = 0.25
        sl_to = 2.0
        sl_step = 0.05
        run_sl_scan = False
        run_wf = False

        st.header("TACO Radar")
        radar_assets = st.multiselect(
            "Radar Assets",
            list(ASSET_PRESETS.keys()),
            default=[
                "EURUSD (EURUSD=X)",
                "GBPUSD (GBPUSD=X)",
                "AUDUSD (AUDUSD=X)",
                "NZDUSD (NZDUSD=X)",
                "USDCAD (CAD=X)",
                "USDCHF (CHF=X)",
                "USDJPY (JPY=X)",
                "UK100 proxy: FTSE 100 Index (^FTSE)",
                "GER40 proxy: DAX Index (^GDAXI)",
                "US100 proxy: Nasdaq 100 (^NDX)",
                "S&P500 / US500 proxy: S&P 500 (^GSPC)",
                "US30 proxy: Dow Jones Industrial Average (^DJI)",
            ],
        )
        radar_comps = st.multiselect(
            "Radar Comparison Assets",
            list(COMPARISON_PRESETS.keys()),
            default=["DXY proxy: US Dollar Index (DX-Y.NYB)", "Gold futures (GC=F)", "10Y Treasury Note futures (ZN=F)"],
        )
        radar_cycle_from = st.number_input("Radar Cycle From", 2, 100, 5)
        radar_cycle_to = st.number_input("Radar Cycle To", 2, 100, 30)
        radar_cycle_step = st.number_input("Radar Cycle Step", 1, 20, 1)
        radar_min_trades = st.number_input("Radar Min Trades", 1, 1000, 30)
        radar_max_loss_streak = st.number_input("Radar Max Loss Streak", 0, 100, 5)
        radar_top_cycles = st.number_input("Cycle Vorschlaege pro Signal", 1, 10, 5)
        radar_near_zone = st.number_input("Near Zone Buffer", 0.0, 50.0, 5.0, step=1.0)
        run_radar = st.button("Run TACO Radar", type="primary")
    elif test_mode == "Cycle Scanner":
        st.header("Scanner")
        scan_assets = st.multiselect(
            "Assets",
            list(ASSET_PRESETS.keys()),
            default=[
                "UK100 proxy: FTSE 100 Index (^FTSE)",
                "GER40 proxy: DAX Index (^GDAXI)",
                "US100 proxy: Nasdaq 100 (^NDX)",
                "S&P500 / US500 proxy: S&P 500 (^GSPC)",
                "US30 proxy: Dow Jones Industrial Average (^DJI)",
            ],
        )
        scan_comps = st.multiselect(
            "Comparison Assets",
            list(COMPARISON_PRESETS.keys()),
            default=["DXY proxy: US Dollar Index (DX-Y.NYB)", "Gold futures (GC=F)", "10Y Treasury Note futures (ZN=F)"],
        )
        scan_directions = st.multiselect(
            "Directions",
            ["Long Only", "Short Only", "Long & Short"],
            default=["Long Only", "Short Only"],
        )
        scan_cycle_from = st.number_input("Cycle From", 2, 100, 5)
        scan_cycle_to = st.number_input("Cycle To", 2, 100, 30)
        scan_cycle_step = st.number_input("Cycle Step", 1, 20, 1)
        top_curve_min_trades = st.number_input("Top Curves Min Trades", 1, 1000, 50)
        top_curve_max_loss_streak = st.number_input("Top Curves Max Loss Streak", 0, 100, 5)
        run_scan = st.button("Run Cycle Scan", type="primary")
        sl_asset_preset = list(ASSET_PRESETS.keys())[0]
        sl_comp_preset = list(COMPARISON_PRESETS.keys())[0]
        sl_directions = []
        sl_from = 0.25
        sl_to = 2.0
        sl_step = 0.05
        run_sl_scan = False
        run_wf = False
    elif test_mode == "SL Scanner":
        scan_assets = []
        scan_comps = []
        scan_directions = []
        scan_cycle_from = 5
        scan_cycle_to = 30
        scan_cycle_step = 1
        top_curve_min_trades = 50
        top_curve_max_loss_streak = 5
        run_scan = False
        st.header("SL Scanner")
        sl_asset_preset = st.selectbox("SL Scanner Asset", list(ASSET_PRESETS.keys()))
        sl_comp_preset = st.selectbox("SL Scanner Comparison", list(COMPARISON_PRESETS.keys()))
        sl_directions = st.multiselect("SL Scanner Directions", ["Long Only", "Short Only", "Long & Short"], default=["Long Only"])
        sl_from = st.number_input("SL From %", 0.05, 20.0, 0.25, step=0.05)
        sl_to = st.number_input("SL To %", 0.05, 20.0, 2.00, step=0.05)
        sl_step = st.number_input("SL Step %", 0.05, 5.0, 0.05, step=0.05)
        run_sl_scan = st.button("Run SL Scan", type="primary")
        run_wf = False
    else:
        scan_assets = []
        scan_comps = []
        scan_directions = []
        scan_cycle_from = 5
        scan_cycle_to = 30
        scan_cycle_step = 1
        top_curve_min_trades = 50
        top_curve_max_loss_streak = 5
        run_scan = False
        sl_asset_preset = list(ASSET_PRESETS.keys())[0]
        sl_comp_preset = list(COMPARISON_PRESETS.keys())[0]
        sl_directions = []
        sl_from = 0.25
        sl_to = 2.0
        sl_step = 0.05
        run_sl_scan = False
        run_radar = False
        run_wf = False
