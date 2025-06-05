import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Modülleri import et
# from veri_onisleme import preprocess_data
from structers import find_all_structures, find_trend_by_extremes, visualize_optimized_extremes
from visualize import plot_structures, plot_trend_by_extremes
from indicators import plot_indicators, Indicators
from ml import prepare_ml_data, train_and_evaluate_ml
from strategy import ema_crossover_strategy, simulate_ema_strategy_trades, add_risk_reward_column, sum_risk_reward, plot_ema_strategy_trades
from candlestick import plot_candlestick
from destek_direnc import find_support_resistance_levels, plot_candlestick_with_sr

def data_preparation(uploaded_file):
    # Dosya uzantısına göre oku
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file, delimiter='\t', header=0)
    else:
        data = pd.read_excel(uploaded_file)

    columns = data.columns.tolist()
    if 'TIME' in [col.upper().replace('<','').replace('>','') for col in columns]:
        # Alt zaman dilimi: DATE + TIME var
        data.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVolume', 'Volume', 'Spread']
        data = data[['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVolume']]
        data['Date'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce')
        data = data.drop(['Time'], axis=1)
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'TickVolume']]
    else:
        # Sadece günlük: DATE var
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'TickVolume', 'Volume', 'Spread']
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'TickVolume']]
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data['Date'] = data['Date'].dt.date

    return data

st.set_page_config(page_title="Finansal Analiz & ML Demo", layout="wide")
st.title("📊 Finansal Zaman Serisi Analiz ve Makine Öğrenmesi")

# --- Veri Yükleme ve Ön İşleme ---
st.sidebar.header("Veri Yükle")
uploaded_file = st.sidebar.file_uploader("Excel/CSV dosyası yükle", type=["xlsx", "csv"])
if uploaded_file:
    df = data_preparation(uploaded_file)
else:
    st.warning("Lütfen bir veri dosyası yükleyin.")
    st.stop()

st.subheader("Veri Önizleme")
st.dataframe(df.head())

st.sidebar.header("Parametreler")
risk_reward = st.sidebar.number_input("Risk/Ödül Oranı (R)", min_value=1, max_value=10, value=3, step=1)

# --- Sekmeler ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Yapı Analizi", "İndikatörler", "Strateji", "Makine Öğrenmesi", "Görselleştirme", "Trend Analizi"
])

with tab1:
    st.header("Yapı Analizi")
    structures = find_all_structures(df['Close'], distance=10, dates=df['Date'])
    trend_data = find_trend_by_extremes(structures)
    st.dataframe(trend_data)
    # optimized_local_extremes için parametreleri hazırla
    from structers import optimized_local_extremes
    opt_peaks, opt_troughs, opt_peak_vals, opt_trough_vals, opt_peak_dates, opt_trough_dates = optimized_local_extremes(df['Close'], 10, df['Date'])
    fig = visualize_optimized_extremes(df['Date'], df['Close'], opt_peaks, opt_troughs)
    st.pyplot(fig)

with tab2:
    st.header("İndikatörler")
    indicators = Indicators(df)
    df_with_ind = indicators.get_all_indicators()
    st.dataframe(df_with_ind.tail())
    fig = plot_indicators(df_with_ind, indicators)
    st.pyplot(fig)

with tab3:
    st.header("Strateji")
    result = ema_crossover_strategy(df_with_ind, ema_window=20, risk_reward=risk_reward)
    simulated = simulate_ema_strategy_trades(result)
    simulated = add_risk_reward_column(simulated, risk_reward=risk_reward)
    total_r = sum_risk_reward(simulated['rr_result'], risk_reward=risk_reward)
    # Sadece TP/SL olan işlemleri kontrol et
    valid_trades = simulated[simulated['rr_result'].isin([f"+{risk_reward}R", "-1R"])]
    if valid_trades.empty:
        st.warning(f"Bu veri seti için bu risk/ödül oranı ({risk_reward}) desteklenmiyor. Lütfen daha küçük bir değer giriniz.")
    else:
        st.write(f"Toplam R: **{total_r}**")
        st.dataframe(valid_trades[['Date','Close','signal','stop_loss','take_profit','result','rr_result']])
        fig = plot_ema_strategy_trades(simulated, ema_window=20)
        st.pyplot(fig)

with tab4:
    st.header("Makine Öğrenmesi")
    X, y, trades = prepare_ml_data(df_with_ind, ema_window=20, risk_reward=risk_reward)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    # Sadece X ve y tamamen doluysa modeli eğit
    if X.empty or y.empty or X.isnull().values.any() or y.isnull().values.any():
        st.warning(f"Bu veri seti için bu risk/ödül oranı ({risk_reward}) desteklenmiyor. Lütfen daha küçük bir değer giriniz.")
    else:
        model, cm, ml_acc, cr = train_and_evaluate_ml(X, y)
        trades['ml_pred'] = model.predict(X)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"ML Doğruluk: **{ml_acc:.2%}**")
            fig_cm, ax = plt.subplots(figsize=(3, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Tahmin')
            ax.set_ylabel('Gerçek')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig_cm)
        with col2:
            st.dataframe(trades[['Date','Close','signal','result','ml_pred']].dropna())

with tab5:
    st.header("Görselleştirme")
    st.subheader("Fiyat Mum Grafiği + Destek/Direnç")
    supports, resistances = find_support_resistance_levels(df, price_col='Close', order=10, tolerance=0.002)
    if len(supports) == 0 or len(resistances) == 0 or np.any(np.isnan(supports)) or np.any(np.isnan(resistances)):
        st.warning(f"Bu veri seti için bu risk/ödül oranı ({risk_reward}) desteklenmiyor. Lütfen daha küçük bir değer giriniz.")
    else:
        fig = plot_candlestick_with_sr(df, supports, resistances)
        st.pyplot(fig)

with tab6:
    st.header("Trend Analizi")
    # Trend analizi için gerekli verileri hazırla
    structures = find_all_structures(df['Close'], distance=10, dates=df['Date'])
    trend_data = find_trend_by_extremes(structures)
    fig = plot_trend_by_extremes(df['Date'], df['Close'], trend_data)
    st.pyplot(fig)
