import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Trader Analytics", layout="wide")
st.title("Trader Analytics Dashboard")

# ── Load files from root ──────────────────────────────────────────────────────
df     = pd.read_csv("Cleaned_Data.csv")
rf     = joblib.load("random_forest_model.pkl")
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# ── Prepare data ──────────────────────────────────────────────────────────────
num_map = {'Extreme Fear': 1, 'Fear': 2, 'Neutral': 3, 'Greed': 4, 'Extreme Greed': 5}
df['sentiment_score'] = df['classification'].map(num_map)
df = df.sort_values(['Account', 'date'])
df['next_day_pnl'] = df.groupby('Account')['daily_pnl'].shift(-1)
df_model = df.dropna(subset=['next_day_pnl']).copy()
df_model['target_bucket'] = pd.qcut(df_model['next_day_pnl'], q=3, labels=['Loss', 'Neutral', 'Profit'])

features = ['trades_per_day', 'avg_trade_size', 'win_rate', 'long_short_ratio', 'value', 'sentiment_score']
X = df_model[features]
y = df_model['target_bucket']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = rf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

trader_profiles = df.groupby('Account').agg({
    'trades_per_day': 'mean', 'avg_trade_size': 'mean',
    'win_rate': 'mean', 'long_short_ratio': 'mean'
}).dropna()
trader_profiles['archetype'] = kmeans.predict(scaler.transform(trader_profiles))
arch_summary = trader_profiles.groupby('archetype').mean()
arch_counts  = trader_profiles['archetype'].value_counts().sort_index()

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Traders",  f"{df['Account'].nunique():,}")
c2.metric("Records",  f"{len(df):,}")
c3.metric("Accuracy", f"{report['accuracy']:.0%}")
c4.metric("Macro F1", f"{report['macro avg']['f1-score']:.2f}")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Model Performance", "Trader Archetypes", "Predict"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=['Loss', 'Neutral', 'Profit'])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Loss', 'Neutral', 'Profit'],
                    yticklabels=['Loss', 'Neutral', 'Profit'], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig); plt.close(fig)

    with col2:
        st.subheader("Feature Importance")
        imp = pd.Series(rf.feature_importances_, index=features).sort_values()
        fig, ax = plt.subplots()
        imp.plot.barh(ax=ax, color='steelblue')
        ax.set_xlabel("Importance")
        st.pyplot(fig); plt.close(fig)

    st.subheader("Classification Report")
    report_df = (pd.DataFrame(report).T.round(2)
                   .loc[['Loss', 'Neutral', 'Profit', 'accuracy', 'macro avg', 'weighted avg']])
    st.dataframe(report_df, use_container_width=True)

with tab2:
    st.subheader("Archetype Summary")
    summary_display = arch_summary.copy()
    summary_display.index = [f"Archetype {i}  (n={arch_counts[i]})" for i in summary_display.index]
    summary_display.columns = ['Trades/Day', 'Avg Trade Size', 'Win Rate', 'Long/Short Ratio']
    st.dataframe(summary_display.round(2), use_container_width=True)

    col1, col2 = st.columns(2)
    colors = ['#f0b429', '#3b82f6', '#10b981']

    with col1:
        st.subheader("Trades/Day vs Win Rate")
        fig, ax = plt.subplots()
        for i in range(3):
            sub = trader_profiles[trader_profiles['archetype'] == i]
            ax.scatter(sub['trades_per_day'], sub['win_rate'],
                       label=f"Archetype {i}", color=colors[i], alpha=0.6, s=15)
        ax.set_xlabel("Trades/Day"); ax.set_ylabel("Win Rate"); ax.legend()
        st.pyplot(fig); plt.close(fig)

    with col2:
        st.subheader("Archetype Distribution")
        fig, ax = plt.subplots()
        arch_counts.plot.bar(ax=ax, color=colors, edgecolor='none')
        ax.set_xlabel("Archetype"); ax.set_ylabel("# Traders")
        ax.set_xticklabels([f"Archetype {i}" for i in arch_counts.index], rotation=0)
        st.pyplot(fig); plt.close(fig)

with tab3:
    st.subheader("Predict Next-Day PnL Bucket")
    c1, c2, c3 = st.columns(3)
    with c1:
        trades   = st.number_input("Trades per Day",    value=100)
        size     = st.number_input("Avg Trade Size ($)", value=5000)
    with c2:
        wr       = st.slider("Win Rate", 0.0, 1.0, 0.45, 0.01)
        ls_ratio = st.number_input("Long/Short Ratio",  value=1.5)
    with c3:
        val  = st.number_input("Fear & Greed Value", value=50)
        sent = st.selectbox("Sentiment", list(num_map.keys()), index=2)

    if st.button("Predict"):
        row = pd.DataFrame([{
            'trades_per_day': trades, 'avg_trade_size': size, 'win_rate': wr,
            'long_short_ratio': ls_ratio, 'value': val, 'sentiment_score': num_map[sent]
        }])
        pred  = rf.predict(row)[0]
        proba = rf.predict_proba(row)[0]
        st.success(f"Predicted bucket: **{pred}**")
        prob_df = pd.DataFrame({'Class': rf.classes_, 'Probability': proba}).set_index('Class')
        st.bar_chart(prob_df)