import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 設定網頁標題
st.set_page_config(page_title="酒類資料集預測平台", layout="wide")

# 加載酒類資料集
@st.cache_data
def get_wine_data():
    wine = load_wine(as_frame=True)
    return wine

wine_data = get_wine_data()
df_features = wine_data.data
df_target = wine_data.target
target_names = wine_data.target_names

# 準備測試資料供預測與計算準確度
X_train, X_test, y_train, y_test = train_test_split(
    df_features, df_target, test_size=0.2, random_state=42
)

# --- Sidebar ---
st.sidebar.title("控制面板")

# 1. 選擇已訓練好的 joblib
model_files = [f for f in os.listdir('.') if f.endswith('.joblib')]
selected_model_name = st.sidebar.selectbox("請選擇預測模型 (.joblib)", model_files)

# 2. 顯示「酒類」資料集資訊
st.sidebar.markdown("---")
st.sidebar.subheader("🍷 酒類資料集資訊")
st.sidebar.info(f"""
- **資料集名稱**: Wine recognition dataset
- **類別數量**: {len(target_names)} ({', '.join(target_names)})
- **特徵數量**: {len(df_features.columns)}
- **總樣本數**: {len(df_features)}
""")
with st.sidebar.expander("詳細敘述"):
    st.text(wine_data.DESCR)

# --- Main Area ---
st.title("🍹 酒類分類預測系統")

# 3. 顯示資料集前5筆內容與特徵統計值資訊
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 資料集前5筆內容")
    st.dataframe(df_features.head())

with col2:
    st.subheader("📊 特徵統計資訊")
    st.dataframe(df_features.describe())

st.markdown("---")

# 4. 按下按鈕進行預測並顯示結果與準確度
st.subheader("🚀 模型預測與準確度評估")

if st.button("進行全量預測 (測試集)"):
    if selected_model_name:
        try:
            # 載入選定的模型
            model = joblib.load(selected_model_name)
            
            # 進行預測
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            # 顯示結果
            st.success(f"✅ 模型 **{selected_model_name}** 載入成功！")
            
            # 顯示準確度
            st.metric(label="測試集總準確度 (Accuracy)", value=f"{acc:.2%}")
            
            # 隨機選取一筆顯示預測結果
            # 我們這裡直接取第一筆測試資料
            sample_idx = 0 
            sample_features = X_test.iloc[sample_idx:sample_idx+1]
            sample_true = int(y_test.iloc[sample_idx])
            sample_pred = int(y_pred[sample_idx])
            
            st.subheader("🎯 樣本預測實例 (測試集第一筆)")
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.write("**真實類別**")
                st.code(target_names[sample_true])
            with res_col2:
                st.write("**預測類別**")
                st.code(target_names[sample_pred])
            with res_col3:
                st.write("**結果判定**")
                if sample_true == sample_pred:
                    st.write("✨ 命中 (Correct)")
                else:
                    st.write("❌ 誤報 (Incorrect)")
            
            # 顯示更詳細的預測表
            with st.expander("查看前10筆測試預測對照"):
                results_df = pd.DataFrame({
                    "真實標籤": [target_names[int(i)] for i in y_test[:10]],
                    "預測標籤": [target_names[int(i)] for i in y_pred[:10]]
                })
                st.table(results_df)

        except Exception as e:
            st.error(f"預測過程中出錯：{e}")
    else:
        st.warning("請先在側邊欄選擇模型檔案。")
else:
    st.write("點擊上方按鈕開始計算模型在測試集 (36筆) 上的表現。")
