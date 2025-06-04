# streamlit_mnist_app.py

import streamlit as st
import joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="MNIST Handwritten Digits Classification by SGDClassifier", layout="centered")
st.title("🔢 MNIST SGDClassifier 예측 + 확률 시각화")

# -------------------------------------------------------------
# 1) 모델 로드 (캐싱)
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path="model.pkl"):
    """
    joblib으로 저장된 SGDClassifier(loss='log_loss') 모델을 로드합니다.
    """
    return joblib.load(path)

model = load_model("model.pkl")

# -------------------------------------------------------------
# 2) 이미지 업로드 및 전처리
# -------------------------------------------------------------
st.subheader("▶ MNIST 형식 이미지 업로드 (28×28 흑백)")

uploaded_file = st.file_uploader(
    label="0~9 숫자가 담긴 28×28 흑백 이미지 업로드 (.png / .jpg / .jpeg)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # 1) PIL로 이미지 열기 → 그레이스케일 변환 → 28×28 리사이즈
    img = Image.open(uploaded_file).convert("L").resize((28, 28))
    # 2) NumPy 배열로 변환 (uint8, 0~255)
    arr = np.array(img).astype(np.uint8)  # shape = (28, 28)

    # 3) MNIST 전처리: (255 - 픽셀값) → float32 → 0~1 정규화 → flatten
    arr_inverted = 255 - arr                     # 뒤집기
    arr_scaled = arr_inverted.astype("float32") / 255.0
    X_input = arr_scaled.reshape(1, -1)          # shape = (1, 784)

    # 업로드된 이미지를 화면에 표시
    st.image(img, caption="Uploaded 28×28 Grayscale Image", use_column_width=False)

    # ---------------------------------------------------------
    # 3) 예측 버튼: predict + predict_proba + 확률 그래프
    # ---------------------------------------------------------
    if st.button("▶ PREDICT!"):
        # 3-1) predict
        pred_class = int(model.predict(X_input)[0])

        # 3-2) predict_proba (확률 배열)
        try:
            proba = model.predict_proba(X_input)[0]  # shape = (10,)
        except AttributeError:
            st.error("❌ 이 모델은 predict_proba를 지원하지 않습니다.")
            st.stop()

        # 가장 높은 확률과 해당 클래스
        best_idx = np.argmax(proba)
        best_prob = proba[best_idx]

        st.write(f"**모델이 예측된 숫자:** {pred_class}")
        st.write(f"**모델 확신도 (해당 클래스 확률):** {best_prob:.4f}")

        # -----------------------------------------------------
        # 4) Matplotlib으로 확률 분포 Bar Chart 그리기
        # -----------------------------------------------------
        classes = np.arange(len(proba))  # 0부터 9까지
        colors = ["skyblue"] * len(proba)
        colors[best_idx] = "orange"  # 가장 확률이 큰 클래스만 다른 색상

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(classes, proba, color=colors)
        ax.set_xticks(classes)
        ax.set_xlabel("클래스 (0~9)")
        ax.set_ylabel("확률")
        ax.set_title("클래스별 예측 확률 분포")

        # y축 범위를 [0, 1]로 고정
        ax.set_ylim(0, 1.0)

        # 그래프를 Streamlit에 출력
        st.pyplot(fig)

else:
    st.info("먼저 상단의 File Uploader를 이용하여 28×28 이미지를 업로드하세요.")