# streamlit_mnist_sgd.py
import streamlit as st
import joblib
import numpy as np
from PIL import Image

st.set_page_config(page_title="MNIST SGDClassifier 예측", layout="centered")
st.title("📝 SGDClassifier(log_loss) 모델로 MNIST 예측 + 확신도 출력")

# -------------------------------------------------------------
# 1) 학습해둔 모델 불러오기 (cache 처리)
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path="model.pkl"):
    model = joblib.load(path)
    return model

model = load_model("model.pkl")


# -------------------------------------------------------------
# 2) MNIST 형식의 이미지 업로드 받기
# -------------------------------------------------------------
st.subheader("▶ MNIST 형식 이미지 업로드 (28×28 흑백)")
uploaded_file = st.file_uploader(
    label="0~9 숫자가 들어있는 28×28 흑백 이미지 업로드 (.png / .jpg / .jpeg)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # 2-1) PIL로 열어서 그레이스케일로 변환 → 28×28 크기로 리사이즈
    img = Image.open(uploaded_file).convert("L").resize((28, 28))
    # 2-2) NumPy 배열로 변환 (uint8, 0~255)
    arr = np.array(img).astype(np.uint8)  # shape = (28, 28)

    # 2-3) MNIST 전처리: (255 - 픽셀값) → float32 → 0~1 스케일링 → flatten
    arr_inverted = 255 - arr                        # 뒤집기
    arr_scaled = arr_inverted.astype("float32") / 255.0
    X_input = arr_scaled.reshape(1, -1)             # shape = (1, 784)

    # 이미지 미리보기
    st.image(img, caption="Uploaded Original Image (28×28 Grayscale)", use_column_width=False)

    # ---------------------------------------------------------
    # 3) 예측 버튼: predict + predict_proba
    # ---------------------------------------------------------
    if st.button("▶ 모델 예측하기"):
        # 3-1) predict (예측된 클래스)
        pred = model.predict(X_input)[0]

        # 3-2) predict_proba (클래스별 확률)
        try:
            proba = model.predict_proba(X_input)[0]  # shape = (n_classes,)
            best_idx = np.argmax(proba)
            best_prob = proba[best_idx]

            st.write(f"**예측된 숫자:** {pred}")
            st.write(f"**모델 확신도 (해당 클래스 확률):** {best_prob:.4f}")

            # (선택) 전체 클래스 확률 분포를 표로 표시
            import pandas as pd
            df_proba = pd.DataFrame({
                "class (0~9)": np.arange(len(proba)),
                "probability": proba
            })
            st.subheader("클래스별 확률 분포")
            st.dataframe(df_proba.style.format({"probability": "{:.4f}"}))

        except AttributeError:
            st.error(
                "❌ 이 모델은 `predict_proba`를 지원하지 않습니다.\n"
                "`SGDRegressor`나 분류기라도 loss가 확률 출력을 지원하지 않는 경우에는 `predict_proba`가 없습니다."
            )

else:
    st.info("먼저 상단의 File Uploader에서 28×28 이미지를 업로드하세요.")
