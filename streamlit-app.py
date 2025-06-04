# streamlit_predict.py
import streamlit as st
import joblib
import numpy as np
from PIL import Image

st.set_page_config(page_title="Model Predict + Confidence", layout="centered")
st.title("✅ Streamlit으로 모델 예측하기 (predict + predict_proba)")

# -------------------------------------------------------------
# 1) 미리 학습된 모델 불러오기
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path="model.pkl"):
    """
    joblib으로 저장한 model.pkl을 로드합니다.
    - 이 예제는 SGDClassifier(loss="log")처럼 predict_proba를 지원하는 분류 모델을 가정합니다.
    - 만약 실제로 SGDRegressor를 쓰고 있다면 아래 predict_proba 호출 코드는 작동하지 않습니다.
    """
    model = joblib.load(path)
    return model

model = load_model("model.pkl")


# -------------------------------------------------------------
# 2) 사용자에게 이미지 파일 업로드 받기
# -------------------------------------------------------------
st.subheader("▶ 분류할 이미지 업로드")
uploaded_file = st.file_uploader(
    label="이미지 파일(.png, .jpg, .jpeg)을 업로드하세요",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # PIL 이미지로 열기 → 그레이스케일 변환 → 28×28 리사이즈 (필요 시 조정)
    img = Image.open(uploaded_file).convert("L").resize((28, 28))
    img_array = np.array(img).astype(np.float32).flatten().reshape(1, -1)  # shape=(1, 784)

    # 화면에 업로드된 이미지 표시
    st.image(img, caption="업로드된 이미지 (28×28 그레이스케일)", use_column_width=False)

    # ---------------------------------------------------------
    # 3) 예측 버튼: predict + predict_proba
    # ---------------------------------------------------------
    if st.button("▶ 모델 예측하기"):
        # 3-1) 클래스 예측
        pred_class = model.predict(img_array)[0]

        # 3-2) 확률(Confidence) 구하기
        #     - SGDClassifier(loss="log")나 predict_proba를 지원하는 분류기만 작동
        try:
            proba = model.predict_proba(img_array)[0]  # 각 클래스별 확률
            # 최대 확률 및 인덱스 추출
            max_idx = np.argmax(proba)
            max_prob = proba[max_idx]
            st.write(f"**예측된 클래스:** {pred_class}")
            st.write(f"**모델 확신도 (해당 클래스 확률):** {max_prob:.4f}")
            
            # (선택) 전체 클래스 확률 분포를 표 형태로 보여주기
            st.subheader("클래스별 확률 분포")
            import pandas as pd
            df_proba = pd.DataFrame({
                "class": np.arange(len(proba)),
                "probability": proba
            })
            st.dataframe(df_proba.style.format({"probability": "{:.4f}"}))
        except AttributeError:
            st.error("❌ 이 모델은 predict_proba를 지원하지 않습니다.\n"
                     "SGDRegressor나 loss='squared_loss' 같은 회귀형 모델에는 확률 출력 기능이 없습니다.")

        # 3-3) (Optional) 결정 함수(decision_function) 값 출력
        #       - 이진 분류일 경우 decision_function을 통해 마진(margin) 점수를 얻을 수 있음
        if hasattr(model, "decision_function"):
            margin = model.decision_function(img_array)[0]
            st.write(f"**Decision function (마진 점수):** {margin:.4f}")

else:
    st.info("먼저 상단에서 이미지를 업로드하세요.")

st.markdown("---")