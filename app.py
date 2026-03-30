import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load your trained model
pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

# Emoji dictionary
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱",
    "happy": "🤗", "joy": "😂", "neutral": "😐",
    "sad": "😔", "sadness": "😔", "shame": "😳",
    "surprise": "😮"
}

# ----------------------- CSS -----------------------
def load_css():
    st.markdown("""
    <style>
    /* Page Background */
    .stApp {
        background: linear-gradient(135deg, #c084fc, #60a5fa);
        display: flex;
        justify-content: center;
    }



      /* Title */
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #4f46e5;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 20px;
    }

    /* Input box */
    textarea {
        border-radius: 12px !important;
        border: 2px solid #ddd !important;
        padding: 12px !important;
    }

    /* Button */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #6366f1);
        color: white;
        border-radius: 12px;
        padding: 12px;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
        border: none;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb, #4f46e5);
    }

    /* Result Box */
    .result-box {
        background: #fff3e6;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------- Prediction Functions -------------------
def predict_emotions(docx):
    return pipe_lr.predict([docx])[0]

def get_prediction_proba(docx):
    return pipe_lr.predict_proba([docx])

# ------------------- Main App -------------------
def main():
    load_css()  # Apply custom CSS


    # Title
    st.markdown('<h1 class="title">Text Emotion Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Detect emotions in your text with NLP</p>', unsafe_allow_html=True)

    # Input Form
    with st.form(key='my_form'):
        raw_text = st.text_area("Type Your Text Here")
        submit_text = st.form_submit_button(label='Analyze Emotion')

    if submit_text:
        if not raw_text.strip():
            st.warning("Please enter some text to analyze!")
        else:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            # ---------------- Left Column: Text & Prediction ----------------
            with col1:
            
                st.subheader("Original Text")
                st.write(raw_text)

                st.subheader("Predicted Emotion")
                emoji_icon = emotions_emoji_dict.get(prediction, "")
                st.write(f"{prediction.capitalize()} {emoji_icon}")
                st.write(f"Confidence: {np.max(probability):.2f}")
                st.markdown('</div>', unsafe_allow_html=True)

            # ---------------- Right Column: Probability Chart ----------------
            with col2:
                st.subheader("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='emotions',
                    y='probability',
                    color=alt.Color('probability', scale=alt.Scale(scheme='plasma')),
                    tooltip=['emotions', alt.Tooltip('probability', format='.2f')]
                ).properties(title='Emotion Probabilities')
                st.altair_chart(fig, use_container_width=True)

    # Close the white container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()