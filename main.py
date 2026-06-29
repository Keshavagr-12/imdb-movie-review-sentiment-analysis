import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]


import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negtive')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')


# import streamlit as st

# # ---------------- PAGE CONFIG ---------------- #
# st.set_page_config(
#     page_title="IMDB Sentiment Studio",
#     page_icon="🎬",
#     layout="wide"
# )

# # ---------------- CUSTOM CSS ---------------- #
# st.markdown("""
# <style>

# /* Background */
# .stApp{
#     background: linear-gradient(135deg,#141E30,#243B55);
# }

# /* Remove Streamlit default menu */
# #MainMenu {visibility:hidden;}
# footer {visibility:hidden;}
# header {visibility:hidden;}

# /* Main title */
# .title{
#     text-align:center;
#     color:white;
#     font-size:52px;
#     font-weight:800;
#     margin-bottom:5px;
# }

# .subtitle{
#     text-align:center;
#     color:#d6d6d6;
#     font-size:20px;
#     margin-bottom:35px;
# }

# /* Card */
# .card{
#     background:rgba(255,255,255,0.08);
#     backdrop-filter: blur(15px);
#     padding:30px;
#     border-radius:20px;
#     border:1px solid rgba(255,255,255,0.15);
#     box-shadow:0px 8px 25px rgba(0,0,0,.35);
#     margin-bottom:30px;
# }

# /* Result card */
# .result-card{
#     background:white;
#     border-radius:18px;
#     padding:25px;
#     box-shadow:0px 8px 25px rgba(0,0,0,.15);
# }

# /* Metric */
# .metric{
#     text-align:center;
#     padding:20px;
#     border-radius:15px;
#     background:#f6f6f6;
# }

# /* Footer */
# .footer{
#     text-align:center;
#     color:#d0d0d0;
#     margin-top:40px;
#     font-size:15px;
# }

# </style>
# """, unsafe_allow_html=True)

# # ---------------- HEADER ---------------- #

# st.markdown("<div class='title'>🎬 IMDB Sentiment Studio</div>", unsafe_allow_html=True)

# st.markdown(
#     "<div class='subtitle'>AI Powered Movie Review Sentiment Analysis</div>",
#     unsafe_allow_html=True,
# )

# # ---------------- INPUT CARD ---------------- #

# st.markdown("<div class='card'>", unsafe_allow_html=True)

# st.subheader("📝 Write a Movie Review")

# user_input = st.text_area(
#     "",
#     height=220,
#     placeholder="Example:\n\nThis movie was absolutely brilliant. The acting, direction and soundtrack were amazing..."
# )

# col1, col2, col3 = st.columns([1,2,1])

# with col2:
#     predict = st.button(
#         "🔍 Analyze Review",
#         use_container_width=True
#     )

# st.markdown("</div>", unsafe_allow_html=True)

# # ---------------- RESULT ---------------- #

# if predict:

#     if user_input.strip()=="":
#         st.warning("Please enter a movie review.")

#     else:

#         preprocessed_input = preprocess_text(user_input)

#         prediction = model.predict(preprocessed_input, verbose=0)

#         sentiment = "Positive" if prediction[0][0]>0.5 else "Negative"

#         score=float(prediction[0][0])

#         confidence = score if sentiment=="Positive" else 1-score

#         st.markdown("<div class='result-card'>", unsafe_allow_html=True)

#         st.subheader("📊 Analysis Result")

#         if sentiment=="Positive":
#             st.success("😊 Positive Review")
#         else:
#             st.error("😞 Negative Review")

#         c1,c2,c3=st.columns(3)

#         with c1:
#             st.metric(
#                 "Prediction Score",
#                 f"{score:.4f}"
#             )

#         with c2:
#             st.metric(
#                 "Confidence",
#                 f"{confidence*100:.2f}%"
#             )

#         with c3:
#             st.metric(
#                 "Sentiment",
#                 sentiment
#             )

#         st.write("### Confidence Meter")

#         st.progress(confidence)

#         if confidence>0.90:
#             st.success("Very High Confidence")
#         elif confidence>0.75:
#             st.info("High Confidence")
#         elif confidence>0.60:
#             st.warning("Moderate Confidence")
#         else:
#             st.error("Low Confidence")

#         st.markdown("</div>", unsafe_allow_html=True)

# st.markdown(
#     "<div class='footer'>Built with ❤️ using TensorFlow • Keras • Streamlit</div>",
#     unsafe_allow_html=True,
# )