# step1: import libraries and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# load the imdb dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key,value in word_index.items()}

# load the pre-trained model with relu activation
model = load_model('simple_rnn_imdb.h5')


# step 2: Helper functions
# Function to decode the review
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words ]
    padded_review = sequence.pad_sequences([encoded_review], maxlen = 500)
    return padded_review


# Streamlit App UI

st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(90deg,#ff4b4b,#ff8c42);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #B0B0B0;
    margin-bottom: 30px;
}

.stTextArea textarea {
    border-radius: 15px;
    border: 2px solid #ff4b4b;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #1E1E1E;
    text-align: center;
    margin-top: 20px;
}

.footer {
    text-align:center;
    color:gray;
    margin-top:40px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📌 About")
    st.write("""
    This application uses a Simple RNN model trained on the IMDB Movie Reviews Dataset.

    ### Features
    - Sentiment Analysis
    - Deep Learning (RNN)
    - Real-time Prediction
    - Confidence Score
    """)

# Main Title
st.markdown(
    "<div class='title'>🎬 IMDB Movie Review Sentiment Analysis</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Analyze movie reviews using Deep Learning</div>",
    unsafe_allow_html=True
)

# User Input
user_input = st.text_area(
    "✍️ Enter Your Movie Review",
    height=200,
    placeholder="Type your movie review here..."
)

# Button Center Alignment
col1, col2, col3 = st.columns([1,1,1])

with col2:
    classify = st.button(
        "🔍 Analyze Sentiment",
        use_container_width=True
    )

# Prediction
if classify:

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a movie review.")
    else:

        preprocessed_input = preprocess_text(user_input)

        prediction = model.predict(preprocessed_input)

        score = float(prediction[0][0])

        sentiment = "Positive 😊" if score > 0.5 else "Negative 😞"

        confidence = score if score > 0.5 else 1 - score

        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

        if score > 0.5:
            st.success(f"🎉 Sentiment: {sentiment}")
        else:
            st.error(f"📉 Sentiment: {sentiment}")

        st.write(f"### Confidence Score: {confidence:.2%}")

        st.progress(float(confidence))

        st.write(f"Raw Prediction Value: {score:.4f}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Optional review statistics
        st.subheader("📊 Review Statistics")

        words = user_input.split()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Words", len(words))

        with col2:
            st.metric("Characters", len(user_input))

        with col3:
            st.metric("Confidence", f"{confidence:.2%}")

else:
    st.info("Enter a movie review and click Analyze Sentiment.")

# Footer
st.markdown(
    "<div class='footer'>Built with ❤️ using TensorFlow, Keras and Streamlit</div>",
    unsafe_allow_html=True
)