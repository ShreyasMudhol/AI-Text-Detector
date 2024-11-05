import joblib
import streamlit as st

def model(text):
    try:
        PredictorModel = joblib.load('lr.pkl')
        tfidf = joblib.load("tfidf_lr.pkl")
        
        text_transformed = tfidf.transform([text])
        res = PredictorModel.predict(text_transformed)
        return res
    except Exception as e:
        st.error(f"Error while processing: {e}")
        return None

def load_css():
    with open("style.css", "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Streamlit UI
st.markdown("<h1>AI Text Detector</h1>", unsafe_allow_html=True)
st.markdown("<h3>Enter some text and find out if it's AI-generated or Human-generated!</h3>", unsafe_allow_html=True)

def streamlit_app():
    text_input = st.text_area("", height=200, help="Type or paste your text in this area.",placeholder="Enter your text...")
    click = st.button("Classify Text")
    
    if click:
        if text_input.strip():
            result = model(text_input)
            if result is not None:
                if result[0] == 1:
                    st.markdown('<p class="success">✅ This text is likely AI-generated.</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="success">✅ This text is likely human-generated.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="warning">⚠️ Please enter some text before classifying.</p>', unsafe_allow_html=True)

# Run the Streamlit app
streamlit_app()