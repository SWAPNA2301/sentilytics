"""Sentilytics review explorer. Inference only; no training or feature fitting."""

import logging

import streamlit as st

from clean_reviews import clean_text
from predict_sentiment import predict_sentiment

SAMPLES = {
    'Write my own': '',
    'Loved it': 'Excellent food! Fresh, delicious, and worth every penny. I love it.',
    'Disappointed': 'Terrible taste and stale ingredients. I will never buy this again.',
    'Mixed feelings': 'The packaging was nice, but the flavor was not good and delivery was slow.',
}


def choose_sample():
    st.session_state.review = SAMPLES[st.session_state.sample]
    st.session_state.pop('prediction', None)


def clear_prediction():
    st.session_state.pop('prediction', None)


def main():
    st.set_page_config(page_title='Sentilytics | Review insights', page_icon='✦', layout='centered')
    st.markdown('''<style>
    .stApp {background: linear-gradient(150deg, #f5f8ff 0%, #f8fafc 65%, #eefbf6 100%);}
    .block-container {max-width: 900px; padding-top: 3rem; padding-bottom: 2rem;}
    .eyebrow {color: #2563eb; font-size: .8rem; font-weight: 700; letter-spacing: .15em;}
    h1 {letter-spacing: -.045em;}
    div.stButton > button[kind="primary"] {border-radius: 12px; min-height: 3rem;}
    @media (max-width: 640px) {.block-container {padding-top: 1.5rem;}}
    </style>''', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">SENTILYTICS / REVIEW INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title('Every review has a story.')
    st.write('Explore the sentiment behind a customer review in seconds.')

    with st.container(border=True):
        st.subheader('Try a review')
        st.selectbox('Start with an example', list(SAMPLES), key='sample', on_change=choose_sample)
        review = st.text_area('Your review', key='review', height=180,
                              placeholder='What did you think of the product?', on_change=clear_prediction)
        st.caption('Best suited to English food and product reviews.')
        if st.button('Predict sentiment', type='primary', use_container_width=True):
            clear_prediction()
            if not clean_text(review):
                st.warning('Please enter a review with some text before predicting.')
            else:
                try:
                    with st.spinner('Reading your review…'):
                        st.session_state.prediction = predict_sentiment(review)
                except FileNotFoundError:
                    st.error('Prediction is unavailable: saved model files are missing. Follow the artifact setup in the README.')
                except Exception:
                    logging.getLogger(__name__).exception('Sentiment prediction failed')
                    st.error('We could not analyze this review. Please try again or check the local model setup.')

    result = st.session_state.get('prediction')
    if result:
        with st.container(border=True):
            st.subheader('Your result')
            if result['sentiment'] == 'Positive':
                st.success('Positive sentiment')
            else:
                st.error('Negative sentiment')
            st.metric('Predicted-class probability', f"{result['confidence']:.1%}")
            st.progress(result['confidence'])
            st.caption('Model probability, not a guarantee. Mixed or sarcastic reviews can be misclassified.')

    with st.expander('About this prediction'):
        st.write('Sentilytics uses a saved logistic regression classifier with word and two-word TF-IDF features. '
                 'Reviews are cleaned using the same preprocessing as training. Predictions run locally.')
        st.write('Test accuracy: 89.64% · Positive-class F1: 0.8959. '
                 'The model was selected using this test set, so these are comparison results rather than an independent final estimate.')
        st.caption('Two classes: Positive and Negative. Neutral sentiment is not modeled. '
                   'An unfamiliar vocabulary can yield a prediction based mostly on the model’s baseline.')
    st.caption('Sentilytics · From words to insight')


if __name__ == '__main__':
    main()
