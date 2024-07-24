import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import streamlit as st

from functions import *

#Title
# st.set_page_config(layout='wide')
st.title("Customer Review Dashboard")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
# submit_1= st.button('submit')
if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True)
    unique_product_count = df['Product Name'].nunique()
    df_reviews = df[['Product Name', 'Reviews']] 
    # Get unique items from the 'Category' column
    unique_items = df['Product Name'].unique().tolist()
    # Create a new DataFrame with unique products and prices
    unique_df = df.drop_duplicates(subset=['Product Name']).reset_index(drop=True)
    # Count the products by brand
    brand_counts = df['Brand Name'].value_counts()


    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label='No. of Products', value = df['Product Name'].nunique())
    kpi2.metric(label = 'Total No. of Reviews', value = df.shape[0])
    kpi3.metric(label = 'No. of Brands', value = unique_df['Brand Name'].nunique())

        
    # Assuming df is the DataFrame containing the data
    # plt.hist(df['Price'], bins=20, edgecolor='black')
    # plt.xlabel('Price')
    # plt.ylabel('Frequency')
    # plt.title('Price Range of Products')

    # # Use st.pyplot to display the plot in Streamlit
    # st.pyplot(plt)


    st.title('Price Range of Products')

    # Assuming df is the DataFrame containing the data
    plt.figure(figsize=(8, 6))  # Set the figure size

    # Plot the histogram with customized colors and transparency
    plt.hist(unique_df['Price'], bins=20, edgecolor='black', color='skyblue', alpha=0.8)

    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.title('Price Range of Products', loc= 'center')

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.5)

    # Customize tick labels
    plt.xticks(rotation=45)  # Rotate x-axis tick labels by 45 degrees

    # Use st.pyplot to display the plot in Streamlit
    st.pyplot(plt)

    # Add a search box to the selectbox
    selected_item = st.selectbox('Select Product', options=unique_items, index=0, key='category_select', help="")
    st.write(selected_item)
    # Filter the DataFrame for the specified product and calculate the average rating
    average_rating = round(df[df['Product Name'] == selected_item]['Rating'].mean(), 2)
    # Filter the DataFrame for the specified product and count the reviews
    review_count = df[df['Product Name'] == selected_item]['Rating'].count()
    # Filter the DataFrame for the specified product and extract the brand
    brand = df[df['Product Name'] == selected_item]['Brand Name'].iloc[0]



    kpi4, kpi5, kpi6 = st.columns(3)
    kpi4.metric(label = 'Average Rating', value = average_rating)
    kpi5.metric(label= 'No. of Reviews', value = review_count )
    kpi6.metric(label= 'Brand', value = brand)

    # Filter the DataFrame for the selected product
    selected_df = df_reviews[df_reviews['Product Name'] == selected_item].copy()
    #Cleaning
    selected_df['cleaned'] = selected_df['Reviews'].apply(preprocess_review)
    #Calculating the polarity and subjectivity of the tweet using Textblob Analyzer
    selected_df[['polarity', 'subjectivity']]=selected_df['cleaned'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
    selected_df['Sentiment'] = selected_df['polarity'].apply(lambda x: textblob_sentiment(x))
    sentiment_counts = count_values_in_column(selected_df, "Sentiment")
    names = sentiment_counts.index
    size = sentiment_counts["Percentage"]

    st.markdown("<div style='text-align: center;'><h3>Sentiment Analysis</h3></div>", unsafe_allow_html=True)
    df_1, plot_1 = st.columns(2)

    with df_1:

        st.dataframe(sentiment_counts)

    with plot_1:
        # Create a circle for the center of the plot
        my_circle = plt.Circle((0, 0), 0.7, color='white')

        # Create the Pie Chart
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ['#66b3ff', '#99ff99', '#ff9999']  # Custom colors for the Pie Chart slices
        wedges, texts, autotexts = ax.pie(size, labels=names, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures the Pie Chart looks like a circle
        ax.set_title('Sentiment Analysis', fontsize=16)

        # Customizing text inside Pie Chart slices
        for text, autotext in zip(texts, autotexts):
            text.set(size=12, color='black', weight='bold')  # Change font size, color, and weight
            autotext.set(size=12, color='white', weight='bold')  # Change font size, color, and weight

        # Add the circle to the plot
        ax.add_artist(my_circle)

        # Display the plot using st.pyplot with improved styles
        st.pyplot(fig)


    #Word Cloud
    st.markdown("<div style='text-align: center;'><h4>Word Cloud</h4></div>", unsafe_allow_html=True)

    # Filter positive reviews
    neutral_reviews = selected_df[selected_df['Sentiment'] == 'neutral']['cleaned']
    text = " ".join(neutral_reviews)

    # Generate Word Cloud
    word_cloud = WordCloud(max_words=100, width=1600, height=800, collocations=False)
    word_cloud.generate(text)

    # Convert Word Cloud to image
    word_cloud_image = word_cloud.to_image()

    #Display the Word Cloud in Streamlit
    st.image(word_cloud_image)


    #Text Analysis
    st.markdown("<div style='text-align: center;'><h4>Text Analysis</h4></div>", unsafe_allow_html=True)
    #Calculating review's lenght and word count
    selected_df['text_len'] = selected_df['cleaned'].astype(str).apply(len)
    selected_df['text_word_count'] = selected_df['cleaned'].apply(lambda x: len(str(x).split()))
    col_1_df = round(pd.DataFrame(selected_df.groupby("Sentiment")['text_len'].mean()),2)
    col_2_df = round(pd.DataFrame(selected_df.groupby("Sentiment")['text_word_count'].mean()),2)

    col_1, col_2 = st.columns(2)

    with col_1:
        st.dataframe(col_1_df)

    with col_2:
        st.dataframe(col_2_df)
    
    st.markdown("<div style='text-align: center;'><h4>Review Sentiment</h4></div>", unsafe_allow_html=True)
    review = st.text_input("Enter the review for the product:")
    # submit = st.button("Submit")
    if review:
        submit_1 = 1
        prep_review = preprocess_review(review)
        polarity, subjectivity = TextBlob(prep_review).sentiment
        sentiment = textblob_sentiment(polarity)
        score = SentimentIntensityAnalyzer().polarity_scores(prep_review)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        pd_senti= pd.DataFrame({
                        "Sentiment Score": [polarity],
                        "Positive %": [pos*100] ,
                        "Neutral %": [neu*100],
                        "Negative %": [neg*100]
        })
        senti_word , positive_percent, neutral_percent, negative_percent = st.columns(4)

        senti_word.metric(label = 'Sentiment of Review', value = sentiment.upper())
        # polarity_score.metric(label = "Sentiment Score", value = round(polarity, 1))
        positive_percent.metric(label= 'Positive Word %', value = round(pos*100,1))
        neutral_percent.metric(label= 'Neutral Word %', value = round(neu*100, 1))
        negative_percent.metric(label= 'Negative Word %', value = round(neg*100,1))

        