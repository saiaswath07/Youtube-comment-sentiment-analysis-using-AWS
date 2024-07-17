import streamlit as st
import demoji
import requests
# import googleapiclient.discovery
from urllib.parse import urlparse, parse_qs
# import matplotlib.pyplot as plt

demoji.download_codes()

apiKey = 'AIzaSyB98DzwZstIoZAHjmOkpzvV9w55Gr7MU0Y'
api_service_name = "youtube"
api_version = "v3"

# youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=apiKey)

def get_video_id(video_url):
    parsed_url = urlparse(video_url)
    video_id = parse_qs(parsed_url.query).get('v')
    return video_id[0] if video_id else None

def analyze_text_sentiment(text):
    aws_endpoint = 'https://4h2km8bo4d.execute-api.us-east-1.amazonaws.com/prod/'
    payload = {"body": text}
    response = requests.post(aws_endpoint, json=payload)

    try:
        sentiment_result_str = response.json().get("body")
        if sentiment_result_str is not None:
            sentiment_result = float(sentiment_result_str)
            return sentiment_result
        else:
            return None
    except (KeyError, ValueError) as e:
        st.error(f"Error extracting sentiment: {str(e)}\nResponse text: {response.text}")
        return None

st.title("YouTube Video Comments Analyzer")

analysis_option = st.selectbox("Select analysis option:", ["Text", "Video"])

if analysis_option == "Text":
    text_input = st.text_area("Enter text for sentiment analysis:")

    if st.button("Analyze Text"):
        if text_input:
            sentiment_result = analyze_text_sentiment(text_input)
            if sentiment_result is not None:
                st.write(f"Sentiment: {'Positive 😃' if sentiment_result == 1 else 'Negative 😠'}")

elif analysis_option == "Video":
    video_url = st.text_input("Enter the YouTube video URL:")

    retrieve_option = st.radio("Select comment retrieval option:", ["Retrieve Full", "Retrieve Custom"])

    if retrieve_option == "Retrieve Custom":
        num_comments = st.number_input("Enter the number of comments to retrieve:", min_value=1, value=10)
    else:
        num_comments = 100

    if st.button("Get Comments"):
        video_id = get_video_id(video_url)

        if video_id:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=num_comments
            )

            response = request.execute()

            comments_data = []
            positive_count = 0
            negative_count = 0

            for item in response['items']:
                author_name = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']

                processed_comment = demoji.replace(comment_text, '')

                if 'http' not in processed_comment and 'Question of the day:' not in processed_comment:
                    sentiment_result = analyze_text_sentiment(processed_comment)
                    if sentiment_result is not None:
                        emoji = '😃' if sentiment_result == 1 else '😠'

                        comments_data.append({
                            'Username': author_name,
                            'Comment': processed_comment,
                            'Sentiment': 'Positive' if sentiment_result == 1 else 'Negative',
                            'Emoji': emoji
                        })

                        if sentiment_result == 1:
                            positive_count += 1
                        else:
                            negative_count += 1

            st.table(comments_data)

            st.write(f"Total Positive Comments: {positive_count}")
            st.write(f"Total Negative Comments: {negative_count}")

            labels = ['Positive', 'Negative']
            sizes = [positive_count, negative_count]
            colors = ['#66ff66', '#ff6666']
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')
            st.pyplot(fig)

        else:
            st.error("Invalid video URL. Please provide a valid YouTube video URL.")
