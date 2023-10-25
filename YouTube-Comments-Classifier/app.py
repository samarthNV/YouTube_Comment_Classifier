from flask import Flask, render_template, request
from googleapiclient.discovery import build
import numpy as np
import pickle

# Creating Flask Server
app = Flask(__name__)

with open('spam_detection_model.pkl', 'rb') as model_file:
    spam_detection_model = pickle.load(model_file)

with open('sentiment_analysis_model.pkl', 'rb') as model_file:
    sentiment_analysis_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Function to extract youtube comments from ID
yt_link = ""

def get_youtube_comments(video_id):
    youtube = build('youtube', 'v3', developerKey='AIzaSyDKeOTZPWuLpSIHSkPPbtsxqEDF73-Nc8U')
    comments = []
    counter = 10

    next_page_token = None
    while True:
        if counter == 0:
            break
        else:
            counter = counter - 1

        results = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,
            pageToken=next_page_token
        ).execute()

        for item in results.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        # Check if there are more comments
        next_page_token = results.get("nextPageToken")
        if not next_page_token:
            break

    return comments


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_link', methods=['POST'])
def process_link():
    youtube_link = request.form.get('youtube_link')
    global yt_link
    yt_link = youtube_link

    return render_template('class.html')

@app.route('/all', methods=['POST'])
def all():
    button_type = request.form['button_name']
    if(button_type=='All'):
        comments = get_youtube_comments(video_id=yt_link)

        label_col = []
        for c in comments:
            comment_features = [c]
            features = np.array(comment_features)
            data_tfidf = tfidf_vectorizer.transform(features)
            prediction = spam_detection_model.predict(data_tfidf)
            label_col.append(prediction)

        return render_template('comments-page.html', comments=comments, type="All")

    elif(button_type=='Relevant'):
        comments = get_youtube_comments(video_id=yt_link)
        comments = np.array(comments)
        label_col = []
        cp = []
        for c in comments:
            cp.append([c])
            comment_features = [c]
            features = np.array(comment_features)
            data_tfidf = tfidf_vectorizer.transform(features)
            prediction = spam_detection_model.predict(data_tfidf)
            label_col.append(prediction)

        label_array = np.array(label_col)
        cp = np.array(cp)
        relevant_comments = cp[label_array == 0]

        return render_template('comments-page.html', comments=relevant_comments, type="Relevant")
    
    elif (button_type == 'Spam'):
        comments = get_youtube_comments(video_id=yt_link)
        comments = np.array(comments)
        label_col = []
        cp = []
        for c in comments:
            cp.append([c])
            comment_features = [c]
            features = np.array(comment_features)
            data_tfidf = tfidf_vectorizer.transform(features)
            prediction = spam_detection_model.predict(data_tfidf)
            label_col.append(prediction)

        label_array = np.array(label_col)
        cp = np.array(cp)
        spam_comments = cp[label_array == 1]

        return render_template('comments-page.html', comments=spam_comments, type="Spam")
 
    elif (button_type == 'Appreciation'):
        comments = get_youtube_comments(video_id=yt_link)
        comments = np.array(comments)
        label_col = []
        cp = []
        for c in comments:
            cp.append(c)
            prediction_dict = sentiment_analysis_model.polarity_scores(c);
            prediction = prediction_dict["compound"]
            label_col.append(prediction)

        label_array = np.array(label_col)
        cp = np.array(cp)
        good_comments = cp[label_array >= 0.5]

        return render_template('comments-page.html', comments=good_comments, type="Appreciation")
    
    elif (button_type == 'Grievances'):
        comments = get_youtube_comments(video_id=yt_link)
        comments = np.array(comments)
        label_col = []
        cp = []
        for c in comments:
            cp.append(c)
            prediction_dict = sentiment_analysis_model.polarity_scores(c);
            prediction = prediction_dict["compound"]
            label_col.append(prediction)

        label_array = np.array(label_col)
        cp = np.array(cp)
        bad_comments = cp[label_array <= -0.5]

        return render_template('comments-page.html', comments=bad_comments, type="Grievance")


if __name__ == '__main__':
    app.run(debug=True)

# For trial, use the ID of Eminem song video : 2BeksVabsCk