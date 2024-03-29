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
    youtube = build('youtube', 'v3',
                    developerKey='AIzaSyDKeOTZPWuLpSIHSkPPbtsxqEDF73-Nc8U')
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


def classify_count_comments(comments):

    global all_comments, relevant_comments, spam_comments, good_comments, bad_comments

    all_comments = np.array(comments)
    cnt1 = len(all_comments)

    # Array for relevant and spam comments
    label_col = []
    cp = []
    for c in all_comments:
        cp.append([c])
        comment_features = [c]
        features = np.array(comment_features)
        data_tfidf = tfidf_vectorizer.transform(features)
        prediction = spam_detection_model.predict(data_tfidf)
        label_col.append(prediction)

    label_array = np.array(label_col)
    relevant_comments = np.array(cp)[label_array == 0]
    spam_comments = np.array(cp)[label_array == 1]
    cnt2 = len(relevant_comments)
    cnt3 = len(spam_comments)

    # Array for comments of appreciation and griviences
    label_col = []
    cp = []
    for c in all_comments:
        cp.append(c)
        prediction_dict = sentiment_analysis_model.polarity_scores(c)
        prediction = prediction_dict["compound"]
        label_col.append(prediction)

    label_array = np.array(label_col)
    good_comments = np.array(cp)[label_array >= 0.5]
    bad_comments = np.array(cp)[label_array <= -0.5]
    cnt4 = len(good_comments)
    cnt5 = len(bad_comments)
    cnt_array = [cnt1, cnt2, cnt3, cnt4, cnt5]

    return cnt_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_link', methods=['POST'])
def process_link():
    youtube_link = request.form.get('youtube_link')
    global yt_link
    yt_link = youtube_link
    comments = get_youtube_comments(video_id=yt_link)
    cnt_array = []
    cnt_array = classify_count_comments(comments)

    return render_template('class.html', cnt_array=cnt_array)


@app.route('/all', methods=['POST'])
def all():
    button_type = request.form['button_name']
    global all_comments, relevant_comments, spam_comments, good_comments, bad_comments
    if (button_type == 'All'):
        required_comments = all_comments

    elif (button_type == 'Relevant'):
        required_comments = relevant_comments

    elif (button_type == 'Spam'):
        required_comments = spam_comments

    elif (button_type == 'Appreciation'):
        required_comments = good_comments

    elif (button_type == 'Grievance'):
        required_comments = bad_comments

    return render_template('comments-page.html', comments=required_comments, type=button_type)


if __name__ == '__main__':
    app.run(debug=True)

# For trial, use the ID of Eminem song video : 2BeksVabsCk
