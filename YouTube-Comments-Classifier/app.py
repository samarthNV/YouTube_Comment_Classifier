from flask import Flask, render_template, request, url_for, redirect
from googleapiclient.discovery import build
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
import numpy as np
import pickle
import os

load_dotenv()

# Generate a secure random secret key
secret_key = os.urandom(24)

# Convert the bytes to a string for easy use in Flask
secret_key_str = secret_key.hex()

# Creating Flask Server
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SECRET_KEY'] = secret_key_str
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(200), nullable=False, unique=True)
    password = db.Column(db.String(255), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)])

    password = PasswordField(
        validators=[InputRequired(), Length(min=8, max=20)])

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'Username already exists.')


class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)])

    password = PasswordField(
        validators=[InputRequired(), Length(min=8, max=20)])

    submit = SubmitField('Login')


with app.app_context():
    db.create_all()

with open('models/spam_detection_model.pkl', 'rb') as model_file:
    spam_detection_model = pickle.load(model_file)

with open('models/sentiment_analysis_model.pkl', 'rb') as model_file:
    sentiment_analysis_model = pickle.load(model_file)

with open('models/vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Function to extract youtube comments from ID
yt_link = ""


def get_youtube_comments(video_id):
    youtube = build('youtube', 'v3', developerKey=os.getenv('DEVELOPER_KEY'))
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
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(
            form.password.data).decode('utf-8')
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return render_template('home.html')


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('index.html')


@app.route('/process_link', methods=['POST'])
@login_required
def process_link():
    youtube_link = request.form.get('youtube_link')
    global yt_link
    yt_link = youtube_link
    comments = get_youtube_comments(video_id=yt_link)
    cnt_array = []
    cnt_array = classify_count_comments(comments)

    return render_template('class.html', cnt_array=cnt_array)


@app.route('/all', methods=['POST'])
@login_required
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