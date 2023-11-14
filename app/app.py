# pip install Flask
from flask import Flask, render_template, request
import joblib

# In[6]:

label_dict = {0: '__label__thể_thao', 1: '__label__âm_nhạc', 2: '__label__nhịp_sống',3: '__label__công_nghệ', 4: '__label__thời_sự', 
              5: '__label__thế_giới', 6: '__label__thời_trang', 7: '__label__du_lịch', 8: '__label__sống_trẻ',9: '__label__giáo_dục', 
              10: '__label__kinh_doanh', 11: '__label__pháp_luật', 12: '__label__giải_trí', 13: '__label__phim_ảnh', 14: '__label__xe_360',
             15: '__label__ẩm_thực', 16: '__label__xuất_bản', 17: '__label__sức_khỏe'}

# Load model from file
model = joblib.load('naive_bayes.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define home route
@app.route('/')
def home():
    return render_template('home.html')

# Define classify route
@app.route('/classify', methods=['POST'])
def classify():
    # Get input text from form
    text = request.form['text']

    # Predict label for input text
    predicted_label = model.predict([text])[0]
    predicted_text_label = label_dict[predicted_label]

    # Render result template with predicted label
    return render_template('result.html', text=text, label=predicted_text_label)

if __name__ == '__main__':
    app.run(debug=True)
    






