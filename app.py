from flask import Flask, render_template, request, redirect, url_for
import os
from run_analysis import run_analysis  # ✅ 분석 함수 불러오기

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/shorts_output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['video']
    if file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)

        run_analysis(path)  # ✅ 분석 실행

        return render_template('loading.html')  # 분석 중 페이지 (자동 redirect to /result)
    return '업로드 실패'

@app.route('/result')
def result():
    files = os.listdir(RESULT_FOLDER)
    videos = [f for f in files if f.endswith('.mp4')]
    return render_template('result.html', videos=videos)

if __name__ == '__main__':
    app.run(debug=True)
