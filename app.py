# app.py

from flask import Flask, send_from_directory
import os

app = Flask(__name__, static_folder='.')

# 홈 페이지 제공
@app.route('/')
def serve_home():
    return send_from_directory('.', 'index.html')

# 다른 정적 페이지 제공
@app.route('/books')
def serve_bookshelf():
    return send_from_directory('.', 'bookshelf.html')

@app.route('/magazine')
def serve_magazine():
    return send_from_directory('.', 'magazine.html')

@app.route('/likes')
def serve_likes():
    return send_from_directory('.', 'likes.html')

@app.route('/chatbot')
def serve_chatbot():
    return send_from_directory('.', 'chatbot.html')

# 정적 파일 제공 (CSS, JS, 이미지 등)
@app.route('/static/<path:path>')
def serve_static_files(path):
    return send_from_directory('static', path)

# 404 에러 핸들러
@app.errorhandler(404)
def page_not_found(e):
    return "페이지를 찾을 수 없습니다.", 404

if __name__ == '__main__':
    app.run(debug=True)
