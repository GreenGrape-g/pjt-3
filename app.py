# app.py

from flask import Flask, jsonify, request
from flask_cors import CORS
from utils.books import get_all_books, get_book_by_title
from utils.magazine import get_all_articles, get_article_by_id
from utils.favorites import get_favorites_message
from utils.chatbot import handle_chatbot_request  # 챗봇 핸들러 함수 가져오기
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)
CORS(app)

# 테스트 라우트
@app.route('/hi')
def hi():
    return jsonify({'message': 'hi'})

# 책 목록 API
@app.route('/books', methods=['GET'])
def books():
    book_list = get_all_books()
    return jsonify(book_list)

# 책 제목으로 책 상세 정보 검색 API
@app.route('/book', methods=['GET'])
def book_by_title_route():
    title = request.args.get('title')
    if not title:
        return jsonify({'error': '책 제목을 입력해주세요.'}), 400
    book = get_book_by_title(title)
    if not book:
        return jsonify({'error': '해당 제목의 책을 찾을 수 없습니다.'}), 404
    return jsonify(book)

# 잡지 콘텐츠 목록 API
@app.route('/magazine', methods=['GET'])
def magazine():
    articles = get_all_articles()
    return jsonify(articles)

# 잡지 상세 콘텐츠 API
@app.route('/magazine/<article_id>', methods=['GET'])
def magazine_article(article_id):
    article = get_article_by_id(article_id)
    if not article:
        return jsonify({'error': '해당 기사를 찾을 수 없습니다.'}), 404
    return jsonify(article)

# 책 신청 게시판 (화면만 구현)
@app.route('/requests_list', methods=['GET'])
def book_request():
    return jsonify({'message': '여기는 책 신청 게시판 목록 화면입니다.'})

# 즐겨찾기 목록 API
@app.route('/favorites/<int:user_id>', methods=['GET'])
def favorites(user_id):
    message = get_favorites_message(user_id)
    return jsonify({'favorites': message})

# 챗봇 API 엔드포인트
@app.route('/chatbot', methods=['POST'])
def chatbot_route():
    data = request.get_json()
    response, status_code = handle_chatbot_request(data)
    return jsonify(response), status_code

if __name__ == '__main__':
    app.run(debug=True)
