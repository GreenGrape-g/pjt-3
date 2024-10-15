from flask import Flask, jsonify, request
from flask_cors import CORS
from utils.books import get_all_books, get_book_by_title
from utils.graph import websearch_rag
from utils.magazine import get_all_articles, get_article_by_id
from utils.favorites import get_favorites_message, get_new_favorite_message


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
def book_by_title():
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

@app.route('/requests/new', methods=['GET'])
def new_book_request():
    return jsonify({'message': '여기는 새로운 책을 신청하는 화면입니다.'})

# 즐겨찾기 목록 화면 (화면만 구현)
@app.route('/favorites', methods=['GET'])
def favorites():
    message = get_favorites_message()
    return jsonify({'message': message})

# 새로운 즐겨찾기 추가 화면 (화면만 구현)
@app.route('/favorites/new', methods=['GET'])
def new_favorite():
    message = get_new_favorite_message()
    return jsonify({'message': message})

# 챗봇 페이지
@app.route('/chatbot', methods=['POST'])
def ask():
    question = request.json['message']
    # utils/graph.py 에서 최종 질문 검색하는 함수
    ans = websearch_rag(question)
    return {'llm': ans}

if __name__ == '__main__':
    app.run(debug=True)