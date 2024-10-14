from flask import Flask, jsonify, request
from flask_cors import CORS
from utils.llm import query_llm
from utils.books import get_all_books, get_book_by_title
from utils.magazine import get_all_articles, get_article_by_id
from utils.favorites import get_favorites_message, get_new_favorite_message

app = Flask(__name__)
CORS(app)

# LLM 라우트
@app.route('/', methods=['POST'])
def index():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': '메시지를 입력해주세요.'}), 400
    ans = query_llm(user_input)
    return jsonify({'llm': ans})

# 테스트 라우트
@app.route('/hi')
def hi():
    return jsonify({'message': 'hi'})

# 책 목록 API
@app.route('/api/books', methods=['GET'])
def books():
    book_list = get_all_books()
    return jsonify(book_list)

# 책 제목으로 책 상세 정보 검색 API
@app.route('/api/book', methods=['GET'])
def book_by_title():
    title = request.args.get('title')
    if not title:
        return jsonify({'error': '책 제목을 입력해주세요.'}), 400
    book = get_book_by_title(title)
    if not book:
        return jsonify({'error': '해당 제목의 책을 찾을 수 없습니다.'}), 404
    return jsonify(book)

if __name__ == '__main__':
    app.run(debug=True)

# 잡지 콘텐츠 목록 API
@app.route('/api/magazine', methods=['GET'])
def magazine():
    articles = get_all_articles()
    return jsonify(articles)

# 잡지 상세 콘텐츠 API
@app.route('/api/magazine/<article_id>', methods=['GET'])
def magazine_article(article_id):
    article = get_article_by_id(article_id)
    if not article:
        return jsonify({'error': '해당 기사를 찾을 수 없습니다.'}), 404
    return jsonify(article)

# 책 신청 게시판 (화면만 구현)
@app.route('/api/requests', methods=['GET'])
def book_request():
    return jsonify({'message': '여기는 책 신청 게시판 목록 화면입니다.'})

@app.route('/api/requests/new', methods=['GET'])
def new_book_request():
    return jsonify({'message': '여기는 새로운 책을 신청하는 화면입니다.'})

# 즐겨찾기 목록 화면 (화면만 구현)
@app.route('/api/favorites', methods=['GET'])
def favorites():
    message = get_favorites_message()
    return jsonify({'message': message})

# 새로운 즐겨찾기 추가 화면 (화면만 구현)
@app.route('/api/favorites/new', methods=['GET'])
def new_favorite():
    message = get_new_favorite_message()
    return jsonify({'message': message})

if __name__ == '__main__':
    app.run(debug=True)
