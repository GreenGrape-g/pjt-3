from flask import Flask, send_from_directory, jsonify, request
from dotenv import load_dotenv

# 'handle_chatbot_request' 대신 'graph_main'을 임포트합니다.
from utils.graph import graph_main  # graph.py의 graph_main 임포트

# 환경 변수 로드
load_dotenv()

app = Flask(__name__, static_folder='.')

# 홈 페이지 제공
@app.route('/')
def serve_home():
    return send_from_directory('.', 'index.html')

# 다른 정적 페이지 제공
@app.route('/books')
def serve_book():
    return send_from_directory('.', 'book.html')

@app.route('/bookshelf')
def serve_bookshelf():
    return send_from_directory('.', 'bookshelf.html')

@app.route('/magazine')
def serve_magazine():
    return send_from_directory('.', 'magazine.html')

@app.route('/likes')
def serve_likes():
    return send_from_directory('.', 'likes.html')

# 챗봇 라우트 정의
@app.route('/chatbot', methods=['POST'])
def chatbot_route():
    data = request.get_json()
    question = data.get('message')
    history = data.get('history')
    if not question:
        return jsonify({'error': '메시지를 입력해주세요.'}), 400

    if history:
    # 초기 상태 설정
        state = {
            "messages": [
                *history,
                {"role": "user", "content": question}
            ]
        }
    else:
        state = {
            "messages": [
                {"role": "user", "content": question}
            ]
        }
    
    
    

    # 그래프 실행
    result = graph_main(state)

    # 최종 응답 가져오기
    final_response = result.get('generation', result.get('response', '죄송하지만, 답변을 생성할 수 없습니다.'))

    return jsonify({'llm': final_response}), 200

# 정적 파일 제공 (CSS, JS, 이미지 등)
@app.route('/static/<path:path>')
def serve_static_files(path):
    return send_from_directory('static', path)

# 404 에러 핸들러
@app.errorhandler(404)
def page_not_found(e):
    return "페이지를 찾을 수 없습니다.", 404

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

