from flask import Flask, send_from_directory, jsonify, request
from dotenv import load_dotenv
from utils.graph import graph_main  # graph.py의 graph_main 임포트

# 환경 변수 로드
load_dotenv()

# Flask 애플리케이션 생성
app = Flask(__name__, static_folder='.')

# 홈 페이지 제공
@app.route('/')
def serve_home():
    """
    메인 페이지를 제공하는 라우트입니다.
    """
    return send_from_directory('.', 'index.html')

# 챗봇 라우트 정의
@app.route('/chatbot', methods=['POST'])
def chatbot_route():
    """
    챗봇 요청을 처리하는 라우트입니다.
    클라이언트로부터 메시지와 히스토리를 받아 그래프를 실행하고 응답을 반환합니다.
    """
    data = request.get_json()
    question = data.get('message')
    history = data.get('history')

    if not question:
        return jsonify({'error': '메시지를 입력해주세요.'}), 400

    # 초기 상태 설정
    if history:
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

if __name__ == '__main__':
    # 애플리케이션 실행
    app.run(debug=True)
