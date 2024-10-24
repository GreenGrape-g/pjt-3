# app.py

from flask import Flask, jsonify, request
from flask_cors import CORS
from utils.chatbot import handle_chatbot_request

app = Flask(__name__)
CORS(app)

# 챗봇 API 엔드포인트
@app.route('/chatbot', methods=['POST'])
def chatbot_route():
    data = request.get_json()
    conversation_history = data.get('conversation_history', [])
    response, status_code = handle_chatbot_request(data, conversation_history)
    return jsonify(response), status_code

if __name__ == '__main__':
    app.run(debug=True)
