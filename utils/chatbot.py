# utils/chatbot.py

from utils.graph import websearch_rag
from utils.optimization import Optimization  # 'optimization.py'의 위치에 따라 경로 수정 필요
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def handle_chatbot_request(data):
    """
    챗봇 요청을 처리하는 함수.

    Parameters:
        data (dict): 클라이언트로부터 받은 JSON 데이터.

    Returns:
        tuple: JSON 응답 데이터와 상태 코드.
    """
    if not data or 'message' not in data:
        return {'error': '메시지를 입력해주세요.'}, 400

    question = data['message']

    # 책 추천 요청인지 확인
    if "추천" in question or "추천해" in question:
        # 항상 1권만 추천하도록 설정
        tone = "친근함"
        style = "전문적"
        num_books = 1  # 항상 1권 추천
        additional_instructions = "추천 이유를 전문적이고 깊이 있게 작성하세요."

        optimizer = Optimization(tone, style, num_books, additional_instructions)
        generated_response = question  # 사용자 입력을 기반으로 응답 생성
        optimized_response = optimizer.optimize_response(generated_response)
        return {'llm': optimized_response}, 200
    else:
        # 일반적인 질문에 대한 응답 처리
        ans = websearch_rag(question)
        return {'llm': ans}, 200
