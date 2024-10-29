# utils/judgement.py
from typing import Dict

def is_about_books(response: str) -> bool:
    """챗봇의 응답이 책과 관련된지 여부를 판단합니다."""
    book_keywords = [
        '책', '소설', '문학', '작가', '읽다', '독서',
        '출판사', '장르', '챕터', '이야기', '도서관', '베스트셀러',
        '추천', '저자', '서적', '도서'
    ]
    return any(keyword in response for keyword in book_keywords)

def decide_next_node(state: Dict) -> str:
    """
    다음 노드를 결정하는 함수.
    
    Args:
        state (Dict): 현재 상태.
    
    Returns:
        str: 다음 노드 이름.
    """
    if state.get("is_book_question", False):
        return "transform_query"
    else:
        return "end"
