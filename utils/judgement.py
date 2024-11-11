from typing import Dict
import re

def is_about_books(response: str) -> bool:
    """챗봇의 응답이 책과 관련된지 여부를 판단합니다."""
    book_keywords = [
        '책', '소설', '문학', '작가', '읽다', '독서',
        '출판사', '장르', '챕터', '이야기', '도서관', '베스트셀러',
        '추천', '저자', '서적', '도서'
    ]
    return any(keyword in response for keyword in book_keywords)

def is_about_author(response: str) -> bool:
    """챗봇의 응답이 작가와 관련된지 여부를 판단합니다."""
    # 간단한 패턴 매칭을 통해 작가 이름 감지 (실제 구현 시 더 정교한 방법 필요)
    # 예: '작가: 김영하' 또는 '저자 김영하'
    author_patterns = [
        r'작가[:：]\s*[\w가-힣]+',
        r'저자[:：]\s*[\w가-힣]+'
    ]
    for pattern in author_patterns:
        if re.search(pattern, response):
            return True
    return False

def is_about_negative(response: str) -> bool:
    """챗봇의 응답에 부정적인 단어가 포함되어 있는지 여부를 판단합니다."""
    negative_keywords = [
        '불가능', '받고', '이 중에서', '?', '등'
    ]
    return any(keyword in response for keyword in negative_keywords)

def decide_next_node(state: Dict) -> str:
    """
    다음 노드를 결정하는 함수.
    
    Args:
        state (Dict): 현재 상태.
    
    Returns:
        str: 다음 노드 이름.
    """
    # 부정적인 응답을 먼저 확인
    if state.get("is_negative", False):
        return "end"
    # 작가 관련 질문을 먼저 확인
    elif state.get("is_author_question", False):
        return "web_search_node_author"
    # 그다음 책 관련 질문을 확인
    elif state.get("is_book_question", False):
        return "web_search_node"
    else:
        return "end"
