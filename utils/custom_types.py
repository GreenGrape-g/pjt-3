# utils/types.py

from typing import List, Dict
from typing_extensions import TypedDict

class State(TypedDict):
    # 사용자 대화 내역
    messages: List[Dict[str, str]]
    ask_human: bool
    about_books: bool
    book_info: str
    other_info: str
    documents: List[str]

