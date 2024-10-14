books = [
    {
        "id": "1",
        "title": "파친코",
        "author": "이민진",
        "genre": "소설",
        "publisher": "문학동네",
        "summary": "한 한국 이민자 가족의 대서사시를 그린 소설.",
        "video": "https://example.com/pachinko-summary-video",
        "purchase_link": "https://bookstore.com/pachinko",
        "reviews_summary": "감동적이고 깊이 있는 서사를 극찬했습니다.",
        "cover_image": "https://example.com/pachinko-cover.jpg"
    },
    {
        "id": "2",
        "title": "데미안",
        "author": "헤르만 헤세",
        "genre": "철학 소설",
        "publisher": "민음사",
        "summary": "자아의 정체성을 찾아가는 과정을 담은 소설.",
        "video": "https://example.com/demian-summary-video",
        "purchase_link": "https://bookstore.com/demian",
        "reviews_summary": "자아 탐구의 여정을 탁월하게 그린 작품으로 평가합니다.",
        "cover_image": "https://example.com/demian-cover.jpg"
    }
]

def get_book_by_title(title):
    """책 제목으로 책 정보 검색"""
    return next((book for book in books if book["title"] == title), None)

def get_all_books():
    """책 목록 반환"""
    return [{"id": book["id"], "title": book["title"], "cover_image": book["cover_image"], "genre": book["genre"]} for book in books]
