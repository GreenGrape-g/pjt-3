articles = [
    {'id': '1', 'title': '가을에 읽기 좋은 책', 'content': '가을에 어울리는 소설 추천 리스트입니다.'},
    {'id': '2', 'title': '문학 속의 사랑', 'content': '사랑을 주제로 한 고전 문학 소개'}
]

def get_all_articles():
    return articles

def get_article_by_id(article_id):
    return next((article for article in articles if article['id'] == article_id), None)
