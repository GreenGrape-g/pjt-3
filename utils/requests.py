book_requests = []

def get_requests():
    return book_requests

def add_request(data):
    new_request = {
        'id': str(len(book_requests) + 1),
        'title': data['title'],
        'author': data['author']
    }
    book_requests.append(new_request)
    return new_request