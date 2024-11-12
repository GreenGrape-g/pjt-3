# utils/chatbot.py

from .graph import compiled_graph

def handle_chatbot_request(data, conversation_history=None):
    if not data or 'message' not in data:
        return {'error': '메시지를 입력해주세요.'}, 400

    message = data['message']
    checkpoint_id = data.get('checkpoint_id')

    if conversation_history is None:
        conversation_history = []

    conversation_history.append({"role": "user", "content": message})

    state = {
        "messages": conversation_history,
        "ask_human": False,
        "question": None,
        "generation": None,
        "documents": [],
        "checkpoint_id": checkpoint_id
    }

    try:
        result = compiled_graph.invoke(state)
        llm_response = result.get('generation', "죄송하지만, 요청을 처리할 수 없습니다.")

        if llm_response:
            conversation_history.append({"role": "assistant", "content": llm_response})

        response = {
            'llm': llm_response,
            'conversation_history': conversation_history
        }

        # 체크포인트 ID가 반환되었는지 확인
        if '__checkpoint_id__' in result:
            response['checkpoint_id'] = result['__checkpoint_id__']
        return response, 200
    except Exception as e:
        return {'error': str(e)}, 500
