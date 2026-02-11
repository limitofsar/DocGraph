import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="Agent Chat")

# Стилизация
st.markdown(
    """
    <style>
    /* Светлый фон всей страницы */
    body {
        background-color: #f7f7f7;
        margin: 0;
    }

    /* Контейнер чата */
    .chat-container {
        height: 75vh;          /* фиксированная высота чата */
        overflow-y: auto;
        padding: 10px;
        display: flex;
        flex-direction: column;
    }

    /* Сообщения */
    .message {
        padding: 10px;
        border-radius: 12px;
        margin-bottom: 8px;
        max-width: 75%;
        word-wrap: break-word;
        font-family: sans-serif;
        font-size: 15px;
    }
    .user {
        background-color: #00ccff;  /* твое сообщение */
        align-self: flex-end;
    }
    .agent {
        background-color: #ff9666;  /* ответ агента */
        align-self: flex-start;
    }

    /* Поле ввода всегда внизу */
    .input-container {
        position: fixed;
        bottom: 0;
        width: 100%;
        padding: 10px 20px;
        background-color: #f7f7f7;
        border-top: 1px solid #ddd;
    }

    /* Убираем стандартный outline при фокусе */
    textarea:focus, input:focus {
        outline: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)

#  История сообщений
if "messages" not in st.session_state:
    st.session_state.messages = []

#  Контейнер чата
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        st.markdown(f'<div class="message {role}">{content}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

#  Функция отправки
def send_message():
    if user_input := st.session_state.input_text.strip():
        # добавляем сообщение пользователя
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.input_text = ""

        # отправляем на бэкенд
        try:
            response = requests.post(BACKEND_URL, json={"question": user_input})
            answer = response.json().get("answer", "")
        except Exception as e:
            answer = f"Ошибка: {e}"

        # добавляем ответ агента
        st.session_state.messages.append({"role": "agent", "content": answer})

#  Поле ввода внизу
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.text_input(
    "",
    key="input_text",
    on_change=send_message,
    placeholder="Напишите сообщение и нажмите Enter...",
)
st.markdown('</div>', unsafe_allow_html=True)
