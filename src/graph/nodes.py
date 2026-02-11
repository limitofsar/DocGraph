from typing import List
import pandas as pd

from src.graph.types import State, Docs
from src.llm.llm_tavily import llm, tavily
from src.vector_db.vector_store import VECTORSTORE
from src.classifier.catboost_clf import clf

## helpers
def build_clf_text(state_question: str, *parts) -> str:
    '''Собирает безопасный clf_text для классификатора, пропуски игнорируются.'''
    safe_parts = [str(p) for p in parts if p and str(p).strip()]
    return f"{state_question}\n\n" + "\n\n".join(safe_parts)

## routers
def chat_router(state: State) -> State:
    prompt = f'''
    Ты определяешь, нужен ли поиск.

    Ответь:
    - chat — если это общение, рассуждение, объяснение
    - retrieval — если нужны факты, рекомендации, актуальная инфа

    Вопрос:
    {state['question']}

    Ответь ОДНИМ словом
    '''
    state['route'] = llm.invoke(prompt).content.strip()
    return state

def route_decision(state: State) -> str:
    return state["route"]

def has_docs(state: State) -> bool:
    return len(state["docs"]) > 0

## vector db
def vector_search(state: State) -> State:
    results = VECTORSTORE.similarity_search(state['question'], k=5)

    docs: List[Docs] = []

    for r in results:
        # raw_text (Для LLM)
        raw_parts = [r.page_content]

        if r.metadata.get('summarized'):
            raw_parts.append("Краткое описание:\n" + r.metadata['summarized'])

        if r.metadata.get('addres'):
            raw_parts.append(f"Адрес: {r.metadata['address']}")

        raw_text = '\n\n'.join(raw_parts)

        # сlf_text (для классификатора)
        clf_text = build_clf_text(state['question'], r.page_content, r.metadata.get('summarized'))

        docs.append({
            'raw_text': raw_text,
            'clf_text': clf_text,
            'source': 'vector_db',
            'url': None,
            'score': None,
        })

    state['docs'] = docs
    return state

def rerank_vector(state: State, threshold=0.545) -> State:
    passed: List[Docs] = []

    for d in state['docs']:
        score = clf.predict_proba(pd.Series([d['clf_text']]))[:, 1][0]

        if score > threshold:
            d['score'] = score
            passed.append(d)

    passed.sort(key=lambda x: x['score'], reverse=True)
    state['docs'] = passed
    state['source'] = 'vector_db' if passed else 'none'
    return state

## web
def web_search(state: State) -> State:
    results = tavily.invoke(state['question'], max_results=5)
    print(results)

    docs: List[Docs] = []
    for r in results['results']:
        parts = []

        if r.get('title'):
            parts.append(r['title'])

        if r.get('content'):
            parts.append(r['content'])

        raw_text = '\n\n'.join(parts)

        clf_text = build_clf_text(state['question'], r.get('title'), r.get('content'))

        docs.append({
            'raw_text': raw_text,
            'clf_text': clf_text,
            'source': 'web',
            'url': r.get('url'),
            'score': None
        })
    state['docs'] = docs
    return state

def rerank_web(state: State, threshold=0.176) -> State:
    passed: List[Docs] = []

    for d in state['docs']:
        score = clf.predict_proba(pd.Series([d['clf_text']]))[:, 1][0]

        if score > threshold:
            d['score'] = score
            passed.append(d)

    passed.sort(key=lambda x: x['score'], reverse=True)

    state['docs'] = passed
    state['source'] = 'web' if passed else 'none'
    return state

## answer
def answer(state: State) -> State:
    """
    Универсальный ответ:
    - Один вызов LLM для всех документов.
    - Формирует структурированный список заведений с особенностями и ссылками.
    - Если документов нет, просто общение в чате.
    """
    if state.get("docs"):
        docs = state.get("docs")

        # Формируем контекст для LLM
        context_lines = []
        for d in docs:
            parts = []

            # Название заведения (можно взять первую строку)
            name_line = d["raw_text"].splitlines()[0].strip()
            if name_line:
                parts.append(f"Название: {name_line}")

            # Особенности / краткое описание
            if "Краткое описание" in d["raw_text"]:
                # вырезаем текст после "Краткое описание:"
                desc = d["raw_text"].split("Краткое описание:")[-1].strip()
                parts.append(f"Особенности: {desc}")

            # URL, если есть
            if d.get("url"):
                parts.append(f"Ссылка: {d['url']}")

            context_lines.append("\n".join(parts))

        context = "\n\n".join(context_lines)

        prompt = f"""
Ты помощник по информации и рекомендациям.
Твоя задача — составить **один связный ответ** на вопрос, используя найденные документы.

Инструкции:
- Начни с фразы: "Я нашел следующие варианты для вас:"
- Для каждого документа создай отдельный пункт с нумерацией: 1., 2., 3. и так далее.
- Для каждого пункта:
  - Название выделяй жирным.
  _ Адрес если есть 
  - Если есть ссылка, всегда её указывай после названия, в формате: [Ссылка](URL)
  - Если есть особенности, кратко расскажи их.
- Пустые поля пропускай.
- Пиши естественно, как рассказ пользователю, а не формальный список.
- НЕ ПРИДУМЫВАЙ ИНФОРМАЦИЮ, ИСПОЛЬЗУЙ ТО, ЧТО ЕСТЬ В ДОКУМЕНТАХ.

Контекст документов:
{context}

Вопрос:
{state['question']}
"""

    else:
        # Просто чат без документов
        prompt = f"""
Ты помощник по информации и рекомендациям.
Ты умеешь общаться и давать полезные советы.

Вопрос:
{state['question']}
"""

    state["answer"] = llm.invoke(prompt).content.strip()
    return state



