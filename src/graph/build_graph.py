from langgraph.graph import StateGraph, START, END

from src.graph.types import State, InputState, OutputState
from src.graph.nodes import (
    chat_router,
    route_decision,
    vector_search,
    rerank_vector,
    web_search,
    rerank_web,
    answer,
    has_docs,
)

def build_graph():
    graph = StateGraph(State, input_schema=InputState, output_schema=OutputState)

    graph.add_node("chat_router", chat_router)
    graph.add_node("vector_search", vector_search)
    graph.add_node("rerank_vector", rerank_vector)
    graph.add_node("web_search", web_search)
    graph.add_node("rerank_web", rerank_web)
    graph.add_node("answer", answer)

    graph.add_edge(START, "chat_router")
    graph.add_conditional_edges(
        "chat_router",
        route_decision,
        {"chat": "answer", "retrieval": "vector_search"},
    )

    graph.add_edge("vector_search", "rerank_vector")
    graph.add_conditional_edges(
        "rerank_vector",
        has_docs,
        {True: "answer", False: "web_search"},
    )

    graph.add_edge("web_search", "rerank_web")
    graph.add_edge("rerank_web", "answer")
    graph.add_edge("answer", END)

    return graph.compile()
