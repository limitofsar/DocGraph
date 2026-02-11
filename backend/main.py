from fastapi import FastAPI, Depends
from backend.schemas import QueryRequest, QueryResponse
from backend.deps import get_graph

app = FastAPI(title='Agent API')

@app.post('/query', response_model=QueryResponse)
def query_agent(req: QueryRequest, graph=Depends(get_graph)):
    result = graph.invoke({'question': req.question})
    return {'answer': result['answer']}

