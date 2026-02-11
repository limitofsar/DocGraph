from pathlib import Path
from langchain_community.vectorstores import  FAISS
from src.embeddings.hf_embeddings import embeddings

# Папка, где лежит текущий скрипт
current_file = Path(__file__).resolve()
current_dir = current_file.parent

FAISS_DIR = current_dir.parent / 'vector_db' / 'faiss_index'

def load_vectorstore():
    '''Загружаем локальную FAISS базу'''
    if not FAISS_DIR.exists() or not any(FAISS_DIR.iterdir()):
        raise FileNotFoundError(f"FAISS база не найдена в {FAISS_DIR}")

    vectorstore = FAISS.load_local(
        str(FAISS_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    #print(f"FAISS база загружена из {FAISS_DIR}")
    return vectorstore

VECTORSTORE = load_vectorstore()
