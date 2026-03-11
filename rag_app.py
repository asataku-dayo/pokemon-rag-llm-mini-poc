# rag_app.py
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader as DocxLoader
from langchain.document_loaders import UnstructuredExcelLoader as ExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os

# --- 設定 ---
DOCS_DIR = "docs"          # 文書フォルダ
DB_DIR = "db"              # ベクトルDB保存先
EMBED_MODEL = "intfloat/multilingual-e5-small"
LLM_MODEL = "qwen2:7b"
CHUNK_SIZE = 600           # 文書分割サイズ
CHUNK_OVERLAP = 50         # 文書分割重複

# --- 1. ドキュメント読み込み ---
# txt
loader_txt = DirectoryLoader(DOCS_DIR, glob="**/*.txt")
docs_txt = loader_txt.load()

# Word
loader_word = DirectoryLoader(DOCS_DIR, glob="**/*.docx", loader_cls=DocxLoader)
docs_word = loader_word.load()

# Excel
loader_excel = DirectoryLoader(DOCS_DIR, glob="**/*.xlsx", loader_cls=ExcelLoader)
docs_excel = loader_excel.load()

# すべての文書をまとめる
all_docs = docs_txt + docs_word + docs_excel
print(f"読み込んだ文書数: {len(all_docs)}")

# --- 2. 文書分割 ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
texts = splitter.split_documents(all_docs)
print(f"分割後の文書数: {len(texts)}")

# --- 3. ベクトルDB化 ---
embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
db = Chroma.from_documents(
    texts,
    embeddings,
    persist_directory=DB_DIR
)
db.persist()
print("ベクトルDB作成完了")

# --- 4. RAGのRetrieverとLLM設定 ---
llm = Ollama(model=LLM_MODEL)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)

# --- 5. ユーザー質問に回答 ---
def answer_question(query: str):
    return qa.run(query)

# --- 6. テスト ---
if __name__ == "__main__":
    pass
    # while True:
    #     q = input("質問を入力: ")
    #     if q.lower() in ["exit", "quit"]:
    #         break
        # a = answer_question(q)
        # print("回答:", a)