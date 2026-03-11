import streamlit as st
from rag_app import answer_question  # rag_app.pyの関数を利用

st.title("ローカルRAGアプリ")

query = st.text_input("質問を入力してください:")

if query:
    answer = answer_question(query)
    st.write("### 回答")
    st.write(answer)