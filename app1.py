import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_core.messages import HumanMessage
from PyPDF2 import PdfReader
from tempfile import NamedTemporaryFile

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


from langchain_core.runnables.history import RunnableWithMessageHistory


import dotenv
dotenv.load_dotenv()


llm=ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'),model="gpt-3.5-turbo")
embeddings=OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
store={}
def get_and_convert_pdf(input_pdf):
    bytes_data = input_pdf.read()
    with NamedTemporaryFile(delete=False) as tmp: 
        tmp.write(bytes_data)                      
        data = PyPDFLoader(tmp.name).load_and_split()
    os.remove(tmp.name)
    return data 

    loader = PyPDFLoader(input_pdf)
    pages = loader.load_and_split()


    return pages

def get_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split= text_splitter.split_documents(docs)
    return split
    #for page in pages:
    #    text+=page.extract_text(page)
    #return text

def create_Vectorstore(chunks):
    
    vectorstore=FAISS.from_documents(chunks,embeddings)
    vectorstore.save_local('faiss_index')


def get_rag_chain(vectorstore):

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    retriever=vectorstore.as_retriever()
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt="""
    you are the assistant for question and answering tasks. use the following given retrieved context to answer the question.if you dont know the 
    answer just say that you don't know. give answer simply in three lines.
    {context}
    """
    qa_prompt=ChatPromptTemplate.from_messages(
        [
            ('system',qa_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human',"{input}"),
        ]
    )
    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)

    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

    return rag_chain

chat_history=[]

def get_response(user_input,rag_chain):
    ai_msg=rag_chain.invoke({'input':user_input,'chat_history':chat_history})
    chat_history.extend([HumanMessage(content=user_input), ai_msg["answer"]])
    print(chat_history)
    return ai_msg['answer']

def preprocess ():
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text=[]
                for pdfs in pdf_docs:
                    raw_text.append(get_and_convert_pdf(pdfs))
                x=[]
                for i in raw_text:
                    x=x+i
                text_chunks = get_chunks(x)
                create_Vectorstore(text_chunks)
                st.success("Done")

database = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
rag_chain=get_rag_chain(database)


msgs = StreamlitChatMessageHistory(key="special_app_key")

def main():
    st.set_page_config("Chat with multiple PDFS")
    st.header("Chat with PDF using OpenAI")
    preprocess()
    session_state = st.session_state
    if 'chat_history' not in session_state:
        session_state.chat_history = []
    user_question = st.text_input("Ask question related to the given context")
    if user_question:
        submit=st.button('submit')
        if submit:
            with st.spinner("processing..."):
                answer=get_response(user_question,rag_chain)
                session_state.chat_history.extend([HumanMessage(content=user_question), answer])
            print(len(chat_history))
            for i in chat_history:
                print(i)
            st.write(answer)
            st.success("done")
            

if __name__=='__main__':
    main()