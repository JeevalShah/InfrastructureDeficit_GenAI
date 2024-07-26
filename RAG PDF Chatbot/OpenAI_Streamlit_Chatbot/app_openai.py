import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlfiles import css, user_template, bot_template




def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks



def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(
        texts=text_chunks, 
        embedding=embeddings)
    return vectorstore



def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    #st.write(response)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content),unsafe_allow_html=True)



def main():
    # load env
    load_dotenv()

    st.set_page_config(page_title='PDF Chatbot', page_icon='🤖')
    st.write(css,unsafe_allow_html=True)
    
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    
    st.header('Chat with your PDF files!! 📄')
    user_question = st.text_input('What do you want to know from your document?')
    if user_question:
        handle_userinput(user_question)
    #st.write(user_template.replace('{{MSG}}', 'Hello Robot'),unsafe_allow_html=True)
    #st.write(bot_template.replace('{{MSG}}', 'Hello Human'),unsafe_allow_html=True)


    with st.sidebar:
        st.subheader('Your files')
        pdf_docs = st.file_uploader(
            'Upload your PDF files here', accept_multiple_files=True)
        if st.button('Load'):
            with st.spinner('Loading'):

                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                st.write(raw_text)

                #get chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)

                #create vector store
                vectorstore = get_vectorstore(text_chunks)

                #coversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
            

if __name__ == '__main__':
    main()