import os
from dotenv import load_dotenv
import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl

load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

llm_groq = ChatGroq(groq_api_key=groq_api_key, model_name='gemma-7b-it')

@cl.on_chat_start
async def on_chat_start():
    
    files = None # Initialize variable to store uploaded files

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload your pdf files to begin!",
            accept=["application/pdf"],
            max_size_mb=100,
            max_files=10,
            timeout=180, 
        ).send()

    # Process each uploaded file
    texts = []
    metadatas = []
    for file in files:
        print(file)

        pdf = PyPDF2.PdfReader(file.path)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        file_texts = text_splitter.split_text(pdf_text)
        texts.extend(file_texts)

        # Create metadata for each chunk
        file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
        metadatas.extend(file_metadatas)
        
    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )
    
    # Initialize message history for conversation
    message_history = ChatMessageHistory()
    
    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Display an image with the number of files
    elements = [
        cl.Image(name="image", display="inline", path="pdf bot.png")
    ]

    # Inform user that the processing has ended
    processed_files = ', '.join([file.name for file in files])
    msg = cl.Message(content=f"Processing {len(files)} files done. Files processed: {processed_files}. You can now ask questions!", elements=elements)
    await msg.send()

    # # Inform user that the processing has ended
    # msg = cl.Message(content=f"Processing {len(files)} files done. You can now ask questions!", elements=elements)
    # await msg.send()

    # Store the chain in user session
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
        
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain") 
    # Callbacks happen asynchronously/parallel 
    cb = cl.AsyncLangchainCallbackHandler()
    
    # Call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"] 

    text_elements = [] # Initialize list to store text elements
    
    # # Process source documents if available
    # if source_documents:
    #     for source_idx, source_doc in enumerate(source_documents):
    #         source_name = f"source_{source_idx}"
    #         # Create the text element referenced in the message
    #         text_elements.append(
    #             cl.Text(content=source_doc.page_content, name=source_name)
    #         )
    
    # Return results without source references
    await cl.Message(content=answer, elements=text_elements).send()
