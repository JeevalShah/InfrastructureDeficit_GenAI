import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
import os

# Initialize the language model
llm = ChatGroq(model_name="llama3-70b-8192", api_key="GROQ_API_KEY")

# Available datasets
datasets = {
    "AIDI": "Africa Infrastructure Development Index AIDI.csv",
    "WSSC": "Water and Sanitation Service (WSS) Composite Index (2).csv",
    "ELECTRICITY": "Electricity composite Index.csv",
    "ICT": "ICT composite Index.csv",
    "TRANSPORT": "Transport Composite Index.csv",
    "FUSED": "combined_dfCountry-aidi_mean-eci_mean-wssci_mean%.csv"
}

# Initialize session state for question history
if 'question_history' not in st.session_state:
    st.session_state.question_history = []

# Function to load image as base64
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Streamlit interface
st.title("T5 Data Analytics Chatbot")

# Sidebar for dataset selection
st.sidebar.header("Dataset Selection")
selected_dataset_name = st.sidebar.selectbox("Select a dataset", list(datasets.keys()))

# Load selected dataset
selected_dataset_path = datasets[selected_dataset_name]
selected_df = pd.read_csv(selected_dataset_path)
smart_selected_df = SmartDataframe(selected_df, config={"llm": llm})

# Display dataset
st.subheader(f"{selected_dataset_name} DATA")
st.dataframe(selected_df.head(20))

# Input for user question
user_question = st.text_input("Ask a question about the data.")

# Display the response to the user's question
if st.button("Get Answer"):
    answer = smart_selected_df.chat(user_question)
    st.write(f"{selected_dataset_name} Answer:", answer)
    st.session_state.question_history.append(user_question)

    # Display any generated images
    image_path = r"exports/charts/temp_chart.png"
    if os.path.exists(image_path):
        image_base64 = get_image_base64(image_path)
        st.markdown(f'<img src="data:image/png;base64,{image_base64}" alt="Sample Chart" style="width:100%">', unsafe_allow_html=True)
        os.remove(image_path)

# Display the history of questions
st.subheader("Question History")
for i, question in enumerate(st.session_state.question_history, 1):
    st.write(f"{i}. {question}")
