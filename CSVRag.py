
import streamlit as st
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers import JsonOutputKeyToolsParser
import ollama
import logging
from typing import List, Tuple, Dict, Any, Optional

# df = pd.read_csv('./data/mortgage-300h-360m-5r.csv')
# print(df.shape)
# print(df.columns.tolist())

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=True)
def extract_model_names(
    _models_info: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        models_info (Dict[str, List[Dict[str, Any]]]): Dictionary containing information about available models.

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")
    model_names = tuple(model["model"] for model in _models_info["models"])
    logger.info(f"Extracted model names: {model_names}")
    return model_names

# Streamlit App
st.title("CSV File Explorer with LLM")

# Get list of avialable LLM models
models_info = ollama.list()
available_models = extract_model_names(models_info)

# Model selection
if available_models:
    selected_model = st.sidebar.selectbox(
        "Select local model", 
        available_models,
        key="model_select"
    )

# LLM from Ollama
# local_model = "mistral:latest"
llm = ChatOllama(model=selected_model, temperature=0)

# Sidebar for file uploads
st.sidebar.header("Upload CSV Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more CSV files", type="csv", accept_multiple_files=True
)

# Store the uploaded dataframes in a dictionary
dataframes = {}
if uploaded_files:
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            dataframes[file.name] = df
            st.sidebar.success(f"{file.name} uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading {file.name}: {e}")

# Display uploaded files
if dataframes:
    st.header("Uploaded Files")
    for name, df in dataframes.items():
        st.subheader(f"Preview of {name}")
        st.dataframe(df.head())

# Main section for questions
st.header("Ask Questions About Your CSV Files")
question = st.text_input("Enter your question below:")



if question and dataframes:
    # Combine all dataframes into one context
    context = "\n\n".join([f"### {name}\n{df.to_string(index=False)}" for name, df in dataframes.items()])
    prompt = f"You are a data analysis assistant. Based on the following CSV file data, answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    
    tool = PythonAstREPLTool(locals={"df": df})
    llm_with_tools = llm.bind_tools([tool], tool_choice=tool.name)
    tool.invoke("df['Period'].mean()")
    parser = JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True)

    system = f"""You have access to a pandas dataframe `df`. 
    Here is the output of `df.head().to_markdown()`:

    {df.head().to_markdown()}

    Given a user question, write the Python code to answer it. 
    Return ONLY the valid Python code and nothing else. 
    Don't assume you have access to any libraries other than built-in Python ones and pandas."""
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
    code_chain = prompt | llm_with_tools | parser

    # Function to query the LLM
    def query_llm(question):
        try:
            response = code_chain.invoke({"question": question})
            print(response)
            return response
        except Exception as e:
            return f"Error querying LLM: {e}"  

    with st.spinner("Processing your question..."):
        answer = query_llm(prompt)
        st.subheader("Answer")
        st.write(answer)
else:
    st.info("Upload CSV files and ask a question to get started!")


