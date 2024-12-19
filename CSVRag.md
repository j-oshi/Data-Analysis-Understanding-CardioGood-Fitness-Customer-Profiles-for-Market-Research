<h1>RAG System for CSV</h1>


```python
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

```

<h2>Import Data</h2>

The data was acqired from https://demo.oshinit.com/pages/mortgage-calculator.html. 300000 for amount, 360 months and Interest rate of 5%.


```python
file_path = './data/mortage-300h-360-5.csv'

data = pd.read_csv(file_path, sep=';', encoding='utf-8')

data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Period</th>
      <th>Monthly Payment</th>
      <th>Computed Interest Due</th>
      <th>Principal Due</th>
      <th>Principal Balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>300000.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1610.464869</td>
      <td>1250.000000</td>
      <td>360.464869</td>
      <td>299639.535131</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1610.464869</td>
      <td>1248.498063</td>
      <td>361.966806</td>
      <td>299277.568325</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1610.464869</td>
      <td>1246.989868</td>
      <td>363.475001</td>
      <td>298914.093324</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1610.464869</td>
      <td>1245.475389</td>
      <td>364.989480</td>
      <td>298549.103844</td>
    </tr>
  </tbody>
</table>
</div>



load and process data


```python
loader = CSVLoader(
    file_path=file_path,
    csv_args={
        "delimiter": ";",
    },
    encoding="windows-1252",
)

try:
    docs = loader.load()
except Exception as e:
    print("Error during data loading:", e)

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(docs)
print(chunks[0])

# for record in docs[:2]:
#     print(record)
# documents =  [Document(page_content=text) for text in docs]
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
# chunks = text_splitter.split_documents(documents)


```

    page_content='Period: 0
    Monthly Payment: 0
    Computed Interest Due: 0
    Principal Due: 0
    Principal Balance: 300000' metadata={'source': './data/mortage-300h-360-5.csv', 'row': 0}
    


    The Kernel crashed while executing code in the current cell or a previous cell. 
    

    Please review the code in the cell(s) to identify a possible cause of the failure. 
    

    Click <a href='https://aka.ms/vscodeJupyterKernelCrash'>here</a> for more info. 
    

    View Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details.



```python
# Create vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="mistral:latest"),
    collection_name="local-rag"
)
```


```python
# LLM from Ollama
local_model = "llama3.1:latest"
llm = ChatOllama(model=local_model)
```


```python
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five question-answering variations of the given user question to retrieve relevant documents from a vector database. By framing the query as potential answers to a question, your goal is to identify documents that directly address the user's information need. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)
```


```python
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
```


```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```


```python
chain.invoke("What is the total monthly payment?")
```
