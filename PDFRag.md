<h1>PDF RAG</h1>


```python
!pip list
```

    Package                                  Version
    ---------------------------------------- ------------------
    aiofiles                                 24.1.0
    aiohappyeyeballs                         2.4.4
    aiohttp                                  3.11.10
    aiosignal                                1.3.2
    altair                                   5.5.0
    annotated-types                          0.7.0
    anyio                                    4.6.2.post1
    argon2-cffi                              23.1.0
    argon2-cffi-bindings                     21.2.0
    arrow                                    1.3.0
    asgiref                                  3.8.1
    asttokens                                2.4.1
    async-lru                                2.0.4
    attrs                                    24.2.0
    babel                                    2.16.0
    backoff                                  2.2.1
    bcrypt                                   4.2.1
    beautifulsoup4                           4.12.3
    bleach                                   6.2.0
    blinker                                  1.9.0
    branca                                   0.8.0
    build                                    1.2.2.post1
    cachetools                               5.5.0
    certifi                                  2024.8.30
    cffi                                     1.17.1
    chardet                                  5.2.0
    charset-normalizer                       3.4.0
    chroma-hnswlib                           0.7.6
    chromadb                                 0.5.23
    click                                    8.1.7
    colorama                                 0.4.6
    coloredlogs                              15.0.1
    comm                                     0.2.2
    contourpy                                1.3.1
    cryptography                             44.0.0
    cycler                                   0.12.1
    dataclasses-json                         0.6.7
    debugpy                                  1.8.9
    decorator                                5.1.1
    defusedxml                               0.7.1
    Deprecated                               1.2.15
    durationpy                               0.9
    emoji                                    2.14.0
    eval_type_backport                       0.2.0
    executing                                2.1.0
    fastapi                                  0.115.6
    fastjsonschema                           2.20.0
    filelock                                 3.16.1
    filetype                                 1.2.0
    flatbuffers                              24.3.25
    folium                                   0.18.0
    fonttools                                4.55.0
    fqdn                                     1.5.1
    frozenlist                               1.5.0
    fsspec                                   2024.10.0
    geographiclib                            2.0
    geopandas                                1.0.1
    geopy                                    2.4.1
    gitdb                                    4.0.11
    GitPython                                3.1.43
    google-auth                              2.37.0
    googleapis-common-protos                 1.66.0
    grpcio                                   1.68.1
    h11                                      0.14.0
    httpcore                                 1.0.7
    httptools                                0.6.4
    httpx                                    0.27.2
    httpx-sse                                0.4.0
    huggingface-hub                          0.26.5
    humanfriendly                            10.0
    idna                                     3.10
    importlib_metadata                       8.5.0
    importlib_resources                      6.4.5
    ipykernel                                6.29.5
    ipython                                  8.29.0
    isoduration                              20.11.0
    jedi                                     0.19.2
    Jinja2                                   3.1.4
    joblib                                   1.4.2
    json5                                    0.9.28
    jsonpatch                                1.33
    jsonpath-python                          1.0.6
    jsonpointer                              3.0.0
    jsonschema                               4.23.0
    jsonschema-specifications                2024.10.1
    jupyter_client                           8.6.3
    jupyter_core                             5.7.2
    jupyter-events                           0.10.0
    jupyter-lsp                              2.2.5
    jupyter_server                           2.14.2
    jupyter_server_terminals                 0.5.3
    jupyterlab                               4.3.1
    jupyterlab_pygments                      0.3.0
    jupyterlab_server                        2.27.3
    kiwisolver                               1.4.7
    kubernetes                               31.0.0
    langchain                                0.3.12
    langchain-community                      0.3.12
    langchain-core                           0.3.25
    langchain-ollama                         0.2.1
    langchain-text-splitters                 0.3.3
    langdetect                               1.0.9
    langsmith                                0.2.3
    lxml                                     5.3.0
    markdown-it-py                           3.0.0
    MarkupSafe                               3.0.2
    marshmallow                              3.23.1
    matplotlib                               3.9.2
    matplotlib-inline                        0.1.7
    mdurl                                    0.1.2
    mistune                                  3.0.2
    mmh3                                     5.0.1
    monotonic                                1.6
    mpmath                                   1.3.0
    multidict                                6.1.0
    mypy-extensions                          1.0.0
    narwhals                                 1.14.2
    nbclient                                 0.10.0
    nbconvert                                7.16.4
    nbformat                                 5.10.4
    nest-asyncio                             1.6.0
    nltk                                     3.9.1
    notebook_shim                            0.2.4
    numpy                                    2.1.3
    oauthlib                                 3.2.2
    ollama                                   0.4.4
    onnx-weekly                              1.18.0.dev20241210
    onnxruntime                              1.20.1
    opencv-python                            4.10.0.84
    opentelemetry-api                        1.29.0
    opentelemetry-exporter-otlp-proto-common 1.29.0
    opentelemetry-exporter-otlp-proto-grpc   1.29.0
    opentelemetry-instrumentation            0.50b0
    opentelemetry-instrumentation-asgi       0.50b0
    opentelemetry-instrumentation-fastapi    0.50b0
    opentelemetry-proto                      1.29.0
    opentelemetry-sdk                        1.29.0
    opentelemetry-semantic-conventions       0.50b0
    opentelemetry-util-http                  0.50b0
    orjson                                   3.10.12
    overrides                                7.7.0
    packaging                                24.2
    pandas                                   2.2.3
    pandocfilters                            1.5.1
    parso                                    0.8.4
    pdf2image                                1.17.0
    pdfminer                                 20191125
    pdfminer.six                             20240706
    pillow                                   11.0.0
    pip                                      24.3.1
    platformdirs                             4.3.6
    posthog                                  3.7.4
    prometheus_client                        0.21.0
    prompt_toolkit                           3.0.48
    propcache                                0.2.1
    protobuf                                 5.28.3
    psutil                                   6.1.0
    pure_eval                                0.2.3
    pyarrow                                  18.0.0
    pyasn1                                   0.6.1
    pyasn1_modules                           0.4.1
    pycparser                                2.22
    pycryptodome                             3.21.0
    pydantic                                 2.9.2
    pydantic_core                            2.23.4
    pydantic-settings                        2.7.0
    pydeck                                   0.9.1
    Pygments                                 2.18.0
    PyMuPDF                                  1.25.1
    pymupdf4llm                              0.0.17
    pyogrio                                  0.10.0
    pyparsing                                3.2.0
    pypdf                                    5.1.0
    PyPika                                   0.48.9
    pypng                                    0.20220715.0
    pyproj                                   3.7.0
    pyproject_hooks                          1.2.0
    pyreadline3                              3.5.4
    python-dateutil                          2.9.0.post0
    python-dotenv                            1.0.1
    python-iso639                            2024.10.22
    python-json-logger                       2.0.7
    python-magic                             0.4.27
    pytz                                     2024.2
    pywin32                                  308
    pywinpty                                 2.0.14
    PyYAML                                   6.0.2
    pyzmq                                    26.2.0
    RapidFuzz                                3.10.1
    referencing                              0.35.1
    regex                                    2024.11.6
    requests                                 2.32.3
    requests-oauthlib                        2.0.0
    requests-toolbelt                        1.0.0
    rfc3339-validator                        0.1.4
    rfc3986-validator                        0.1.1
    rich                                     13.9.4
    rpds-py                                  0.21.0
    rsa                                      4.9
    scikit-learn                             1.5.2
    scipy                                    1.14.1
    seaborn                                  0.13.2
    Send2Trash                               1.8.3
    setuptools                               75.6.0
    shapely                                  2.0.6
    shellingham                              1.5.4
    six                                      1.16.0
    smmap                                    5.0.1
    sniffio                                  1.3.1
    soupsieve                                2.6
    SQLAlchemy                               2.0.36
    stack-data                               0.6.3
    starlette                                0.41.3
    streamlit                                1.40.1
    sympy                                    1.13.3
    tabulate                                 0.9.0
    tenacity                                 9.0.0
    terminado                                0.18.1
    threadpoolctl                            3.5.0
    tinycss2                                 1.4.0
    tokenizers                               0.20.3
    toml                                     0.10.2
    tornado                                  6.4.2
    tqdm                                     4.67.1
    traitlets                                5.14.3
    typer                                    0.15.1
    types-python-dateutil                    2.9.0.20241003
    typing_extensions                        4.12.2
    typing-inspect                           0.9.0
    tzdata                                   2024.2
    unstructured                             0.11.8
    unstructured-client                      0.28.1
    uri-template                             1.3.0
    urllib3                                  2.2.3
    uvicorn                                  0.34.0
    watchdog                                 6.0.0
    watchfiles                               1.0.3
    wcwidth                                  0.2.13
    webcolors                                24.11.1
    webencodings                             0.5.1
    websocket-client                         1.8.0
    websockets                               14.1
    wrapt                                    1.17.0
    XlsxWriter                               3.2.0
    xyzservices                              2024.9.0
    yarl                                     1.18.3
    zipp                                     3.21.0
    

Import Libraries


```python
import os, sys
import pymupdf
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from IPython.display import display as Markdown

# Set directory paths
parent_dir = os.path.abspath("..")
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# import chromadb
# chroma_client = chromadb.Client()
# embeddings = OllamaEmbeddings(model="llama3")
# embeddings.embed_query("What is the meaning of life?")
```

<h2>Load PDF</h2>


```python
# file_name = 'developer_job.pdf'
# folder_path = 'data'
# file_path = parent_dir + os.sep + folder_path + os.sep + file_name

# # PDF file uploads
# if file_path:
#   loader = UnstructuredPDFLoader(file_path) #Wheel issue with 3.13
#   docs = loader.load()
# else:
#   print("Upload a PDF file")

file_name = 'developer_job.pdf'
folder_path = 'data'
file_path = os.path.join(os.getcwd(), folder_path, file_name)

def load_pdf_with_pymupdf(file_path):
    """
    Load text from a PDF file using PyMuPDF (fitz).
    Returns the extracted text as a string.
    """
    try:
        doc = pymupdf.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None
    
# PDF file uploads
if os.path.exists(file_path):
    pdf_text = load_pdf_with_pymupdf(file_path)
    if pdf_text:
        print(f"PDF loaded successfully! Extracted text:\n{pdf_text[:500]}...")  
    else:
        print("Failed to extract text.")
else:
    print("Upload a valid PDF file.")

```

    PDF loaded successfully! Extracted text:
    Search...
     Search results
    ome
    >
    Information Technology
    >
    Developer
    >
    Fullstack Developer
    > Job details
    Apply now
    Register and upload your CV to apply with just one click
    View all jobs
    Senior Full Stack Developer
    Featured
    Senior Full Stack Developer
    Posted 3 days ago by Kensington Mortgage Company
    Work from home
    Salary
    negotiable
    London, South East
    England
    Permanent,
    full-time
    Register CV
    Sign in
    Saved jobs
    Senior Full Stack Developer in London - Reed.co.uk
    https://www.reed.co.uk/jobs/senior-full...
    

List of local LLMS for Ollama


```python
!ollama list
```

    NAME                   ID              SIZE      MODIFIED     
    llama3.1:latest        f66fc8dc39ea    4.7 GB    3 months ago    
    phi3:medium            cf611a26b048    7.9 GB    3 months ago    
    mistral-nemo:latest    994f3b8b7801    7.1 GB    4 months ago    
    mistral:latest         2ae6f6dd7a3d    4.1 GB    6 months ago    
    


```python
# Split and chunk 
documents = [Document(page_content=pdf_text, metadata={})]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
```


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
chain.invoke("What are the skills needed for the job?")
```




    'According to the job description, the required skills are:\n\n* Strong background in software development building cloud based applications\n* Experience developing c# models (yes or no question)\n* Strong C# / .NET Framework Programming skills\n* Strong knowledge of design patterns and experienced in designing software components\n* Strong experience in Microsoft .NET Parallel programming\n* Experience working with Azure batch is desirable\n* Demonstrable experience using python. Experience with any of the following is beneficial: numpy, pandas\n* Programming with C# / Microsoft Excel\n* Proficient in Database Development on MS SQL Server with T-SQL (not explicitly stated but can be inferred)\n* Cloud technology and senior developer skills for a Full Stack Developer role.'




```python
chain.invoke("What is the least amount of year experience needed for the job role?")
```




    '2 years. The job description specifies "2+ years demonstrable experience with Microsoft Azure" as one of the requirements.'


