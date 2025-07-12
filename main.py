import os
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re

DATA_PATH = "data/"
PDF_FILENAME = "sample.pdf"

def load_documents():
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Page Count: {len(documents)}")
    return documents

docs = load_documents()

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Chunk count:  {len(all_splits)} chunks")
    return all_splits

chunks = split_documents(docs)


def get_embedding_function(model_name="nomic-embed-text"):
    embeddings = OllamaEmbeddings(model=model_name)
    print(f"Embedding Model: {model_name}")
    return embeddings

embedding_function = get_embedding_function()

CHROMA_PATH = "chroma_db"

def get_vector_store(embedding_function, persist_directory=CHROMA_PATH):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    print(f"Vector store: {persist_directory}")
    return vectorstore


def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
    print(f"Indexing {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    vectorstore.persist() # Ensure data is saved
    print(f"Indexing data saved to: {persist_directory}")
    return vectorstore

if os.path.isdir(CHROMA_PATH):
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
else:
    vector_store = index_documents(chunks, embedding_function)

def create_rag_chain(vector_store, llm_model_name="llama3.2", context_window=8192):
    # Initialize the LLM
    llm = ChatOllama(
        model=llm_model_name,
        temperature=0, # Lower temperature for more factual RAG answers
    )
    print(f"Initialized ChatOllama with model: {llm_model_name}, context window: {context_window}")

    # Create the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", # Or "mmr"
        search_kwargs={'k': 3} # Retrieve top 3 relevant chunks
    )

    template = """You are a helpful assistant trained in homeopathic medicine. Based on the provided context, suggest the most appropriate homeopathic remedy for the given symptoms.
{context}

Question: {question} .
"""
    prompt = ChatPromptTemplate.from_template(template)
    print("Prompt template created.")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
    )
    print("RAG chain created.")
    return rag_chain

def query_rag(chain, question):
    response = chain.invoke(question)
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    print(cleaned)

#Create RAG Chain
rag_chain = create_rag_chain(vector_store, llm_model_name="llama3.2")

symptoms = input("Please enter the symptoms comma separated: ")
query_question = "Given the following symptoms: "+symptoms+" ,suggest few  most suitable homeopathic medicines. Atleast 5. Include the medicine name(s), key indications, and any matching symptoms from the context. Only use the context provided. Do not hallucinate or fabricate any information."
query_rag(rag_chain, query_question)