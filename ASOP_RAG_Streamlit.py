# # Initial set up
import streamlit as st
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# sqlite3
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import sqlite3

# Import the necessary modules
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel # for RAG with source
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import chromadb

# # Set up the title and input
st.title("Actuarial Standards of Practice (ASOP) Q&A Machine using Retrieval Augmented Generation (RAG)")
st.header("Utilize the power of LLM 🧠 to search and get information on Actuarial Standards of Practice")

link1 = "https://github.com/DanTCIM/ASOP_RAG"
st.write(f"The Python code and the documentation of the project are in [GitHub]({link1}).")
link2 = "https://www.actuarialstandardsboard.org/standards-of-practice/"
st.write(f"Please visit [Actuarial Standard Board's ASOP site]({link2}) to see the source.")

st.write("What is your question on ASOP?:")
# Prompt the user for a question on ASOP
usr_input = st.text_input("")

# # Model and directory setup
embeddings_model = OpenAIEmbeddings()
db_directory = "./data/chroma_db1"
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", 
                 temperature=0) # context window size 16k for GPT 3.5 Turbo


# # Get a Chroma vector database with specified parameters
#@st.cache_data
#def create_vectorstore(_embeddings_model, _db_directory):
#    # Get a Chroma vector database with specified parameters
#    return Chroma(embedding_function=_embeddings_model, persist_directory=_db_directory)
vectorstore = Chroma(embedding_function=embeddings_model, persist_directory=db_directory)

# # Retrieve and RAG chain
# Create a retriever using the vector database as the search source
retriever = vectorstore.as_retriever(search_type="mmr", 
                                     search_kwargs={'k': 6, 'lambda_mult': 0.35}) 
# Use MMR (Maximum Marginal Relevance) to find a set of documents that are both similar to the input query and diverse among themselves
# Increase the number of documents to get, and increase diversity (lambda mult 0.5 being default, 0 being the most diverse, 1 being the least)

# Load the RAG (Retrieval-Augmented Generation) prompt
prompt = hub.pull("rlm/rag-prompt")

# Define a function to format the documents with their sources and pages
def format_docs_with_sources(docs):
    formatted_docs = "\n\n".join(doc.page_content for doc in docs)
    sources_pages = "\n".join(f"{doc.metadata['source']} (Page {doc.metadata['page'] + 1})" for doc in docs)
    # Added 1 to the page number assuming 'page' starts at 0 and we want to present it in a user-friendly way

    return f"Documents:\n{formatted_docs}\n\nSources and Pages:\n{sources_pages}"

# Create a RAG chain using the formatted documents as the context
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs_with_sources(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

# Create a parallel chain for retrieving and generating answers
rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)


# # Generate output
def generate_output():
    
    # Invoke the RAG chain with the user input as the question
    output = rag_chain_with_source.invoke(usr_input)

    # Generate the Markdown output with the question, answer, and context
    markdown_output = "### Question\n{}\n\n### Answer\n{}\n\n### Context\n".format(output['question'], output['answer'])

    last_page_content = None  # Variable to store the last page content
    i = 1 # Source indicator

    # Iterate over the context documents to format and include them in the output
    for doc in output['context']:
        current_page_content = doc.page_content.replace('\n', '  \n')  # Get the current page content
        
        # Check if the current content is different from the last one
        if current_page_content != last_page_content:
            markdown_output += "- **Source {}**: {}, page {}:\n\n{}\n".format(i, doc.metadata['source'], doc.metadata['page'], current_page_content)
            i = i + 1
        last_page_content = current_page_content  # Update the last page content
    
    # Display the Markdown output
    st.markdown(markdown_output)

# Let's search ASOP!
if st.button('Search ASOP'):
    st.write("Thank you for being patient! I am running ...")
    generate_output()
