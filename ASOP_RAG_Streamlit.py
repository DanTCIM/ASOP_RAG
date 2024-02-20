# # Initial set up
# Import the necessary modules
import streamlit as st
import os

# sqlite3 related (for Streamlit)
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Langchain and Vector DB
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel # for RAG with source
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import chromadb

## API key setup 
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

## Sidebar
with st.sidebar:
    st.title("ASOP Q&A Machine")
    st.write("Powered by OpenAI's GPT 3.5-Turbo: Harness the capabilities of LLM to search for and retrieve information on ASOP.")
    
    link1 = "http://www.actuarialstandardsboard.org/wp-content/uploads/2023/12/ASOPs-as-of-Decemeber-2023.zip"
    st.write(f"ASOP documents are downloaded from Actuarial Standards Board's [link]({link1}) as of December 2023.")
    
    st.subheader('⚙️ RAG Parameters')
    num_source = st.sidebar.slider('No. of sources to return', min_value=4, max_value=20, value=5, step=1)
    _lambda_mult = st.sidebar.slider('Source precision', min_value=0.0, max_value=1.0, value=0.5, step=0.25)
    st.write("The search algorithm fetches 20 sources regardless of the number of sources to return.")
    st.write("Source precision controls how fetched sources are diverse. 0 being the most diverse, 1 being the least diverse. The diversity helps LLM generate a response considering multiple aspects.")

    st.subheader('📖 Further Notes')
    st.write("Responses are based on LLM's features and search algorithms, and should not be relied upon as definitive or error-free. Users are encouraged to review the source contexts carefully. The sources may appear less relevant to the question due to the diversity of the search.")
    link2 = "https://www.actuarialstandardsboard.org/standards-of-practice/"
    st.write(f"Please visit [Actuarial Standard Board's ASOP site]({link2}) to get the latest ASOP.")
    
    link3 = "https://github.com/DanTCIM/ASOP_RAG"
    st.write(f"The Python codes and documentation of the project are in [GitHub]({link3}).")

# # Set up the title and input
st.header("Actuarial Standards of Practice (ASOP) Q&A Machine using Retrieval Augmented Generation (RAG)")
st.write("Please see sidebar for further information.")
# Prompt the user for a question on ASOP
usr_input = st.text_input("What is your question on ASOP?: ")

# # Model and directory setup
embeddings_model = OpenAIEmbeddings()
db_directory = "./data/chroma_db1"
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", 
                 temperature=0) # context window size 16k for GPT 3.5 Turbo

# # Get a Chroma vector database with specified parameters
vectorstore = Chroma(embedding_function=embeddings_model, persist_directory=db_directory)

# # Retrieve and RAG chain
# Create a retriever using the vector database as the search source
retriever = vectorstore.as_retriever(search_type="mmr", 
                                     search_kwargs={'k': num_source, 'lambda_mult': _lambda_mult}) 
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
    markdown_output = "### Question\n{}\n\n### Search summarized by GPT 3.5-turbo\n{}\n\n### Sources\n".format(output['question'], output['answer'])

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
    with st.spinner('Retrieving info and generating response...'):
        generate_output()
