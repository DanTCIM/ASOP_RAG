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
from langchain_core.prompts import ChatPromptTemplate

## API key setup 
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

## Sidebar
with st.sidebar:
    st.title("ASOP Q&A Machine")
    st.write("Powered by OpenAI's GPT 3.5-Turbo: Harness the capabilities of LLM to search for and retrieve information on ASOP.")
    
    link1 = "http://www.actuarialstandardsboard.org/wp-content/uploads/2023/12/ASOPs-as-of-Decemeber-2023.zip"
    st.caption(f"ASOP documents are downloaded from Actuarial Standards Board's [link]({link1}) as of December 2023.")
    
    def clear_chat_history():
        st.session_state.messages = [{"role": "ai", "content": "What is your question on ASOP?", "type": "text"}]
    st.button('Clear Chat History', on_click=clear_chat_history)
   
    with st.container(border=True):
        st.subheader('⚙️ RAG Parameters')
        num_source = st.slider('Top N sources to view:', min_value=4, max_value=20, value=5, step=1)
        _lambda_mult = st.slider('Source diversity (lambda):', min_value=0.0, max_value=1.0, value=0.5, step=0.25)
        with st.expander("What is diversity?"):
            st.caption("Maximal Marginal Relevance (MMR) tries to reduce redundancy of fetched documents and increase diversity. 0 being the most diverse, 1 being the least diverse. 0.5 is a balanced state.")

    st.subheader('📖 Further Notes')
    st.write("Responses are based on LLM's features and search algorithms, and should not be relied upon as definitive or error-free. Users are encouraged to review the source contexts carefully. The sources may appear less relevant to the question due to the diversity of the search.")
    link2 = "https://www.actuarialstandardsboard.org/standards-of-practice/"
    st.write(f"Please visit [Actuarial Standard Board's ASOP site]({link2}) to get the latest ASOP.")
    
    link3 = "https://github.com/DanTCIM/ASOP_RAG"
    st.write(f"The Python codes and documentation of the project are in [GitHub]({link3}).")

# # Set up the title and input
st.header("Actuarial Standards of Practice (ASOP) Q&A Machine using Retrieval Augmented Generation (RAG)")
st.write("Please see sidebar for further information.")

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
#prompt_concise = hub.pull("rlm/rag-prompt")

qa_system_prompt = """You are a helpful assistant to help actuaries with question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
All the contexts are from Actuarial Standards of Practice (also called ASOP or asop). \
A user can ask related to a specific source (e.g., ASOP No. 14 means sources such as asop014_***.pdf). If so, use the specific source to answer.\
After you answer, provide the sources you used to answer the question. \
If you don't know the answer, just say that you don't know. \

{context}"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{question}"),
    ]
)

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
def generate_output(prompt_input):
    # Invoke the RAG chain with the user input as the question
    output = rag_chain_with_source.invoke(prompt_input)

    # Generate the Markdown output with the question, answer, and context
    markdown_output = "{}".format(output['answer'])

    last_page_content = None  # Variable to store the last page content
    i = 1 # Source indicator
    markdown_source_output = ''

    # Iterate over the context documents to format and include them in the output
    for doc in output['context']:
        current_page_content = doc.page_content.replace('\n', '  \n')  # Get the current page content
        
        # Check if the current content is different from the last one
        if current_page_content != last_page_content:
            markdown_source_output += "#### Source {}: {}, page {}\n\n{}\n".format(i, doc.metadata['source'].split("/data/ASOP/")[-1], doc.metadata['page'], current_page_content)
            i = i + 1
        last_page_content = current_page_content  # Update the last page content
    
    # Display the output for markdown
    return markdown_output, markdown_source_output

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "ai", "content": "What is your question on ASOP?", "type": "text"}]

# Display or clear chat messages
for message in st.session_state.messages:
    if message['type'] == 'source':
        with st.expander("See the top N sources (some may appear less relevant due to the diversity of the search)"):
            st.write(message["content"])    
    else:
        with st.chat_message(message["role"]):
            if message['type'] == 'text':
                st.write(message["content"])

# User-provided prompt
if user_prompt := st.chat_input("What is your question on ASOP?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt, "type": "text"})
    with st.chat_message("user"):
        st.write(user_prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "ai":
    with st.chat_message("ai"):
        with st.spinner("Retrieving info and generating response..."):
            response, sources = generate_output(user_prompt)
            st.write(response)
    with st.expander("See the top N sources (some may appear less relevant due to the diversity of the search)", expanded=False):
        st.write(sources)
    message = {"role": "ai", "content": response, "type": "text"}
    source_expand = {"role": "ai", "content": sources, "type": "source"}
    st.session_state.messages.append(message)
    st.session_state.messages.append(source_expand)
