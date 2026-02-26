import PyPDF2
import langchain
import streamlit as st
from io import StringIO
from langsmith import Client
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()

# st.sidebar.subheader("Streamlit App")
with st.sidebar:
    st.title("Streamlit App")

st.set_page_config(page_title="Streamlit Rag Demo")
st.title("Streamlit Rag Demo")

uploaded_file = st.file_uploader("Choose a file", type="pdf")
# if uploaded_files:
# if uploaded_file is not None:
#     st.success("All Files Uploaded")



def format_docs(docs):
    print(f"Formatting documents{"\n\n".join(doc.page_content for doc in docs)}")
    return "\n\n".join(doc.page_content for doc in docs)

def generate_response(file, query_text):
    if file is not None:
        pdf_reader = PyPDF2.PdfReader(file)
        documents = [page.extract_text() for page in pdf_reader.pages]

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.create_documents(documents)

        llm = ChatOpenAI(model="gpt-4o")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        database = Chroma.from_documents(texts, embeddings)
        retriever = database.as_retriever()

        client = Client()
        prompt = client.pull_prompt("rlm/rag-prompt")
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
        # Create QA chain
        response = rag_chain.invoke(query_text)
        return response


question = st.text_input("Enter your question", placeholder="Ask your questions here")

result = None
with st.form(key='qa_form', clear_on_submit=True, border=False):
    submitted = st.form_submit_button("Submit", disabled=not (uploaded_file and question))
    if submitted:
        with st.spinner("Calculating..."):
            response = generate_response(uploaded_file, question)
            result = response

if result:
    st.info(result)