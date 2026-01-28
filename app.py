import streamlit as st 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.prompts import PromptTemplate
import requests
from dotenv import load_dotenv
load_dotenv()
import os
import tempfile

st.title("Multi Document RAG System")
st.write("Upload PDFs and ask questions")

PROMPT_TEMPLATE = """
You are an expert research assistant.

Context:
{context}

Question:
{question}

Using only the provided context, answer the question clearly and thoroughly.
If multiple documents are involved, compare them logically and highlight similarities and differences.
If the answer is not present in the context, say you do not have enough information. Keep the answer concise.
"""

document_prompt = PromptTemplate(
    input_variables=["page_content", "source_file"],
    template="""
    Source Document: {source_file}
    Content:
    {page_content}    
"""
)

uploaded_files = st.file_uploader("Choose one or more PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_documents = []
    with st.spinner("📄 Processing PDFs..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            # Adding the source file metadata
            for doc in documents:
                doc.metadata['source_file'] = uploaded_file.name
            
            all_documents.extend(documents)
            os.unlink(tmp_path)

        st.write(f"Loaded {len(all_documents)} pages from {len(uploaded_files)} PDF(s)")        

        # Splitting the documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )
        chunks = splitter.split_documents(all_documents)

        # Creating embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2" # using huggingface sentence transformer model for embeddings
        )
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        # Initializing the LLM 
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", # here I'm using google gemini 2.5 flash model
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0 # setting temperature to 0 for more deterministic responses
        )

        # Store in session state
        st.session_state.vector_store = vector_store
        st.session_state.llm = llm
        st.session_state.processed = True

    st.success(f"Processed {len(all_documents)} pages. Ask anything!")

if "processed" in st.session_state:
    question = st.text_input("Your Question:", placeholder="e.g., What do these PDFs talk about?")

    if st.button("Get Answer") and question:
        with st.spinner("Analyzing documents, please hang on..."):
            
            retriever = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": 15}
            )
            
            # Create QA chain with adaptive prompt
            prompt = PromptTemplate(
                template=PROMPT_TEMPLATE,
    
                input_variables=["context", "question"]
            )
            qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={
                    "prompt": prompt,
                    "document_prompt": document_prompt,
                    "document_variable_name": "context"
                },
                return_source_documents=True
            )
            
            result = qa_chain.invoke(question)
            
            st.markdown("### 📝 Answer:")
            st.markdown(result['result'])
            
            st.markdown("### 📚 Sources:")
            sources = []
            for doc in result['source_documents']:
                source_name = doc.metadata.get('source_file', 'Unknown')
                page = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                sources.append(f"• {source_name} (Page{page})")
            
            unique_sources = list(set(sources))
            for source in unique_sources:
                st.markdown(source)
            with st.expander("🔍 Debug Info"):
                # st.write(f"**Question Type Detected**: {get_question_type(question)}")
                docs = retriever.invoke(question)
                st.write(f"**Documents Retrieved**: {len(docs)}")

