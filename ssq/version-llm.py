import logging
import os
import streamlit as st
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
import openai

# Setup logging
logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

# Initialize document store
document_store = InMemoryDocumentStore(use_bm25=True)

# UI to upload text files
st.title("Document Upload and Question Answering")
uploaded_files = st.file_uploader("Choose text files", type=["txt"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save uploaded file to the directory
        with open(os.path.join("data/build_your_first_question_answering_system", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

# Indexing the documents
doc_dir = "data/build_your_first_question_answering_system"
files_to_index = [os.path.join(doc_dir, f) for f in os.listdir(doc_dir) if f.endswith('.txt')]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)

# Initialize retriever and reader
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
pipe = ExtractiveQAPipeline(reader, retriever)

# User input for query
query = st.text_input("Ask a question:")
if query:
    prediction = pipe.run(
        query=query,
        params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 5}}
    )
    
    # Display the answers
    if prediction['answers']:
        for i, answer in enumerate(prediction['answers']):
            st.write(f"Answer {i + 1}: {answer.answer} (score: {answer.score:.4f})")

        # Prepare the context for OpenAI
        context = " ".join([answer.answer for answer in prediction['answers']])
        openai_answer = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or another model
            messages=[
                {"role": "user", "content": f"Based on the context: {context}, answer the question: {query}"}
            ]
        )
        
        # Display the human-readable answer from OpenAI
        human_readable_answer = openai_answer.choices[0].message['content']
        st.write("Human-readable answer:", human_readable_answer)
    else:
        st.write("No answers found.")
