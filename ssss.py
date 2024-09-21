from haystack.nodes import FARMReader
from haystack.utils import convert_files_to_docs, clean_wiki_text
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.pipelines import ExtractiveQAPipeline

# Initialize the document store
document_store = InMemoryDocumentStore()

# Function to read the document (txt in this case)
def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Read the file content
document_text = read_txt('/workspace/yakiapp/data/build_your_first_question_answering_system/ddd.txt')

# Split the text into chunks (Haystack handles chunking by default)
docs = [{"content": document_text}]

# Write documents to the document store
document_store.write_documents(docs)

# Initialize retriever and reader
retriever = DensePassageRetriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# Add retriever embeddings
document_store.update_embeddings(retriever)

# Build the pipeline for question answering
pipe = ExtractiveQAPipeline(reader, retriever)

# Define the question
question = "How do I change the port of code-server?"

# Ask the question
prediction = pipe.run(query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}})

# Display the best answer
print(f"Answer: {prediction['answers'][0].answer}")
# Display all answers
# Display all answers with their texts
for i, answer in enumerate(prediction['answers']):
    print(f"Answer {i + 1}: {answer.answer}")

# Optional: Check if answers have any associated metadata that might help understand the context
for i, answer in enumerate(prediction['answers']):
    print(f"Answer {i + 1}: {answer.answer}, Score: {answer.score}, Context: {answer.context}")
