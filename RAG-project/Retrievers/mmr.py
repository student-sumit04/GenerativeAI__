from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings()

docs=[
    Document(page_content="This is a sample document."),
    Document(page_content="Gradient descent is an optimizaton that minimises  the loss function."),
    Document(page_content="Deep learning is a subset of machine learning that uses neural networks to model and solve complex problems.")
]

embedding_model= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore=Chroma.from_documents(docs,embeddings)

similarity_results=vectorstore.similarity_search("What is deep learning?",k=2)

for result in similarity_results:
    print(result.page_content)

mmr_retriever=vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":2,"fetch_k":4})
mmr_results=mmr_retriever.invoke("What is deep learning?")    

for result in mmr_results:
    print(result.page_content)
