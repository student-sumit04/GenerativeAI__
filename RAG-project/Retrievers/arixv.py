from langchain_community.retrievers import ArxivRetriever



retriever =ArxivRetriever(
    load_max_docs=2,# no of papers to retrieve
    load_all_available_meta=True
)

docs =retriever.invoke("large language models")



for i,doc in enumerate(docs):
    print(f"Document {i+1}:")
    print(f"Title: {doc.metadata.get('title')}")
    print(f"Authors: {doc.metadata.get('authors')}")
    print(f"Abstract: {doc.page_content}")
    print("-" * 40)