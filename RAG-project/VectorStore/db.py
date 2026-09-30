from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings


from dotenv import load_dotenv
from langchain_core.documents  import Document#library

docs =[
    Document(page_content="Python is a programming language that lets you work quickly and integrate systems more effectively.", metadata={"source": "https://www.python.org/doc/essays/blurb/"}),
    Document(page_content="Pandas is used for data analysis and manipulation of data. It provides data structures and functions needed to manipulate structured data.", metadata={"source": "https://pandas.pydata.org/docs/"}),
    Document(page_content="NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.", metadata={"source": "https://numpy.org/doc/stable/"}),
    
]

load_dotenv()

embedding_model = OpenAIEmbeddings()

vectorstore=Chroma.from_documents(
    documents =docs,
    embedding=embedding_model,
    #local
    persist_directory="chroma-db"
)

result=vectorstore.similarity_search("what is used for the data analysis",k=2)
for r in result:
    print(r.page_content)
    print(r.metadata)


retriver=vectorstore.as_retriever()
docs=retriver.invoke("Explain deep learning")


for d in docs:
    print(d.page_content)