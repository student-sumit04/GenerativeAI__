from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

pdf_path = Path(__file__).parent / "document loaders" / "GRU.pdf"
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)

embeddings = MistralAIEmbeddings(model="mistral-embed")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="gru_documents",
    persist_directory=str(Path(__file__).parent / "chroma-db"),
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
#prompt template for RAG chain
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using only the "
            "provided context. If the answer is not in the context, say you "
            "do not know.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

model = ChatMistralAI(model="zai-glm-5-2")


def format_documents(documents):
    return "\n\n".join(document.page_content for document in documents)


rag_chain = (
    {"context": retriever | format_documents, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


if __name__ == "__main__":
    question = input("Ask a question about the document: ")
    answer = rag_chain.invoke(question)
    print("\nAnswer:\n")
    print(answer)



