from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader  # type: ignore[import-not-found]
from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

data=PyPDFLoader("document loaders/GRU.pdf")
docs=data.load()
template=ChatPromptTemplate.from_message(
    [("system","you are a AI that summarizes the text"),
     ("human","{data}")]
)
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks=splitter.split_documents(docs)

model=ChatMistralAI(model="zai-glm-5-2")



