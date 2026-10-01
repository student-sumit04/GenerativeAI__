from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI
from langchain.retrievers.multi_query import MultiQueryRetriever


load_dotenv()

docs = [
	Document(page_content="This is a sample document."),
	Document(
		page_content=(
			"Gradient descent is an optimization algorithm that minimizes "
			"the loss function."
		)
	),
	Document(
		page_content=(
			"Deep learning is a subset of machine learning that uses neural "
			"networks to model and solve complex problems."
		)
	),
]

embedding_model = HuggingFaceEmbeddings(
	model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(docs, embedding_model)

model = ChatMistralAI(model="mistral-large-latest")
multiquery_retriever = MultiQueryRetriever.from_llm(
	retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
	llm=model,
)

multiquery_results = multiquery_retriever.invoke("What is deep learning?")

for result in multiquery_results:
	print(result.page_content)
