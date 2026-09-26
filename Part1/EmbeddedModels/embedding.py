#lets take a example of movie recommendation system 
#if i convert these movie names in to vector embedding and then we can use these embeddings to find similar movies based on their vector representations.


#using openAI

from importlib import import_module
from dotenv import load_dotenv

load_dotenv()

try:
	OpenAIEmbeddings = import_module("langchain_openai").OpenAIEmbeddings
except ModuleNotFoundError as error:
	raise ModuleNotFoundError(
		"Install the required dependency with: pip install langchain-openai"
	) from error

embeddings = OpenAIEmbeddings()

#we can assign dimesions manually inside the model
#in hugging face there are some free embedding models available