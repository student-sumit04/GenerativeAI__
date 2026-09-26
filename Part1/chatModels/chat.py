from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq"
)

response = model.invoke("Give me a brief summary of RAG")
print(response.content)
#temperature less for the logical task
#max_token =20 inside model these will be used
