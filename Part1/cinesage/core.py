from dotenv import load_dotenv
load_dotenv()


import langchain_mistralai 
from langchain_core.prompts import ChatPromptTemplate



model =langchain_mistralai.ChatMistralAI()
prompt =ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "what is the capital of France?")
])

para=input("enter your para:")

response =model.invoke(prompt)
print(response.content)

#refer langchain prompt for more
