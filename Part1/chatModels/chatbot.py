from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

from langchain_mistralai import ChatMistralAI
model=ChatMistralAI(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    model_provider="mistralai",
    temperature=0.1,
    max_token=20)

messages=[
    SystemMessage(content="You are a helpful assistant.")



]

while True:
    print("_____welcome to the chatbot____")
    prompt=input("enter your prompt:")
    messages.append(HumanMessage(content=prompt))
    if prompt=="0":
        break

    response=model.invoke(prompt)
    messages.append(AIMessage(content=response.content))
    print("Bot:",response.content)


#langchain_core.messages  
# i can create the ui for this using the claude ui by giving it prrompt   