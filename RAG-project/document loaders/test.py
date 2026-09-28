#for more use langchain docs

from tempfile import template

from langchain_mistralai import ChatMistralAI

from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

data=TextLoader("document loaders/notes.txt")
docs=data.load()

ChatPromptTemplate.from_message(
    [("")

    ]
)
model=ChatMistralAI(model="mistral-small-2506")
prompt=template.format_message(data=docs[0].page_content)
result =model.invoke(prompt)

print (result.content)#metadata and page content

