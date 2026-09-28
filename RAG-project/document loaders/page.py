from langchain_community.document_loaders import WebBaseLoader


url="https://www.oreilly.com/library/view/learning-python-5th/9781449355722/ch01.html"

data=WebBaseLoader(url)
docs=data.load()

print(len(docs))

