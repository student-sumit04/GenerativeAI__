import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader  # type: ignore[reportMissingImports]
from langchain_text_splitters import TokenTextSplitter




data=PyPDFLoader("document loaders/GRU.pdf")

docs=data.load()

splitter=TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)
chunks=splitter.split_documents(docs)
print(docs[14])
print(len(chunks))
