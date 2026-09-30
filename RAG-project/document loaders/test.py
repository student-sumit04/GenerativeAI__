#for more use langchain docs

import os
from pathlib import Path
from types import SimpleNamespace

from langchain_mistralai import ChatMistralAI


class TextLoader:
    """Minimal text loader compatible with the document splitter."""

    def __init__(self, file_path):
        self.file_path = Path(file_path)

    def load(self):
        return [
            SimpleNamespace(
                page_content=self.file_path.read_text(encoding="utf-8"),
                metadata={"source": str(self.file_path)},
            )
        ]
from langchain_text_splitters import RecursiveCharacterTextSplitter


splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=1,
)

data = TextLoader(Path(__file__).with_name("notes.txt"))
docs = data.load()
chunks = splitter.split_documents(docs)

if os.getenv("MISTRAL_API_KEY"):
    model = ChatMistralAI(model="mistral-small-2506")
    result = model.invoke(docs[0].page_content)
    print(result.content)
else:
    print(f"Loaded {len(docs)} document and created {len(chunks)} chunks.")

#text splitter follows recursively first ,double line break ,then single line braeak
#spaces and then chunks size

