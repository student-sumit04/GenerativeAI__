"""Demonstrate a LangChain RunnableSequence."""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


# Loads MISTRAL_API_KEY (and other values) from a local .env file.
# Keep .env out of version control; API keys must never be committed.
load_dotenv()


def main() -> None:
    # The template declares the input key that the chain will receive.
    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in simple words."
    )

    # The model reads MISTRAL_API_KEY from the environment by default.
    model = ChatMistralAI(model="mistral-small-2506")

    # StrOutputParser converts the model's AIMessage into a plain string.
    parser = StrOutputParser()

    # The | operator creates a RunnableSequence:
    # input dictionary -> prompt messages -> model response -> text output.
    chain = prompt | model | parser

    # Invoke the whole sequence with a dictionary matching {topic}.
    final_output = chain.invoke({"topic": "Machine Learning"})
    print(final_output)


if __name__ == "__main__":
    main()
