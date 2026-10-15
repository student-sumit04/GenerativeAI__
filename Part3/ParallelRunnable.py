"""Demonstrate parallel execution with LangChain's RunnableParallel."""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_mistralai import ChatMistralAI


# Loads MISTRAL_API_KEY from a local .env file.
# Keep the .env file out of version control because it contains a secret.
load_dotenv()


def main() -> None:
    # Both branches below receive the same input dictionary.
    summary_prompt = ChatPromptTemplate.from_template(
        "Give a short and simple summary of {topic}."
    )
    key_points_prompt = ChatPromptTemplate.from_template(
        "List three important points about {topic}."
    )

    # The model is shared by both branches. RunnableParallel can execute
    # independent branches concurrently when the chain is invoked.
    model = ChatMistralAI(model="mistral-small-2506")
    parser = StrOutputParser()

    # Each value in RunnableParallel is a runnable chain:
    # prompt -> model -> parser.
    parallel_chain = RunnableParallel(
        summary=summary_prompt | model | parser,
        key_points=key_points_prompt | model | parser,
    )

    # The input keys must match the {topic} variables in both prompts.
    # The result is a dictionary with "summary" and "key_points" keys.
    result = parallel_chain.invoke({"topic": "Machine Learning"})

    print("Summary:")
    print(result["summary"])
    print("\nKey points:")
    print(result["key_points"])


if __name__ == "__main__":
    main()