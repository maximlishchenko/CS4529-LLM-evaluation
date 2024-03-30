import os
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder

def build_pipeline(document_store, prompt_template, question):
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    retriever = InMemoryBM25Retriever(document_store=document_store)
    prompt_builder = PromptBuilder(template=prompt_template)
    llm = OpenAIGenerator()

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    results = rag_pipeline.run(
        {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }
    )

    print(results["llm"]["replies"])