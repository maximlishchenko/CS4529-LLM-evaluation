# import needed dependencies
import os
from dotenv import load_dotenv
from . import constants
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder

# function to build the pipeline and print the llm's response
def build_pipeline(document_store, prompt_template, question):
    load_dotenv() # load environment variables
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # access the api key env variable

    # create necessary components
    retriever = InMemoryBM25Retriever(document_store=document_store)
    prompt_builder = PromptBuilder(template=prompt_template)
    llm = OpenAIGenerator() # initialise the llm

    # initialise the pipeline and add components
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    # run the pipeline
    results = rag_pipeline.run(
        {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }
    )

    # print the response of the llm
    print(results["llm"]["replies"])

# function to build the document store that will hold traces
def build_document_store(traces):
    document_store = InMemoryDocumentStore() # initialise document store
    # access the traces from the parameter list
    for trace in traces:
        # read each trace and write into document store
        file = open(constants.DATA_DIR + trace + '.json', 'r')
        content = file.read()
        document_store.write_documents([
            Document(content=content)
        ])
        file.close()

    return document_store

# function to read the prompt template
def read_prompt(template_dir, template_name):
    prompt_file = open(template_dir + template_name, 'r')
    prompt_template = prompt_file.read()
    prompt_file.close()
    return prompt_template