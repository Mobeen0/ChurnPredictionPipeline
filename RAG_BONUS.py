import os
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from load_embeddings_BONUS import load_qdrant_client, get_vector_store
from dotenv import load_dotenv

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.qdrant import QdrantTranslator

load_dotenv()

skeleton_query = """Process the following natural language query and retrieve relevant customer data from the vector database:

Query: {}

Return the list of relevant customers, ensuring that each result is directly related to the user's query and provides valuable information.
"""

metadata_field_info = [
    AttributeInfo(
        name="customer_id",
        description="The unique Id of the customer",
        type="string",
    ),
    AttributeInfo(
        name="monthly_charges",
        description="The monthly charge of subscription per customer",
        type="float", # was "integer"
    ),
    AttributeInfo(
        name="total_charges",
        description="The total charges for the customer",
        type="string",
    ),
    AttributeInfo(
        name="tenure", 
        description="The total amount of months as a subscriber", 
        type="integer"
    ),
    AttributeInfo(
        name="contract_type", 
        description="The type of customer contract", 
        type="string"
    )
]

metadata_key = "metadata"

# Create the QdrantTranslator
qdrant_translator = QdrantTranslator(metadata_key=metadata_key)

def initialize_qa():
    qdrant_client = load_qdrant_client()
    vector_store = get_vector_store(qdrant_client)
    # qa = RetrievalQA.from_chain_type(
    #     llm=OpenAI(temprature=0),
    #     chain_type="stuff",
    #     retriever=vector_store.as_retriever(),
    #     return_source_documents=True
    # )
    document_content_description = "Semi structured information of customers who churned or not"
    qa = SelfQueryRetriever.from_llm(
        llm=OpenAI(temperature=0),
        vectorstore=vector_store,
        metadata_field_info = metadata_field_info,
        document_contents = document_content_description,
        verbose = True,
        structured_query_translator=qdrant_translator,
        #return_source_documents=True
    )
    return qa

qa = initialize_qa()