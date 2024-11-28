import os
import qdrant_client
import pandas as pd


from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from qdrant_client.http import models
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()



def load_qdrant_client():
    QDRANT_HOST = os.getenv("QDRANT_HOST")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    client = qdrant_client.QdrantClient(
        QDRANT_HOST,
        api_key = QDRANT_API_KEY
    )

    return client

def create_collection(client):
    vectors_config = models.VectorParams(
        size=1536,
        distance = models.Distance.COSINE
    )

    client.recreate_collection(
        collection_name = "Churn",
        vectors_config = vectors_config
    )

    return client

def get_vector_store(client):
    embeddings = OpenAIEmbeddings()
    vector_store = Qdrant(
        client = client,
        collection_name = "Churn",
        embeddings = embeddings
    )
    return vector_store

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_combined_text(row, columns_to_combine):

    combined_text = []
    for col in columns_to_combine:
        if pd.notna(row[col]):  # Check for non-null values
            combined_text.append(f"{col}: {row[col]}")
    return " | ".join(combined_text)

def load_chunks_to_vector_db(vector_store):
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    counter = 0
    text_columns = [
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]
    
    
    metadata_columns = {
        "customerID": "customer_id",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "tenure": "tenure",
        "Contract": "contract_type"
    }
    
    for index,row in df.iterrows():
        combined_text = get_combined_text(row, text_columns)
        texts = get_chunks(combined_text) # 1000 is the chunk_size
        metadatas = []
        for _ in texts:
            metadata = {}
            for col, meta_key in metadata_columns.items():
                if pd.notna(row[col]):
                    if isinstance(row[col], (int, float)):
                        metadata[meta_key] = float(row[col]) 
                    else:
                        metadata[meta_key] = str(row[col]) 
                else:
                    metadata[meta_key] = None
            metadatas.append(metadata)
        
        
        if texts: 
            vector_store.add_texts(texts, metadatas=metadatas)

def main():

    client = load_qdrant_client()
    client = create_collection(client)
    vector_store = get_vector_store(client)

    load_chunks_to_vector_db(vector_store)

