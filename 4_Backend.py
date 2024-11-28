from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import os
import mysql.connector
from openai import OpenAI
from dotenv import load_dotenv
from langchain.schema import Document
from RAG_BONUS import skeleton_query, qa

app = FastAPI()
load_dotenv()


class ChurnDatabaseSetup:
    def __init__(self, host: str = "localhost",
                 user: str = "root",
                 password: str = os.getenv('MYSQL_PASS'),
                 port: int = 3306):
        self.connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            port=port
        )
        self.cursor = self.connection.cursor()
        
    def close(self):
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()
            

model = load_model(os.path.join('models','NeuralNetwork.keras'))
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class ChurnFeatures(BaseModel):
    features: list

@app.post("/predict")
async def predict(data: ChurnFeatures):
    
    prediction = model.predict(np.array([data.features]))
    return {"prediction": float(prediction[0][0])}


class Query(BaseModel):
    query: str

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': os.getenv('MYSQL_PASS'),
    'database': 'churn'
}

def get_schema() -> Dict:
    with mysql.connector.connect(**DB_CONFIG) as conn:
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute("""
                SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s
                ORDER BY TABLE_NAME
            """, (DB_CONFIG['database'],))
            
            schema = {}
            for row in cursor.fetchall():
                table = row['TABLE_NAME']
                if table not in schema:
                    schema[table] = []
                schema[table].append({
                    'column': row['COLUMN_NAME'],
                    'type': row['DATA_TYPE']
                })
            return schema

@app.post("/generate_sql")
async def generate_sql(query: Query):
    schema = get_schema()
    schema_text = "Database Schema:\n"
    for table, columns in schema.items():
        schema_text += f"\nTable: {table}\n"
        for col in columns:
            schema_text += f"- {col['column']} ({col['type']})\n"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"{schema_text}\nConvert this question to SQL: {query.query}\nReturn only the SQL query."
        }],
        temperature=0
    )
    
    sql = response.choices[0].message.content.strip()
    
    try:
        with mysql.connector.connect(**DB_CONFIG) as conn:
            with conn.cursor(dictionary=True) as cursor:  # Use dictionary cursor
                cursor.execute(sql)
                results = cursor.fetchall()  # Fetch all rows
                # Get column names from cursor description
                columns = [column[0] for column in cursor.description] if cursor.description else []
                
        return {
            "sql": sql, 
            "status": "success",
            "results": results,  # List of dictionary rows
            "columns": columns   # List of column names
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": str(e), "sql": sql})
    
class QueryRequest(BaseModel):
    query: str

class DocumentResponse(BaseModel):
    content: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    sources: List[DocumentResponse]
    confidence: Optional[float] = None

@app.post("/rag_query")
async def process_rag_query(request: QueryRequest):
    try:
        # Format the query using the skeleton
        formatted_query = skeleton_query.format(request.query)
        
        # Get relevant documents using the retriever
        docs = qa.get_relevant_documents(formatted_query)
        
        # Process the results
        sources = []
        for doc in docs:
            # Convert Document objects to digestible format
            source = DocumentResponse(
                content=doc.page_content,
                metadata=doc.metadata
            )
            sources.append(source)
            
        # Calculate a simple confidence score based on number of relevant documents
        confidence = min(len(sources) / 5, 1.0)  # Normalize to 0-1 range, max at 5 docs
        
        # Combine the documents into a coherent answer
        combined_answer = "Based on the query, here are the relevant customers:\n\n"
        for idx, source in enumerate(sources, 1):
            combined_answer += f"Customer {idx}:\n"
            combined_answer += f"- ID: {source.metadata.get('customer_id', 'N/A')}\n"
            combined_answer += f"- Monthly Charges: ${source.metadata.get('monthly_charges', 'N/A')}\n"
            combined_answer += f"- Total Charges: ${source.metadata.get('total_charges', 'N/A')}\n"
            combined_answer += f"- Tenure: {source.metadata.get('tenure', 'N/A')} months\n"
            combined_answer += f"- Contract Type: {source.metadata.get('contract_type', 'N/A')}\n\n"
        
        return QueryResponse(
            answer=combined_answer,
            sources=sources,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

