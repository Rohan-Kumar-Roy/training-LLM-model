from openai import OpenAI

import os
from dotenv import load_dotenv
import numpy
import json

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBED_MODEL = os.getenv("EMBED_MODEL")
KB_FOLDER = os.getenv("KB_FOLDER")
VECTOR_STORE_FILE=os.getenv("VECTOR_STORE_FILE")

def cosine_similarity(A , B):
    dot_prod = numpy.dot(A,B)
    norm_a = numpy.linalg.norm(A)
    norm_b = numpy.linalg.norm(B)
    return dot_prod / (norm_a * norm_b)

def get_embedding(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

def load_documents(folder=KB_FOLDER):
    documents = []
    
    for filename in sorted(os.listdir(folder)):
        
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(folder , filename)
        with open(path , "r" , encoding="utf-8") as f:
            documents.append(
                {
                    "source": filename,
                    "text":f.read()
                }
            )
            
    return documents 


def add_embeddings(chunks):
    for chunk in chunks:
        vector = get_embedding(chunk["text"])
        chunk["vector"] = vector
    
    return chunks
       

def chunk_document(document):

  chunks = []
  chunk_id = 0

  for part in document["text"].split("\n\n"):
    part = part.strip()
    if part:
      chunks.append(
          {
              "source": document["source"],
              "chunk_id": chunk_id,
              "text": part,
          }  
      )

      chunk_id += 1

  return chunks


def create_chunks():
    documents = load_documents()
    chunks = []
    
    for document in documents:
        
        chunks.extend(chunk_document(document))
    
    return chunks 
  
def save_vector_store(chunks):
    
    with open(VECTOR_STORE_FILE , "w" , encoding="utf-8") as f:
        
        json.dump(chunks , f , indent=4)

def load_vector_store():
    
    with open(VECTOR_STORE_FILE , "r" , encoding="utf-8") as f:
        
       return json.load(f)
   
def load_chunks():
    if os.path.exists(VECTOR_STORE_FILE):
         chunks = load_vector_store()
         return chunks

    
    else:
        chunks = create_chunks()
        chunks = add_embeddings(chunks)
        save_vector_store(chunks)  
        return chunks
    
chunks = load_chunks()
    
def search(query , top_k = 3):
    
    query_vector = get_embedding(query)
    result = []
    
    for chunk in chunks:
        score = cosine_similarity(query_vector , chunk["vector"])
        
        result.append(
            {
                "score" : float(score),
                "source": chunk["source"],
                "chunk_id":chunk["chunk_id"],
                "text"  : chunk["text"],
                "size"  : len(chunk["text"])
            }
        )
    
    result.sort(key=lambda rec : rec["score"] , reverse = True)       
    
    return result[:top_k]