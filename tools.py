from datetime import datetime 

from rag import search 

from Utilities.utilities import log

import requests

import json

def get_current_time():
    
    return datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

def get_city_weather(city):
    
    url = f"https://wttr.in/{city}?format=4"
    
    response = requests.get(url , timeout=10)
    
    response.raise_for_status() # This will raise HTTPerror if something went wrong otherwise gets ignored
    
    return response.text

def get_user_city():
    
    return "Katihar"


def search_knowledge_base(query):
    result = search(query, top_k=5)
    
    log(f"🔍 search_knowledge_base: {query}")
    
    # 1. Initialize blocks here before the loop
    blocks = []
    
    for r in result:
        log(f"   {r['score']:.2f} - {r['source']} #{r['chunk_id']}")
        
        sourceFileName = r['source']
        chunkId        = r['chunk_id']
        chunkText      = r['text']
        
        header = f"[{sourceFileName} #{chunkId}]"
        block  = f"{header}\n{chunkText}"
        blocks.append(block)
        
    # 2. Join all collected blocks together after the loop finishes
    final_output = "\n\n---\n\n".join(blocks)
    
    return final_output
    
    
#TOOLS

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns current date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_weather",
            "description": "Returns weather for a given city name. Use the exact city name specified by the user in their prompt. Only call get_user_city if the user asks for their local weather without naming a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The exact name of the city or region requested by the user (e.g. Kashmir, Delhi, etc.).",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_city",
            "description": "Call this FIRST to get the user's home city name whenever the user asks about their local weather or location without specifying a city.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Searches the Zorvex Technologies company knowledge base: leave , medical , HR , hiring , tech , incident"
            "reports and travel expense policies. Use it for any question about company policy or documents."
            "Returns the most relevant passages with their source.",
            
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question, rewritten as a short, self-contained search query.",
                    }
                },
                "required": ["query"]
            }
        }
        
    }
]

FUNCTION_NAME_TO_MAP = {
    "get_current_time" : get_current_time,
    "get_city_weather" : get_city_weather,
    "get_user_city"    : get_user_city,
    "search_knowledge_base" : search_knowledge_base,
}