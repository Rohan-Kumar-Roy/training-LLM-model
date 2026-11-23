from datetime import datetime
import json
import os
from Utilities.utilities import log
from dotenv import load_dotenv
from openai import OpenAI
from types import SimpleNamespace
from tools import TOOLS, FUNCTION_NAME_TO_MAP

load_dotenv()  # by default search for .env file

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant for Zorvex Technologies employees. "
            "Answer general questions from your own knowledge. "
            "For ANY question about Zorvex policies, leave, medical cover, HR, or travel expenses, you MUST first call search_knowledge_base.For example if the user asks what are the leave policies in Zorvex Technologies , than you must definitely call search_knowledge_base function to extract response"
            "Answer company questions ONLY using the text returned by search_knowledge_base. "
            "Always cite the source at the end of your answer (e.g., [leave_policy.txt #3]). "
            "Keep answers under 100 words."
        )
    }
]

def stream_chat(history):
    
    stream = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=history,
        tools=TOOLS,
        stream=True,
    )
    
    full_response = ""
    calls = {}
    label_printed = False
    
    # Delta streaming part
    try:
        
     for chunk in stream:
        if not chunk.choices:
         continue
    
        delta = chunk.choices[0].delta
    
        if delta.content:
        
           if not label_printed:
              print("\n 🤖 Assistant: ", end = "" , flush=True)
              label_printed = True
           print(delta.content , end = "" , flush = True)
           full_response += delta.content
        
    # Tool call building part
    
        for tc in (delta.tool_calls or []):
        
          obj = calls.setdefault(
            tc.index,
            SimpleNamespace(id=None, function=SimpleNamespace(name="",arguments="")),
           )
        
          if tc.id:
            obj.id = tc.id
        
          if tc.function and tc.function.name:
            obj.function.name = tc.function.name
            
          if tc.function and tc.function.arguments:
            obj.function.arguments += tc.function.arguments
            
    except (KeyboardInterrupt, EOFError):
        
        log("\n ⚠️ Interrupted by exit command\n =====EXIT=====")
        calls = {}
               
    if label_printed:
        
        print()  # Move to new line after assistant's response   
          
    calls_list = []
    for key in sorted(calls):
        calls_list.append(calls[key])
        
    return full_response , calls_list
          

def build_tool_calls_payload(tool_calls):
    
    payload = []
    
    for tool_call in tool_calls:
        
        tool_call_dict = {
            "id": tool_call.id,
            "type":"function",
            "function":{
                "name": tool_call.function.name,
                "arguments":tool_call.function.arguments,
            }
        }
        
        payload.append(tool_call_dict)
        
    return payload

while True:
    try:
        user_input = input("\n 👤 You: ")
    except (KeyboardInterrupt, EOFError):
        break

    if user_input.lower() in ["exit", "quit", "bye", "goodbye", "return"]:
        print("\nGoodbye!")
        break
      
    if not user_input.strip():
        continue

    messages.append({
        "role": "user",
        "content": user_input
    })
  
    # 2. Handle tool calls with a safety step limit
    max_calls = 10
    current_call = 0
    
    while True:
              
        full_response , tool_calls = stream_chat(messages)
        
    
        
        # Check if the model actually wants to call a tool
        if not tool_calls:

          messages.append({
        "role": "assistant",
        "content": full_response
          })
          break
        
        log(f"\n Current_call = {current_call} : model requested {len(tool_calls)} tool calls.")
            
        current_call += 1
        if current_call > max_calls:
            
            # Safety breakout if model loops too many times
            fallback = "I got stuck looping tool calls. Let's try something else."
            
            messages.append({
                "role":"assistant",
                "content":fallback
            })
            print("\n 🤖 Assistant: " + fallback)
            break
        
        messages.append(
         {
        "role":"assistant",
        "content":full_response,
        "tool_calls": build_tool_calls_payload(tool_calls)
         })
            
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments or "{}")
        
            if func_name in FUNCTION_NAME_TO_MAP:
                fn = FUNCTION_NAME_TO_MAP[func_name]
                log(f"\n ⚙️   {func_name}({func_args})\n")
                
                try:
                    result = fn(**func_args)
                except Exception as e:
                    result = f"Error executing {func_name}: {str(e)}"
            else:
                result = f"Error: Function {func_name} not found."
                
            log(f"\n 🔧  {result}")
            
            # Append each tool's output back to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
  
