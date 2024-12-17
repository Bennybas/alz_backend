import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import asyncio
import re

load_dotenv()

# Validate API Key
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY must be set in environment variables")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Prompt with More Specific Instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are AIVY, a sophisticated Healthcare AI Assistant.
    Provide clear, concise, and accurate responses about healthcare topics.
    
    Communication Guidelines:
    - Be direct and informative
    - Use professional medical terminology
    - Maintain a compassionate and professional tone
    - Provide comprehensive yet succinct answers"""),
    ("human", "{Question}")
])

# LLM Configuration
llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0.2,
    api_key=GROQ_API_KEY
)

class UserQuery(BaseModel):
    message: str




def clean_text(text: str) -> str:
    """
    Advanced text cleaning to handle streaming text issues
    """
    # Remove excessive whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    
    # Trim leading/trailing spaces
    return text.strip()

async def stream_response(chain, question):
    """
    Improved streaming response with better chunk handling
    """
    try:
        full_response = ""
        async for chunk in chain.astream({"Question": question}):
            if chunk:
                full_response += chunk
                
                # Clean the accumulated response
                cleaned_response = clean_text(full_response)
                
                # Stream the entire cleaned response
                yield f"data: {cleaned_response} \n\n"
                
                # Small delay to simulate natural typing
                await asyncio.sleep(0.05)
        
        # Final streaming signal
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        yield f"data: Error in response: {str(e)}\n\n"

@app.post("/chat")
async def chat_endpoint(query: UserQuery):
    try:
        chain = prompt | llm | StrOutputParser()
        
        return StreamingResponse(
            stream_response(chain, query.message),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}
