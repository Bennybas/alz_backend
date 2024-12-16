import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import asyncio
import re
 
# Load environment variables
load_dotenv()
 
# Initialize FastAPI app with CORS
app = FastAPI()
 
# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - adjust in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
 
# Initialize LangChain components
prompt = ChatPromptTemplate.from_messages([
    ("system", '''You are AIVY, a sophisticated Healthcare-based Q&A Chatbot.
    Your primary goal is to provide detailed, accurate, and helpful information
    related to patient journeys, healthcare processes, medical guidance,
    and healthcare support.
   
    Guidelines:
    - Focus on healthcare-related queries
    - Provide comprehensive and clear answers
    - Maintain a professional and empathetic tone
    - If a query is not healthcare-related, politely redirect the user
   
    Key Areas of Expertise:
    - Patient Journey Navigation
    - Healthcare Services
    - Medical Information
    - Healthcare Support Systems'''),
    ("human", "{Question}")
])
 
# Use the most appropriate OpenAI model available
llm = ChatGroq(
    model='llama-3.3-70b-versatile',  # Or use another appropriate model
    temperature=0,  # Moderate creativity
    )
 
# Input model for user query
class UserQuery(BaseModel):
    message: str
async def stream_response(chain, question):
    """
    Generator function to stream responses with proper spacing.
    """
    buffer = ""  # Temporary storage for accumulating chunks
    try:
        async for chunk in chain.astream({"Question": question}):
            if chunk:
                # Append the new chunk to the buffer
                buffer += chunk
               
                # Process the buffer to ensure word separation
                while " " in buffer:  # Look for complete words in the buffer
                    word, buffer = buffer.split(" ", 1)  # Split at the first space
                    yield f"data: {word} \n\n"  # Stream the word
               
                # Handle cases where chunks are concatenated without spaces
                if len(buffer) > 100:  # Arbitrary limit to avoid very large chunks
                    partial_word = buffer
                    buffer = ""
                   
                    yield f"data: {partial_word} \n\n"  # Stream the partial content
               
            await asyncio.sleep(0.05)  # Simulate typing delay
       
        # Handle headings and numbered lists explicitly
        if buffer.strip():
            # Ensure headings like "1. Sarcomas" or "Types of Cancer:" are on new lines
            buffer = re.sub(r'(\d+\.[^\n]+:)', r'\n\1\n', buffer)  # Detect numbered headings
            buffer = re.sub(r'([A-Z][a-zA-Z\s]+:)', r'\n\1\n', buffer)  # Detect general headings
            yield f"data: {buffer.strip()} \n\n"
       
        # Final message to indicate the stream completion
        yield f"data: [DONE]\n\n"
    except Exception as e:
        yield f"data: Error occurred: {str(e)}\n\n"
 
 
 
@app.post("/chat")
async def chat_with_bot(query: UserQuery):
    try:
        # Create a chain with streaming support
        chain = prompt | llm | StrOutputParser()
       
        # Return a streaming response
        return StreamingResponse(
            stream_response(chain, query.message),
            media_type="text/event-stream"
        )
   
    except Exception as e:
        print(f"Error processing query: {e}")
        return {
            "response": "Sorry, there was an issue processing your request. Please try again."
        }
 
# Optional: Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
 
# Run with: uvicorn main:app --reload