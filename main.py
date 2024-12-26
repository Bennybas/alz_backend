import os
import json
import asyncio
import logging
import time
import datetime
from typing import Dict, Any
import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Key Validations
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
APP_URL = os.getenv('APP_URL', 'https://alzheimers-patient-journey-6uah.onrender.com/')

if not GROQ_API_KEY or not TAVILY_API_KEY:
    raise ValueError("GROQ_API_KEY and TAVILY_API_KEY must be set in environment variables")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for clients
groq_llm = None
tavily_client = None

# Cache for storing recent results
response_cache: Dict[str, Any] = {}
CACHE_EXPIRY = 3600  # 1 hour in seconds

# Tavily Search Function with caching
async def tavily_search_function(q: str):
    cache_key = f"tavily_search_{q}"
    if cache_key in response_cache:
        logger.info("Returning cached search results")
        return response_cache[cache_key]
    
    try:
        # Convert synchronous Tavily call to async context
        search_results = await asyncio.to_thread(
            tavily_client.search,
            q,
            max_results=5,
            include_answer=True
        )
        response_cache[cache_key] = search_results['results']
        return search_results['results']
    except Exception as e:
        logger.error(f"Tavily search error: {str(e)}")
        raise

# Prompt Templates
TEXT_TO_SEARCHQUERY_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Write 3 google search queries to search online that form an 
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            [query 1,query 2,query 3].''')
])

WEB_PAGE_QA_PROMPT = ChatPromptTemplate.from_messages([
    ('system', """{text}
        -----------
        Using the above text, answer in short the following question:
        > {question}
        -----------
        if the question cannot be answered using the text, imply summarize the text. 
        Include all factual information, numbers, stats etc if available.""")
])

WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."

RESEARCH_REPORT_TEMPLATE = """Information:
    --------
    {research_summary}
    --------
    Using the above information, answer the following question or topic: "{question}" in a concise manner -- \
    The answer should focus on the answer to the question, be well-structured, informative, \
    in-depth, with facts and numbers if available.
    You should strive to write the answer as concise as possible, using all relevant and necessary information provided.
    Write the answer with markdown syntax.
    You MUST determine your own concrete and valid opinion based on the given information. Avoid general or vague responses.You must always give more importance to the latest information that is from the year 2024 in your answer.The response must not be in a report format.Do not mention where the information comes from or reference any context in your response.Avoid general or vague responses
    Dont Include Question in your Response"""

# Define final_research_prompt here
final_research_prompt = ChatPromptTemplate.from_messages([
    ('system', WRITER_SYSTEM_PROMPT),
    ('user', RESEARCH_REPORT_TEMPLATE)
])

# Initialize chains during startup
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing server components...")
    global groq_llm, tavily_client, prompt_classifier_llm, similar_question_generator_llm
    
    try:
        groq_llm = ChatGroq(
            model='llama-3.3-70b-versatile',
            temperature=0.2,
            api_key=GROQ_API_KEY
        )
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        
        # Initialize classifier and generator
        prompt_classifier_llm = groq_llm.with_structured_output(PromptClassifier)
        similar_question_generator_llm = groq_llm.with_structured_output(SimilarQuestionGenerator)
        
        # Initialize chains
        await initialize_chains()
        
        logger.info("Server components initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

# Initialize chains
async def initialize_chains():
    global text_to_searchquery_chain, web_page_qa_chain, multipage_qa_chain, complete_summarizer_chain
    global final_research_report_chain, prompt_classifier_chain, similar_prompt_generator_chain

    # Wrap synchronous tavily search in async
    async def async_search_wrapper(x):
        results = await tavily_search_function(x['question'])
        return results

    text_to_searchquery_chain = (
        TEXT_TO_SEARCHQUERY_PROMPT 
        | groq_llm 
        | StrOutputParser() 
        | (lambda x: x[1:-1].replace('"',"").split(",")) 
        | (lambda x: [{'question': i} for i in x])
    )

    web_page_qa_chain = (
        RunnablePassthrough.assign(summary=lambda x: x['text']) 
        | (lambda x: f"Summary:{x['summary']}\nURL:{x['url']}")
    )

    multipage_qa_chain = (
        RunnablePassthrough.assign(text=async_search_wrapper)
        | (lambda x: [{'question': x['question'], 'text': i['content'], 'url': i['url']} for i in x['text']])
        | web_page_qa_chain.map()
    )

    complete_summarizer_chain = (
        text_to_searchquery_chain
        | multipage_qa_chain.map()
        | RunnableLambda(summary_list_exploder)
    )

    final_research_report_chain = (
        RunnablePassthrough.assign(research_summary=complete_summarizer_chain)
        | final_research_prompt
        | groq_llm
        | StrOutputParser()
    )
    
    prompt_classifier_chain = (
        PROMPT_CLASSIFIER_PROMPT 
        | prompt_classifier_llm 
        | (lambda x: x.response)
    )
    
    similar_prompt_generator_chain = (
        SIMILAR_QUESTION_PROMPT 
        | similar_question_generator_llm 
        | (lambda x: str(x.response))
    )

# Helper function for summary list
def summary_list_exploder(l):
    if not isinstance(l, list):
        raise TypeError(f"Expected list, got {type(l)}")
    
    final_researched_content = "\n\n".join(map(str, l))
    return final_researched_content

# Prompt Classifier
class PromptClassifier(BaseModel):
    response: str = Field(description="'Yes' if healthcare-related, 'No' otherwise")

PROMPT_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Classify the question: 
    Respond 'Yes' if related to healthcare, medicine, pharma, personal health
    Respond 'No' if unrelated.
    Question:{question}''')
])

prompt_classifier_llm = None  # Will be initialized during startup

# Similar Question Generator
class SimilarQuestionGenerator(BaseModel):
    response: list = Field(description='3 similar questions based on the given question should be in a list')

SIMILAR_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Your task is to generate 3 similar questions based on the given question which should be in a list\nQuestion:{question}''')
])

# Keep-alive mechanism
async def keep_alive():
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(f"{APP_URL}/health") as response:
                    logger.info(f"Keep-alive ping sent. Status: {response.status}")
            except Exception as e:
                logger.error(f"Keep-alive ping failed: {str(e)}")
            await asyncio.sleep(14 * 60)  # 14 minutes

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

# Request model
class UserQuery(BaseModel):
    message: str

# Enhanced error handling for stream response
async def stream_error_response(error_message: str):
    error_response = {
        'message': error_message,
        'suggested_questions': []
    }
    yield f"data: {json.dumps(error_response)}\n\n"
    yield "data: [DONE]\n\n"

# Enhanced response generation
async def generate_response(question: str):
    cache_key = f"response_{question}"
    
    if cache_key in response_cache:
        logger.info("Returning cached response")
        return response_cache[cache_key]
    
    try:
        async with asyncio.timeout(5):
            is_healthcare = await prompt_classifier_chain.ainvoke({'question': question})
        
        if is_healthcare == 'Yes':
            try:
                async with asyncio.timeout(15):
                    research_response = await final_research_report_chain.ainvoke({'question': question})
                    
                    if isinstance(research_response, list):
                        research_response = "\n".join(map(str, research_response))
                    elif not isinstance(research_response, str):
                        research_response = str(research_response)

                async with asyncio.timeout(5):
                    suggested_questions = await similar_prompt_generator_chain.ainvoke({'question': question})
                    
                    if not isinstance(suggested_questions, list):
                        suggested_questions = eval(suggested_questions)  # Safely evaluate string representation of list

                response = {
                    'message': research_response,
                    'suggested_questions': suggested_questions
                }
                
                response_cache[cache_key] = response
                return response
                
            except asyncio.TimeoutError:
                return {
                    'message': "I apologize, but the research is taking longer than expected. Please try again or rephrase your question.",
                    'suggested_questions': ['Could you rephrase your question?', 'Try breaking down your question into smaller parts']
                }
        else:
            return {
                'message': "Hi I'm AIVY, here to help you with the Patient Journey",
                'suggested_questions': ['Explain barriers in Initial Assessment', 'Impact Measures of Diagnosis Stage']
            }
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced streaming response
async def stream_response(question: str):
    try:
        async with asyncio.timeout(20):
            full_response = await generate_response(question)
            
            # Ensure proper response format
            full_response['message'] = str(full_response['message'])
            full_response['suggested_questions'] = list(full_response['suggested_questions'])
            
            # Stream the response
            yield f"data: {json.dumps(full_response)}\n\n"
            yield "data: [DONE]\n\n"
            
    except asyncio.TimeoutError:
        logger.error("Response streaming timed out")
        error_response = {
            'message': "Request timed out. Please try again.",
            'suggested_questions': []
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error streaming response: {str(e)}")
        error_response = {
            'message': f"Error: {str(e)}",
            'suggested_questions': []
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        yield "data: [DONE]\n\n"

# Enhanced chat endpoint
@app.post("/chat")
async def chat_endpoint(query: UserQuery):
    try:
        logger.info(f"Received chat request: {query.message[:50]}...")
        
        async with asyncio.timeout(20):
            return StreamingResponse(
                stream_response(query.message),
                media_type="text/event-stream"
            )
    except asyncio.TimeoutError:
        logger.error("Chat endpoint timed out")
        return StreamingResponse(
            stream_error_response("Request timed out"),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return StreamingResponse(
            stream_error_response(f"Error processing request: {str(e)}"),
            media_type="text/event-stream"
        )

# Periodic cache cleanup
async def cleanup_cache():
    while True:
        try:
            current_time = time.time()
            keys_to_remove = [
                key for key, (timestamp, _) in response_cache.items()
                if current_time - timestamp > CACHE_EXPIRY
            ]
            for key in keys_to_remove:
                del response_cache[key]
            logger.info(f"Cleaned up {len(keys_to_remove)} cached items")
        except Exception as e:
            logger.error(f"Cache cleanup error: {str(e)}")
        await asyncio.sleep(3600)  # Run every hour

if __name__ == "__main__":
    import uvicorn
    
    # Start background tasks
    asyncio.create_task(keep_alive())
    asyncio.create_task(cleanup_cache())
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
