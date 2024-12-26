import os
import json
import asyncio
import logging
import time
import datetime
from typing import Dict, Any, Optional, List
import aiohttp
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
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

# Environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
APP_URL = os.getenv('APP_URL', 'https://alzheimers-patient-journey-6uah.onrender.com/')

# Validate required environment variables
if not GROQ_API_KEY or not TAVILY_API_KEY:
    raise ValueError("GROQ_API_KEY and TAVILY_API_KEY must be set in environment variables")

# Global state management
class GlobalState:
    def __init__(self):
        self.groq_llm: Optional[ChatGroq] = None
        self.tavily_client: Optional[TavilyClient] = None
        self.prompt_classifier_llm = None
        self.similar_question_generator_llm = None
        self.is_initialized: bool = False
        self.initialization_lock: asyncio.Lock = asyncio.Lock()
        self.retry_count: int = 0
        self.max_retries: int = 3
        self.response_cache: Dict[str, Any] = {}
        self.cache_expiry: int = 3600  # 1 hour

global_state = GlobalState()

# Pydantic models
class UserQuery(BaseModel):
    message: str

class PromptClassifier(BaseModel):
    response: str = Field(description="'Yes' if healthcare-related, 'No' otherwise")

class SimilarQuestionGenerator(BaseModel):
    response: List[str] = Field(description='3 similar questions based on the given question')

# Prompt templates
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

WRITER_SYSTEM_PROMPT = """You are an AI critical thinker research assistant. 
Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."""

RESEARCH_REPORT_TEMPLATE = """Information:
    --------
    {research_summary}
    --------
    Using the above information, answer the following question or topic: "{question}" in a concise manner -- \
    The answer should focus on the answer to the question, be well-structured, informative, \
    in-depth, with facts and numbers if available.
    You should strive to write the answer as concise as possible, using all relevant and necessary information provided.
    Write the answer with markdown syntax.
    You MUST determine your own concrete and valid opinion based on the given information. 
    Avoid general or vague responses. You must always give more importance to the latest information that is from the year 2024 in your answer.
    The response must not be in a report format. Do not mention where the information comes from or reference any context in your response.
    Dont Include Question in your Response"""

PROMPT_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Classify the question: 
    Respond 'Yes' if related to healthcare, medicine, pharma, personal health
    Respond 'No' if unrelated.
    Question:{question}''')
])

SIMILAR_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Your task is to generate 3 similar questions based on the given question which should be in a list
    Question:{question}''')
])

final_research_prompt = ChatPromptTemplate.from_messages([
    ('system', WRITER_SYSTEM_PROMPT),
    ('user', RESEARCH_REPORT_TEMPLATE)
])

# Helper functions
def summary_list_exploder(l: List[str]) -> str:
    if not isinstance(l, list):
        raise TypeError(f"Expected list, got {type(l)}")
    return "\n\n".join(map(str, l))

async def tavily_search_function(q: str) -> List[Dict[str, Any]]:
    cache_key = f"tavily_search_{q}"
    
    if cache_key in global_state.response_cache:
        logger.info("Returning cached search results")
        return global_state.response_cache[cache_key]
    
    try:
        search_results = await asyncio.to_thread(
            global_state.tavily_client.search,
            q,
            max_results=5,
            include_answer=True
        )
        global_state.response_cache[cache_key] = search_results['results']
        return search_results['results']
    except Exception as e:
        logger.error(f"Tavily search error: {str(e)}")
        raise

# Initialize chains
async def initialize_chains():
    global text_to_searchquery_chain, web_page_qa_chain, multipage_qa_chain
    global complete_summarizer_chain, final_research_report_chain
    global prompt_classifier_chain, similar_prompt_generator_chain

    async def async_search_wrapper(x):
        results = await tavily_search_function(x['question'])
        return results

    text_to_searchquery_chain = (
        TEXT_TO_SEARCHQUERY_PROMPT 
        | global_state.groq_llm 
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
        | global_state.groq_llm
        | StrOutputParser()
    )

    prompt_classifier_chain = (
        PROMPT_CLASSIFIER_PROMPT 
        | global_state.prompt_classifier_llm 
        | (lambda x: x.response)
    )

    similar_prompt_generator_chain = (
        SIMILAR_QUESTION_PROMPT 
        | global_state.similar_question_generator_llm 
        | (lambda x: str(x.response))
    )

# Service initialization
async def initialize_services():
    try:
        global_state.groq_llm = ChatGroq(
            model='llama-3.3-70b-versatile',
            temperature=0.2,
            api_key=GROQ_API_KEY
        )
        global_state.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        
        # Initialize LLM variants
        global_state.prompt_classifier_llm = global_state.groq_llm.with_structured_output(PromptClassifier)
        global_state.similar_question_generator_llm = global_state.groq_llm.with_structured_output(SimilarQuestionGenerator)
        
        await initialize_chains()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Service initialization error: {str(e)}")
        raise

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    retry_delay = 1
    while not global_state.is_initialized and global_state.retry_count < global_state.max_retries:
        try:
            async with global_state.initialization_lock:
                if not global_state.is_initialized:
                    await initialize_services()
                    global_state.is_initialized = True
        except Exception as e:
            global_state.retry_count += 1
            logger.error(f"Initialization attempt {global_state.retry_count} failed: {str(e)}")
            await asyncio.sleep(retry_delay)
            retry_delay *= 2
    
    yield
    
    # Cleanup
    global_state.is_initialized = False
    logger.info("Application shutdown complete")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request, call_next):
    try:
        if not global_state.is_initialized:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "Service is initializing. Please try again in a few moments."}
            )
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred. Please try again."}
        )

# Response generation
async def generate_response(question: str):
    cache_key = f"response_{question}"
    
    if cache_key in global_state.response_cache:
        logger.info("Returning cached response")
        return global_state.response_cache[cache_key]
    
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
                        suggested_questions = eval(suggested_questions)

                response = {
                    'message': research_response,
                    'suggested_questions': suggested_questions
                }
                
                global_state.response_cache[cache_key] = response
                return response
                
            except asyncio.TimeoutError:
                return {
                    'message': "I apologize, but the research is taking longer than expected. Please try again or rephrase your question.",
                    'suggested_questions': ['Explain barriers in Initial Assessment', 'Impact Measures of Diagnosis Stage']
                }
        else:
            return {
                'message': "Hi I'm AIVY, here to help you with the Patient Journey",
                'suggested_questions': ['Explain barriers in Initial Assessment', 'Impact Measures of Diagnosis Stage']
            }
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Streaming response handlers
async def stream_error_response(error_message: str):
    error_response = {
        'message': error_message,
        'suggested_questions': []
    }
    yield f"data: {json.dumps(error_response)}\n\n"
    yield "data: [DONE]\n\n"

async def stream_response(question: str):
    try:
        async with asyncio.timeout(20):
            full_response = await generate_response(question)
            yield f"data: {json.dumps(full_response)}\n\n"
            yield "data: [DONE]\n\n"
    except asyncio.TimeoutError:
        logger.error("Response streaming timed out")
        async for chunk in stream_error_response("Request timed out. Please try again."):
            yield chunk
    except Exception as e:
        logger.error(f"Error streaming response: {str(e)}")
        async for chunk in stream_error_response(f"Error processing request: {str(e)}"):
            yield chunk

# Endpoints
@app.post("/chat")
async def chat_endpoint(query: UserQuery):
    if not global_state.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is still initializing. Please try again in a few moments."
        )
    
    try:
        logger.info(f"Received chat request: {query.message[:50]}...")
        return StreamingResponse(
            stream_response(query.message),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return StreamingResponse(
            stream_error_response(str(e)),
            media_type="text/event-stream"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if global_state.is_initialized else "initializing",
        "timestamp": datetime.datetime.now().isoformat(),
        "initialization_attempts": global_state.retry_count,
        "services_ready": global_state.is_initialized
    }

# Cache cleanup
async def cleanup_cache():
    while True:
        try:
            current_time = time.time()
            keys_to_remove = []
            for key in global_state.response_cache:
                if current_time - global_state.response_cache[key].get('timestamp', 0) > global_state.cache_expiry:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del global_state.response_cache[key]
            
            logger.info(f"Cleaned up {len(keys_to_remove)} cached items")
        except Exception as e:
            logger.error(f"Cache cleanup error: {str(e)}")
        await asyncio.sleep(3600)  # Run cache cleanup every hour

# Run cache cleanup as a background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_cache())
