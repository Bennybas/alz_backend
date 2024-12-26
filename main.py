import os
import json
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
from tavily import TavilyClient
from typing import Optional
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# API Key Validations with better error messages
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY must be set in environment variables")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY must be set in environment variables")

# LLM Configuration with error handling
try:
    groq_llm = ChatGroq(
        model='llama-3.3-70b-versatile',
        temperature=0.2,
        api_key=GROQ_API_KEY,
        max_retries=3,
        timeout=20
    )
except Exception as e:
    logger.error(f"Failed to initialize Groq LLM: {str(e)}")
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
    You MUST determine your own concrete and valid opinion based on the given information. Avoid general or vague responses.
    You must always give more importance to the latest information that is from the year 2024 in your answer.
    The response must not be in a report format.Do not mention where the information comes from or reference any context in your response.
    Avoid general or vague responses. Don't Include Question in your Response"""

# Chain Definitions
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

# Ensure tavily_search_function is async
async def tavily_search_function(q: str) -> list:
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        search_results = tavily_client.search(q, max_results=5, include_answer=True)
        return search_results.get('results', [])
    except Exception as e:
        logger.error(f"Tavily search error: {str(e)}")
        return []

multipage_qa_chain = (
    RunnablePassthrough.assign(text=lambda x: tavily_search_function(x['question']))
    | (lambda x: [{'question': x['question'], 'text': i['content'], 'url': i['url']} for i in await x['text']])
    | web_page_qa_chain.map()
)

def summary_list_exploder(l):
    if not isinstance(l, list):
        raise TypeError(f"Expected list, got {type(l)}")
    final_researched_content = "\n\n".join(map(str, l))
    return final_researched_content

complete_summarizer_chain = (
    text_to_searchquery_chain
    | multipage_qa_chain.map()
    | RunnableLambda(summary_list_exploder)
)

final_research_prompt = ChatPromptTemplate.from_messages([
    ('system', WRITER_SYSTEM_PROMPT),
    ('user', RESEARCH_REPORT_TEMPLATE)
])

final_research_report_chain = (
    RunnablePassthrough.assign(research_summary=complete_summarizer_chain)
    | final_research_prompt
    | groq_llm
    | StrOutputParser()
)

# Prompt Classifier
class PromptClassifier(BaseModel):
    response: str = Field(description="'Yes' if healthcare-related, 'No' otherwise")

PROMPT_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Classify the question: 
    Respond 'Yes' if related to healthcare, medicine, pharma, personal health
    Respond 'No' if unrelated.
    Question: {question}''')
])

prompt_classifier_llm = groq_llm.with_structured_output(PromptClassifier)

prompt_classifier_chain = (
    PROMPT_CLASSIFIER_PROMPT 
    | prompt_classifier_llm 
    | (lambda x: x.response)
)

# Similar Question Generator
class SimilarQuestionGenerator(BaseModel):
    response: list = Field(description='3 similar questions based on the given question should be in a list')

SIMILAR_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Your task is to generate 3 similar questions based on the given question which should be in a list\nQuestion: {question}''')
])

similar_question_generator_llm = groq_llm.with_structured_output(SimilarQuestionGenerator)

similar_prompt_generator_chain = (
    SIMILAR_QUESTION_PROMPT 
    | similar_question_generator_llm 
    | (lambda x: str(x.response))
)

# FastAPI App and Routes
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup")
    yield
    logger.info("Application shutdown")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    message: str

async def stream_error_response(error_message: str):
    error_response = {
        'message': error_message,
        'suggested_questions': []
    }
    yield f"data: {json.dumps(error_response)}\n\n"
    yield "data: [DONE]\n\n"

# Response Generator
async def generate_response(question: str) -> dict:
    try:
        is_healthcare = await prompt_classifier_chain.ainvoke({'question': question})
        logger.info(f"Classification result: {is_healthcare}")

        if is_healthcare == 'Yes':
            try:
                research_response = await final_research_report_chain.ainvoke({'question': question})
                
                if isinstance(research_response, list):
                    research_response = "\n".join(map(str, research_response))
                
                suggested_questions = await similar_prompt_generator_chain.ainvoke({'question': question})
                
                if isinstance(suggested_questions, str):
                    suggested_questions = json.loads(suggested_questions.replace("'", '"'))

                return {
                    'message': research_response,
                    'suggested_questions': suggested_questions
                }
            except asyncio.TimeoutError:
                logger.warning("Research generation timed out")
                return {
                    'message': "I apologize, but I'm having trouble processing your request. Please try again or rephrase your question.",
                    'suggested_questions': ['Could you rephrase your question?', 'Try asking a more specific question']
                }
        else:
            return {
                'message': "Hi I'm AIVY, here to help you with the Patient Journey. Please ask a healthcare-related question.",
                'suggested_questions': ['What are common barriers in Initial Assessment?', 'How is diagnosis typically conducted?']
            }
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def stream_response(question: str):
    try:
        response = await generate_response(question)
        
        sanitized_response = json.dumps({
            'message': str(response['message']),
            'suggested_questions': list(response.get('suggested_questions', []))
        })
        
        yield f"data: {sanitized_response}\n\n"
        yield "data: [DONE]\n\n"
        
    except asyncio.TimeoutError:
        logger.error("Stream response timed out")
        error_response = json.dumps({
            'message': "Request timed out. Please try again.",
            'suggested_questions': []
        })
        yield f"data: {error_response}\n\n"
        yield "data: [DONE]\n\n"

@app.post("/chat")
async def submit_query(user_query: UserQuery, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(stream_response, user_query.message)
        return StreamingResponse(stream_response(user_query.message), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Error in submit_query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
