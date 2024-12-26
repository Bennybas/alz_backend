import os
import json
import asyncio
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

load_dotenv()

# API Key Validations
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
if not GROQ_API_KEY or not TAVILY_API_KEY:
    raise ValueError("GROQ_API_KEY and TAVILY_API_KEY must be set in environment variables")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tavily Search Function
def tavily_search_function(q):
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    search_results = tavily_client.search(q, max_results=5, include_answer=True)
    return search_results['results']

# LLM Configuration
groq_llm = ChatGroq(
    model='llama-3.3-70b-versatile', 
    temperature=0.2, 
    api_key=GROQ_API_KEY
)

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

# Chains
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
    RunnablePassthrough.assign(text=lambda x: tavily_search_function(x['question']))
    | (lambda x: [{'question': x['question'], 'text': i['content'], 'url': i['url']} for i in x['text']])
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
    Question:{question}''')
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
    ('system', '''Your task is to generate 3 similar questions based on the given question which should be in a list\nQuestion:{question}''')
])

similar_question_generator_llm = groq_llm.with_structured_output(SimilarQuestionGenerator)

similar_prompt_generator_chain = (
    SIMILAR_QUESTION_PROMPT 
    | similar_question_generator_llm 
    | (lambda x: str(x.response))
)

class UserQuery(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(query: UserQuery):
    try:
        # Add timeout for the entire operation
        async with asyncio.timeout(25):  # 25 seconds timeout
            return StreamingResponse(
                stream_response(query.message),
                media_type="text/event-stream"
            )
    except asyncio.TimeoutError:
        return StreamingResponse(
            stream_error_response("Request timed out"),
            media_type="text/event-stream"
        )
    except Exception as e:
        return StreamingResponse(
            stream_error_response(f"Error processing request: {str(e)}"),
            media_type="text/event-stream"
        )

async def stream_error_response(error_message: str):
    error_response = {
        'message': error_message,
        'suggested_questions': []
    }
    yield f"data: {json.dumps(error_response)}\n\n"
    yield "data: [DONE]\n\n"

async def generate_response(question):
    try:
        # Add timeout for classification
        async with asyncio.timeout(5):  # 5 second timeout
            is_healthcare = await prompt_classifier_chain.ainvoke({'question': question})
        
        if is_healthcare == 'Yes':
            try:
                # Add timeout for research response
                async with asyncio.timeout(15):  # 15 second timeout
                    research_response = await final_research_report_chain.ainvoke({'question': question})
                    
                    if isinstance(research_response, list):
                        research_response = "\n".join(map(str, research_response))
                    elif not isinstance(research_response, str):
                        research_response = str(research_response)

                # Add timeout for suggested questions
                async with asyncio.timeout(5):  # 5 second timeout
                    suggested_questions = await similar_prompt_generator_chain.ainvoke({'question': question})
                    
                    if not isinstance(suggested_questions, list):
                        suggested_questions = (suggested_questions[1:-1]).split(",")
                        suggested_questions = [q.strip().replace("'","") for q in suggested_questions]

                return {
                    'message': research_response,
                    'suggested_questions': suggested_questions
                }
            except asyncio.TimeoutError:
                # Fallback response if research times out
                return {
                    'message': "I apologize, but the research is taking longer than expected. Please try asking your question again or rephrase it.",
                    'suggested_questions': ['Could you rephrase your question?', 'Try breaking down your question into smaller parts']
                }
        else:
            return {
                'message': "Hi I'm AIVY, here to help you with the Patient Journey",
                'suggested_questions': ['Explain barriers in Initial Assessment', 'Impact Measures of Diagnosis Stage']
            }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Operation timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

async def stream_response(question):
    """Stream response with error handling"""
    try:
        # Generate full response with timeout
        async with asyncio.timeout(20):  # 20 second timeout
            full_response = await generate_response(question)
        
        # Ensure message is a string and suggested_questions is a list
        full_response['message'] = str(full_response['message'])
        full_response['suggested_questions'] = list(full_response['suggested_questions'])
        
        # Sanitize and stream response
        sanitized_response = json.dumps(full_response)
        yield f"data: {sanitized_response}\n\n"
        yield "data: [DONE]\n\n"
    
    except asyncio.TimeoutError:
        error_response = {
            'message': "Request timed out. Please try again.",
            'suggested_questions': []
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_response = {
            'message': f"Error: {str(e)}",
            'suggested_questions': []
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        yield "data: [DONE]\n\n"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
