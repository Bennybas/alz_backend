import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

import json
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
    allow_origins=["https://alzheimers-patient-journey-6uah.onrender.com"],
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

async def generate_response(question):
    """Generate comprehensive response with research and suggested questions"""
    # Check if healthcare-related
    is_healthcare = await prompt_classifier_chain.ainvoke({'question': question})
    
    if is_healthcare == 'Yes':
        # Perform research
        research_response = await final_research_report_chain.ainvoke({'question': question})
        # Ensure the research response is a string
        if isinstance(research_response, list):
            research_response = "\n".join(map(str, research_response))  # Combine list items into a single string
        elif not isinstance(research_response, str):
            research_response = str(research_response)  # Convert any non-string result to string

        # Generate suggested questions
        suggested_questions = await similar_prompt_generator_chain.ainvoke({'question': question})
        
        # Ensure suggested_questions is a list
        if not isinstance(suggested_questions, list):
            suggested_questions = (suggested_questions[1:-1]).split(",")
            for i in range(0,len(suggested_questions)):
                suggested_questions[i]=suggested_questions[i].replace("'","")


        return {
            'message': research_response,
            'suggested_questions': suggested_questions  
        }
    else:
        return {
            'message': "Hi I'm AIVY, here to help you with the Patient Journey",
            'suggested_questions': ['Explain barriers in Initial Assesment','Impact Measures of Diagnosis Stage']
        }

async def stream_response(question):
    """Stream response with error handling"""
    try:
        # Generate full response
        full_response = await generate_response(question)
        
        # Ensure message is a string and suggested_questions is a list
        full_response['message'] = str(full_response['message'])
        full_response['suggested_questions'] = list(full_response['suggested_questions'])
        
        # Sanitize and stream response
        sanitized_response = json.dumps(full_response)
        yield f"data: {sanitized_response}\n\n"
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        error_response = {
            'message': f"Error: {str(e)}"
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        yield "data: [DONE]\n\n"

@app.post("/chat")
async def chat_endpoint(query: UserQuery):
    return StreamingResponse(
        stream_response(query.message),
        media_type="text/event-stream"
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
