from __future__ import annotations
import sys, os
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass
# Add local libs directory where langchain_core was installed via --target
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../libs')))

import asyncio
import os
import warnings
import json
import re
from typing import List, TypedDict, Any, Dict, Literal

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from .classifier import run_classifier
from ingestion.config import Config
load_dotenv()

class Configuration(TypedDict):
    model: str
    temperature: float
    max_tokens: int

class MultiHopState(TypedDict):
    original_question: str
    decomposed_questions: List[str]
    intermediate_answers: List[str]
    retrieved_docs: List[Document]
    rewritten_query: str
    current_question_index: int
    final_answer: str
    trace: List[Dict[str, Any]]
    # Supervisor fields
    classification: str
    classification_reason: str
    worker_answer: str

class RAGSystem:
    def __init__(self):
        self._vectorstore = None
        self._retriever = None
    
    def _initialize_if_needed(self):
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                persist_directory=Config.CHROMA_PERSIST_DIR,
                embedding_function=OllamaEmbeddings(
                    base_url=Config.OLLAMA_BASE_URL,
                    model=Config.OLLAMA_EMBEDDING_MODEL,
                ),
                collection_name=Config.CHROMA_COLLECTION_NAME,
            )
            self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 5})

    async def retrieve_documents(self, query: str) -> List[Document]:
        try:
            self._initialize_if_needed()
            docs = await asyncio.to_thread(self._retriever.invoke, query)
            return docs
        except Exception as e:
            print(f"❌ RETRIEVAL ERROR: {type(e).__name__}: {e}")
            raise e

rag_system = RAGSystem()

async def decompose_question(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    print("\n" + "="*50)
    print("🧠 STEP 1: DECOMPOSING QUESTION")
    print("="*50)
    
    configuration = config.get("configurable", {})
    
    llm = await asyncio.to_thread(
        ChatOllama,
        model=configuration.get("model", "llama3:latest"),
        temperature=0
    )
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert at breaking down complex questions into a sequence of simpler questions.
        
        IMPORTANT: Only create sub-questions if the original question truly requires multiple steps. 
        If the question can be answered directly, create fewer, more focused sub-questions.

        CRITICAL: You must respond with ONLY a valid JSON object. Do not include any explanations, comments, or additional text. Return ONLY the JSON.

        The user will provide a complex question, and you must generate a JSON object containing a list of sub-questions.
        Each sub-question should be answerable on its own and build upon the previous one.

        Examples:
        
        Question: "Which movie, directed by the person who directed 'Jaws', stars an actor born in the same city as that director?"
        Output: {{"questions": ["Who directed the movie 'Jaws'?", "In which city was the director of 'Jaws' born?", "Which major actor was born in that same city?", "Which movie directed by the director of 'Jaws' stars that actor?"]}}

        Question: "In which year was the actor who played Harry Potter born?"
        Output: {{"questions": ["Who played the character Harry Potter in the movies?", "In what year was this actor born?"]}}

        Now, decompose the following question. Return ONLY the JSON object, no additional text:
        Question: {question}
        
        JSON Output:"""
    )
    
    parser = StrOutputParser()
    chain = prompt | llm | parser
    
    raw_result = await chain.ainvoke({"question": state["original_question"]})
    
    json_match = re.search(r'\{.*\}', raw_result.strip(), re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                result = json.loads(raw_result.strip())
            except json.JSONDecodeError:
                result = {"questions": [state["original_question"]]}
    else:
        result = {"questions": [state["original_question"]]}
    
    print(f"📝 Original: {state['original_question']}")
    print(f"🎯 Decomposed into {len(result['questions'])} sub-questions:")
    for i, q in enumerate(result['questions']):
        print(f"   {i+1}. {q}")
    
    trace_entry = {
        "step": "decompose_question",
        "input": {"original_question": state["original_question"]},
        "output": {"decomposed_questions": result["questions"]},
    }
    return {
        "decomposed_questions": result["questions"],
        "current_question_index": 0,
        "intermediate_answers": [],
        "trace": state.get("trace", []) + [trace_entry],
    }

async def rewrite_query(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    print("\n" + "="*50)
    print("🔄 QUERY REWRITING")
    print("="*50)
    
    current_index = state["current_question_index"]
    current_question = state["decomposed_questions"][current_index]
    current_q_num = current_index + 1
    
    print(f"📍 Question {current_q_num}: {current_question}")

    if state["intermediate_answers"]:
        context = "\n".join(f"- {ans}" for ans in state["intermediate_answers"])
        
        print(f"🧠 Previous answers available: {len(state['intermediate_answers'])}")
        
        configuration = config.get("configurable", {})
        
        llm = await asyncio.to_thread(
            ChatOllama,
            model=configuration.get("model", "llama3:latest"),
            temperature=0
        )
        
        prompt = ChatPromptTemplate.from_template(
            """You are a query rewriting expert. Given previous answers and the current question, create a more specific and searchable query.

            Previous answers: {context}
            Current question: {question}

            Instructions:
            1. If you have specific information from previous answers, use it
            2. If no relevant information exists, keep the original question or make it more general
            3. Do NOT create hypothetical or nonsensical queries
            4. Focus on finding factual information that can be retrieved

            Return only the rewritten query, nothing else.
            """
        )
        chain = prompt | llm | StrOutputParser()
        rewritten_question = await chain.ainvoke({"context": context, "question": current_question})
        
        print(f"🔄 Original: {current_question}")
        print(f"✨ Rewritten: {rewritten_question}")
        
        trace_entry = {
            "step": "rewrite_query",
            "input": {"question": current_question, "previous_answers": state["intermediate_answers"]},
            "output": {"rewritten_query": rewritten_question},
        }
        return {"rewritten_query": rewritten_question, "trace": state.get("trace", []) + [trace_entry]}
    else:
        print("ℹ️  No previous answers - using original question")
        print(f"➡️  Query: {current_question}")
        trace_entry = {
            "step": "rewrite_query",
            "input": {"question": current_question, "previous_answers": []},
            "output": {"rewritten_query": current_question},
        }
        return {"rewritten_query": current_question, "trace": state.get("trace", []) + [trace_entry]}

async def retrieve_documents(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    print("\n" + "="*50)
    print("🔍 STEP 2: RETRIEVING DOCUMENTS")
    print("="*50)
    
    query = state.get("rewritten_query", state["decomposed_questions"][0])
    current_q_num = state["current_question_index"] + 1
    
    print(f"📍 Question {current_q_num}: {state['decomposed_questions'][state['current_question_index']]}")
    print(f"🔍 Search Query: {query}")
    
    docs = await rag_system.retrieve_documents(query)
    
    print(f"📄 Retrieved: {len(docs)} documents")
    
    return {"retrieved_docs": docs}

async def generate_intermediate_answer(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    print("\n" + "="*50)
    print("💭 STEP 3: GENERATING ANSWER")
    print("="*50)
    
    context = "\n\n".join([doc.page_content for doc in state["retrieved_docs"]])
    current_question = state["rewritten_query"]  # Use rewritten_query instead of decomposed_questions
    current_q_num = state["current_question_index"] + 1
    
    print(f"❓ Answering Question {current_q_num}: {current_question}")
    
    if not context.strip():
        print("⚠️  No context available - cannot answer this question")
        answer = f"I don't have enough information in my knowledge base to answer: '{current_question}'"
    else:
        
        configuration = config.get("configurable", {})
        
        llm = await asyncio.to_thread(
            ChatOllama,
            model=configuration.get("model", "llama3:latest"),
            temperature=0
        )
        
        prompt = ChatPromptTemplate.from_template(
            """You are an expert answer generator. Based ONLY on the provided context, answer the following question.

            IMPORTANT INSTRUCTIONS:
            1. Extract the specific information requested from the context
            2. If you find the answer, provide it clearly and concisely
            3. If the answer is not in the context, say "The information is not available in the provided context"
            4. Do not make assumptions or add information not in the context

            Context:
            {context}

            Question: {question}

            Answer:"""
        )
        chain = prompt | llm | StrOutputParser()
        answer = await chain.ainvoke({"context": context, "question": current_question})
    
    print(f"✅ Answer: {answer}")
    
    new_answers = state["intermediate_answers"] + [f"The answer to '{current_question}' is: {answer}"]
    trace_entry = {
        "step": "generate_intermediate_answer",
        "input": {"rewritten_query": current_question},
        "output": {"answer": answer},
    }
    
    return {
        "intermediate_answers": new_answers,
        "current_question_index": state["current_question_index"] + 1,
        "trace": state.get("trace", []) + [trace_entry],
    }

async def generate_final_answer(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    print("\n" + "="*50)
    print("🎉 STEP 4: SYNTHESIZING FINAL ANSWER")
    print("="*50)
    
    context = "\n".join(state["intermediate_answers"])
    original_question = state["original_question"]
    
    print(f"📝 Original Question: {original_question}")
    print(f"🧩 Combining {len(state['intermediate_answers'])} intermediate answers...")

    configuration = config.get("configurable", {})
    
    llm = await asyncio.to_thread(
        ChatOllama,
        model=configuration.get("model", "llama3:latest"),
        temperature=configuration.get("temperature", 0.7),
        num_predict=configuration.get("max_tokens", 500)
    )
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are a master synthesist. You have been provided with a series of facts that were gathered to answer a complex user question.
        Combine these facts into a single, clear, and comprehensive answer.

        IMPORTANT: Do NOT include any introductory, explanatory, or concluding phrases (such as 'Here is my conclusion', 'Based on the context', 'In summary', etc.).
        Only provide the direct cumulative answer to the original question.

        Original Question: {original_question}
        Facts Gathered:
        {context}

        Final Answer (no extra text):
        """
    )
    chain = prompt | llm | StrOutputParser()
    final_answer = await chain.ainvoke({"context": context, "original_question": original_question})
    
    print(f"� Final Answer: {final_answer}")
    print("="*50)
    
    trace_entry = {
        "step": "generate_final_answer",
        "input": {"intermediate_answers": state["intermediate_answers"]},
        "output": {"final_answer": final_answer},
    }
    return {"final_answer": final_answer, "trace": state.get("trace", []) + [trace_entry]}

# ---------------------------------------------------------------------
# Supervisor classification and routing nodes
# ---------------------------------------------------------------------
async def supervisor_classify(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    """Classify the incoming user query using the lightweight Ollama model.

    Stores ``classification`` and ``classification_reason`` in the state.
    """
    question = state.get("original_question", "")
    result = run_classifier(question)
    return {
        "classification": result.classification,
        "classification_reason": result.reason,
        "trace": state.get("trace", []) + [{"step": "supervisor_classify", "input": {"question": question}, "output": {"classification": result.classification, "reason": result.reason}}],
    }

async def out_of_scope(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    msg = "I’m sorry, I can only answer factual, multi‑hop knowledge questions."
    return {"final_answer": msg, "trace": state.get("trace", []) + [{"step": "out_of_scope", "output": {"final_answer": msg}}]}

async def prompt_injection(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    msg = "I’m sorry, I cannot process that request."
    return {"final_answer": msg, "trace": state.get("trace", []) + [{"step": "prompt_injection", "output": {"final_answer": msg}}]}

async def invalid_query(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    msg = "I’m sorry, the query appears to be invalid. Please re‑phrase."
    return {"final_answer": msg, "trace": state.get("trace", []) + [{"step": "invalid_query", "output": {"final_answer": msg}}]}

async def needs_clarification(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    msg = "I need a little more clarification before I can answer. Could you elaborate?"
    return {"final_answer": msg, "trace": state.get("trace", []) + [{"step": "needs_clarification", "output": {"final_answer": msg}}]}

async def supervisor_finalize(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    # Preserve the worker's answer, sanity‑check, and expose as final_answer
    worker_ans = state.get("final_answer", "")
    if not worker_ans:
        worker_ans = "[No answer generated]"
    return {"final_answer": worker_ans, "worker_answer": worker_ans, "trace": state.get("trace", []) + [{"step": "supervisor_finalize", "output": {"final_answer": worker_ans}}]}

def route_based_on_classification(state: MultiHopState) -> Literal["decompose_question", "out_of_scope", "prompt_injection", "invalid_query", "needs_clarification"]:
    label = state.get("classification", "invalid")
    mapping = {
        "in_scope": "decompose_question",
        "out_of_scope": "out_of_scope",
        "prompt_injection": "prompt_injection",
        "invalid": "invalid_query",
        "needs_clarification": "needs_clarification",
    }
    return mapping.get(label, "invalid_query")



def should_continue(state: MultiHopState) -> Literal["rewrite_query", "generate_final_answer"]:
    current = state["current_question_index"]
    total = len(state["decomposed_questions"])
    
    if current < total:
        return "rewrite_query"
    return "generate_final_answer"

workflow = StateGraph(MultiHopState, config_schema=Configuration)

# Supervisor nodes
workflow.add_node("supervisor_classify", supervisor_classify)
workflow.add_node("out_of_scope", out_of_scope)
workflow.add_node("prompt_injection", prompt_injection)
workflow.add_node("invalid_query", invalid_query)
workflow.add_node("needs_clarification", needs_clarification)
workflow.add_node("supervisor_finalize", supervisor_finalize)

# Existing RAG nodes
workflow.add_node("decompose_question", decompose_question)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("generate_intermediate_answer", generate_intermediate_answer)
workflow.add_node("generate_final_answer", generate_final_answer)

# Entry point is supervisor classification
workflow.set_entry_point("supervisor_classify")

# Routing from classifier to appropriate next step
workflow.add_conditional_edges(
    "supervisor_classify",
    route_based_on_classification,
    {
        "decompose_question": "decompose_question",
        "out_of_scope": "out_of_scope",
        "prompt_injection": "prompt_injection",
        "invalid_query": "invalid_query",
        "needs_clarification": "needs_clarification",
    },
)

# Normal RAG flow
workflow.add_edge("decompose_question", "rewrite_query")
workflow.add_edge("rewrite_query", "retrieve_documents")
workflow.add_edge("retrieve_documents", "generate_intermediate_answer")
workflow.add_conditional_edges(
    "generate_intermediate_answer",
    should_continue,
    {
        "rewrite_query": "rewrite_query",
        "generate_final_answer": "generate_final_answer",
    },
)
# After final answer, go through supervisor finalization
workflow.add_edge("generate_final_answer", "supervisor_finalize")
workflow.add_edge("supervisor_finalize", END)

# Compile graph
graph = workflow.compile(name="multi-hop-agent")