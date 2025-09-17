from __future__ import annotations

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
from dotenv import load_dotenv
load_dotenv()

PERSIST_DIR = "./hotpot_QA_distractor_1k_ollama"

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

class RAGSystem:
    def __init__(self):
        self._vectorstore = None
        self._retriever = None
    
    def _initialize_if_needed(self):
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest")
            )
            self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 5})

    async def retrieve_documents(self, query: str) -> List[Document]:
        try:
            self._initialize_if_needed()
            docs = await asyncio.to_thread(self._retriever.invoke, query)
            return docs
        except Exception as e:
            print(f"❌ RETRIEVAL ERROR: {e}")
            return []

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
    
    # Clean the response to extract JSON
    # Remove any text before the first { and after the last }
    json_match = re.search(r'\{.*\}', raw_result.strip(), re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: try to parse the entire response as JSON
            try:
                result = json.loads(raw_result.strip())
            except json.JSONDecodeError:
                # Last resort: create a simple single-question response
                result = {"questions": [state["original_question"]]}
    else:
        # If no JSON found, treat the whole question as one question
        result = {"questions": [state["original_question"]]}
    
    print(f"📝 Original: {state['original_question']}")
    print(f"🎯 Decomposed into {len(result['questions'])} sub-questions:")
    for i, q in enumerate(result['questions']):
        print(f"   {i+1}. {q}")
    
    return {
        "decomposed_questions": result["questions"],
        "current_question_index": 0,
        "intermediate_answers": [],
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
        print(f"📝 Context: {context[:1000]}..." if len(context) > 200 else f"📝 Context: {context}")
        
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
        
        return {"rewritten_query": rewritten_question}
    else:
        print("ℹ️  No previous answers - using original question")
        print(f"➡️  Query: {current_question}")
        
    return {"rewritten_query": current_question}

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
    if docs:
        print(f"📝� Preview: {docs[0].page_content[:1000]}...")
    
    return {"retrieved_docs": docs}

async def generate_intermediate_answer(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    print("\n" + "="*50)
    print("💭 STEP 3: GENERATING ANSWER")
    print("="*50)
    
    context = "\n\n".join([doc.page_content for doc in state["retrieved_docs"]])
    current_question = state["rewritten_query"]  # Use rewritten_query instead of decomposed_questions
    current_q_num = state["current_question_index"] + 1
    
    print(f"❓ Answering Question {current_q_num}: {current_question}")
    print(f"📄 Context: {len(context)} characters")
    
    if not context.strip():
        print("⚠️  No context available - cannot answer this question")
        answer = f"I don't have enough information in my knowledge base to answer: '{current_question}'"
    else:
        print(f"📝 Context preview: {context[:1000]}...")
        
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
    
    return {
        "intermediate_answers": new_answers,
        "current_question_index": state["current_question_index"] + 1,
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
        """You are a master synthesist. You have been provided with a series of facts that were gathered
        to answer a complex user question. Combine these facts into a single, clear, and comprehensive answer.

        Original Question: {original_question}

        Facts Gathered:
        {context}

        Final Answer:
        """
    )
    chain = prompt | llm | StrOutputParser()
    final_answer = await chain.ainvoke({"context": context, "original_question": original_question})
    
    print(f"� Final Answer: {final_answer}")
    print("="*50)
    
    return {"final_answer": final_answer}

def should_continue(state: MultiHopState) -> Literal["rewrite_query", "generate_final_answer"]:
    current = state["current_question_index"]
    total = len(state["decomposed_questions"])
    
    if current < total:
        return "rewrite_query"
    return "generate_final_answer"

workflow = StateGraph(MultiHopState, config_schema=Configuration)

workflow.add_node("decompose_question", decompose_question)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("generate_intermediate_answer", generate_intermediate_answer)
workflow.add_node("generate_final_answer", generate_final_answer)

workflow.set_entry_point("decompose_question")
workflow.add_edge("decompose_question", "rewrite_query")
workflow.add_edge("rewrite_query", "retrieve_documents")
workflow.add_edge("retrieve_documents", "generate_intermediate_answer")

workflow.add_conditional_edges(
    "generate_intermediate_answer",
    should_continue,
    {
        "rewrite_query": "rewrite_query",
        "generate_final_answer": "generate_final_answer"
    }
)
workflow.add_edge("generate_final_answer", END)

graph = workflow.compile(name="multi-hop-agent")