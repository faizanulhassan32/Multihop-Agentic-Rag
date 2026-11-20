from __future__ import annotations
import sys, os
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

# Add local libs directory where dependencies were installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../libs')))

import asyncio
import warnings
import json
import re
from difflib import SequenceMatcher
from typing import List, TypedDict, Any, Dict, Literal, Set

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from .classifier import arun_classifier
from ingestion.config import Config
from pydantic import BaseModel, ConfigDict, Field, ValidationError
load_dotenv()

class Configuration(TypedDict):
    model: str
    temperature: float
    max_tokens: int

class MultiHopState(TypedDict):
    original_question: str
    answer_target: Dict[str, Any]
    decomposed_questions: List[str]
    retrieved_docs: List[Document]
    retrieved_evidence: str
    recovery_evidence: str
    recovery_docs: List[Document]
    retrieval_matches: Dict[str, List[str]]
    followup_rounds: int
    retrieval_queries: List[str]
    followup_queries: List[str]
    evidence_sufficient: bool
    evidence_relevant: bool
    established_entities: List[str]
    established_facts: List[str]
    established_relationships: List[str]
    evidence_gaps: List[Dict[str, str]]
    final_answer: str
    trace: List[Dict[str, Any]]
    # Supervisor fields
    classification: str
    classification_reason: str
    worker_answer: str


class EvidenceGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str = ""
    missing_fact: str = ""
    missing_relationship: str = ""
    retrieval_query: str


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevant_document_ids: List[int]
    sufficient: bool = False
    supported_relationship_ids: List[str] = Field(default_factory=list)
    answer_attribute_supported: bool = False
    resolved_answer_entity: str = ""
    answer_value: str = ""
    established_entities: List[str] = Field(default_factory=list)
    established_facts: List[str] = Field(default_factory=list)
    established_relationships: List[str] = Field(default_factory=list)
    gaps: List[EvidenceGap] = Field(default_factory=list)


class RecoveryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: List[str] = Field(default_factory=list)


class RequiredRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    subject: str
    predicate: str
    object: str
    direction: str
    evidence_requirement: str
    retrieval_query: str


class AnswerTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_entity: str
    requested_attribute: str
    required_relationships: List[RequiredRelationship] = Field(default_factory=list)
    evidence_requirements: List[str] = Field(default_factory=list)
    resolved_answer_entity: str = ""
    verified_answer_value: str = ""


# Reusable shared model and vectorstore instances (avoids repeated re-instantiation)
_generator_llm: ChatOllama | None = None


async def _ainvoke_with_model_timeout(runnable: Any, input: Any) -> Any:
    return await asyncio.wait_for(
        runnable.ainvoke(input),
        timeout=Config.MODEL_TIMEOUT,
    )


def _query_preserves_question_relations(
    question: str,
    query: str,
    established_relationships: List[str] | None = None,
) -> bool:
    question_words = set(re.findall(r"[a-z0-9]+", question.casefold()))
    query_words = set(re.findall(r"[a-z0-9]+", query.casefold()))
    established_words = [
        set(re.findall(r"[a-z0-9]+", relation.casefold()))
        for relation in (established_relationships or [])
    ]
    required_groups = [
        group for group in _QUESTION_RELATION_GROUPS
        if question_words & group
    ]
    return all(
        bool(query_words & group)
        or any(bool(relationship_words & group) for relationship_words in established_words)
        for group in required_groups
    )


def _evidence_supports_question_relations(
    question: str,
    docs: List[Document],
) -> bool:
    evidence = "\n".join(doc.page_content for doc in docs)
    return _query_preserves_question_relations(question, evidence)


def get_generator_llm(model: str = "llama3:latest", temperature: float = 0.0) -> ChatOllama:
    global _generator_llm
    if _generator_llm is None or _generator_llm.model != model:
        _generator_llm = ChatOllama(
            base_url=Config.OLLAMA_BASE_URL,
            model=model,
            temperature=temperature,
        )
    return _generator_llm

class RAGSystem:
    def __init__(self):
        self._vectorstore = None
        self._retriever = None
        self._semaphore = asyncio.Semaphore(max(1, Config.MAX_RETRIEVAL_CONCURRENCY))
    
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
            self._retriever = self._vectorstore.as_retriever(
                search_kwargs={"k": Config.RETRIEVAL_TOP_K}
            )

    async def retrieve_documents(self, query: str) -> List[Document]:
        async with self._semaphore:
            try:
                self._initialize_if_needed()
                docs = await asyncio.to_thread(self._retriever.invoke, query)
                return docs
            except Exception as e:
                print(f"❌ RETRIEVAL ERROR: {type(e).__name__}: {e}")
                raise e

rag_system = RAGSystem()

# Helper for deduplicating documents safely by title and content
def document_key(doc: Document) -> str:
    content_sample = doc.page_content.strip()[:100]
    title = str(doc.metadata.get("title", ""))
    return f"{title}::{content_sample}"


def deduplicate_documents(docs: List[Document]) -> List[Document]:
    seen: Set[str] = set()
    unique_docs: List[Document] = []
    for doc in docs:
        key = document_key(doc)
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs


def format_retrieval_evidence(
    docs: List[Document],
    retrieval_matches: Dict[str, List[str]],
) -> str:
    sections = []
    for doc in docs:
        queries = retrieval_matches.get(document_key(doc), [])
        query_label = ", ".join(queries) if queries else "initial retrieval"
        sections.append(
            f"Retrieved for query: {query_label}\n"
            f"Source: {doc.metadata.get('title', 'Document')}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(sections)


def format_assessment_evidence(
    docs: List[Document],
    retrieval_matches: Dict[str, List[str]],
) -> str:
    sections = []
    for index, doc in enumerate(docs):
        queries = retrieval_matches.get(document_key(doc), [])
        query_label = ", ".join(queries) if queries else "initial retrieval"
        sections.append(
            f"Document ID: {index}\n"
            f"Retrieved for query: {query_label}\n"
            f"Source: {doc.metadata.get('title', 'Document')}\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n".join(sections)


def _named_entity_phrases(text: str) -> List[str]:
    entities = re.findall(
        r"\b(?:[A-Z][A-Za-z0-9'.’\-]*\s+){1,}[A-Z][A-Za-z0-9'.’\-]*\b",
        text,
    )
    return [re.sub(r"['’]s$", "", entity) for entity in entities]


def _normalized_entity_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold().replace("'s", "").replace("’s", "")))


def _text_mentions_entity(text: str, entity: str) -> bool:
    normalized_text = f" {_normalized_entity_text(text)} "
    normalized_entity = _normalized_entity_text(entity)
    return bool(normalized_entity) and f" {normalized_entity} " in normalized_text


def _grounded_entities(
    entities: List[str],
    docs: List[Document],
) -> List[str]:
    source_text = "\n".join(
        f"{doc.metadata.get('title', '')}\n{doc.page_content}"
        for doc in docs
    )
    source_phrases = {
        _normalized_entity_text(entity)
        for entity in _named_entity_phrases(source_text)
    }
    source_words = set(re.findall(r"[a-z0-9]+", source_text.casefold()))
    grounded: List[str] = []
    seen: Set[str] = set()
    for entity in entities:
        normalized = _normalized_entity_text(entity)
        source_match = (
            normalized in source_phrases
            if len(normalized.split()) > 1
            else normalized in source_words
        )
        if normalized and normalized not in seen and source_match:
            grounded.append(entity.strip())
            seen.add(normalized)
    return grounded


def _query_uses_only_grounded_entities(query: str, grounded_entities: List[str]) -> bool:
    known_entities = {
        _normalized_entity_text(entity)
        for entity in grounded_entities
        if _normalized_entity_text(entity)
    }
    return all(
        _normalized_entity_text(entity) in known_entities
        for entity in _named_entity_phrases(query)
    )


def _rewrite_question_with_grounded_aliases(
    question: str,
    docs: List[Document],
) -> List[str]:
    rewritten_queries: List[str] = []
    question_entities = _named_entity_phrases(question)
    alias_pattern = re.compile(
        r"\b(?:also known as|better known as|better known by|known as|stage name|pseudonym|alias)\b",
        re.IGNORECASE,
    )
    for doc in docs:
        title_entities = _named_entity_phrases(str(doc.metadata.get("title", "")))
        for sentence in re.split(r"(?<=[.!?])\s+", doc.page_content):
            if not alias_pattern.search(sentence):
                continue
            for subject in question_entities:
                if not _text_mentions_entity(sentence, subject):
                    continue
                aliases = [
                    entity
                    for entity in title_entities + _named_entity_phrases(sentence)
                    if _normalized_entity_text(entity) != _normalized_entity_text(subject)
                    and _text_mentions_entity(sentence, entity)
                ]
                for alias in aliases:
                    subject_pattern = re.compile(
                        rf"(?<!\w){re.escape(subject)}(?P<possessive>['’]s)?(?!\w)",
                        re.IGNORECASE,
                    )
                    rewritten = subject_pattern.sub(
                        lambda match: alias + (match.group("possessive") or ""),
                        question,
                    )
                    if rewritten != question and rewritten not in rewritten_queries:
                        rewritten_queries.append(rewritten)
    return rewritten_queries


def _alias_bridge_documents(
    question: str,
    rewritten_questions: List[str],
    docs: List[Document],
) -> List[Document]:
    original_entities = {
        _normalized_entity_text(entity)
        for entity in _named_entity_phrases(question)
    }
    alias_entities = {
        _normalized_entity_text(entity)
        for rewritten in rewritten_questions
        for entity in _named_entity_phrases(rewritten)
        if _normalized_entity_text(entity) not in original_entities
    }
    if not alias_entities:
        return []

    alias_pattern = re.compile(
        r"\b(?:also known as|better known as|better known by|known as|stage name|pseudonym|alias)\b",
        re.IGNORECASE,
    )
    bridge_docs = []
    for doc in docs:
        for sentence in re.split(r"(?<=[.!?])\s+", doc.page_content):
            if (
                alias_pattern.search(sentence)
                and any(
                    _text_mentions_entity(sentence, entity)
                    for entity in _named_entity_phrases(question)
                )
                and any(
                    _normalized_entity_text(entity) in alias_entities
                    and _text_mentions_entity(sentence, entity)
                    for entity in _named_entity_phrases(sentence)
                )
            ):
                bridge_docs.append(doc)
                break
    return bridge_docs


def requested_answer_type(question: str) -> str:
    if re.search(r"\bwho(?:m)?\b", question, re.IGNORECASE):
        return "person or entity name"
    if re.search(r"\bnationality\b", question, re.IGNORECASE):
        return "nationality"
    return "the exact fact or property requested"


_QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "did", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "the", "to", "was",
    "were", "what", "when", "where", "which", "who", "whom", "with",
}
_QUERY_SYNONYMS = {
    "wife": "spouse",
    "husband": "spouse",
    "spouse": "spouse",
    "married": "spouse",
    "marriage": "spouse",
    "citizen": "nationality",
    "citizenship": "nationality",
    "nationality": "nationality",
}
_QUESTION_RELATION_GROUPS = (
    {"wife", "husband", "spouse", "married", "marriage"},
    {"mother", "father", "parent"},
    {"son", "daughter", "child"},
    {"sister", "brother", "sibling"},
)


def _query_fingerprint(query: str) -> tuple[str, ...]:
    words = re.findall(r"[a-z0-9]+", query.casefold().replace("'s", ""))
    return tuple(sorted(
        _QUERY_SYNONYMS.get(word, word)
        for word in words
        if word not in _QUERY_STOP_WORDS
    ))


def _queries_equivalent(left: str, right: str) -> bool:
    left_tokens = _query_fingerprint(left)
    right_tokens = _query_fingerprint(right)
    if not left_tokens or not right_tokens:
        return False
    if left_tokens == right_tokens:
        return True

    left_set = set(left_tokens)
    right_set = set(right_tokens)
    shared = len(left_set & right_set)
    union = len(left_set | right_set)
    token_similarity = shared / union if union else 0
    text_similarity = SequenceMatcher(
        None,
        " ".join(left_tokens),
        " ".join(right_tokens),
    ).ratio()
    return shared >= 3 and (token_similarity >= 0.85 or text_similarity >= 0.9)


def _unique_contract_queries(
    queries: List[str],
    previous_queries: List[str],
    limit: int,
) -> List[str]:
    unique_queries: List[str] = []
    seen_queries = list(previous_queries)
    for query in queries:
        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized or any(
            _queries_equivalent(normalized, previous)
            for previous in seen_queries
        ):
            continue
        unique_queries.append(normalized)
        seen_queries.append(normalized)
        if len(unique_queries) >= limit:
            break
    return unique_queries


def _unique_retrieval_queries(
    candidates: List[str],
    previous_queries: List[str],
    limit: int,
    grounded_entities: List[str] | None = None,
    original_question: str | None = None,
    established_relationships: List[str] | None = None,
) -> List[str]:
    if limit <= 0:
        return []
    unique_queries: List[str] = []
    seen_queries = list(previous_queries)
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        normalized = candidate.strip()
        if grounded_entities is not None and not _query_uses_only_grounded_entities(
            normalized,
            grounded_entities,
        ):
            continue
        if original_question is not None and not _query_preserves_question_relations(
            original_question,
            normalized,
            established_relationships,
        ):
            continue
        if any(_queries_equivalent(normalized, previous) for previous in seen_queries):
            continue
        unique_queries.append(normalized)
        seen_queries.append(normalized)
        if len(unique_queries) >= limit:
            break
    return unique_queries


def build_answer_target(question: str, supplied: Dict[str, Any] | None = None) -> AnswerTarget:
    named_after = re.search(
        r"\b(?P<named>[A-Z][A-Za-z0-9'.’\-]*)\s*,?\s+who\s+"
        r"(?P<namer>[A-Z][A-Za-z0-9'.’\-]*(?:\s+[A-Z][A-Za-z0-9'.’\-]*){0,3})"
        r"\s+named\s+after\s+who\b",
        question,
    )
    if named_after:
        named_entity = named_after.group("named")
        namer = named_after.group("namer")
        return AnswerTarget(
            answer_entity=f"person {namer} named {named_entity} after",
            requested_attribute="identity",
            required_relationships=[RequiredRelationship(
                id="named-after-target",
                subject=named_entity,
                predicate="named_after",
                object="$answer_entity",
                direction="named entity to inspiration",
                evidence_requirement=(
                    f"Evidence must establish that {namer} named {named_entity} after the "
                    "resolved answer entity, preserving that direction."
                ),
                retrieval_query=f"Who did {namer} name {named_entity} after?",
            )],
            evidence_requirements=[
                f"Identify the person {namer} used as the naming inspiration for {named_entity}."
            ],
        )

    named_after_active = re.search(
        r"\bwho\s+did\s+(?P<namer>[A-Z][A-Za-z0-9'.’\-]*(?:\s+[A-Z][A-Za-z0-9'.’\-]*){0,3})"
        r"\s+name\s+(?P<named>[A-Z][A-Za-z0-9'.’\-]*)\s+after\b",
        question,
        re.IGNORECASE,
    )
    if named_after_active:
        namer = named_after_active.group("namer")
        named_entity = named_after_active.group("named")
        return AnswerTarget(
            answer_entity=f"person {namer} named {named_entity} after",
            requested_attribute="identity",
            required_relationships=[RequiredRelationship(
                id="named-after-target",
                subject=named_entity,
                predicate="named_after",
                object="$answer_entity",
                direction="named entity to inspiration",
                evidence_requirement=(
                    f"Evidence must establish that {namer} named {named_entity} after the "
                    "resolved answer entity, preserving that direction."
                ),
                retrieval_query=f"Who did {namer} name {named_entity} after?",
            )],
            evidence_requirements=[
                f"Identify the person {namer} used as the naming inspiration for {named_entity}."
            ],
        )

    spouse_attribute = re.search(
        r"\bwhat\s+(?P<attribute>[A-Za-z][A-Za-z -]*?)\s+was\s+"
        r"(?P<subject>[A-Z][A-Za-z0-9'.’\-]*(?:\s+[A-Z][A-Za-z0-9'.’\-]*){0,4})"
        r"['’]s\s+(?P<relation>wife|husband)\b",
        question,
        re.IGNORECASE,
    )
    if spouse_attribute:
        subject = spouse_attribute.group("subject")
        relation = spouse_attribute.group("relation").casefold()
        attribute = spouse_attribute.group("attribute").strip().casefold()
        return AnswerTarget(
            answer_entity=f"{relation} of {subject}",
            requested_attribute=attribute,
            required_relationships=[RequiredRelationship(
                id="requested-related-entity",
                subject=subject,
                predicate="spouse",
                object="$answer_entity",
                direction="question subject to requested related entity",
                evidence_requirement=(
                    f"Evidence must establish that the resolved answer entity is the {relation} "
                    f"of {subject}, directly or through a source-supported identity bridge."
                ),
                retrieval_query=f"Who was {subject}'s {relation}?",
            )],
            evidence_requirements=[
                f"Establish the {relation} of {subject}.",
                f"Establish the {attribute} of that {relation}, not of {subject}.",
            ],
        )

    if supplied:
        try:
            return AnswerTarget.model_validate(supplied)
        except ValidationError as error:
            raise ValueError("Question decomposition returned an invalid answer target.") from error

    return AnswerTarget(
        answer_entity="the entity or value requested by the original question",
        requested_attribute=requested_answer_type(question),
        evidence_requirements=[
            "Resolve the answer target in the original question and establish the requested value."
        ],
    )


# ---------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------

async def decompose_question(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    print("\n" + "="*50)
    print("🧠 STEP 1: CONCURRENT QUESTION DECOMPOSITION")
    print("="*50)
    
    configuration = config.get("configurable", {})
    llm = get_generator_llm(
        model=configuration.get("model", "llama3:latest"),
        temperature=0.0
    )
    
    max_sub_qs = max(1, Config.MAX_SUB_QUESTIONS)
    max_generated_questions = max_sub_qs - 1
    inferred_target = build_answer_target(state["original_question"].strip())
    prompt = ChatPromptTemplate.from_template(
        """You are an expert at creating concise retrieval queries for multi-hop questions.
        
        Generate at most {max_sub_qs} concise, retrieval-ready sub-questions in addition to the
        original question. Preserve the exact entities, attributes, and relationships asked about.
        Preserve relationship direction: "who X named Y after" asks for the person X used as the
        inspiration for Y, not who X was named after.
        Each query must resolve a fact needed by the original question. For bridge questions, seek
        the intermediate entity and the requested property of that entity; do not replace the
        requested property with a generic assumption or an unrelated fact. Queries in this initial
        set should be independently useful; dependent gaps can be queried after evidence is assessed.
        For a question asking a property of a related person or thing (for example, "What nationality
        was A's wife?"), keep the query focused on that related target, not on A or an unrelated
        person with a similar name.
        Structured answer target: {answer_target}
        Return an answer_target object with answer_entity, requested_attribute,
        required_relationships (each with id, subject, predicate, object, direction,
        evidence_requirement, retrieval_query), and evidence_requirements. Use "$answer_entity"
        for an unresolved entity whose identity must be retrieved. Preserve subject/object direction.
        Respond with ONLY a valid JSON object matching:
        {{"questions": ["sub-question 1"], "answer_target": {{"answer_entity": "...",
        "requested_attribute": "...", "required_relationships": [], "evidence_requirements": []}}}}
        Do not include any explanations, comments, or extra text.
        
        Question: {question}
        
        JSON Output:"""
    )
    
    original_question = state["original_question"].strip()
    questions = [original_question]
    supplied_target: Dict[str, Any] | None = None
    if max_generated_questions:
        chain = prompt | llm | StrOutputParser()
        raw_result = await _ainvoke_with_model_timeout(chain, {
            "question": state["original_question"],
            "answer_target": json.dumps(inferred_target.model_dump(), ensure_ascii=False),
            "max_sub_qs": max_generated_questions,
        })

        json_match = re.search(r'\{.*\}', raw_result.strip(), re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    result = json.loads(raw_result.strip())
                except json.JSONDecodeError:
                    result = {"questions": []}
        else:
            result = {"questions": []}

        candidate_target = result.get("answer_target")
        if isinstance(candidate_target, dict):
            supplied_target = candidate_target
        raw_questions = result.get("questions", [])
        if isinstance(raw_questions, list):
            seen_questions = {original_question.casefold()}
            for question in raw_questions:
                if not isinstance(question, str) or not question.strip():
                    continue
                normalized = question.strip()
                if normalized.casefold() in seen_questions:
                    continue
                if inferred_target.required_relationships and not _query_matches_answer_target(
                    normalized,
                    inferred_target,
                ):
                    relationship_predicates = {
                        relationship.predicate
                        for relationship in inferred_target.required_relationships
                    }
                    is_identity_bridge_query = (
                        "spouse" in relationship_predicates
                        and _text_mentions_entity(normalized, inferred_target.required_relationships[0].subject)
                        and bool(re.search(
                            r"\b(also known as|better known as|stage name|alias|pseudonym)\b",
                            normalized,
                            re.IGNORECASE,
                        ))
                    )
                    if not is_identity_bridge_query:
                        continue
                seen_questions.add(normalized.casefold())
                questions.append(normalized)
                if len(questions) >= max_sub_qs:
                    break

    answer_target = build_answer_target(original_question, supplied_target)
        
    print(f"📝 Original: {state['original_question']}")
    print(f"🎯 Generated {len(questions)} bounded sub-questions:")
    for i, q in enumerate(questions):
        print(f"   {i+1}. {q}")
    
    trace_entry = {
        "step": "decompose_question",
        "input": {"original_question": state["original_question"]},
        "output": {"decomposed_questions": questions},
    }
    return {
        "answer_target": answer_target.model_dump(),
        "decomposed_questions": questions,
        "followup_rounds": 0,
        "retrieval_queries": questions,
        "followup_queries": [],
        "evidence_sufficient": False,
        "evidence_relevant": False,
        "established_entities": [],
        "established_facts": [],
        "established_relationships": [],
        "evidence_gaps": [],
        "retrieval_matches": {},
        "retrieved_docs": [],
        "retrieved_evidence": "",
        "recovery_evidence": "",
        "recovery_docs": [],
        "trace": state.get("trace", []) + [trace_entry],
    }


async def parallel_retrieve(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    print("\n" + "="*50)
    print("🔍 STEP 2: CONCURRENT SUB-QUESTION RETRIEVAL")
    print("="*50)
    
    sub_questions = state.get("decomposed_questions", [])[:max(1, Config.MAX_SUB_QUESTIONS)]
    if not sub_questions:
        sub_questions = [state["original_question"]]
        
    # Execute retrieval for all sub-questions concurrently using asyncio.gather
    tasks = [rag_system.retrieve_documents(q) for q in sub_questions]
    retrieval_batches = await asyncio.gather(*tasks)
    
    all_docs: List[Document] = []
    retrieval_matches: Dict[str, List[str]] = {}
    for q, docs in zip(sub_questions, retrieval_batches):
        print(f"📍 Query: {q} -> Retrieved: {len(docs)} documents")
        all_docs.extend(docs)
        for doc in docs:
            matches = retrieval_matches.setdefault(document_key(doc), [])
            if q not in matches:
                matches.append(q)
        
    unique_docs = deduplicate_documents(all_docs)
    evidence_text = format_retrieval_evidence(unique_docs, retrieval_matches)
    print(f"📦 Total Deduplicated Evidence Chunks: {len(unique_docs)}")
    
    trace_entry = {
        "step": "parallel_retrieve",
        "input": {"sub_questions": sub_questions},
        "output": {"total_retrieved": len(all_docs), "unique_chunks": len(unique_docs)},
    }
    return {
        "retrieved_docs": unique_docs,
        "retrieved_evidence": evidence_text,
        "retrieval_matches": retrieval_matches,
        "trace": state.get("trace", []) + [trace_entry],
    }


async def _assess_evidence_with_llm(
    question: str,
    retrieval_queries: List[str],
    evidence: str,
    answer_type: str,
    answer_target: Dict[str, Any],
    config: RunnableConfig,
) -> EvidenceAssessment:
    configuration = config.get("configurable", {})
    llm = get_generator_llm(
        model=configuration.get("model", "llama3:latest"),
        temperature=0.0,
    )
    structured_llm = llm.with_structured_output(
        EvidenceAssessment,
        method="json_schema",
    )
    prompt = ChatPromptTemplate.from_template(
        """        Grade document relevance separately from answer sufficiency and the structured answer target.

For each supplied document, return its Document ID in relevant_document_ids only if its contents
support a fact or relationship on the exact path needed to answer the original question. Exclude
documents that merely share a name, mention a similar entity, or describe an unrelated person or
relationship. Retrieval-query wording is not evidence.

Evaluate every required_relationship and evidence_requirement in the answer target. Report the IDs
of only those required relationships supported in the supplied evidence. Identify the resolved
answer entity and its exact requested value only when those are explicitly supported. Set
answer_attribute_supported only when evidence supports the requested attribute for the resolved
answer entity, not an intermediate entity. Set sufficient=true only when every required relationship
and requested attribute is supported. Relevant documents alone never imply sufficiency.

Preserve relationship direction in both facts and gaps. For "who X named Y after", the requested
person is the inspiration for Y; do not reverse this into asking who X was named after. A gap must
name an unmet fact required to answer the original question, not an unrelated fact about how an
entity in the question got its own name. If an established fact already answers the requested
question, do not report that fact as a gap.

List concise established entities, facts, and relationships supported by the evidence. For each
missing fact or relationship, return a structured gap and a targeted retrieval_query that can
resolve that gap. Base queries on the original question, established entities and relationships,
the gap, and previous queries. Use known bridge entities to formulate dependent queries. Do not
query for an entity unless the relevant document text establishes its link to the original question.
Do not repeat a previous query or substitute a bridge entity's property for the requested target's
property.
Treat names in apposition, parentheses, or quoted nicknames as references to the same entity.
Use explicit relationship statements such as "was married to" as evidence for the corresponding
spouse relation. Do not invent an alternate name, surname, or relationship not stated by a source.
For every unsupported target requirement, return a gap with its requirement_id, missing fact or
relationship, and a query preserving the target's entity and relationship direction. If evidence
is insufficient, never return zero gaps: include a gap for every unsupported target requirement.
If the evidence is sufficient, return no gaps. Do not provide chain-of-thought.

Question: {question}
Structured answer target: {answer_target}
Previous retrieval queries: {retrieval_queries}
Requested answer type: {answer_type}
Evidence:
{evidence}

Each evidence section is labeled with the retrieval query that returned its sources. Use the source
content to validate facts; the query text alone is not evidence. A demonym describing a person
(such as "American" or "British") establishes that person's nationality when that is the requested
property.
"""
    )
    result = await _ainvoke_with_model_timeout(prompt | structured_llm, {
        "question": question,
        "answer_target": json.dumps(answer_target, ensure_ascii=False),
        "answer_type": answer_type,
        "retrieval_queries": json.dumps(retrieval_queries, ensure_ascii=False),
        "evidence": evidence or "(No evidence retrieved.)",
    })
    if not isinstance(result, EvidenceAssessment):
        raise TypeError("Evidence assessment did not return an EvidenceAssessment.")
    return result


def _answer_value_is_grounded(
    answer_value: str,
    resolved_entity: str,
    docs: List[Document],
) -> bool:
    if not answer_value.strip() or not resolved_entity.strip():
        return False
    for doc in docs:
        source_text = f"{doc.metadata.get('title', '')}. {doc.page_content}"
        for sentence in re.split(r"(?<=[.!?])\s+", source_text):
            if (
                _text_mentions_entity(sentence, resolved_entity)
                and _text_mentions_entity(sentence, answer_value)
            ):
                return True
    return False


def _contract_gaps(
    target: AnswerTarget,
    assessment: EvidenceAssessment,
) -> List[EvidenceGap]:
    if not target.required_relationships:
        return []
    supported_ids = set(assessment.supported_relationship_ids)
    gaps = [
        EvidenceGap(
            requirement_id=relationship.id,
            missing_relationship=relationship.evidence_requirement,
            retrieval_query=relationship.retrieval_query,
        )
        for relationship in target.required_relationships
        if relationship.id not in supported_ids
    ]
    answer_target_resolved = bool(
        assessment.resolved_answer_entity.strip()
        and assessment.answer_value.strip()
        and assessment.answer_attribute_supported
    )
    if not answer_target_resolved:
        resolved_entity = assessment.resolved_answer_entity.strip()
        query = (
            f"What {target.requested_attribute} was {resolved_entity}?"
            if resolved_entity
            else target.required_relationships[0].retrieval_query
            if target.required_relationships
            else " ".join((target.requested_attribute, target.answer_entity)).strip()
        )
        gaps.append(EvidenceGap(
            requirement_id="answer-attribute",
            missing_fact=(
                f"The {target.requested_attribute} of the resolved answer entity "
                f"({target.answer_entity})"
            ),
            retrieval_query=query,
        ))
    return gaps


def _query_matches_answer_target(query: str, target: AnswerTarget) -> bool:
    normalized_query = " ".join(query.casefold().split())
    for relationship in target.required_relationships:
        if relationship.predicate == "named_after":
            named_entity = re.escape(relationship.subject.casefold())
            if not re.search(
                rf"\bname(?:d)?\s+{named_entity}\s+after\b",
                normalized_query,
            ):
                return False
        elif relationship.predicate == "spouse":
            subject = _normalized_entity_text(relationship.subject)
            query_text = _normalized_entity_text(query)
            resolved_entity = _normalized_entity_text(target.resolved_answer_entity)
            attribute_query = (
                bool(resolved_entity)
                and resolved_entity in query_text
                and _normalized_entity_text(target.requested_attribute) in query_text
            )
            if not attribute_query and (
                subject not in query_text
                or not re.search(
                    r"\b(wife|husband|spouse|married)\b",
                    normalized_query,
                )
            ):
                return False
    return True


async def assess_evidence(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    max_followup_queries = max(0, Config.MAX_FOLLOWUP_QUERIES)
    retrieved_docs = deduplicate_documents(
        state.get("retrieved_docs", []) + state.get("recovery_docs", [])
    )
    retrieval_matches = state.get("retrieval_matches", {})
    question_entities = _named_entity_phrases(state["original_question"])
    allowed_entities = list(dict.fromkeys(
        question_entities + state.get("established_entities", [])
    ))
    candidate_docs = [
        doc for doc in retrieved_docs
        if not allowed_entities
        or any(
            _text_mentions_entity(
                f"{doc.metadata.get('title', '')}\n{doc.page_content}",
                entity,
            )
            for entity in allowed_entities
        )
    ]
    resolved_questions = _rewrite_question_with_grounded_aliases(
        state["original_question"],
        candidate_docs,
    )
    if resolved_questions:
        original_entity_text = {
            _normalized_entity_text(entity)
            for entity in question_entities
        }
        resolved_entities = {
            _normalized_entity_text(entity)
            for rewritten in resolved_questions
            for entity in _named_entity_phrases(rewritten)
            if _normalized_entity_text(entity) not in original_entity_text
        }
        alias_docs = [
            doc for doc in candidate_docs
            if any(
                _text_mentions_entity(
                    f"{doc.metadata.get('title', '')}\n{doc.page_content}",
                    entity,
                )
                for entity in resolved_entities
            )
        ]
        if alias_docs:
            candidate_docs = alias_docs
    assessment_question = resolved_questions[0] if resolved_questions else state["original_question"]
    assessment = await _assess_evidence_with_llm(
        question=assessment_question,
        retrieval_queries=state.get("retrieval_queries", []),
        evidence=format_assessment_evidence(candidate_docs, retrieval_matches),
        answer_type=requested_answer_type(state["original_question"]),
        answer_target=state.get("answer_target", {}),
        config=config,
    )
    relevant_ids = {
        document_id
        for document_id in assessment.relevant_document_ids
        if isinstance(document_id, int) and not isinstance(document_id, bool)
        and 0 <= document_id < len(candidate_docs)
    }
    anchored_docs = candidate_docs
    relevant_docs = [
        doc for index, doc in enumerate(candidate_docs)
        if index in relevant_ids
    ]
    alias_bridge_docs = _alias_bridge_documents(
        state["original_question"],
        resolved_questions,
        anchored_docs,
    )
    recovery_evidence = format_retrieval_evidence(
        anchored_docs,
        retrieval_matches,
    )
    entity_candidates = (
        question_entities
        + state.get("established_entities", [])
        + assessment.established_entities
        + [
            entity
            for doc in anchored_docs
            for entity in _named_entity_phrases(str(doc.metadata.get("title", "")))
        ]
    )
    grounded_entities = _grounded_entities(
        list(dict.fromkeys(entity_candidates)),
        anchored_docs,
    )
    grounded_entities = list(dict.fromkeys(question_entities + grounded_entities))
    def claims_are_grounded(claim: str) -> bool:
        return all(
            _normalized_entity_text(entity) in {
                _normalized_entity_text(known)
                for known in grounded_entities
            }
            for entity in _named_entity_phrases(claim)
        )

    grounded_relationships = [
        relationship for relationship in assessment.established_relationships
        if claims_are_grounded(relationship)
    ]
    answer_target = AnswerTarget.model_validate(state.get("answer_target", {}))
    target_gaps = _contract_gaps(answer_target, assessment)
    answer_value_grounded = (
        not answer_target.required_relationships
        or _answer_value_is_grounded(
            assessment.answer_value,
            assessment.resolved_answer_entity,
            relevant_docs,
        )
    )
    if answer_target.required_relationships and not answer_value_grounded and not any(
        gap.requirement_id == "answer-attribute"
        for gap in target_gaps
    ):
        target_gaps.append(EvidenceGap(
            requirement_id="answer-value-grounding",
            missing_fact=(
                f"Evidence directly linking {assessment.answer_value} to "
                f"{assessment.resolved_answer_entity}"
            ),
            retrieval_query=(
                f"What {answer_target.requested_attribute} was "
                f"{assessment.resolved_answer_entity}?"
                if assessment.resolved_answer_entity
                else state["original_question"]
            ),
        ))
    explicit_target_sufficient = (
        bool(answer_target.required_relationships)
        and bool(relevant_docs)
        and not target_gaps
        and not assessment.gaps
        and answer_value_grounded
        and all(
            relationship.id in assessment.supported_relationship_ids
            for relationship in answer_target.required_relationships
        )
    )
    legacy_sufficient = (
        not answer_target.required_relationships
        and bool(relevant_docs)
        and assessment.sufficient
        and not assessment.gaps
        and _evidence_supports_question_relations(
            state["original_question"],
            relevant_docs,
        )
    )
    evidence_sufficient = (
        (explicit_target_sufficient or legacy_sufficient)
        and (not resolved_questions or bool(alias_bridge_docs))
    )
    actionable_gaps = list({
        (
            gap.requirement_id,
            gap.missing_fact,
            gap.missing_relationship,
            gap.retrieval_query,
        ): gap
        for gap in target_gaps + assessment.gaps
    }.values())
    if not evidence_sufficient and not actionable_gaps:
        actionable_gaps = [EvidenceGap(
            requirement_id="answer-target",
            missing_fact=(
                f"Evidence establishing {answer_target.answer_entity} and "
                f"{answer_target.requested_attribute}"
            ),
            retrieval_query=(
                answer_target.required_relationships[0].retrieval_query
                if answer_target.required_relationships
                else state["original_question"]
            ),
        )]
    if evidence_sufficient:
        answer_target.resolved_answer_entity = assessment.resolved_answer_entity
        answer_target.verified_answer_value = assessment.answer_value
    elif (
        answer_target.required_relationships
        and all(
            relationship.id in assessment.supported_relationship_ids
            for relationship in answer_target.required_relationships
        )
    ):
        answer_target.resolved_answer_entity = assessment.resolved_answer_entity
    if evidence_sufficient and alias_bridge_docs:
        relevant_docs = deduplicate_documents(relevant_docs + alias_bridge_docs)
    relevant_matches = {
        document_key(doc): list(retrieval_matches.get(document_key(doc), []))
        for doc in relevant_docs
    }
    relevant_evidence = format_retrieval_evidence(relevant_docs, relevant_matches)
    followup_queries = []
    if not evidence_sufficient and max_followup_queries:
        missing_relationships = [
            relationship
            for relationship in answer_target.required_relationships
            if relationship.id not in assessment.supported_relationship_ids
        ]
        aliases_by_relationship = {
            relationship.id: _rewrite_question_with_grounded_aliases(
                state["original_question"],
                anchored_docs,
            )
            for relationship in missing_relationships
        }
        contract_alias_queries = [
            alias_query
            for alias_queries in aliases_by_relationship.values()
            for alias_query in alias_queries
        ]
        contract_queries = [
            relation.retrieval_query
            for relation in missing_relationships
            if not aliases_by_relationship[relation.id]
        ]
        attribute_queries = [
            gap.retrieval_query
            for gap in target_gaps
            if gap.requirement_id in {"answer-attribute", "answer-value-grounding"}
            and answer_target.resolved_answer_entity
            and all(
                relationship.id in assessment.supported_relationship_ids
                for relationship in answer_target.required_relationships
            )
        ]
        previous_queries = state.get("retrieval_queries", [])
        safe_contract_queries = _unique_contract_queries(
            contract_alias_queries + contract_queries + attribute_queries,
            previous_queries,
            max_followup_queries,
        )
        recovery_gap_queries = (
            []
            if contract_alias_queries
            else [gap.retrieval_query for gap in actionable_gaps]
        )
        remaining_queries = _unique_retrieval_queries(
            recovery_gap_queries,
            previous_queries + safe_contract_queries,
            max_followup_queries - len(safe_contract_queries),
            grounded_entities,
            state["original_question"],
            grounded_relationships,
        )
        followup_queries = safe_contract_queries + remaining_queries
        followup_queries = [
            query for query in followup_queries
            if _query_matches_answer_target(query, answer_target)
            or query in safe_contract_queries
        ][:max_followup_queries]
    evidence_gaps = [
        {
            "requirement_id": gap.requirement_id,
            "missing_fact": gap.missing_fact,
            "missing_relationship": gap.missing_relationship,
            "retrieval_query": gap.retrieval_query,
        }
        for gap in actionable_gaps
    ]

    trace_entry = {
        "step": "assess_evidence",
        "output": {
            "relevant_document_count": len(relevant_docs),
            "sufficient": evidence_sufficient,
            "gap_count": len(evidence_gaps),
            "followup_query_count": len(followup_queries),
        },
    }
    return {
        "answer_target": answer_target.model_dump(),
        "retrieved_docs": relevant_docs,
        "retrieved_evidence": relevant_evidence,
        "recovery_evidence": recovery_evidence,
        "recovery_docs": anchored_docs,
        "retrieval_matches": retrieval_matches,
        "evidence_relevant": bool(relevant_docs),
        "evidence_sufficient": evidence_sufficient,
        "established_entities": grounded_entities,
        "established_facts": [
            fact for fact in assessment.established_facts
            if claims_are_grounded(fact)
        ],
        "established_relationships": grounded_relationships,
        "evidence_gaps": evidence_gaps,
        "followup_queries": followup_queries,
        "trace": state.get("trace", []) + [trace_entry],
    }


def route_after_assessment(
    state: MultiHopState,
) -> Literal[
    "followup_retrieve",
    "fallback_query_planner",
    "generate_final_answer",
    "insufficient_evidence",
]:
    if state.get("evidence_sufficient", False):
        return "generate_final_answer"
    if state.get("followup_rounds", 0) >= max(0, Config.MAX_FOLLOWUP_ROUNDS):
        return "insufficient_evidence"
    if state.get("followup_queries"):
        return "followup_retrieve"
    return "fallback_query_planner"


async def fallback_query_planner(
    state: MultiHopState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    max_queries = max(0, Config.MAX_FOLLOWUP_QUERIES)
    if max_queries == 0:
        return {"followup_queries": []}

    configuration = config.get("configurable", {})
    llm = get_generator_llm(
        model=configuration.get("model", "llama3:latest"),
        temperature=0.0,
    )
    planner = llm.with_structured_output(RecoveryPlan, method="json_schema")
    prompt = ChatPromptTemplate.from_template(
        """Plan bounded retrieval queries to recover evidence for an unanswered question.

The grader found the current evidence insufficient and supplied no usable query. Do not answer the
question or reveal reasoning. Use the original question, supported entities/facts/relationships,
explicit gaps, the evidence itself, and previous queries. Identify the next missing fact or
relationship needed for the answer, including a dependent bridge when necessary. Preserve the
requested property and relation from the original question. If the evidence establishes an alias
for the question's subject, rewrite the question using that alias without changing who or what the
requested property belongs to. For example, for "What nationality was X's wife?", never query X's
nationality; query the wife's nationality using a source-established alias for X. Return up to
{max_queries} concise, retrieval-ready queries that target only facts not already established.
Do not repeat or paraphrase any previous query.

Original question: {question}
Structured answer target: {answer_target}
Established entities: {established_entities}
Established facts: {established_facts}
Established relationships: {established_relationships}
Missing facts or relationships: {evidence_gaps}
Previous retrieval queries: {retrieval_queries}
Evidence:
{evidence}
"""
    )
    result = await _ainvoke_with_model_timeout(
        prompt | planner,
        {
            "question": state["original_question"],
            "answer_target": json.dumps(state.get("answer_target", {}), ensure_ascii=False),
            "established_entities": json.dumps(state.get("established_entities", []), ensure_ascii=False),
            "established_facts": json.dumps(state.get("established_facts", []), ensure_ascii=False),
            "established_relationships": json.dumps(
                state.get("established_relationships", []),
                ensure_ascii=False,
            ),
            "evidence_gaps": json.dumps(state.get("evidence_gaps", []), ensure_ascii=False),
            "retrieval_queries": json.dumps(state.get("retrieval_queries", []), ensure_ascii=False),
            "evidence": (
                state.get("recovery_evidence")
                or state.get("retrieved_evidence", "")
                or "(No evidence retrieved.)"
            ),
            "max_queries": max_queries,
        },
    )
    if not isinstance(result, RecoveryPlan):
        raise TypeError("Recovery planner did not return a RecoveryPlan.")
    alias_queries = _rewrite_question_with_grounded_aliases(
        state["original_question"],
        state.get("recovery_docs", []),
    )
    answer_target = AnswerTarget.model_validate(state.get("answer_target", {}))
    missing_relationships = [
        relationship
        for relationship in answer_target.required_relationships
        if relationship.id in {
            gap.get("requirement_id")
            for gap in state.get("evidence_gaps", [])
        }
    ]
    aliases_by_relationship = {
        relationship.id: _rewrite_question_with_grounded_aliases(
            state["original_question"],
            state.get("recovery_docs", []),
        )
        for relationship in missing_relationships
    }
    contract_alias_queries = [
        alias_query
        for alias_queries in aliases_by_relationship.values()
        for alias_query in alias_queries
    ]
    target_queries = [
        relationship.retrieval_query
        for relationship in missing_relationships
        if not aliases_by_relationship[relationship.id]
    ]
    previous_queries = state.get("retrieval_queries", [])
    safe_contract_queries = _unique_contract_queries(
        contract_alias_queries + target_queries,
        previous_queries,
        max_queries,
    )
    fallback_candidates = (
        []
        if contract_alias_queries and safe_contract_queries
        else alias_queries + result.queries
    )
    followup_queries = safe_contract_queries + _unique_retrieval_queries(
        fallback_candidates,
        previous_queries + safe_contract_queries,
        max_queries - len(safe_contract_queries),
        list(dict.fromkeys(
            _named_entity_phrases(state["original_question"])
            + state.get("established_entities", [])
        )),
        state["original_question"],
        state.get("established_relationships", []),
    )
    followup_queries = [
        query for query in followup_queries
        if _query_matches_answer_target(query, answer_target)
        or query in contract_alias_queries
        or query in alias_queries
    ][:max_queries]
    return {
        "followup_queries": followup_queries,
        "trace": state.get("trace", []) + [{
            "step": "fallback_query_planner",
            "output": {"query_count": len(followup_queries)},
        }],
    }


def route_after_recovery_planning(
    state: MultiHopState,
) -> Literal["followup_retrieve", "insufficient_evidence"]:
    if state.get("followup_queries"):
        return "followup_retrieve"
    return "insufficient_evidence"


async def insufficient_evidence(
    state: MultiHopState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    message = "I don't have enough information in my knowledge base to answer this question."
    return {
        "evidence_sufficient": False,
        "final_answer": message,
        "trace": state.get("trace", []) + [{
            "step": "insufficient_evidence",
            "output": {"final_answer": message},
        }],
    }


def route_after_followup(state: MultiHopState) -> Literal["assess_evidence"]:
    return "assess_evidence"


async def followup_retrieve(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    print("\n" + "="*50)
    print("🔄 CONDITIONAL FOLLOW-UP RETRIEVAL")
    print("="*50)
    
    queries = state.get("followup_queries", [])[:max(0, Config.MAX_FOLLOWUP_QUERIES)]
    if not queries:
        raise RuntimeError("Follow-up retrieval was scheduled without any retrieval queries.")

    retrieval_batches = await asyncio.gather(
        *(rag_system.retrieve_documents(query) for query in queries)
    )
    docs = [doc for batch in retrieval_batches for doc in batch]
    print(f"📄 Follow-up Retrieved: {len(docs)} documents across {len(queries)} evidence-driven queries")
    
    combined_docs = deduplicate_documents(state.get("retrieved_docs", []) + docs)
    retrieval_matches = {
        key: list(matched_queries)
        for key, matched_queries in state.get("retrieval_matches", {}).items()
    }
    for query, batch in zip(queries, retrieval_batches):
        for doc in batch:
            matches = retrieval_matches.setdefault(document_key(doc), [])
            if query not in matches:
                matches.append(query)
    evidence_text = format_retrieval_evidence(combined_docs, retrieval_matches)
    
    trace_entry = {
        "step": "followup_retrieve",
        "output": {"query_count": len(queries), "retrieved_count": len(docs)},
    }
    return {
        "retrieved_docs": combined_docs,
        "retrieved_evidence": evidence_text,
        "retrieval_matches": retrieval_matches,
        "followup_rounds": state.get("followup_rounds", 0) + 1,
        "retrieval_queries": state.get("retrieval_queries", []) + queries,
        "followup_queries": [],
        "trace": state.get("trace", []) + [trace_entry],
    }


async def generate_final_answer(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    print("\n" + "="*50)
    print("🎉 STEP 3: DIRECT CUMULATIVE FINAL SYNTHESIS")
    print("="*50)

    if not state.get("evidence_sufficient", False):
        return await insufficient_evidence(state, config)
    
    original_question = state["original_question"]
    evidence = state.get("retrieved_evidence", "")
    answer_type = requested_answer_type(original_question)
    answer_target = AnswerTarget.model_validate(state.get("answer_target", {}))
    if answer_target.required_relationships and not answer_target.verified_answer_value:
        return await insufficient_evidence(state, config)
    
    print(f"📝 Original Question: {original_question}")
    
    if not evidence.strip():
        return await insufficient_evidence(state, config)
    
    configuration = config.get("configurable", {})
    llm = get_generator_llm(
        model=configuration.get("model", "llama3:latest"),
        temperature=0.0
    )
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the user's question accurately and concisely using ONLY the provided evidence.

        Structured answer target: {answer_target}
        The requested entity is "{target_entity}". The requested property is
        "{requested_attribute}". The evidence assessor resolved that entity to
        "{resolved_answer_entity}" and verified its value as "{verified_answer_value}".
        Return that verified value only if the evidence below supports it for that exact entity.
        Never transfer an attribute from an intermediate or related entity.

        Requested answer type: {answer_type}. For a person or entity name, return only the
        requested name, not a property belonging to that person or a description of the relation.

        IMPORTANT:
        1. Identify the entity whose property is requested in the question.
        2. Follow aliases and relationships stated in the evidence to resolve that target entity.
        3. Give the requested property's value for that target, not for an intermediate entity or
           another person with a similar name.
        4. If asked who or whom, return only the requested person's name. When evidence expresses
           the relation through an attribute (for example, "X was named after P's middle name"),
           return person P, not P's attribute value or a description of the relation.
        5. Do not substitute an assumed or typical property, or add adjacent facts.
        6. Do NOT reveal reasoning or include introductory phrases such as 'Based on the context'.
        7. If the evidence does not establish the target and requested property, say so clearly.

        For example, if asked "What nationality was A's wife?", give the wife's nationality, not
        A's nationality. Keep every final answer limited to the requested value or entity name.

        Evidence:
        {evidence}

        Question: {original_question}

        Final Answer:"""
    )
    
    chain = prompt | llm | StrOutputParser()
    final_answer = await _ainvoke_with_model_timeout(chain, {
        "answer_type": answer_type,
        "answer_target": json.dumps(answer_target.model_dump(), ensure_ascii=False),
        "target_entity": answer_target.answer_entity,
        "requested_attribute": answer_target.requested_attribute,
        "resolved_answer_entity": answer_target.resolved_answer_entity,
        "verified_answer_value": answer_target.verified_answer_value,
        "evidence": evidence,
        "original_question": original_question,
    })

    if answer_target.required_relationships:
        normalized_answer = _normalized_entity_text(final_answer)
        normalized_verified = _normalized_entity_text(answer_target.verified_answer_value)
        if not normalized_answer or not normalized_verified or not (
            normalized_answer == normalized_verified
            or f" {normalized_verified} " in f" {normalized_answer} "
        ):
            message = (
                "I couldn't verify a response for the answer target in the available evidence."
            )
            return {
                "final_answer": message,
                "trace": state.get("trace", []) + [{
                    "step": "generate_final_answer",
                    "output": {"final_answer": message, "target_verified": False},
                }],
            }
        final_answer = answer_target.verified_answer_value

    print(f"🏁 Final Answer: {final_answer}")
    print("="*50)
    
    trace_entry = {
        "step": "generate_final_answer",
        "input": {"original_question": original_question},
        "output": {"final_answer": final_answer},
    }
    return {"final_answer": final_answer, "trace": state.get("trace", []) + [trace_entry]}


# ---------------------------------------------------------------------
# Supervisor classification and routing nodes
# ---------------------------------------------------------------------

async def supervisor_classify(state: MultiHopState, config: RunnableConfig) -> Dict[str, Any]:
    question = state.get("original_question", "")
    result = await asyncio.wait_for(
        arun_classifier(question),
        timeout=Config.MODEL_TIMEOUT,
    )
    return {
        "classification": result.classification,
        "classification_reason": result.reason,
        "trace": state.get("trace", []) + [{
            "step": "supervisor_classify",
            "input": {"question": question},
            "output": {"classification": result.classification, "reason": result.reason}
        }],
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
    worker_ans = state.get("final_answer", "")
    if not worker_ans:
        worker_ans = "[No answer generated]"
    return {
        "final_answer": worker_ans,
        "worker_answer": worker_ans,
        "trace": state.get("trace", []) + [{"step": "supervisor_finalize", "output": {"final_answer": worker_ans}}]
    }

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


# ---------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------

workflow = StateGraph(MultiHopState, config_schema=Configuration)

# Supervisor nodes
workflow.add_node("supervisor_classify", supervisor_classify)
workflow.add_node("out_of_scope", out_of_scope)
workflow.add_node("prompt_injection", prompt_injection)
workflow.add_node("invalid_query", invalid_query)
workflow.add_node("needs_clarification", needs_clarification)
workflow.add_node("supervisor_finalize", supervisor_finalize)

# Parallel RAG nodes
workflow.add_node("decompose_question", decompose_question)
workflow.add_node("parallel_retrieve", parallel_retrieve)
workflow.add_node("assess_evidence", assess_evidence)
workflow.add_node("followup_retrieve", followup_retrieve)
workflow.add_node("fallback_query_planner", fallback_query_planner)
workflow.add_node("insufficient_evidence", insufficient_evidence)
workflow.add_node("generate_final_answer", generate_final_answer)

# Routing
workflow.set_entry_point("supervisor_classify")

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

# Parallel retrieval, evidence assessment, and bounded dependent follow-up.
workflow.add_edge("decompose_question", "parallel_retrieve")
workflow.add_edge("parallel_retrieve", "assess_evidence")
workflow.add_conditional_edges(
    "assess_evidence",
    route_after_assessment,
    {
        "followup_retrieve": "followup_retrieve",
        "fallback_query_planner": "fallback_query_planner",
        "generate_final_answer": "generate_final_answer",
        "insufficient_evidence": "insufficient_evidence",
    },
)
workflow.add_conditional_edges(
    "fallback_query_planner",
    route_after_recovery_planning,
    {
        "followup_retrieve": "followup_retrieve",
        "insufficient_evidence": "insufficient_evidence",
    },
)
workflow.add_conditional_edges(
    "followup_retrieve",
    route_after_followup,
    {
        "assess_evidence": "assess_evidence",
    },
)

# Final answer goes through supervisor finalization
workflow.add_edge("generate_final_answer", "supervisor_finalize")
workflow.add_edge("insufficient_evidence", "supervisor_finalize")
workflow.add_edge("supervisor_finalize", END)

# Terminal edges for non-RAG supervisor routes
workflow.add_edge("out_of_scope", END)
workflow.add_edge("prompt_injection", END)
workflow.add_edge("invalid_query", END)
workflow.add_edge("needs_clarification", END)

# Compile graph
graph = workflow.compile(name="low-latency-multi-hop-agent")