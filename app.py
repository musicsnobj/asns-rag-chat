import json
import os
import re
import textwrap
import boto3
import instructor
import time
from datetime import datetime
from anthropic import AnthropicBedrock
from typing import List, Dict, Any
from pydantic import BaseModel, Field


bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
)
bedrock_client = instructor.from_bedrock(bedrock_runtime)

s3vectors = boto3.client("s3vectors")

anthropic_client = instructor.from_anthropic(AnthropicBedrock(aws_region="us-east-1"))

EMBED_MODEL_ID = os.environ["EMBED_MODEL_ID"]
CHAT_MODEL_ID = os.environ["CHAT_MODEL_ID"]
RELEVANCE_MODEL_ID = os.environ["RELEVANCE_MODEL_ID"]
VECTOR_BUCKET_NAME = os.environ["VECTOR_BUCKET_NAME"]
VECTOR_INDEX_NAME = os.environ["VECTOR_INDEX_NAME"]
CHAT_TABLE_NAME = os.environ["CHAT_TABLE_NAME"]
TRANSCRIPT_TABLE_NAME = os.environ["TRANSCRIPT_TABLE_NAME"]

TOP_K = 8

dynamodb = boto3.resource("dynamodb")

chat_request_table = dynamodb.Table(CHAT_TABLE_NAME)
transcript_table = dynamodb.Table(TRANSCRIPT_TABLE_NAME)

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(description="Overall determination of whether a piece of context is useful for answering user query")
    confidence: int = Field(description="Confidence score (0-100) from the LLM regarding determination of relevance", ge=0, le=100)
    reason: str = Field(description="A one-sentence explanation of the LLM's reasoning behind determination of relevance")

class SearchQuery(BaseModel):
    query: str = Field(
        description="Revised semantic search query excluding filter information for cleaner results"
    )
    filter_exp: Dict[str, Any] | None = Field(
        description=(
            "Dict that can be used as the 'filter' part of an Elasticsearch/OpenSearch bool query"
            " or None if no filters are applicable"
        )
    )

class TranscriptLine(BaseModel):
    episode_id: str = Field(description="Episode ID of transcript")
    date: str = Field(description="Date of recording")
    speaker: str = Field(description="Speaker of the line (Jess or Jack Brett)")
    timestamp: str = Field(description="Timestamp of the line relative to start of episode")
    line_id: str = Field(description="Unique line_id from S3 Vectors index, based on hash of episode_id:timestamp:speaker:text")
    timestamp_sec: int = Field(description="Numeric timestamp of the line in number of seconds from start of episode")
    text: str = Field(description="A line from the episode transcript")
    line_position: int = Field(description="Numeric position of line relative to its neighbors")

class TranscriptExchange(BaseModel):
    episode_id: str = Field(description="Episode ID of exchange")
    date: str = Field(description="Date of recording")
    lines: list[TranscriptLine] = Field(description="A sequence of lines from episode transcript")
    score: int = Field(description="K-NN similarity score of the central line of the exchange")

# regex patterns to detect when user prompt refers to earlier chat history
PRONOUN_PATTERN = re.compile(
    r"\b(he|she|they|them|his|her|their|it|its)\b",
    re.IGNORECASE,
)

DEICTIC_PATTERN = re.compile(
    r"\b(this|that|these|those|there|here)\b",
    re.IGNORECASE,
)

FOLLOWUP_PATTERN = re.compile(
    r"\b(expand|elaborate|explain|clarify|compare|contrast|follow up|go on)\b",
    re.IGNORECASE,
)

CONTINUATION_PATTERN = re.compile(
    r"\b(what about|why is that|how about|and what|what else)\b",
    re.IGNORECASE,
)

QUERY_TO_KNN_PROMPT = (
    "You are a helpful research assistant tasked with analyzing a semantic search query"
    " and identifying any filters that should be applied in data retrieval from an S3 "
    "Vectors bucket. Each vector in the S3 Vectors bucket contains an embedded dialogue "
    "line from the 'However Comma' podcast hosted by Jack Brett and Jess. The vector "
    "collection has three filterable metadata fields:\n"
    "speaker (string) - name of podcast host who spoke the line, must be either 'Jess' or 'Jack Brett', e.g. 'Jess'\n"
    "date (string) - date of podcast episode in YYYY-MM-DD format, e.g. '2025-10-31'\n"
    "date_ts (number) - numeric representation of the podcast date in YYYYMMDD format, e.g. 20251031\n"
    "With this in mind, your task is to read the input query, determine if any of the above "
    "filters should be applied to the semantic search. If there are applicable filters, you must also "
    "revise the input query to scrub out any references to the metadata we plan to filter (this is to achieve"
    " cleaner semantic search results given e.g. lines spoken by Jess on 2025-02-04 will not likely contain Jess'"
    " name or the podcast date, best to leave them out of the semantic search query). Your output "
    "must be a JSON-parsable object with two fields:\n"
    "query (string) - revised semantic search query excluding references to metadata filters (or "
    "the original query if no filters are applicable)\n"
    "filter_exp (JSON object) - an elasticsearch filter expression to apply the identified filters "
    "(or None if no filters are applicable).\n"
    "Examples:\n"
    "1. For query 'Fetch any statements that indicate Jack Brett's position on free will', you would output:\n"
    "{\"query\": \"opinions about free will\", \"filter_exp\": {\"speaker\": {\"$eq\": \"Jack Brett\"}}}\n"
    "2. For query 'Fetch any statements Jess made about Trump's alleged pandering to White Supremacists in early 2025 (January-March)',"
    " you would output:\n"
    "{\"query\": \"Trump alleged pandering to White Supremacists\", \"filter_exp\": {\"$and\": [{\"speaker\": {\"$eq\": \"Jess\"}}, {\"date_ts\": {\"$gte\": 20250101}}, {\"date_ts\": {\"$lte\": 20250331}}]}}\n"
    "3. For query 'Fetch any discussions about the murder of Charlie Kirk', you would output:\n"
    "{\"query\": \"murder of Charlie Kirk\", \"filter_exp\": \"None\"}\n"
    "Return only the JSON filter expression, no prose or additional reasoning."
)

def is_referential(query: str) -> bool:
    """
    Returns True if the query likely depends on prior conversational context.
    """

    q = query.strip().lower()

    # Very short questions are often referential
    if len(q.split()) <= 4:
        return True

    # Follow-up phrases are strong signals
    if FOLLOWUP_PATTERN.search(q):
        return True

    if CONTINUATION_PATTERN.search(q):
        return True

    # Pronouns or deictic words
    has_pronoun = PRONOUN_PATTERN.search(q) is not None
    has_deictic = DEICTIC_PATTERN.search(q) is not None

    if (has_pronoun or has_deictic):
        return True

    return False


def _now() -> int:
    return int(time.time())

def to_date_ts(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d"))

def episode_id_to_name(episode_id: str) -> str:
    """
    Extracts the episode name from an episode_id
    """
    # Regex breakdown:
    # \d{2}-\d{2}-\d{4}      → MM-DD-YYYY
    # (?:_.+?)?              → optional underscore + characters (e.g., _pt1)
    # \.(wav|mp3)            → file extension
    match = re.search(r"(\d{2}-\d{2}-\d{4}(?:_.+?)?\.(?:wav|mp3))", episode_id)

    if not match:
        raise ValueError(f"Invalid episode_id format: {episode_id}")

    return match.group(1)

def embed_text(text: str) -> List[float]:
    response = bedrock_runtime.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({
            "inputText": text,
            "dimensions": 512
        })
    )
    body = json.loads(response["body"].read())
    return body["embedding"]


def retrieve_context(query: str) -> List[Dict[str, Any]]:
    knn_query = anthropic_client.messages.create(
        model=CHAT_MODEL_ID,
        response_model=SearchQuery,
        max_tokens=500,
        temperature=0.2,
        system=QUERY_TO_KNN_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Query:\n{query}"
            }
        ]
    )
    print(f"query: {knn_query.query}")
    print(f"filter_exp: {knn_query.filter_exp}")
    query_embedding = embed_text(knn_query.query)
    
    response = s3vectors.query_vectors(
        vectorBucketName=VECTOR_BUCKET_NAME,
        indexName=VECTOR_INDEX_NAME,
        queryVector={"float32": query_embedding},
        topK=TOP_K,
        returnMetadata=True,
        returnDistance=True,
        filter=knn_query.filter_exp
    )
    return response.get("vectors", [])


def get_search_tasks_for_query(query: str) -> List[str]:
    """
    Break down query into a set of retrieval instructions for our vector collection
    """
    separate_tasks_prompt = textwrap.dedent(f"""
        You are a helpful research assistant with the job of helping retrieve the most relevant RAG context to answer a user question. The data source where context will be retrieved from is an S3 Vectors bucket with embedded dialogue lines from the However Comma podcast (hosted by Jack Brett and Jess). The vectors are stored with three filterable metadata fields: speaker (name of host who spoke the line, must be either 'Jack Brett' or 'Jess'), date (string date of podcast episode in 'YYYY-MM-DD' format), date_ts (numeric date of podcast episode in YYYYMMDD format). These metadata fields allow filtering vectors down to one of the two hosts, or a particular date or date range. Your task is to help overcome some of the limitations of distance-based similarity search by optimizing the user question for k-NN search of this specific vector collection. You will do this by splitting the user query into a set of instructions for semantic search by identifying any unique topics and/or filterable fields. i.e. there should be as many instructions as there are combinations of unique topics and relevant filterable values.
        Examples:
        1. For user query "Give a summary of both hosts' positions on American intervention in the Russia-Ukraine conflict, highlighting any areas of disagreement", your instructions might be:
        Fetch any statements Jess has made regarding the US intervening in Russia-Ukraine conflict
        Fetch any statements Jack Brett has made regarding the US intervening in Russia-Ukraine conflict
        2. For user query "How did Jess' position on the electoral college evolve over 2025?", your instructions might be:
        Fetch any statements Jess made about the electoral college in early 2025 (January-March)
        Fetch any statements Jess made about the electoral college in mid 2025 (April-June)
        Fetch any statements Jess made about the electoral college in late 2025 (July-September)
        Fetch any statements Jess made about the electoral college in end of 2025 (October-December)
        Fetch any statements Jess made about changes in her views on the electoral college
        Provide these retrieval instructions separated by newlines.
        Do not output any prose or additional reasoning.
        USER QUESTION: {query}
    """)
    separate_tasks_resp = bedrock_runtime.invoke_model(
        modelId=CHAT_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 600,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": separate_tasks_prompt
                }
            ]
        })
    )
    resp_body = json.loads(separate_tasks_resp["body"].read())
    llm_output = resp_body["content"][0]["text"]
    
    return llm_output.split('\n')

def dedupe_vectors_by_key(
    vectors: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    seen_keys: set[str] = set()
    deduped: list[Dict[str, Any]] = []

    for v in vectors:
        key = v.get("key")
        if key is None:
            # ignore malformed entries
            continue

        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(v)

    return deduped

def get_vector_relevance(
    original_query: str,
    vector: Dict[str, Any],
) -> RelevanceDecision:
    prompt = textwrap.dedent(f"""
        You are a strict relevance judge for a research assistant.
        Your job is to decide whether a transcript excerpt is useful for answering the user's original question.
        Be conservative.
        If the excerpt is only tangentially related, judge it as NOT RELEVANT.

        USER QUESTION:
        {original_query}

        TRANSCRIPT EXCERPT:
        Speaker: {vector['speaker']}
        Date: {vector['date']}
        Timestamp: {vector['timestamp']}
        Text:
        "{vector['text']}"

        Is this excerpt useful for answering the user's question?

        Respond ONLY in JSON with this schema:
        {{
        "relevant": boolean,
        "confidence": number,   // 0-100
        "reason": string        // short, max 1 sentence
        }}
    """)
    return bedrock_client.create(
        model=RELEVANCE_MODEL_ID,
        response_model=RelevanceDecision,
        max_tokens=300,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

def get_n_surrounding_lines(vector: Dict[str, Any], n: int = 3) -> TranscriptExchange:
    """
    Fetch n transcript lines immediately before and after line_id
    (in order to place the lines returned by semantic search in
    their appropriate context within the conversation to determine
    relevance)
    """
    line_id = vector.get("key")
    k_distance = vector.get("distance")
    score = 100 - int(k_distance * 100)
    try:
        fetch_line_resp = transcript_table.query(
            IndexName="line_id_index",
            KeyConditionExpression="line_id = :lid",
            ExpressionAttributeValues={":lid": line_id},
            Limit=1
        )
        item = fetch_line_resp["Items"][0]
        episode_id = item["episode_id"]
        episode_date = item["date"]
        line_position = item["line_position"]

        adjacent_lines_resp = transcript_table.query(
            KeyConditionExpression=
                "episode_id = :ep AND line_position BETWEEN :start AND :end",
            ExpressionAttributeValues={
                ":ep": episode_id,
                ":start": line_position - n,
                ":end": line_position + n,
            },
            ScanIndexForward=True
        )
        adjacent_lines = adjacent_lines_resp.get("Items")

        return TranscriptExchange(
            episode_id=episode_id,
            date=episode_date,
            score=score,
            lines=adjacent_lines
        )
    except Exception as e:
        print(f"Error occurred fetching surrounding lines: {e}")
        raise

def stringify_exchange(exchange: TranscriptExchange) -> str:
    formatted_lines = "\n".join(
        f"({line.timestamp}) {line.speaker}: {line.text}" for line in exchange.lines
    )
    return textwrap.dedent(f"""
        EXCERPT FROM {exchange.date}:
        {formatted_lines}
    """)

def format_exchange_relevance_prompt(user_query: str, exchange: TranscriptExchange) -> str:
    
    str_exchange = stringify_exchange(exchange)
    return textwrap.dedent(f"""
        You are a strict relevance judge for a research assistant.
        Your job is to decide whether a transcript excerpt is useful for answering the user's original question.
        Be conservative.
        If the excerpt is only tangentially related, judge it as NOT RELEVANT.

        USER QUESTION:
        {user_query}

        {str_exchange}

        Is this excerpt useful for answering the user's question?

        Respond ONLY in JSON with this schema:
        {{
        "relevant": boolean,
        "confidence": number,   // 0-100
        "reason": string        // short, max 1 sentence
        }}
    """)

def get_exchange_relevance(user_query: str, exchange: TranscriptExchange) -> RelevanceDecision:
    """
    Determine relevance of a particular exchange from transcript
    by comparing to user query via LLM
    """
    prompt = format_exchange_relevance_prompt(user_query, exchange)

    return bedrock_client.create(
            model=RELEVANCE_MODEL_ID,
            response_model=RelevanceDecision,
            max_tokens=300,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

def get_curated_context(query: str, max_hits: int = 25) -> List[TranscriptExchange]:
    all_context = []
    relevant_exchanges: List[TranscriptExchange] = []
    search_tasks = get_search_tasks_for_query(query)
    for task in search_tasks:
        print(f"TASK: {task}")
        task_context_items = retrieve_context(task)
        all_context.extend(task_context_items)
    # deduplicate by key
    deduped_context = dedupe_vectors_by_key(all_context)
    # order by k-NN distance
    ordered_context = sorted(deduped_context, key=lambda x: x['distance'], reverse=True)
    # filter by relevance
    for i in range(len(ordered_context)):
        if len(relevant_exchanges) >= max_hits:
            # return early if we have enough relevant hits
            return relevant_exchanges

        vector = ordered_context[i]
        exchange = get_n_surrounding_lines(vector, n=5)
        relevance: RelevanceDecision = get_exchange_relevance(query, exchange)
        print(stringify_exchange(exchange))
        print(f"RELEVANCE: {relevance}")
        if relevance.is_relevant:
            relevant_exchanges.append(exchange)
    # we didn't meet the max_hits benchmark. return what we've got
    return relevant_exchanges


def build_rag_prompt_with_context(
    latest_user_query: str,
    context: List[TranscriptExchange],
    messages: list[dict] | None = None,
    max_history_turns: int = 4,
) -> str:
    """
    Builds the final LLM prompt.
    Includes recent conversation history ONLY if the query is referential.
    """

    system_prompt = (
        "You are answering questions using ONLY the provided podcast transcript excerpts.\n"
        "Rules:\n"
        "- Do not use outside knowledge.\n"
        "- Do not speculate or guess.\n"
        "- If the excerpts do not support an answer, say so clearly.\n"
        "- Be concise and factual.\n"
    )

    if not context:
        context_block = "No relevant podcast excerpts were found.\n"
    else:
        blocks = [stringify_exchange(exchange) for exchange in context]
        context_block = "Podcast transcript excerpts:\n\n" + "\n".join(blocks)

    # Include recent conversation history if referential language detected
    conversation_block = ""

    if messages and is_referential(latest_user_query):
        recent_messages = messages[-max_history_turns:]

        formatted_history = []
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg['content']}")

        conversation_block = (
            "\nConversation so far (for context only):\n"
            + "\n".join(formatted_history)
            + "\n\n"
            "The conversation history is provided ONLY to clarify meaning.\n"
            "Do NOT treat it as factual evidence.\n"
        )

    user_instruction = (
        f"\nUser question:\n{latest_user_query}\n\n"
        "Answer the user's question using the podcast excerpts above.\n"
        "If multiple points are relevant, list them clearly.\n"
        "If the excerpts do not contain the answer, say so explicitly."
    )

    prompt = (
        system_prompt
        + "\n"
        + context_block
        + "\n"
        + conversation_block
        + user_instruction
    )

    return prompt


def call_llm(prompt: str) -> str:
    response = bedrock_runtime.invoke_model(
        modelId=CHAT_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 600,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
    )

    body = json.loads(response["body"].read())
    return body["content"][0]["text"]


def lambda_handler(event, context):
    request_id = event.get("request_id")

    if not request_id:
        # Nothing we can do without a request_id
        raise Exception("Missing required parameter: request_id")

    # Fetch request from DynamoDB
    resp = chat_request_table.get_item(Key={"request_id": request_id})
    db_record = resp.get("Item")
    
    if not db_record:
        raise Exception(f"No db record found for request_id: {request_id}")

    if db_record.get("status") in {"complete", "error"}:
        # quiet short-circuit - this request_id is already handled
        return

    try:
        chat_request_table.update_item(
            Key={"request_id": request_id},
            UpdateExpression=(
                "SET #status = :status, "
                "updated_at = :now"
            ),
            ExpressionAttributeNames={
                "#status": "status"
            },
            ExpressionAttributeValues={
                ":status": "processing",
                ":now": _now()
            },
        )

        messages = db_record.get("messages")
        if not messages:
            raise Exception("Required field 'messages' missing from db record")

        user_query = messages[-1]["content"]
        if not user_query:
            raise Exception("User query missing from request")

        # retrieve relevant transcript chunks
        relevant_exchanges = get_curated_context(user_query)
        # assemble RAG prompt
        prompt = build_rag_prompt_with_context(
            user_query,
            relevant_exchanges,
            messages,
        )
        # get answer from LLM
        answer = call_llm(prompt)
        sources = []
        for exchange in relevant_exchanges:
            sources.append({
                "episode_id": exchange.episode_id,
                "episode_name": episode_id_to_name(exchange.episode_id),
                "text": stringify_exchange(exchange),
                "date": exchange.date,
                "score": exchange.score,
            })

        # Write success result
        chat_request_table.update_item(
            Key={"request_id": request_id},
            UpdateExpression=(
                "SET #status = :status, "
                "answer = :answer, "
                "sources = :sources, "
                "updated_at = :now "
                "REMOVE error_message"
            ),
            ExpressionAttributeNames={
                "#status": "status"
            },
            ExpressionAttributeValues={
                ":status": "complete",
                ":answer": answer,
                ":sources": sources,
                ":now": _now()
            },
        )

    except Exception as exc:
        # ---- Capture error state ----
        chat_request_table.update_item(
            Key={"request_id": request_id},
            UpdateExpression=(
                "SET #status = :status, "
                "error_message = :error_message, "
                "updated_at = :now"
            ),
            ExpressionAttributeNames={
                "#status": "status"
            },
            ExpressionAttributeValues={
                ":status": "error",
                ":error_message": str(exc),
                ":now": _now()
            },
        )

        # Re-raise so Lambda marks this invocation as failed
        raise
