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
from enum import Enum
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
LINE_INDEX_NAME = os.environ["LINE_INDEX"]
DIALOG_INDEX_NAME = os.environ["DIALOG_INDEX"]
CHAT_TABLE_NAME = os.environ["CHAT_TABLE_NAME"]
TRANSCRIPT_TABLE_NAME = os.environ["TRANSCRIPT_TABLE_NAME"]

TOP_K = 8

dynamodb = boto3.resource("dynamodb")

chat_request_table = dynamodb.Table(CHAT_TABLE_NAME)
transcript_table = dynamodb.Table(TRANSCRIPT_TABLE_NAME)

class DialogLabel(str, Enum):
    TOPIC = "TOPIC"
    TONE = "TONE"
    MIXED = "MIXED"

class DialogTone(str, Enum):
    DEBATE = "DEBATE"
    BANTER = "BANTER"
    LOGISTICS = "LOGISTICS"

class LabelDecision(BaseModel):
    label: DialogLabel = Field(description="the classification assigned to user query which will inform RAG strategy")

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(description="Overall determination of whether a piece of context is useful for answering user query")
    confidence: int = Field(description="Confidence score (0-100) from the LLM regarding determination of relevance", ge=0, le=100)
    reason: str = Field(description="A one-sentence explanation of the LLM's reasoning behind determination of relevance")

class LineQueryKNN(BaseModel):
    query: str = Field(
        description="Revised semantic search query excluding filter information for cleaner results"
    )
    filter_exp: Dict[str, Any] | None = Field(
        description=(
            "Dict that can be used as the 'filter' part of an Elasticsearch/OpenSearch bool query"
            " or None if no filters are applicable"
        )
    )
class DialogQueryKNN(BaseModel):
    query: str = Field(
        description="Revised semantic search query excluding filter information for cleaner results"
    )
    tone: DialogTone | None = Field(description="the tone of conversation specified in user query (BANTER, DEBATE, LOGISTICS, or None)")

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
    timestamp: str = Field(description="Timestamp of the exchange relative to start of episode")
    lines: list[TranscriptLine] = Field(description="A sequence of lines from episode transcript")
    score: int = Field(description="K-NN similarity score of the central line of the exchange")

class RelevantSource(BaseModel):
    episode_id: str = Field(description="Episode ID of exchange")
    date: str = Field(description="Date of recording")
    timestamp: str = Field(description="Timestamp of the exchange relative to start of episode")
    text: str = Field(description="Stringified sequence of lines from episode transcript")
    score: int = Field(description="K-NN similarity score of the central line of the exchange")

class QueryType(str, Enum):
    DIRECT_SEARCH = "DIRECT_SEARCH"
    RAG_SEARCH = "RAG_SEARCH"

class QueryTypeDecision(BaseModel):
    label: QueryType = Field(description="The type of search query identified by LLM, which will inform search strategy")

class QueryResponse(BaseModel):
    answer: str = Field(description="Final answer to user query")
    sources: List[RelevantSource] = Field(description="The list of search hits from vector index determined to be relevant to query")

class ChatMessage(BaseModel):
    role: str = Field(description="Message role (user or system)")
    content: str = Field(description="Message content")

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

QUERY_TO_LINE_KNN_PROMPT = textwrap.dedent("""
    You are a helpful research assistant tasked with analyzing a user query and helping to optimize it for semantic search via K-NN.
    This means determining which if any available metadata filters that should be applied in data retrieval.
    The data source is an S3 Vectors bucket containing all the embedded dialogue from the 'However Comma' podcast hosted by Jack Brett and Jess. The vector collection has three filterable metadata fields:\n
    speaker (string) - name of podcast host who spoke the line, must be either 'Jess' or 'Jack Brett', e.g. 'Jess'\n
    date (string) - date of podcast episode in YYYY-MM-DD format, e.g. '2025-10-31'\n
    date_ts (number) - numeric representation of the podcast date in YYYYMMDD format, e.g. 20251031\n
    With this in mind, your task is to read the input query, determine if any of the above filters should be applied to the semantic search. If there are applicable filters, you must also revise the input query to scrub out any references to the metadata we plan to filter (this is to achieve cleaner semantic search results given e.g. lines spoken by Jess on 2025-02-04 will not likely contain Jess' name or the podcast date, best to leave them out of the semantic search query). Your output must be a JSON-parsable object with two fields:\n
    query (string) - revised semantic search query excluding references to metadata filters (or the original query if no filters are applicable)\n
    filter_exp (JSON object) - an elasticsearch filter expression to apply the identified filters (or None if no filters are applicable).\n
    Examples:\n
    1. For query 'Fetch any statements that indicate Jack Brett's position on free will', you would output:\n
    {"query": "opinions about free will", "filter_exp": {"speaker": {"$eq": "Jack Brett"}}}\n
    2. For query 'Fetch any argumentative statements Jess made about Trump's alleged pandering to White Supremacists in early 2025 (January-March)', you would output:\n
    {"query": "Argumentative statements about Trump pandering to White Supremacists", "filter_exp": {"$and": [{"speaker": {"$eq": "Jess"}}, {"date_ts": {"$gte": 20250101}}, {"date_ts": {"$lte": 20250331}}]}}\n
    3. For query 'Fetch any discussions about the murder of Charlie Kirk', you would output:\n
    {"query": "murder of Charlie Kirk", "filter_exp": "None"}\n
    4. For query 'Jack Brett joking about his sparse sexual history', you would output:\n
    {"query": "jokes about sexual history", "filter_exp": {"speaker": {"$eq": "Jack Brett"}}}\n
    Return only the JSON object, no prose or additional reasoning.
""")

QUERY_TO_DIALOG_KNN_PROMPT = textwrap.dedent("""
    You are a helpful research assistant tasked with analyzing a user query and helping to optimize it for semantic search via K-NN.
    This means determining whether to apply metadata filters in data retrieval.
    The data source is an S3 Vectors bucket containing all the embedded dialogue, minus speaker labels, from the 'However Comma' podcast hosted by Jack Brett and Jess.
    The vector collection has one filterable metadata field:\n
    tone (string) - the tone of transcript segment (BANTER, DEBATE, or LOGISTICS)
    With this in mind, your task is to read the input query and determine if results should be filtered down to one of the three specified dialog tones:\n
    BANTER - personal, playful, joking, anecdotes, interests\n
    DEBATE - political, argumentative, topical rather than personal, opinionated, persuasive\n
    LOGISTICS - planning, strategizing about the life events or content for the podcast, goal-oriented, productive\n
    If 'tone' filter should be applied, you must also revise the input query to scrub out any references to the metadata we plan to filter (so as to achieve cleaner semantic search results given e.g. dialogue exchanges of type BANTER won't likely contain the word 'banter', best to leave them out of the semantic search query).
    If input query concerns which host said something e.g. "Fetch any statements by Jess calling for an alien takeover of earth", those host name(s) must be scrubbed from the query as well (as speaker labels are excluded from the embedded dialog).
    Conversely, if input query concerns the actual mention of a host's name in the dialog, e.g. "Fetch instances where Jack Brett refers to himself in third person", the host name should remain in the query.
    Your output must be a JSON-parsable object with two fields:\n
    query (string) - revised semantic search query excluding references to speaker names or tonal information (or the original query if no host names mentioned and 'tone' filter not applicable)\n
    tone (string) - the tone best represented by the query (must be BANTER, DEBATE, LOGISTICS, or None)\n
    Examples:\n
    1. For query 'Fetch any mentions of TV or movies about serial killers', you would output:\n
    {"query": "TV or movies about serial killers", "tone": "None"}\n
    2. For query 'Fetch any arguments about Trump's alleged pandering to White Supremacists', you would output:\n
    {"query": "Trump pandering to White Supremacists", "tone": "DEBATE"}\n
    3. For query 'Fetch any planning sessions for future episodes', you would output:\n
    {"query": "Planning future episodes", "tone": "LOGISTICS"}\n
    4. For query 'Fetch any jokes about Jack Brett's sparse sexual history', you would output:\n
    {"query": "sparse sexual history", "tone": "BANTER"}\n
    5. For query 'Fetch instances where Jess calls Jack Brett by name while coaching him on Zelda tactics', you would output:\n
    {"query": "Jack Brett Zelda tips and tactics", "tone": "LOGISTICS"}\n
    Return only the JSON object, no prose or additional reasoning.
""")

TONE_DESCRIPTIONS = {
    DialogTone.DEBATE: "political, argumentative, topical rather than personal, opinionated, persuasive",
    DialogTone.BANTER: "personal, playful, joking, anecdotes, interests",
    DialogTone.LOGISTICS: "planning, strategizing about the life events or content for the podcast, goal-oriented, productive",
}
TONE_EXAMPLES = {
    DialogTone.DEBATE: textwrap.dedent("""
        Well if you're going to tell people that what you believe that this is what you actually believe is going on you lying about your podcast is gotta be if it it may not be illegal but it is absolutely 1000% unequivocally unethical, Would you agree with that, Yeah Ok So then you're telling me all of these people all of these mothers with already too much on their plate All of these like scientists who you know want to be banned for saying what they want to try to like no one takes you seriously if you're into this and just nobody does it and like all of this stuff every single one of them is feeding into this, perversion out of monetary gain, the level one the level of unethical just astounds me I don't understand how we as a as a internet savvy technology species would not have immediately sniffed that shit out and it would have blown up all over the internet Like, that doesn't seem that seems like a weird thing that has not happened and should have if that's the case Does that make sense
        Yes Which is what I'm saying is they if that's the case they must be paid actors and if they are paid actors all of them it doesn't make sense that that would not have gotten found out by now and spread all over the internet
        And I'm saying that like they don't have to be paid actors more than uh dear Band and Bors the the like uh that the COVID vaccine skeptic that my father sends me that I you know sent me the links to um like the fact that he was lying through his teeth has resulted in explosive scandal like having an advanced degree, and making a false claim as it happens does not do anything in the in in this day and age Like I mean maybe you've had a bill nice sort of reputation that that might make a difference But some neurologist no one's heard of uh like change like or at least you've never heard of before This podcast changes their mind Oh wow, Like there's just so many ways that
        if this were a courtroom setting like I would say objection asked and answered because you keep on accusing me or you keep on telling me that I'm accusing all these people of lying and being part of a vast conspiracy When I have said no they probably believe that their kid is special They probably believe they have a special way of communicating with them, And in in every single one of these cases there is some explanation other than uh they have telepathic abilities something which is an extraordinary claim which has never been proven And this is not proof that that's all I'm saying I'm not saying they're lying and I don't have to prove anyone's lying I just have to say they don't have proof show me the proof then I'm interested
        Yes But in my mind like I interpreted as like it's a it's a rigged system like the the sheep is the the the voters who the system of democracy makes them think that they have a voice when in fact like elections are controlled by forces beyond our control by the fat cats and, uh in in in the shadows who are just funneling money into these into whatever campaigns are going to support their their pork barrel causes And so like we don't actually have a choice of what's for dinner We just you know we we cast our vote Uh But uh we're we're uh we're dreaming if we think it means a damn thing
        So but but the fact that it's winner take all in almost every state like that and the fact that the like the electoral college being a thing allows for that as opposed to just counting every vote even like third party votes like making it so that your vote actually does count, Um Like, so you you you would still support the electoral college
        Yes I I would say I would say that the winner take all system does more to hurt than help but that the electoral college system itself is still better than popular vote
    """),
    DialogTone.BANTER: textwrap.dedent("""
        I'm trying to figure out what to do tomorrow because like half of it is gonna be taken up by, errands
        As in your friend Erin Talberg or errands as in going to the store and getting things
        the latter, So, I can't do like a full Wednesday thing
        Do you ever run errands with Erin
        No but we couch rot sometimes
        What's cow trotting
        She just comes over for a few hours and we sit on the couch and do nothing
        Oh couch rotting Oh I thought you said cow trotting Like you go out to the ranch and each of you picks out a cow and you just trot them along the trail while you know doing girl talk
        you don't do that
        I mean I'd never heard of the practice of couch rotting which is why I was so Yeah
        The the gift of Gary Oldman's amusement that he gave to the world was just, delightful
        It was amazing, Uh The far took was great I mean it was more than just one but one particularly memorable after the death scene, I'm just gonna add something to this, Yeah
        Yeah That's the reason that I don't feel like you would enjoy the character Jackson and Lamb Are you because like he's Gary Oldman and you can't be sexually attracted to him because of how often he farts Uh but it is a very well written character But anyway um what I what I especially liked about the interview Yeah I guess I mean if you've seen it there's nothing I can tell you about it that you don't know But like I love his tip about playing a evil looking character or a uh um, at least a tough looking character where you ii I wanted to try, you bring up the bring up the head first, than the eyes
        Yeah Yeah he does That that's a thing he does
        How's your workout regimen
        I have not kept up mostly I think because of this fucking crazy project Like I'm I'm working, until the evening times, Still, It yeah they're tomorrow's the last day for it because um the their their presentation is Thursday So it's it's going to end shortly But um and then I will hopefully try to be able to, get back into it, Uh, but yeah I'm gonna and then I'm gonna talk to Jeremy, about quitting, either Thursday or Friday
        Ok, I bet you are super looking forward to that
        so, much
        I told that I told that to my therapist, I was like, I mean it's fine I just you know I am rooting for the aliens I don't care if it's education or a peaceful whatever it is Just let's get this going
        We clearly cannot be trusted to handle our own uh shit So
        I mean she seems like someone who really does her homework and uh totally legit, I would feel more comfortable if like there was a second source or or you know maybe a few more sources verified But of course you know if the alien has a special connection with her, and, you know we'll only communicate through her then I guess nothing we can do about that But you know where have I heard that sort of system before I think, I think like in the Bible with the burning bush and Moses the 10 Commandments on the top of the mountain and all that like it always seems to be just one prophet whose job it is to carry these messages from on high down to us Oh yeah And every other person who's like claimed to uh to to have an alien sighting they they have like a special message for us, Uh but it only goes through them and I feel like I don't know that that's that's just like the whole prophet model, It's hard to put much stock in that because believe it or not a few of these guys have been full of shit
        Some are yeah for sure But not her
        not her Of course I mean I can't I can't she cannot be full of shit You you have no idea how much I need this
        it fills me with hope that we'll see spaceships in the sky visible day and night within a week and a week of a week More of Trump I can do
        And and there's like the leggings as pants phenomena where like that is such a ubiquitous thing Now it's like I mean we're supposed to treat it, as normal, because like sure I've been captivated by the female buttocks but it and it was always such a rare sight to see but now it is very frequent now it is everywhere I look
        If are you saying maybe it's like wait, if you've been seeing women's butts and leggings since birth you might not have the attraction or are you trying to say
        and honestly it's like well I mean you know this about me that it's actually more attractive to see them in the pants oftentimes because it is the modern day spandex Like Lulu's they have the shaping, Yeah they like actually hug you in a flattering way And which is yeah to me seeing that is sexier than seeing a naked body, but I don't know what it means for my sexual identity, and like it was a fetish for me before before it became a ubiquitous fashion trend
    """),
    DialogTone.LOGISTICS: textwrap.dedent("""
        Well you did mention uh the possibility of, releasing the video just like the the human video of us in full length with uh just the uh like um ha have that be the podcast and have the uh cartoon be a segment of it and use it for promotional purposes Um does does that idea is that idea still something you're thinking of
        Um if that's still something you're interested in exploring
        I was thinking that could be a, uh that that like yeah I mean I guess, it depends on the level of interest in uh maybe we we start with the the cartoon version and if there's enough interest in that if if like uh let's let's let somebody demand that hey are the full scale or the full length videos of these uh uh conversations available anywhere online and and then and if there's enough call for that we uh we maybe go in that direction I don't know That sounds that seems like, well yeah
        It would take longer to do the longer animated segments just in terms of how much free time I have So if we wanted quantity, with as best quality as we can right now, it would be, us on video edited into segments into one long thing and then the animated pieces in the middles in the front middle and end, for the sections or whatever that would be much much more manageable time wise, if you would agree to that So then I will then next just take one of these episodes and do that, and then send them to you separately and see, if you think they work as segments for a full, thing or if you just want to release them as separate things or
        Yeah Lovely, Yeah Totally Like since since you've posed this idea of like my brain's been thinking of like how how we would segment up these episode how we would like uh, um cut it down to like the the real meat and potatoes of it Um uh But yeah I I think there's yeah and and I'd be interested in like experimenting with that and seeing how, uh how long a political segment like how long a uh a show is when we trim the ft, Yeah But yeah all that to say I agree with the strategy and I'm happy to be a part of it
        yes, Dragon Con and the fact that Rachel will not be staying with us, we'll get to meet Max and hopefully they are cool
        I'm mostly worried because like the airbnb people are notorious for canceling shit around Dragon Con So like if they don't have a place to stay, Like, what a day in a day
        That is a good well I don't know I I hadn't known that about airbnb but it makes perfect sense Given the, a high value of real estate around the, Atlanta area at the time of Dragon Con historically speaking
        Yes I mean that seems like the only logical, thing to shed And so perhaps, could you push his buttons What what are his buttons that you could, push Maybe like renew your sort of loyalty to Richter Studios in the same conversation that you announced that you can't be with him on this thing
        Yeah How would I do, that
        Um So I I would think like make a video like the sort of video that he would have wanted you to present um, that you know sells the company in the way whatever way he wants you want you to sell it at this presentation and yourself is in it like personal testimonial
        Yeah I think that's a good idea, because it requires it only requires my time to sacrifice and it would alleviate a huge thing So maybe I can like, record it somehow, and just be like hey but I still I still need a reason why I can't go
        Um, and that that that's where the having to lie comes in unless you actually do get COVID or something Um, yeah But like yeah because I guess it's an untenable situation Like it is not untenable but it is one where you have to do the unfortunate thing of lying as you know, we as as humans like to avoid but you do have obligation to your employer and obligation to those freelancers and neither of them know about each other So yeah
        I have put myself in the chaos yes
        something's gotta give, Uh And unfortunately that means that you have to tell a falsehood of some kind to free up a bit of your time Um And yeah I would do that in your position without hesitation
        Oh, I looked up a prompt engineer job
        Oh Yeah The Prompt engineer Um, It's not impossible for people without a degree in it to get it and you don't necessarily need to know how to code but it is really helpful, Um, And so I would have to dig into it far deeper to know if it would be worth my time
        If you want some coding lessons I can do that, I like the coding lessons that are like specific to the field of um you know interacting with large language models It's you would be you would pick it up like with your like experience with animation and you know just kind of like programmatic elements I feel like you you would you would pick up on it
        I'm thinking, about it I I don't even know if it's worth the time to delve into it yet So I have to still consider, that part
        But yeah like even if you think that just setting up a local Python environment would inspire you to, explore on your own I could do that just like take that simple step and you know just give you the playground to do what you will
    """),
}

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

def get_query_type(query: str) -> QueryTypeDecision:
    GET_QUERY_TYPE_PROMPT = textwrap.dedent("""
        You are a helpful research assistant for the However Comma podcast hosted by Jack Brett and Jess.
        Your task is to analyze a search query and classify it as either RAG_SEARCH or DIRECT_SEARCH, which will determine whether to embed it as is for semantic search (K-NN) against a vector index, or use it to plan out a more sophisticated RAG strategy including metadata filters.\n
        You will classify it as follows:\n
        RAG_SEARCH - query is a clear instruction from a user to a research tool (as opposed to a raw search string) and/or contains information that could be used to filter search results by host name, episode date, etc.\n
        DIRECT_SEARCH - query appears to be a raw search string intended to be used directly in semantic search, contains no instructions on how to perform search or what metadata field(s) to filter by\n
        Respond ONLY a JSON object with a single key 'label' whose value is your classification:\n
        {"label": string}\n
        Examples:\n
        1. For query "Please find examples of Jack Brett telling Jess to 'get out' and Jess replying 'i will not'", you would output:\n
        {"label": "RAG_SEARCH"}\n
        2. For query "Please give examples of Jack and Jess in heated debate over which political party is more guilty of fear mongering", you would output:\n
        {"label": "RAG_SEARCH"}\n
        3. For query "Trump is not opposed to introducing his cabinet members to the undercarriage of a bus", you would output:\n
        {"label": "DIRECT_SEARCH"}\n
        4. For query "It's hard to give our elected leaders the benefit of the doubt because...history?", you would output:\n
        {"label": "DIRECT_SEARCH"}\n
    """)
    query_type_decision = anthropic_client.messages.create(
        model=CHAT_MODEL_ID,
        response_model=QueryTypeDecision,
        max_tokens=500,
        temperature=0.2,
        system=GET_QUERY_TYPE_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Query:\n{query}"
            }
        ]
    )
    return query_type_decision

def get_hyde_prompt_for_query_and_tone(query: str, tone: str) -> str:
    description = TONE_DESCRIPTIONS[tone]
    sample_dialog = TONE_EXAMPLES[tone]
    return textwrap.dedent(f"""
        Your task is to generate realistic dialogue exchanges matching a specific tone to answer a user query.
        TONE: {tone}
        FEATURES: {description}
        EXAMPLES:
        {sample_dialog}
        Generate 3 exchanges separated by new lines, matching this tone and relevant to the query: {query}
    """)

def generate_hypothetical_documents(query: str, tone: DialogTone) -> list[str]:
    hy_docs_resp = bedrock_runtime.invoke_model(
        modelId=CHAT_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 600,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": get_hyde_prompt_for_query_and_tone(query, tone)
                }
            ]
        })
    )
    resp_body = json.loads(hy_docs_resp["body"].read())
    llm_output = resp_body["content"][0]["text"]
    # filter out any blank/whitespace lines from result
    return [line for line in llm_output.split('\n') if line.strip()]

def get_knn_query_for_line_index(query: str) -> LineQueryKNN:
    line_query = anthropic_client.messages.create(
        model=CHAT_MODEL_ID,
        response_model=LineQueryKNN,
        max_tokens=500,
        temperature=0.2,
        system=QUERY_TO_LINE_KNN_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Query:\n{query}"
            }
        ]
    )
    return line_query

def get_knn_query_for_dialog_index(query: str) -> DialogQueryKNN:
    line_query = anthropic_client.messages.create(
        model=CHAT_MODEL_ID,
        response_model=DialogQueryKNN,
        max_tokens=500,
        temperature=0.2,
        system=QUERY_TO_DIALOG_KNN_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Query:\n{query}"
            }
        ]
    )
    return line_query

def get_top_k_vectors(index_name: str, text: str, filter_exp: Dict[str, Any] | None = None, k: int = 8) -> List[Dict[str, Any]]:
    query_embedding = embed_text(text)
    
    response = s3vectors.query_vectors(
        vectorBucketName=VECTOR_BUCKET_NAME,
        indexName=index_name,
        queryVector={"float32": query_embedding},
        topK=k,
        returnMetadata=True,
        returnDistance=True,
        filter=filter_exp
    )
    return response.get("vectors", [])

def retrieve_line_vectors(query: str, max_hits: int = 25) -> List[RelevantSource]:
    all_vectors = []
    relevant_sources: List[RelevantSource] = []
    # break down query into search tasks
    search_tasks = get_search_tasks_for_query(query)
    # run kNN search for each task
    for task in search_tasks:
        print(f"TASK: {task}")
        knn_query = get_knn_query_for_line_index(task)
        print(f"query: {knn_query.query}")
        print(f"filter_exp: {knn_query.filter_exp}")
        all_vectors.extend(get_top_k_vectors(
            LINE_INDEX_NAME,
            text=knn_query.query,
            filter_exp=knn_query.filter_exp,
            k=TOP_K
        ))
    # deduplicate by key
    deduped_vectors = dedupe_line_vectors_by_key(all_vectors)
    # order by k-NN distance
    ordered_vectors = sorted(deduped_vectors, key=lambda x: x['distance'], reverse=True)
    # filter results by relevance to query
    for i in range(len(ordered_vectors)):
        if len(relevant_sources) >= max_hits:
            # return early if we have enough relevant hits
            return relevant_sources

        vector = ordered_vectors[i]
        try:
            transcript_exchange = get_n_surrounding_lines(vector, n=5)
        except ValueError:
            metadata = vector.get("metadata")
            print(f"Line not found in DynamoDB: episode_id: {metadata.get("episode_id")}, text: {metadata.get("text")}, line_id: {vector.get("key")}")
            continue

        str_exchange = stringify_exchange(transcript_exchange)
        relevance: RelevanceDecision = get_exchange_relevance(query, str_exchange)
        print(f"RELEVANCE: {relevance}")
        if relevance.is_relevant:
            relevant_sources.append(RelevantSource(
                episode_id=transcript_exchange.episode_id,
                date=transcript_exchange.date,
                timestamp=transcript_exchange.timestamp,
                text=str_exchange,
                score=transcript_exchange.score
            ))
    # we didn't meet the max_hits benchmark. return what we've got
    return relevant_sources

def retrieve_dialog_vectors(query: str, max_hits: int = 25) -> List[RelevantSource]:
    all_vectors = []
    relevant_vectors: List[RelevantSource] = []
    # remove host names from query (no speaker labels in dialog index)
    no_hostname_query = scrub_host_names_from_query(query)
    print(f"no_hostname_query: {no_hostname_query}")
    all_vectors = get_top_k_vectors(
        DIALOG_INDEX_NAME,
        text=no_hostname_query,
        filter_exp=None,
        k=TOP_K
    )
    # deduplicate by key
    deduped_vectors = dedupe_line_vectors_by_key(all_vectors)
    # order by k-NN distance
    ordered_vectors = sorted(deduped_vectors, key=lambda x: x['distance'], reverse=True)
    # filter results by relevance to query
    for i in range(len(ordered_vectors)):
        if len(relevant_vectors) >= max_hits:
            # return early if we have enough relevant hits
            return relevant_vectors

        vector = ordered_vectors[i]
        # transcript_exchange = get_n_surrounding_lines(vector, n=5)
        metadata = vector.get("metadata")
        str_exchange = metadata.get("text")
        relevance: RelevanceDecision = get_exchange_relevance(no_hostname_query, str_exchange)
        print(str_exchange)
        print(f"RELEVANCE: {relevance}")
        if relevance.is_relevant:
            k_distance = vector.get("distance", 1)
            score = 100 - int(k_distance * 100)
            relevant_vectors.append(RelevantSource(
                episode_id=metadata["episode_id"],
                date=metadata["date"],
                timestamp=str(metadata["start_time"]),
                text=str_exchange,
                score=score
            ))
    # we didn't meet the max_hits benchmark. return what we've got
    return relevant_vectors

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

def scrub_host_names_from_query(query: str) -> str:
    SCRUB_HOST_NAMES_PROMPT = textwrap.dedent(f"""
        You are a helpful research assistant for the However Comma podcast hosted by Jack Brett and Jess.
        Your task is to scrub the user query of references to the two hosts, preserving as much of the semantic meaning as possible.
        Only scrub references to Jack Brett and Jess. Any other proper nouns must be preserved.
        Examples:\n
        1. For the query "Fetch any statements that indicate Jack Brett's position on free will", you would output:\n
        "Fetch any statements that indicate a position on free will"\n
        2. For the query "Fetch any argumentative statements Jess made about Trump's alleged pandering to White Supremacists in early 2025 (January-March)", you would output:\n
        "Fetch any argumentative statements about Trump's alleged pandering to White Supremacists in early 2025 (January-March)"
        3. For the query "Fetch any heated discussions about the death of Charlie Kirk", you would output the same query unaltered:\n
        "Fetch any heated discussions about the death of Charlie Kirk"
        Output only the edited query, no prose or additional reasoning.
        USER QUERY: {query}
    """)
    response = bedrock_runtime.invoke_model(
            modelId=CHAT_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 600,
                "temperature": 0.2,
                "messages": [
                    {
                        "role": "user",
                        "content": SCRUB_HOST_NAMES_PROMPT
                    }
                ]
            })
        )
    resp_body = json.loads(response["body"].read())
    llm_output = resp_body["content"][0]["text"]
    return llm_output

def dedupe_line_vectors_by_key(
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

def get_n_surrounding_lines(vector: Dict[str, Any], n: int = 3) -> TranscriptExchange:
    """
    Fetch up to n transcript lines immediately before and after line_id.
    If the line is within fewer than n lines of the episode boundary,
    only the available lines in that direction are returned.
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
        items = fetch_line_resp.get("Items", [])
        if not items:
            raise ValueError(f"No transcript line found for line_id: {line_id}")

        item = items[0]
        episode_id = item["episode_id"]
        episode_date = item["date"]
        line_position = item["line_position"]

        start_position = max(0, line_position - n)

        adjacent_lines_resp = transcript_table.query(
            KeyConditionExpression=
                "episode_id = :ep AND line_position BETWEEN :start AND :end",
            ExpressionAttributeValues={
                ":ep": episode_id,
                ":start": start_position,
                ":end": line_position + n,
            },
            ScanIndexForward=True
        )
        adjacent_lines = adjacent_lines_resp.get("Items", [])
        if not adjacent_lines:
            raise ValueError(f"No adjacent lines found for episode: {episode_id}, position: {line_position}")

        start_timestamp = adjacent_lines[0]["timestamp"]
        return TranscriptExchange(
            episode_id=episode_id,
            date=episode_date,
            score=score,
            lines=adjacent_lines,
            timestamp=start_timestamp
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

def format_exchange_relevance_prompt(user_query: str, exchange: str) -> str:
    
    return textwrap.dedent(f"""
        You are a strict relevance judge for a research assistant.
        Your job is to decide whether a transcript excerpt is useful for answering the user's original query.
        Be conservative.
        If the excerpt is only tangentially related, judge it as NOT RELEVANT.

        USER QUERY:
        {user_query}

        {exchange}

        Is this excerpt useful for answering the user's question?

        Respond ONLY in JSON with this schema:
        {{
        "relevant": boolean,
        "confidence": number,   // 0-100
        "reason": string        // short, max 1 sentence
        }}
    """)

def get_exchange_relevance(user_query: str, exchange: str) -> RelevanceDecision:
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

def get_curated_context(query: str, max_hits: int = 25) -> List[RelevantSource]:
    relevant_sources: List[RelevantSource] = []
    # Run query against LINE INDEX first
    line_vectors = retrieve_line_vectors(query, max_hits)
    print(f"{len(line_vectors)} relevant line vectors")
    for lv in line_vectors:
        print(lv.text)
    relevant_sources.extend(line_vectors)

    dialog_vectors = retrieve_dialog_vectors(query, max_hits)
    print(f"{len(dialog_vectors)} relevant dialog vectors")
    for dv in dialog_vectors:
        print(dv.text)
    relevant_sources.extend(dialog_vectors)
    return relevant_sources

def build_rag_prompt_with_context(
    latest_user_query: str,
    context: List[RelevantSource],
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
        blocks = [exchange.text for exchange in context]
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

def direct_search(query: str) -> QueryResponse:
    top_hits = get_top_k_vectors(LINE_INDEX_NAME, query)
    deduped_vectors = dedupe_line_vectors_by_key(top_hits)
    # order by k-NN distance
    ordered_vectors = sorted(deduped_vectors, key=lambda x: x['distance'], reverse=True)
    sources = []
    for vector in ordered_vectors:
        metadata = vector.get("metadata")
        k_distance = vector.get("distance", 1)
        score = 100 - int(k_distance * 100)

        sources.append(RelevantSource(
            episode_id=metadata.get("episode_id"),
            date=metadata.get("date"),
            timestamp=metadata.get("timestamp"),
            text=metadata.get("text"),
            score=score
        ))
    return QueryResponse(
        answer="Your search returned the following excerpts:",
        sources=sources
    )

def rag_search(query: str, messages: List[ChatMessage]) -> QueryResponse:
    relevant_sources = get_curated_context(query)
    # assemble RAG prompt
    prompt = build_rag_prompt_with_context(
        query,
        relevant_sources,
        messages,
    )
    # get answer from LLM
    answer = call_llm(prompt)

    return QueryResponse(
        answer=answer,
        sources=relevant_sources
    )

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

        query_response: QueryResponse | None = None
        # determine if this is a DIRECT or RAG search
        query_type = get_query_type(user_query)
        if query_type.label == QueryType.DIRECT_SEARCH:
            query_response = direct_search(user_query)
        elif query_type.label == QueryType.RAG_SEARCH:
            query_response = rag_search(user_query, messages)

        if not query_response:
            raise Exception("Search failed. Could not identify query type")

        sources = []
        # format sources for storage in dynamodb table
        for vector in query_response.sources:
            sources.append({
                "episode_id": vector.episode_id,
                "episode_name": episode_id_to_name(vector.episode_id),
                "text": vector.text,
                "date": vector.date,
                "timestamp": vector.timestamp,
                "score": vector.score,
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
                ":answer": query_response.answer,
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
