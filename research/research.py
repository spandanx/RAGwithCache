import time

from langchain_openai import OpenAIEmbeddings

import redis

import os
#------------------
from configparser import ConfigParser
parser = ConfigParser()
config_file_path = '../config.properties'
with open(config_file_path) as f:
    file_content = f.read()
parser.read_string(file_content)
#------------------
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings(dimensions=768, model="text-embedding-3-small")
#------------------
from redisvl.extensions.cache.llm import SemanticCache

# )
# parser['CACHE']['cache_key']
redis_client = redis.Redis(host=parser['CACHE']['cache_url'], port=parser['CACHE']['cache_port'],
                           db=parser['CACHE']['cache_db'], password=parser['CACHE']['cache_password'])

llmcache = SemanticCache(
    name="llmcache",  # The name of the search index in Redis
    # redis_url="redis://103.180.212.180:6379",  # Connection URL for Redis
    redis_client=redis_client,
    distance_threshold=0.1  # Similarity threshold for cache matching
)

def embed_query(text):
    embd_qry = embedding.embed_query(text)
    return embd_qry



def answer_question_with_cache(query_text):
    start_time = time.time()

    # Embed the query
    query_vector = embed_query(query_text)

    # Check cache first
    cached_result = llmcache.check(vector=query_vector)

    if cached_result:
        # Cache hit - return immediately
        elapsed = time.time() - start_time
        print(f"Cache HIT - Response time: {elapsed:.2f}s")
        return cached_result[0]['response']

    # Cache miss - run full RAG pipeline
    # answer = await answer_question(index, query_text)
    answer = "Here is the sample result for the query - " + query_text

    # Store in cache for future queries
    llmcache.store(
        prompt=query_text,
        response=answer,
        vector=query_vector
    )

    elapsed = time.time() - start_time
    print(f"Cache MISS - Response time: {elapsed:.2f}s")
    return answer

if __name__ == "__main__":
    # embds = []
    # query = "What is the longest river in world?"
    # embds.append(embed_query(query))
    # query = "Which is the longest river in the world?"
    # embds.append(embed_query(query))
    query = "What is the longest river in Earth"
    # query = "Tell me the name of the longest river in the world."
    # embds.append(embed_query(query))
    # x = 1
    answer = answer_question_with_cache(query)
    print(answer)
