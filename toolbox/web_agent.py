from neo4j import GraphDatabase

# 设置Neo4j数据库连接
NEO4J_URI = "bolt://localhost:7687"  # 替换为你的Neo4j URI
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
import openai
from openai import AzureOpenAI
from openai import OpenAI
import os
from dotenv import load_dotenv
import traceback
from PIL import Image
load_dotenv()
import base64,io
import requests
from bs4 import BeautifulSoup
import re
import os
import requests
import matplotlib.pyplot as plt
from io import BytesIO
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import BedrockChat

from langchain_community.graphs import Neo4jGraph

from langchain_community.vectorstores import Neo4jVector

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_community.llms import Ollama
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import Neo4jVector
from langchain_community.chat_models import ChatOllama
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from typing import List, Any
from toolbox.utils import BaseLogger, extract_title_and_question
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from toolbox.utils import ImageDownloader, convert_to_base64
from PIL import Image
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

import ast


query="""
MATCH (n1)-[r]->(n2)
WHERE n1.wikidata_id IS NOT NULL AND n1.name IS NOT NULL AND n2.wikidata_id IS NOT NULL AND n2.name IS NOT NULL
RETURN n1.wikidata_id + " (" + n1.name + ")" + "->" + type(r) + "->" + n2.wikidata_id + " (" + n2.name + ")" AS edge_representation
LIMIT $limit
    """
#with driver.session() as session:
#    result = session.run(query, limit=2000)
#    for record in result:
#        print(f"edge: {record['edge_representation']}")

base_url="https://www.wikidata.org/wiki/"

def retrieve_content(url):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    try:
        driver.get(url)
        time.sleep(2)  # Wait for the page to load

        # Get page content
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract text content
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])

        return content

    except Exception as e:
        print(f"Failed to retrieve content from {url}: {e}")
        return None

    finally:
        driver.quit()

def extract_top_k_links(weburl, topK, question):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(weburl)
    time.sleep(3)
    # Parse the page source with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Extract all links
    links = soup.find_all('a', href=True)

    template="""
    Your task is to find and output the top K urls that most closely match a given question. You should do this even if some of the required data is not explicitly included in the provided dictionary.
    You are provided with amount of links, K number requirement and question
    Don't generate anything else! Follow the Output Requirement!
    Output Requirements:

    Only output the top K link values. Do not provide any other information or responses.
    The output format should be a list, for example: [https://, ..., https://].
    """


    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    base_url = os.getenv('AZURE_OPENAI_ENDPOINT')
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-06-01",
        azure_endpoint=base_url
        )

    system_input = {
        "role": "system",
        "content": template
    }

    user_input = {
        "role": "user",
        "content": f"The question is {question} Top K number is {topK} The links pool is {links}"
    }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[system_input, user_input],
        temperature=0
    )
    # Filter and select top K links based on criteria (e.g., domain reliability, link text relevance)
    response_text = response.choices[0].message.content
    #print(response_text)
    return response_text

def website_selection(question, website_list, topK):
    template="""
    Your task is to find and output the top K urls that most closely match a given question. You should do this even if some of the required data is not explicitly included in the provided list pool.
    You are provided with amount of links, K number requirement and question.
    Note that only provide websites that are readable, so don't give any link that contains 'jpg' or other image format in it.
    Don't generate anything else! Follow the Output Requirement!
    Output Requirements:

    Only output the top K link values. Do not provide any other information or responses.
    The output format should be a list, for example: [https://, ..., https://].
    """


    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    base_url = os.getenv('AZURE_OPENAI_ENDPOINT')
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-06-01",
        azure_endpoint=base_url
        )

    system_input = {
        "role": "system",
        "content": template
    }

    user_input = {
        "role": "user",
        "content": f"The question is {question} Top K number is {topK} The links pool is {website_list}"
    }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[system_input, user_input],
        temperature=0
    )
    # Filter and select top K links based on criteria (e.g., domain reliability, link text relevance)
    response_text = response.choices[0].message.content
    #print(response_text)
    return response_text



def judgement(question, content):
    template="""
    Your task is to respond to the given question using the content that has already been carefully retrieved to align with the question's specific requirements.

You must strictly base your answer on the provided materials, ensuring no additional or speculative information is generated.
The content you are provided with has been specifically gathered to match the question's criteria, so focus entirely on what is available.
Do not introduce any external data or assumptions—stick closely to the provided Output Guidelines.
Output Guidelines:

If you determine that the retrieved content sufficiently answers the question, respond with 'Yes' and clearly state the answer.
If the content is inadequate or incomplete for answering the question, simply respond with 'No' and nothing else.
Output Example:

'No' 
'Yes, the answer to the question is, ...' 
    """
    template2="""
Your task is to respond to the given question using the content that has already been carefully retrieved to align with the question's specific requirements, like retreived from a certain location or time based on the question's requirement.
Thus just give an answer, don't always refer the material, it's just a back-up.
Even if the exact information isn't found in the website content, you are still expected to produce an informed response.
You have access to a wide range of website content, which should help you craft your answer.
Strictly adhere to the output guidelines below.

Output Requirements:

Deliver a to the question, answer it, and material is just an addition.
Output Example:

"The answer to the question is, ..."
    """
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    base_url = os.getenv('AZURE_OPENAI_ENDPOINT')
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-06-01",
        azure_endpoint=base_url
        )

    # Prepare the system and user prompts
    system_input = {
        "role": "system",
        "content": template2
    }

    user_input = {
        "role": "user",
        "content": f"The question is {question} The website contents is {content}"[:10000]
    }

    #We need some algorithm that simplify the html content.
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[system_input, user_input],
        temperature=0
    )

    return response.choices[0].message.content



def iterative_searcher(question,wikidata_list):
    #print(wikidata_list)
    #print(type(wikidata_list))
    wikidata_list = re.search(r'\[(.*?)\]', wikidata_list).group(1)
    elements = [element.strip() for element in wikidata_list.split(',')]
    # 对每个元素添加引号
    quoted_elements = [f'"{element}"' for element in elements]
    # 将处理后的内容放回原来的字符串
    wikidata_list = "[" + ", ".join(quoted_elements) + "]"
    #print(wikidata_list)
    wikidata_list=ast.literal_eval(wikidata_list)
    website_to_brows=[]
    for wikidata_key in wikidata_list:
        browse_website=base_url+wikidata_key
        website_to_brows.append(browse_website)
    while(True): 
        all_content_list=[]
        website_list=[]
        for link in website_to_brows:
            #browse_website=base_url+wikidata_key
            website_list.append(link)
            web_content=retrieve_content(link)
            print(web_content)
            all_content_list.append(web_content)

    
        #length_limit=10000
        #all_content_list=all_content_list
        temp_answer=judgement(question, all_content_list)
        #print(temp_answer)
        #print(a)
        if temp_answer!="No":
            return temp_answer
        else:
            websit_list_all=[]
            for websit in website_list:
                print(websit)
                relevant_websites=extract_top_k_links(websit,2,question)
                websit_list_all.append(relevant_websites)
            Extracted_topK_websites=website_selection(question,websit_list_all,2)
            print(Extracted_topK_websites)
            website_to_brows=Extracted_topK_websites

    
from sentence_transformers import SentenceTransformer
from py2neo import Graph
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def web_search_agent(llm, question, early_stop, limit_N, topK, retriev_option=0):
    #Sample all wikidata-name pairs 
    if retriev_option ==0:
        query="""
    MATCH (n1)-[r]->(n2)
    RETURN n1.wikidata_id + " (" + n1.name + ")" + "->" + type(r) + "->" + n2.wikidata_id + " (" + n2.name + ")" AS edge_representation
    LIMIT 500
        """

        #Need to retreive based on query
        with driver.session() as session:
            result = session.run(query, question=question, limit=limit_N)
            print(result)

            #First and most naive method, just use LLM to tell which top K methods are best.
            result_dict=[]
            for record in result:
                node_data = {
                    "edge": record["edge_representation"],
                }
                result_dict.append(node_data)
    elif retriev_option==1:
        query="""
    MATCH (n1)-[r]->(n2)
    RETURN n1.wikidata_id + " (" + n1.name + ")" + "->" + type(r) + "->" + n2.wikidata_id + " (" + n2.name + ")" AS edge_representation
        """
        with driver.session() as session:
            edges=session.run(query)
            print(edges)

            question_embedding = sentence_model.encode(question).reshape(1, -1)
        
            # For simplicity, we'll use dummy embeddings for edges here.
            # In a real scenario, you should calculate or fetch edge embeddings.
            #edge_embeddings = [sentence_model.encode(edge) for edge in edges]
            
            # Compute similarities between question and each edge
            edge_embeddings = np.array([sentence_model.encode(edge) for edge in edges])
    
            # Compute similarities
            similarities = []
            for edge, edge_embedding in zip(edges, edge_embeddings):
                edge_embedding = edge_embedding.reshape(1, -1)
                similarity = cosine_similarity(edge_embedding, question_embedding)[0][0]
                similarities.append((edge, similarity))
            
            result_dict = []
            for edge, similarity in similarities:
                node_data = {
                    "edge": edge,
                    "similarity": similarity
                }
                result_dict.append(node_data)

            # Sort and get top K results
            top_k = 500
            top_k_results = sorted(result_dict, key=lambda x: x["similarity"], reverse=True)[:top_k]

            print(top_k_results)
            return top_k_results
                
    

    template = """
    Your task is to find and output the top K wikidata_id values that most closely match a given question. You should do this even if some of the required data is not explicitly included in the provided dictionary.
    You are provided with a dictionary of edges and each is with its wikidata_id
    Don't generate anything else! Follow the Output Requirement!
    Output Requirements:

    Only output the top K wikidata_id values. Do not provide any other information or responses.
    The output format should be a list, for example: [Q2697746, Q12345, Q89667, ...].
    """
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "The question is {question} Top K number is {topK} The dictionary is {result_dict}"
    #topK_template= 
    #dict_template=
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    #topK_message_prompt = HumanMessagePromptTemplate.from_template(topK_template)
    #dict_message_prompt = HumanMessagePromptTemplate.from_template(dict_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    def generate_llm_output(prompt=chat_prompt) -> str:
        chain = prompt | llm
        prompt_input = {
        "question": question,
        "topK": topK,
        "result_dict": result_dict
        }
        answer = chain.invoke(prompt_input).content
        #print(answer)
        return answer
    
    #result_wikidatas=generate_llm_output()
    """
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    base_url = os.getenv('AZURE_OPENAI_ENDPOINT')
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-06-01",
        azure_endpoint=base_url
        )

    # Prepare the system and user prompts
    system_input = {
        "role": "system",
        "content": template
    }

    user_input = {
        "role": "user",
        "content": f"The question is {question} Top K number is {topK} The dictionary is {result_dict}"
    }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[system_input, user_input],
        temperature=0
    )

    response_text = response.choices[0].message.content
    print(response_text)
    #print(a)
    final_result=iterative_searcher(question=question, wikidata_list=response_text)
    #目前不用做这一部分
    return final_result



