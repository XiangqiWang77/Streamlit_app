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
from toolbox.chains import Find_URLs,image_url_finder
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from toolbox.utils import ImageDownloader, convert_to_base64
from PIL import Image
from bs4 import BeautifulSoup
import json
from typing import Dict

import openai
from openai import AzureOpenAI
from openai import OpenAI
import os
from dotenv import load_dotenv
import traceback
from PIL import Image
load_dotenv()
import ast


def extract_keywords_from_question(llm, question: str) -> Dict[str, str]:
    print("question is", question)
    """
    Use the LLM to extract relevant keywords from a user's question and map them to the required kwargs.
    
    Args:
        question (str): The user question from which to extract keywords.

    Returns:
        Dict[str, str]: A dictionary of kwargs for the Cypher query.
    """
    # Define a prompt to guide the LLM in extracting relevant information
    prompt = """
    You are an intelligent assistant that helps with querying a Neo4j database, note that all multimedia information must be given in the mutimedia_id, not in wikidata_id.
   Instructions for Neo4j Query Assistance:

Use multimedia_id for Multimedia References:
Always refer to multimedia information using multimedia_id. Avoid using wikidata_id unless it is specifically mentioned in the question.

Handling Multimedia Information:
If information can only be conveyed through multimedia, do not generate or return it as a plain value. Instead, ensure it is appropriately referenced via multimedia_id.
Wildcard Representation:
If no specific criteria or requirements are provided for relationships (e.g., no edge type specified), use name to represent the relationship type and value just use nothing.
Minimize Use of wikidata_id:
Refrain from referring to wikidata_id unless explicitly required or mentioned. Focus on using multimedia_id as the primary identifier.
    Given a question about animal observations in Florida, identify the relevant values for the following fields:
    - specy_type: The type of species (only formatted like, Bird_name, Reptile_name with _name suffix).
    - location_attribute: An attribute of the location (only contain attribute, name, wikidata_id).
    - location_value: The value for the location attribute (e.g., name is like Alachua).
    - specy_attribute: An attribute of the species (only contain attribute name, wikidata_id).
    - specy_value: The value for the species attribute (e.g., name is like Panther, Eagle).
    - observation_attribute: An attribute of the observation edge (only contain date, multimedia, observed_times).
    - observation_value: The value for the observation attribute (e.g., date like 2024-04-16T11:49:47, multimedia like https://www.inaturalist.org/observations/207625017 , observed_times like 1).

    Output Requirements:

    You  must generate output following the output requirement of a dict formatted like this:

    {
    "specy_type": "Reptile_name",
    "location_attribute": "name",
    "location_value": "Alachua",
    "specy_attribute": "name",
    "specy_value": "Great Egret",
    "observation_attribute": "observed_times"
    "observation_value": 1
    }

    Extract and return these fields as a str(not json) based on the given question.


    """

    # Get the response from the LLM
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    base_url = os.getenv('AZURE_OPENAI_ENDPOINT')
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-06-01",
        azure_endpoint=base_url
        )

    system_input = {
        "role": "system",
        "content": prompt
    }

    user_input = {
        "role": "user",
        "content": f"The question is {question}"
    }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[system_input, user_input],
        temperature=0
    )
    # Filter and select top K links based on criteria (e.g., domain reliability, link text relevance)
    response_text = response.choices[0].message.content

    # Return the extracted keywords
    return response_text


def extract_interact_keywords(llm, question: str) -> Dict[str, str]:
    print("question is", question)
    prompt="""
You are an intelligent assistant specializing in querying a Neo4j database for animal interactions, with the following key instructions:
Wildcard Representation
When no specific criteria or requirements are provided (e.g., for relationship types), represent the relationship by name, and leave the value unspecified.
Minimize wikidata_id Usage
Refrain from using wikidata_id unless it is explicitly required. Focus on multimedia_id as the primary identifier.
Query Instructions for Animal Interaction in Florida:
When tasked with identifying relevant values for a species interaction, you should extract the following fields based on the given question:

Output Requirements:

    You  must generate output following the output requirement of a dict formatted like this:

    {
    "source_name": "Reptile_name",
    "target_name": "Amphibian_name",
    "source_attribute": "wikidata_id",
    "source_value": Q123897,
    "target_attribute": "wikidata_id",
    "target_value": Q641909
    }

    Extract and return these fields as a str(not json) based on the given question.
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
        "content": prompt
    }

    user_input = {
        "role": "user",
        "content": f"The question is {question}"
    }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[system_input, user_input],
        temperature=0
    )
    # Filter and select top K links based on criteria (e.g., domain reliability, link text relevance)
    response_text = response.choices[0].message.content

    # Return the extracted keywords
    return response_text

def extract_temporal_keywords(llm, question: str) -> Dict[str, str]:
    print("question is", question)
    prompt="""
You are an intelligent assistant specializing in querying a Neo4j database for animal location observations with temporal requirement, with the following key instructions:
Wildcard Representation
When no specific criteria or requirements are provided (e.g., for relationship types), represent the relationship by name, and leave the value unspecified.
Minimize wikidata_id Usage
Refrain from using wikidata_id unless it is explicitly required. Focus on multimedia_id as the primary identifier.
Query Instructions for Animal Location observations in Florida:
When tasked with identifying relevant values for a specy animal observations, you should extract the following fields based on the given question:
You need to note that specy type has '_name' as suffix, so specy_type should formatted differently.
Output Requirements:

    You  must generate output following the output requirement of a dict formatted like this:

    {
    "specy_type": "Reptile_name",
    "location_attribute": "name",
    "location_value": "Alachua",
    "specy_attribute": "name",
    "specy_value": "",
    "observed_times_relationship": "CONTAINS",
    "observed_dates_relationship": "CONTAINS"
    "latest_observed": ""
    "observation_times": 
    }

    Extract and return these fields as a str(not json) based on the given question.
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
        "content": prompt
    }

    user_input = {
        "role": "user",
        "content": f"The question is {question}"
    }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[system_input, user_input],
        temperature=0
    )
    # Filter and select top K links based on criteria (e.g., domain reliability, link text relevance)
    response_text = response.choices[0].message.content

    # Return the extracted keywords
    return response_text

def construct_cypher_query(kwargs,multi_option):
    # Set default values or extract from kwargs
    specy_type = kwargs.get("specy_type", "")
    location_attribute = kwargs.get("location_attribute", "")
    location_value = kwargs.get("location_value", "")
    specy_attribute = kwargs.get("specy_attribute", "")
    specy_value = kwargs.get("specy_value", "")
    observation_attribute = kwargs.get("observation_attribute", "")
    observation_value = kwargs.get("observation_times", "")

    conditions = []
    # Construct the query with the provided or default values
    if location_value:  # Check if location_value is not empty
        conditions.append(f'l.{location_attribute} CONTAINS "{location_value}"')
    
    if specy_value:  # Check if specy_value is not empty
        conditions.append(f'r.{specy_attribute} CONTAINS "{specy_value}"')
    
    if observation_value:  # Check if observation_value is not empty
        conditions.append(f'o.{observation_attribute} CONTAINS "{observation_value}"')

    # Combine conditions into the WHERE clause
    where_clause = " AND ".join(conditions)
    
    cypher_query = f"""
    MATCH (r:{specy_type})-[o:OBSERVED_AT]->(l:Location)
    """
    if where_clause:
        cypher_query += f"WHERE {where_clause}\n"

    if multi_option==1:
        cypher_query += """
        RETURN r.name as specy_name,o.multimedia as Multimedia,l.name as location_name
        """
    else:
        cypher_query += """
        RETURN r,o,l
        """

    return cypher_query




def create_relationship_query(kwargs):
     # Set default values or extract from kwargs
    source_name = kwargs.get("source_name", "")
    target_name = kwargs.get("target_name", "")
    source_attribute = kwargs.get("source_attribute", "")
    source_value = kwargs.get("source_value", "")
    target_attribute = kwargs.get("target_attribute", "")
    target_value = kwargs.get("target_value", "")

    conditions = []
    # Construct the query with attribute values of source_specy and target_specy
    if source_value:  # Check if source_value is not empty
        conditions.append(f's.{source_attribute} CONTAINS "{source_value}"')

    if target_value:  # Check if target_value is not empty
        conditions.append(f't.{target_attribute} CONTAINS "{target_value}"')

    # Combine conditions into the WHERE clause
    where_clause = " AND ".join(conditions)

    # Construct Cypher query for 'interactsWith' and 'preysOn' edges
    cypher_query = f"""
    MATCH (s:{source_name})-[i]->(t:{target_name})
    """

    if where_clause:
        cypher_query += f"WHERE {where_clause}\n"

    cypher_query += """
    RETURN s.name AS source_specy, i.type AS interaction, t.name AS target_specy, p.type AS prey_interaction
    """

    return cypher_query

def create_temporal_query(kwargs):
    specy_type = kwargs.get("specy_type", "")
    location_attribute = kwargs.get("location_attribute", "")
    location_value = kwargs.get("location_value", "")
    specy_attribute = kwargs.get("specy_attribute", "")
    specy_value = kwargs.get("specy_value", "")
    latest_observed = kwargs.get("latest_observed", "")
    observation_value = kwargs.get("observed_times", "")
    relationship_1=kwargs.get("observed_times_relationship", "")
    relationship_2=kwargs.get("observed_dates_relationship", "")

    conditions = []
    # Construct the query with the provided or default values
    if location_value:  # Check if location_value is not empty
        conditions.append(f'l.{location_attribute} CONTAINS "{location_value}"')
    
    if specy_value:  # Check if specy_value is not empty
        conditions.append(f'r.{specy_attribute} CONTAINS "{specy_value}"')
    
    if observation_value:  # Check if observation_value is not empty
        conditions.append(f'o.observed_times {relationship_1} "{observation_value}"')
    
    if latest_observed:
        conditions.append(f'o.dates {relationship_2} "{latest_observed}"')

    # Combine conditions into the WHERE clause
    where_clause = " AND ".join(conditions)

    cypher_query = f"""
    MATCH (r:{specy_type})-[o:OBSERVED_AT]->(l:Location)
    """
    if where_clause:
        cypher_query += f"WHERE {where_clause}\n"

    cypher_query += """
    RETURN r.name,o.observed_times,o.dates,l.name
    """

    return cypher_query



def generate_aspect_chain(vllm_name, llm_name, question, multimedia_option,  aspect, callbacks):
    if multimedia_option=="Multimedia Enabled":
        print(multimedia_option)
        graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="12345678")
        kwargs = extract_keywords_from_question(vllm_name,question)
        print("kwargs",kwargs)
        kwargs=json.loads(kwargs)
        # Now use the `kwargs` in the Cypher query construction
        formatted_query = construct_cypher_query(kwargs=kwargs, multi_option=1)
        print(formatted_query)
        Kg_output = graph.query(formatted_query)
        print("Kg_output is",Kg_output)
        KG_results="""
        You are a helpful assistant that answers questions about animals in Florida. Try your best to provide accurate and informative responses.
        """
        #general_user_template = "Question:{questions}, Kg_result:{messages}"

        # Get the response from the LLM
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        base_url = os.getenv('AZURE_OPENAI_ENDPOINT')
        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-06-01",
            azure_endpoint=base_url
            )

        system_input = {
            "role": "system",
            "content": KG_results
        }

        user_input = {
            "role": "user",
            "content": f"The question is {question} and KG result is {Kg_output}"
        }

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[system_input, user_input],
            temperature=0
        )
        # Filter and select top K links based on criteria (e.g., domain reliability, link text relevance)
        response_text = response.choices[0].message.content
        
        if "http" in str(Kg_output):
            #print(messages)
            #print(type(messages))
            URLs=Find_URLs(Kg_output)
        #vllm=Ollama(model="llava-phi3")
            image_list=[]
            image_URLs=[]
            #cols = st.columns(len(image_urls))
            for i, url in enumerate(URLs):
                #print('old url',url)
                if i<10:
                    url=image_url_finder(url)
                    print("new url",url)
                    #print(a)
                    if url:
                        #print("new url",url)
                        #print(a)
                        image_URLs.append(url)
                        response = requests.get(url)
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                        plt.imshow(image)
                        plt.show()
                        #cols[i].image(image, caption=f"Image {i+1}", use_column_width=True)
                        image_b64 = convert_to_base64(image)
                        image_list.append(image_b64)
            
            #Final_urls=image_URLs
        return {"answer": response_text, "URLs": image_URLs}
    elif multimedia_option=="Multimedia Disabled":
        print(aspect)
        if "Animal Observations and Distribution" in aspect:
            print(multimedia_option)
            graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="12345678")
            kwargs = extract_keywords_from_question(vllm_name,question)
            print("kwargs",kwargs)
            kwargs=json.loads(kwargs)
            # Now use the `kwargs` in the Cypher query construction
            formatted_query = construct_cypher_query(kwargs=kwargs, multi_option=0)
            print(formatted_query)
            Kg_output = graph.query(formatted_query)
            print("Kg_output is",Kg_output)
            KG_results="""
            You are a helpful assistant that answers questions about animals in Florida. Try your best to provide accurate and informative responses.
            """
            #general_user_template = "Question:{questions}, Kg_result:{messages}"

            # Get the response from the LLM
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            base_url = os.getenv('AZURE_OPENAI_ENDPOINT')
            client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-06-01",
                azure_endpoint=base_url
                )

            system_input = {
                "role": "system",
                "content": KG_results
            }

            user_input = {
                "role": "user",
                "content": f"The question is {question} and KG result is {Kg_output}"
            }

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[system_input, user_input],
                temperature=0
            )
            # Filter and select top K links based on criteria (e.g., domain reliability, link text relevance)
            response_text = response.choices[0].message.content
            print(response_text)
            return response_text
            #return str(response_text)
        elif "Relationships and Interactions Between Animals" in aspect:
            print(multimedia_option)
            graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="12345678")
            kwargs = extract_interact_keywords(vllm_name,question)
            print("kwargs",kwargs)
            kwargs=json.loads(kwargs)
            # Now use the `kwargs` in the Cypher query construction
            formatted_query = create_relationship_query(kwargs=kwargs)
            print(formatted_query)
            Kg_output = graph.query(formatted_query)
            print("Kg_output is",Kg_output)
            KG_results="""
            You are a helpful assistant that answers questions about animals in Florida. Try your best to provide accurate and informative responses.
            """
            #general_user_template = "Question:{questions}, Kg_result:{messages}"

            # Get the response from the LLM
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            base_url = os.getenv('AZURE_OPENAI_ENDPOINT')
            client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-06-01",
                azure_endpoint=base_url
                )

            system_input = {
                "role": "system",
                "content": KG_results
            }

            user_input = {
                "role": "user",
                "content": f"The question is {question} and KG result is {Kg_output}"
            }

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[system_input, user_input],
                temperature=0
            )
            # Filter and select top K links based on criteria (e.g., domain reliability, link text relevance)
            response_text = response.choices[0].message.content
            return response_text
        elif "Time and Trend Analysis" in aspect:
            print(multimedia_option)
            graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="12345678")
            kwargs = extract_temporal_keywords(vllm_name,question)
            print("kwargs",kwargs)
            kwargs=json.loads(kwargs)
            formatted_query = create_temporal_query(kwargs=kwargs)
            print(formatted_query)
            Kg_output = graph.query(formatted_query)
            print("Kg_output is",Kg_output)
            KG_results="""
            You are a helpful assistant that answers questions about animals in Florida. Try your best to provide accurate and informative responses.
            """
            #general_user_template = "Question:{questions}, Kg_result:{messages}"

            # Get the response from the LLM
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            base_url = os.getenv('AZURE_OPENAI_ENDPOINT')
            client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-06-01",
                azure_endpoint=base_url
                )

            system_input = {
                "role": "system",
                "content": KG_results
            }

            user_input = {
                "role": "user",
                "content": f"The question is {question} and KG result is {Kg_output}"
            }

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[system_input, user_input],
                temperature=0
            )
            # Filter and select top K links based on criteria (e.g., domain reliability, link text relevance)
            response_text = response.choices[0].message.content
            return response_text
        else:
            pass
    pass