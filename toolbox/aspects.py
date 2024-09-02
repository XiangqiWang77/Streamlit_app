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
    Don't easily refer to wikidata_id, unless it's mentioned in the question!
    For any information that can only be conveyed in multimedia information, don't generate it as value.
    Given a question about animal observations in Florida, identify the relevant values for the following fields:
    - specy_type: The type of species (only formatted like, Bird_name, Reptile_name with _name suffix).
    - location_attribute: An attribute of the location (only contain attribute, name, wikidata_id).
    - location_value: The value for the location attribute (e.g., name is like Alachua).
    - specy_attribute: An attribute of the species (only contain attribute name, wikidata_id).
    - specy_value: The value for the species attribute (e.g., name is like Panther, Eagle).
    - observation_attribute: An attribute of the observation edge (only contain date, multimedia, observed_times).
    - observation_value: The value for the observation attribute (e.g., date like 2024-04-16T11:49:47, multimedia like https://www.inaturalist.org/observations/207625017 , observed_times like 1).

    This is an feasible output example, generate like this:

    {
    "specy_type": "Reptile_name",
    "location_attribute": "name",
    "location_value": "Alachua",
    "specy_attribute": "name",
    "specy_value": "Great Egret",
    "observation_attribute": "observed_times"
    "observation_value": 1
    }

    Extract and return these fields as a JSON object based on the given question.


    Question:{question}
    """

    # Get the response from the LLM
    chat_prompt = ChatPromptTemplate.from_messages(
        [prompt]
    )

    chain = chat_prompt | llm
    response = chain.invoke({"question": question}).content

    print(response)
    # Assume the LLM returns a JSON string, parse it to a dictionary
    try:
        extracted_keywords = json.loads(response)
    except json.JSONDecodeError:
        # Handle parsing errors or unexpected responses
        extracted_keywords = {}

    # Return the extracted keywords
    return extracted_keywords

def construct_cypher_query(**kwargs):
    # Set default values or extract from kwargs
    specy_type = kwargs.get("specy_type", "")
    location_attribute = kwargs.get("location_attribute", "")
    location_value = kwargs.get("location_value", "")
    specy_attribute = kwargs.get("specy_attribute", "")
    specy_value = kwargs.get("specy_value", "")
    observation_attribute = kwargs.get("observation_attribute", "")
    observation_value = kwargs.get("observation_value", "")

    # Construct the query with the provided or default values
    cypher_query = f"""
    MATCH (r:{specy_type})-[o:OBSERVED_AT]->(l:Location)
    WHERE l.{location_attribute} = "{location_value}" 
          AND r.{specy_attribute} = "{specy_value}" 
          AND o.{observation_attribute} = "{observation_value}"
    RETURN r.name AS specy, o.multimedia AS MultimediaInfo, l.name AS location
    """

    return cypher_query

def generate_aspect_chain(vllm_name, llm_name, question, multimedia_option,  aspect, callbacks):
    if multimedia_option=="Multimedia Enabled":
        print(multimedia_option)
        graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="12345678")
        kwargs = extract_keywords_from_question(vllm_name,question)

        # Now use the `kwargs` in the Cypher query construction
        formatted_query = construct_cypher_query(**kwargs)
        print(formatted_query)
        Kg_output = graph.query(formatted_query)
        print("Kg_output is",Kg_output)
        KG_results="""
        You are a helpful assistant that answers questions of animals in Florida.
        Try your best to answer the question with the figures given.
        """
        general_user_template = "Question:"+question+"knowledge graph retrieval results, "+Kg_output

        messages = [
        SystemMessagePromptTemplate.from_template(KG_results),
        HumanMessagePromptTemplate.from_template(general_user_template),
        ]
        chat_prompt = ChatPromptTemplate.from_messages(messages)
        def generate_vllm_output(
            user_input: str, callbacks: List[Any], prompt=chat_prompt, messages=Kg_output
        ) -> str:
            chain = prompt | vllm_name
            if "http" in str(messages):
                print(messages)
                #print(type(messages))
                URLs=Find_URLs(messages)
            #vllm=Ollama(model="llava-phi3")
                image_list=[]
                #cols = st.columns(len(image_urls))
                image_URLs=[]
                for i, url in enumerate(URLs):
                    #print('old url',url)
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
                        vllm_name.bind(images=image_list)
                    
                answer=chain.invoke(
                        {"question": user_input, "messages": Kg_output},config={"callbacks": callbacks}
                    ).content
                print(answer)
                return {"answer": answer, "URLs": image_URLs}
        return generate_vllm_output
    elif multimedia_option=="Multimedia Disabled":
        if aspect=="Animal Observations and Distribution":
            templte=""
            print(aspect)
        elif aspect=="Relationships and Interactions Between Animals":
            template=""
            print(aspect)
        elif aspect=="Conservation Status and Ecological Threats":
            template=""
            print(aspect)
        elif aspect=="Time and Trend Analysis":
            template=""
            print(aspect)
        else:
            pass
    pass