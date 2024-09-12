import os
import streamlit as st
import json
from io import BytesIO
from PIL import Image
import requests
from streamlit.logger import get_logger
from langchain_community.graphs import Neo4jGraph
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import BedrockChat
from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from neo4j import GraphDatabase

# 设置Neo4j数据库连接
NEO4J_URI = "bolt://localhost:7687"  # 替换为你的Neo4j URI
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
from toolbox.chains import generate_ticket
import re
import openai
from toolbox.utils import ImageDownloader, convert_to_base64
from toolbox.aspects import generate_aspect_chain
from toolbox.web_agent import web_search_agent
from toolbox.chains import configure_llm_only_chain, prompt_cypher, configure_qa_rag_chain,generate_llava_output,classfic
from toolbox.ToG import ToG_retrieval_pipeline
from dotenv import load_dotenv
from typing import List, Any
from langchain.callbacks.base import BaseCallbackHandler
from openai import AzureOpenAI
from openai import OpenAI
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
load_dotenv(".env")

LLM=os.getenv("LLM") #or any Ollama model tag, gpt-4, gpt-3.5, or claudev2
EMBEDDING_MODEL=os.getenv("EMBEDDING_MtODEL")  #or google-genai-embedding-001 openai, ollama, or aws
VLLM=os.getenv("VLLM") 
NEO4J_URL=os.getenv("NEO4J_URL") 
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME") 
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD") 
OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL") 
EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL") 

embeddings = OllamaEmbeddings(
    base_url=OLLAMA_BASE_URL, model=LLM
        )




# Configure the OpenAI client to use Azure

openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT


def get_autocomplete_suggestions(question):
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-06-01",
        azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

    """Call Azure OpenAI API for auto-completion."""
    prompt="""
    You are given a graph containing information about animals and locations in Florida. The graph consists of the following types of relationships and data:

Animal-Location (OBSERVED_AT edge):

Animals and locations are connected by an OBSERVED_AT edge.
Both animals and locations have wikidata_id.
Animals also have an IUCN_id, though not all animals have this identifier.
The OBSERVED_AT edge contains additional details such as observed_times, dates, and multimedia information (like images or videos).
Animal-Animal Relationships:

Animals are connected through relationships like PreysOn and InteractWith, representing ecological interactions.
Event-Location (Happened_In edge):

Events (e.g., natural disasters) are connected to locations through the Happened_In relationship.
Given this graph structure and a partial question entered by the user, your task is to automatically complete the question in a way that is most relevant to the data available in the graph. Focus on generating a natural, coherent question that corresponds to the relationships and attributes described above.

Input:
A partially completed question based on the Florida graph (e.g., "Where was the...").

Output Requirement:
Complete the most likely question related to the graph. Only output the final question, don't generate anything else despite the question, and output the question even the question is far from complete, just output one.

Example Input:
"Where was the"

Example Output:
"Where was the Florida Panther observed?"
"""
    system_input = {
        "role": "system",
        "content": prompt
    }

    user_input = {
        "role": "user",
        "content": f"The question is {question}"
    }
    try:
        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[system_input, user_input],
        temperature=0
        )
        # Filter and select top K links based on criteria (e.g., domain reliability, link text relevance)
        response_text = response.choices[0].message.content
        return response_text
    except Exception as e:
        return f"Error: {e}"



dimension = 4096

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="12345678")

def load_llm(llm_name: str, config={}):
    if llm_name == "gpt-4":
        return ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
    elif llm_name == "gpt-3.5":
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
    elif llm_name == "claudev2":
        return BedrockChat(
            model_id="anthropic.claude-v2",
            model_kwargs={"temperature": 0.0, "max_tokens_to_sample": 1024},
            streaming=True,
        )
    return ChatOllama(
        temperature=100,
        base_url=config["ollama_base_url"],
        model=llm_name,
    )



styl = f"""
<style>
    /* not great support for :has yet (hello FireFox), but using it for now */
    .stChatFloatingInputContainer {{
        bottom: 20px;
    }}

    /* Generate ticket text area */
    textarea[aria-label="Description"] {{
        height: 200px;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)


llm=load_llm(llm_name=LLM, config={"ollama_base_url": OLLAMA_BASE_URL})
vllm=load_llm(llm_name=VLLM, config={"ollama_base_url": OLLAMA_BASE_URL})
#vllm_chian=llava_chain(vllm)
llm_chain = configure_llm_only_chain(llm)
query_chain=prompt_cypher(llm)
#aspects_chain=generate_aspect_chain(vllm_name=vllm,llm_name=llm)

import json

# Load JSON data from a file
with open('aspects.json', 'r') as f:
    json_data = json.load(f)



def chat_input():
   #classification_category=re.findall(r'\{(.*?)\}', classification_category)
   # print(classification_category)
    #print(a)
    result = None  
    user_input_text = st.text_input("What do you want to know about animals in Florida?", key="user_input_text", value="")

    #print("user_input",user_input)
    # Initialize autocomplete suggestion
    autocomplete_suggestion = ""
    user_input=None
    if user_input_text:
        # Call Azure OpenAI for auto-completion when user types something
        autocomplete_suggestion = get_autocomplete_suggestions(user_input_text)
        print("autocomplete",autocomplete_suggestion)
        # Display autocomplete suggestion in Beijing font
        if autocomplete_suggestion:
            st.markdown(f"Suggestion: {autocomplete_suggestion}", unsafe_allow_html=True)

        # Check if user wants to accept the suggestion by pressing Tab
        if st.button("Press Tab to Accept Suggestion"):
            # Append the suggestion to the user input
            user_input = autocomplete_suggestion
        elif st.button("Press to Cancel Suggestion"):
            user_input=user_input_text
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        #autocomplete_suggestion = get_autocomplete_suggestions(user_input)
        with st.chat_message("assistant"):
            st.caption(f"bot: {name}")
            stream_handler = StreamHandler(st.empty())
            if name[0] == "RAG Disabled":
                #print("assigned RAG")
                output_function=llm_chain
                result = output_function(
                    {"question": user_input},callbacks=[stream_handler]
                )["answer"]
            elif name[0] == "LLM Translator RAG":
                translate_function=query_chain
                temp_result=translate_function(
                    {"question": user_input}
                )["answer"]
                print(temp_result)
                #print(temp_result[3:-3])
                if name[1]=="Multimedia Disabled":
                    rag_chain = configure_qa_rag_chain(
                    llm, Graph_url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, embeddings=embeddings, query=temp_result
                    )
                    output_function = rag_chain
                    feed_result=output_function(
                        {"question": user_input},callbacks=[stream_handler]
                    )
                    result=feed_result["answer"]
                    #st.session_state[f"user_input"].append(user_input)
                    #st.session_state[f"generated"].append(result)
                    #st.session_state[f"rag_mode"].append(name)
                if name[1]=="Multimedia Enabled":
                    vlla_rag_chain=generate_llava_output(
                    vllm, Graph_url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, embeddings=embeddings, query=temp_result
                    )
                    output_function=vlla_rag_chain
                    feed_result=output_function(
                        {"question": user_input},callbacks=[stream_handler]
                    )
                    result=feed_result["answer"]
                    image_urls=feed_result["URLs"][:10]
                    if image_urls:
                        cols = st.columns(len(image_urls))
                        for i, url in enumerate(image_urls):
                            response = requests.get(url)
                            image = Image.open(BytesIO(response.content)).convert("RGB")
                            cols[i].image(image, caption=f"Image {i+1}", use_column_width=True)
                    #st.session_state[f"user_input"].append(user_input)
                    #st.session_state[f"generated"].append(result)
                    #st.session_state[f"rag_mode"].append(name)
            elif name[0] == "Aspects matching RAG":
                classification_category=classfic(user_input, json_data, llm)
                print("classification_category",classification_category)
                #print(a)
                if name[1]=="Multimedia Enabled":
                    temp_chain=generate_aspect_chain(vllm_name=vllm,llm_name=llm, question=user_input, multimedia_option=name[1], aspect=classification_category, callbacks=[stream_handler])
                    #output_function=temp_chain
                    feed_result=temp_chain
                    result=feed_result["answer"]
                    st.write(result)
                    #result=None
                    image_urls=feed_result["URLs"]
                    cols = st.columns(len(image_urls))
                    for i, url in enumerate(image_urls):
                        response = requests.get(url)
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                        cols[i].image(image, caption=f"Image {i+1}", use_column_width=True)
                    #st.session_state[f"user_input"].append(user_input)
                    #st.session_state[f"generated"].append(result)
                    #st.session_state[f"rag_mode"].append(name)
                if name[1]=="Multimedia Disabled":
                    temp_chain=generate_aspect_chain(vllm_name=vllm,llm_name=llm, question=user_input, multimedia_option=name[1], aspect=classification_category, callbacks=[stream_handler])
                    #output_function=temp_chain
                    feed_result=temp_chain
                    result=feed_result
                    st.write(result)
                    #result=None
                    #print("result is", result)
                    #st.session_state[f"user_input"].append(user_input)
                    #st.session_state[f"generated"].append(result)
                    #st.session_state[f"rag_mode"].append(name)
                #result=infer_function(question=user_input, multimedia_option=name[1], aspect=classification_category, callbacks=[stream_handler])["answer"]
            elif name[0] == "Wikidata based web agent":
                result=web_search_agent(llm=llm, question=user_input, early_stop=10, limit_N=200, topK=2, retriev_option=0)
                st.write(result)
            print("loaded into array")
            st.session_state[f"user_input"].append(user_input)
            st.session_state[f"generated"].append(result)
            st.session_state[f"rag_mode"].append(name)

        
def display_chat():
    # Session state
    if "generated" not in st.session_state:
        st.session_state[f"generated"] = []

    if "user_input" not in st.session_state:
        st.session_state[f"user_input"] = []

    if "rag_mode" not in st.session_state:
        st.session_state[f"rag_mode"] = []

    if st.session_state[f"generated"]:
        print(st.session_state[f"generated"])
        size = len(st.session_state[f"generated"])
        print(size)
        # Display only the last three exchanges
        for i in range(max(size - 100, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input"][i])

            with st.chat_message("assistant"):
                st.caption(f"RAG: {st.session_state[f'rag_mode'][i]}")
                st.write(st.session_state[f"generated"][i])

        with st.container():
            st.write("&nbsp;")

def open_sidebar():
    st.session_state.open_sidebar = True


def close_sidebar():
    st.session_state.open_sidebar = False



def mode_select() -> list:
    options = ["RAG Disabled","LLM Translator RAG", "Aspects matching RAG", "Wikidata based web agent"]
    multimediaoptions = ["Multimedia Disabled", "Multimedia Enabled"]
    #ToGoptions= ["Yes","No"]
    selected_multimedia_mode = st.radio("Select multimedia mode", multimediaoptions, horizontal=True)
    mode_selected=st.radio("Select RAG mode", options, horizontal=True)
    #ToG_selection = st.radio("Select ToG mode", ToGoptions, horizontal=True)
    return [mode_selected, selected_multimedia_mode]




name = mode_select()
if name[0] == "RAG Disabled":
    #print("assigned RAG")
    output_function = llm_chain
elif name[0] == "LLM Translator RAG":
    translate_function = query_chain
#elif name[0] == "Aspects matching RAG":
#    output_function = aspects_chain


open_sidebar()
display_chat()
chat_input()
#display_chat()
