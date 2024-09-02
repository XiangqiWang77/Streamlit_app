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
NEO4J_URI = "bolt://10.7.218.37:7687"  # 替换为你的Neo4j URI
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
from toolbox.chains import generate_ticket
import re

from toolbox.utils import ImageDownloader, convert_to_base64
from toolbox.aspects import generate_aspect_chain
from toolbox.web_agent import web_search_agent
from toolbox.chains import configure_llm_only_chain, prompt_cypher, configure_qa_rag_chain,generate_llava_output,classfic
from toolbox.ToG import ToG_retrieval_pipeline
from dotenv import load_dotenv
from typing import List, Any
from langchain.callbacks.base import BaseCallbackHandler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
load_dotenv(".env")

LLM=os.getenv("LLM") #or any Ollama model tag, gpt-4, gpt-3.5, or claudev2
EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL")  #or google-genai-embedding-001 openai, ollama, or aws
VLLM=os.getenv("VLLM") 
NEO4J_URL=os.getenv("NEO4J_URL") 
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME") 
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD") 
OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL") 
EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL") 

embeddings = OllamaEmbeddings(
    base_url=OLLAMA_BASE_URL, model=LLM
        )


dimension = 4096

graph = Neo4jGraph(url="bolt://10.7.218.37:7687", username="neo4j", password="12345678")

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
    result = None  
    user_input = st.chat_input("What do you want to know of animals in Florida?")
   #classification_category=re.findall(r'\{(.*?)\}', classification_category)
   # print(classification_category)
    #print(a)
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.caption(f"bot: {name}")
            stream_handler = StreamHandler(st.empty())
            if name[0] == "RAG Disabled":
                result = output_function(
                    {"question": user_input},callbacks=[stream_handler]
                )["answer"]
            elif name[0] == "LLM Translator RAG":
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
                if name[1]=="Multimedia Enabled":
                    vlla_rag_chain=generate_llava_output(
                    vllm, Graph_url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, embeddings=embeddings, query=temp_result
                    )
                    output_function=vlla_rag_chain
                    feed_result=output_function(
                        {"question": user_input},callbacks=[stream_handler]
                    )
                    result=feed_result["answer"]
                    image_urls=feed_result["URLs"]
                    cols = st.columns(len(image_urls))
                    for i, url in enumerate(image_urls):
                        response = requests.get(url)
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                        cols[i].image(image, caption=f"Image {i+1}", use_column_width=True)
            elif name[0] == "Aspects matching RAG":
                classification_category=classfic(user_input, json_data, llm)
                print(classification_category)
                #print(a)
                result=generate_aspect_chain(vllm_name=vllm,llm_name=llm, question=user_input, multimedia_option=name[1], aspect=classification_category, callbacks=[stream_handler])["answer"]
                #result=infer_function(question=user_input, multimedia_option=name[1], aspect=classification_category, callbacks=[stream_handler])["answer"]
            elif name[0] == "Wikidata based web agent":
                result=web_search_agent(llm=llm, question=user_input, early_stop=10, limit_N=200, topK=2)
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
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input"][i])

            with st.chat_message("assistant"):
                st.caption(f"RAG: {st.session_state[f'rag_mode'][i]}")
                st.write(st.session_state[f"generated"][i])

        with st.expander("Not finding what you're looking for?"):
            st.write(
            "Automatically generate a draft for an internal ticket to our support team."
            )
            st.button(
            "Generate ticket",
            type="primary",
            key="show_ticket",
            on_click=open_sidebar,
            )
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
    output_function = llm_chain
elif name[0] == "LLM Translator RAG":
    translate_function = query_chain
#elif name[0] == "Aspects matching RAG":
#    output_function = aspects_chain


open_sidebar()
display_chat()
chat_input()
