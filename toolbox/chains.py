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

def image_url_finder(webpage_url):
    response = requests.get(webpage_url)
    response.raise_for_status()  # Check if the request was successful
    # Parse the web page content
    soup = BeautifulSoup(response.text, 'html.parser')
    #print("soup",soup)
    # Find all image tags
    img_tags = soup.find_all('meta')
    # Download images containing 'jpg' in the URL
    #print("image_tags",img_tags)
    for img in img_tags:
        #print("img",img)
        img_url = img.get('content')
        print("img_url",img_url)
        if 'large.jpg'in img_url:
            #result=img.get('content')
            return img_url
        elif 'large.jpeg' in img_url:
            result=img.get('content')
            return img_url

def Find_URLs(KG_text):
    print(type(KG_text))
    url_list = []
    for item in KG_text:
        url_list.extend(re.findall(r'https:\/\/www\.inaturalist\.org\/observations\/\d+', str(item)))
    print(url_list)
    return url_list
#Don't convert it into a Cypher query, this is a simple combination between natural language query and search in KG.



def classfic(user_input, json_data, llm):
    template = """
    You are a helpful assistant that answers questions of animals in Florida.
    Please think step by step of what scope of dictionary can solve this question.
    Now you are given a question and a json dictionary, you need to output which part of dictionary best covers the scope of the question.
    Note that trends or similar words are for tendency studies, colors or other visible information are for multimedia.
    Distribution is for interation between animals and locations. For interation between animals, refer Relationships and Interactions Between Animals. And for interation between locations, use Habitats and Ecosystems.
    You are only required to output the scope only, don't generate anything else!
    For example, generate 'Animal Observations and Distribution' only instead of some dict containing it.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "question is {question}. The aspects ditionary is {diction}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = chat_prompt | llm
    answer = chain.invoke(
        {"question": user_input, "diction": json_data}
    ).content
    #print(answer)
    #return {"answer": answer}

    return answer

def configure_llm_only_chain(llm):
    # LLM only response
    template = """
    You are a helpful assistant that answers questions of animals in Florida.
    If you don't know the answer, just say that you don't know, you must not make up an answer.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    def generate_llm_output(
        user_input: str, callbacks: List[Any], prompt=chat_prompt
    ) -> str:
        chain = prompt | llm
        answer = chain.invoke(
            {"question": user_input},config={"callbacks": callbacks}
        ).content
        #print(answer)
        return {"answer": answer}

    return generate_llm_output


def prompt_cypher(llm):
    #Use llama2 to convert question into Cypher query.
    template = """
    Generate the code only, don't generate anything else!
    Generate step by step and the code is used to Neo4j Cypher query!
    You are given a Neo4j graph of animals in Florida, USA, and you need to convert the question into Cypher query code of the question. 
    The configuration of the Neo4j graph is given below:
    The nodes can be divided into five classes, 'Amphibian_name', 'Bird_name', 'Reptile_name', 'Fish_name' and 'Location'.
    They are connected with 'OBSERVED_AT' relationship property, which means directed edge indicating 'Amphibian_name', 'Reptile_name' or 'Bird_name'  observed at specific 'Location'.
    The connection is formatted like 
    (r:Reptile_name)->[:OBSERVED_AT]->(l:Location)
    (r:Amphibian_name)->[:OBSERVED_AT]->(l:Location)
    (r:Bird_name)->[:OBSERVED_AT]->(l:Location)
    and (r:Fish_name)->[:OBSERVED_AT]->(l:Location)
    The 'OBSERVED_AT' relationship property has attributes 'multimedia', 'observed_times' and 'dates'. 
    If any query takes multimedia information like color or other visible features, search and output the multimedia information, which contains URLs for further process.
    The nodes have 'wikidata_id' and animal nodes have 'IUCN_id' of it.
    Output Cypher query code only! Don't use ''' and ''', just code text!
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    def generate_llm_output(
        user_input: str, prompt=chat_prompt
    ) -> str:
        chain = prompt | llm
        answer = chain.invoke(
            {"question": user_input}
        ).content
        #print("converted Cypher query is", answer)
        return {"answer": answer}
    #query_output=
    return generate_llm_output

def configure_qa_rag_chain(llm, Graph_url, username, password, embeddings, query):
    #print("multi is",multi)
    print("query is", query)
    kg = Neo4jVector.from_existing_relationship_index(
        embedding=embeddings,
        url=Graph_url,
        username=username,
        password=password,
        database="neo4j",  # neo4j by default
        index_name="relationship_vector",
        #node_label=["Bird_name", "Location"
        # , "Amphibian_name"],
        #retrieval_query="""
#MATCH (r:Reptile_name)-[o:OBSERVED_AT]->(l:Location)
#WHERE l.name = "Alachua"           
#RETURN r.name as Reptiles
#        """,
        retrieval_query=query
    )
    
    #Retrieval query is to be modified.
    #pass
    KFSystemtemplate = """
    Use the following pieces of context to answer the question at the end.
    The context contains Animal, Location related information from GBIF, iNaturalist dataset.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Don't generate URL links.
    ----
    {summaries}
    ----
    The context is about Florida, USA.
    """
    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(KFSystemtemplate),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    #print(messages)
    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )


    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=999999,
    )

    return kg_qa

def generate_llava_output(vllm,  Graph_url, username, password, embeddings, query):
        #print(re.search(r'\{(.+?)\}',"{Neo4jquery}").group(1))
        #print("multi is",multi)
        graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="12345678")
        Kg_output=graph.query(
            """
MATCH (r:Bird_name)-[o:OBSERVED_AT]->(l:Location)
WHERE l.name = "Alachua"           
RETURN r.name AS Bird, o.multimedia AS MultimediaInfo
            """
        )

        print("Kg_output is",Kg_output)
        KG_results="""
        You are a helpful assistant that answers questions of animals in Florida.
        Try your best to answer the question with the figures given.
        """
        general_user_template = "Question:```{question}```, knowledge graph retrieval results, {messages}"

        messages = [
        SystemMessagePromptTemplate.from_template(KG_results),
        HumanMessagePromptTemplate.from_template(general_user_template),
        ]
        chat_prompt = ChatPromptTemplate.from_messages(messages)
        def generate_vllm_output(
            user_input: str, callbacks: List[Any], prompt=chat_prompt, messages=Kg_output
        ) -> str:
            chain = prompt | vllm
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
                        vllm.bind(images=image_list)
                    
                answer=chain.invoke(
                        {"question": user_input, "messages": Kg_output},config={"callbacks": callbacks}
                    ).content
                print(answer)
                return {"answer": answer, "URLs": image_URLs}
        return generate_vllm_output



def generate_ticket(neo4j_graph, llm_chain, input_question):
    # Get high ranked questions
    records = neo4j_graph.query(
        "MATCH (q:Question) RETURN q.title AS title, q.body AS body ORDER BY q.score DESC LIMIT 3"
    )
    questions = []
    for i, question in enumerate(records, start=1):
        questions.append((question["title"], question["body"]))
    # Ask LLM to generate new question in the same style
    questions_prompt = ""
    for i, question in enumerate(questions, start=1):
        questions_prompt += f"{i}. \n{question[0]}\n----\n\n"
        questions_prompt += f"{question[1][:150]}\n\n"
        questions_prompt += "----\n\n"

    gen_system_template = f"""
    You're an expert in formulating high quality questions. 
    Formulate a question in the same style and tone as the following example questions.
    {questions_prompt}
    ---

    Don't make anything up, only use information in the following question.
    Return a title for the question, and the question post itself.

    Return format template:
    ---
    Title: This is a new title
    Question: This is a new question
    ---
    """
    # we need jinja2 since the questions themselves contain curly braces
    system_prompt = SystemMessagePromptTemplate.from_template(
        gen_system_template, template_format="jinja2"
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            SystemMessagePromptTemplate.from_template(
                """
                Respond in the following template format or you will be unplugged.
                ---
                Title: New title
                Question: New question
                ---
                """
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    llm_response = llm_chain(
        f"Here's the question to rewrite in the expected format: ```{input_question}```",
        [],
        chat_prompt,
    )
    new_title, new_question = extract_title_and_question(llm_response["answer"])
    return (new_title, new_question)
