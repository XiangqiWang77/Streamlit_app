
def ToG_retrieval_pipeline(llm, Graph_url, username, password, embeddings, query):

    max_length =256 
    temperature_exploration =0.4 # the temperature in exploration stage.
    temperature_exploration= 0 # the temperature in reasoning stage.
    width= 3 # choose the search width of ToG, 3 is the default setting.
    depth= 3 # choose the search depth of ToG, 3 is the default setting.s
    remove_unnecessary_rel=True # whether removing unnecessary relations.
    #LLM_type gpt-3.5-turbo # the LLM you choose
    #opeani_api_keys sk-xxxx # your own api keys, if LLM_type == llama, this parameter would be rendered ineffective.
    num_retain_entity=5 # Number of entities retained during entities search.
    prune_tools=llm # prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.

    graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="12345678")

    return KG_outputs
