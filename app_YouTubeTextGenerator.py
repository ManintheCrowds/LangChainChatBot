import os
apikey = 'INSERTAPIKEYHERE'

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

#prompt templates
title_template = PromptTemplate(
    input_variables= ['topic'],
    template='write me a title for this {topic}.'
)

script_template = PromptTemplate(
    input_variables= ['title', 'wikipedia_research'],
    template='write me a script based on the title. TITLE:{title} while leveraging this wikipedia research:{wikipedia_research}'
)

bibliography_links_template = PromptTemplate(
    input_variables=['topic'],
    template='provide me links to relevant knowledge sources about {topic}.'
)

#Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

#app framework
st.title('🦜🔗 GPT YouTube Video Script & Info Gatherer')
prompt = st.text_input('Plug in your prompt here')

# LLMs
llm = OpenAI(temperature=0.9)

title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key= 'script', memory=script_memory)
bibliography_links_chain = LLMChain(llm=llm, prompt=bibliography_links_template, verbose=True, output_key='bibliography')

wiki = WikipediaAPIWrapper()

#Running two LLM chains independantly now, so we don't need this to be sequential.
#sequential_chain = SequentialChain(chains=[title_chain, script_chain, bibliography_links_chain], input_variables=['topic'], output_variables=['title', 'script', 'bibliography'], verbose=True)

#Show Prompt to screen
if prompt:
    # Run the title chain to generate a title.
    title = title_chain.run(prompt)

    # Run the wikipedia research to get some information about the topic.
    wiki_research = wiki.run(prompt)

    # Run the script chain to generate a script based on the title and wikipedia research.
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    # Run the bibliography chain to generate a list of links.
    bibliography = bibliography_links_chain.run(topic=prompt)

    st.write(title)
    st.write(script)
    st.write(bibliography)

    with st.expander('Message History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
else:
    st.warning("Please enter a prompt.")
