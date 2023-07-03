import streamlit as st
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import streamlit as st
import os
import base64
from langchain.chains import RetrievalQA


def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents


def split_docs(documents, chunk_size=500, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

@st.cache_resource
def embedDocs(directory):
    documents = load_docs(directory)
    docs = split_docs(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embeddings)
    return db

@st.cache_resource
def initializeLLM():
    checkpoint = "LaMini-T5-738M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 512,
        do_sample=True,
        temperature = 0.3,
        top_p = 0.95
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def main():
  ## Streamlit Page
  st.set_page_config(page_title='Cyberthon')
  st.title('Smart EDiscovery using LLM')
  buttons = st.container()
  st.divider()
  logs = st.container()
  tab1,tab2  = logs.tabs(['Files','Content'])

  directory = buttons.text_input('**Enter the directory path :**')
  validity = st.empty()
  if(os.path.exists(directory)):
      with validity.container():
          buttons.write('_Valid Path_')
          with tab1:
            for i in os.listdir(directory):
              logs.markdown('* '+i)
          with tab2:
            st.write(load_docs(directory))
      if buttons.button('Process Document'):
        pass
  else:
    with validity.container():
      buttons.write('_Invalid Path. Enter a Valid Path_')

  query = buttons.text_input('**Enter your query :**')
  if buttons.button('Find'):
      db= embedDocs(directory)
      llm = initializeLLM()
      retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
      answer = retrieval_chain.run(query)
      buttons.markdown('**Answer :**')
      buttons.markdown(':blue[' + answer + ']' )


if __name__ == '__main__':
   main()
