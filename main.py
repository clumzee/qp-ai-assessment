import cassio
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import CharacterTextSplitter
from conversation import start_conversation
from pdf_conversion import read_pdf_and_extract_text
import sys


file_path = sys.argv[1]
raw_text = read_pdf_and_extract_text(file_path)

print(raw_text)

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
OPENAI_API_KEY =os.getenv('OPENAI_API_KEY')

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)

# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)


texts = text_splitter.split_text(raw_text)

astra_vector_store.add_texts(texts[:50])
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

start_conversation(astra_vector_index, astra_vector_store, llm)