from dotenv import load_dotenv
import os
from preprocess.clear_data import xml2csv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Load the "training" data, which will be the Document Store
CORPUS_DATA_PATH="pan12-sexual-predator/training/conversations.xml"
PREDATORS_DATA_PATH="pan12-sexual-predator/training/predator-ids.txt"
CORPUS_TEST_DATA_PATH="pan12-sexual-predator/test/conversations.xml"
PREDATORS_TEST_DATA_PATH="pan12-sexual-predator/test/predator-ids.txt"

os.makedirs("csv_files", exist_ok=True)
# Load the training data, in this case only load the abusive cases to store them in Qdrant.
xml2csv(nameXML=CORPUS_DATA_PATH,nameCSV="csv_files/abusive_text.csv",predatorsTXT=PREDATORS_DATA_PATH, only_abusive=True)
# Load the test data, in this case load all the cases to test the model.
xml2csv(nameXML=CORPUS_TEST_DATA_PATH,nameCSV="csv_files/abusive_text_test.csv",predatorsTXT=PREDATORS_TEST_DATA_PATH, only_abusive=False)

# Load the training data using the doc loader from langchain, make sure to specify the content, 
# source and metadata columns
loader = CSVLoader(
    file_path="csv_files/abusive_text.csv",
    csv_args={
        "delimiter": ";",
        "fieldnames": ["CONVERSATION_ID", "AUTHORS_IDS", "IS_ABUSIVE", "CONVERSATION_TEXT"],
    },
    content_columns=["CONVERSATION_TEXT"],
    source_column="CONVERSATION_ID",
    metadata_columns=["AUTHORS_IDS"],
)

data = loader.load()

# Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
# It splits text into chunks of 1000 characters each with a 150-character overlap.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# 'splits' holds the text you want to split, split the text into documents using the text splitter.
splits = text_splitter.split_documents(data)

# Use Qdrant to load the data into the vector store, using an embedding function to convert the text into vectors.
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")

embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

qdrant = QdrantVectorStore.from_documents(
    splits,
    embedding_function,
    url=qdrant_url,
    prefer_grpc=True,
    api_key=qdrant_key,
    collection_name="groom_chats",
)