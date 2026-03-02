# Setup
# 1.1 Install and import the required libraries

#(%pip install --force-reinstall --no-deps -r ./requirements.txt)

# Restart kernel
from IPython.core.display import HTML
HTML("<script>Jupyter.notebook.kernel.restart()</script>")

# Standard library imports
import os
import sys
import json
import time
import random

# Third-party imports
import boto3
from botocore.exceptions import ClientError

# Local imports
import utility

# Print SDK versions
print(f"Python version: {sys.version.split()[0]}")
print(f"Boto3 SDK version: {boto3.__version__}")

# 1.2 Initial setup for clinets and global variables

# Create boto3 session and set AWS region
boto_session = boto3.Session()
aws_region = boto_session.region_name

# Create boto3 clients for AOSS, Bedrock, and S3 services
aoss_client = boto3.client('opensearchserverless')
bedrock_agent_client = boto3.client('bedrock-agent')
s3_client = boto3.client('s3')

# Define names for AOSS, Bedrock, and S3 resources
resource_suffix = random.randrange(100, 99999)
s3_bucket_name = f"bedrock-kb-{aws_region}-{resource_suffix}"
aoss_collection_name = f"bedrock-kb-collection-{resource_suffix}"
aoss_index_name = f"bedrock-kb-index-{resource_suffix}"
bedrock_kb_name = f"bedrock-kb-{resource_suffix}"

# Set the Bedrock model to use for embedding generation
embedding_model_id = 'amazon.titan-embed-text-v2:0'
embedding_model_arn = f'arn:aws:bedrock:{aws_region}::foundation-model/{embedding_model_id}'
embedding_model_dim = 1024

# Some temporary local paths
local_data_dir = 'data'

# Print configurations
print("AWS Region:", aws_region)
print("S3 Bucket:", s3_bucket_name)
print("AOSS Collection Name:", aoss_collection_name)
print("Bedrock Knowledge Base Name:", bedrock_kb_name)

# 2. Create an S3 Data Source
# Amazon bedrock knowldge bases can connect to a variety of data sources for downstream RAG applications. Supported data sources include AMazon S3.
# Confluence, MIcrosoft Sharepoint, Salesforce. Web Crawler, and custom data sources.

# In this workshop, we will use Amazon S3 to store unstructured data - specifically. PDF files containing Amazon Shareholder Letters from different years.
# This S3 bucket will serve as the source of documents for our knowldge base. During the ingestion process, Bedrock will parse these documents, convert them into vector embeddings using an embedding model,
# and store them in a vector databasse for efficient retrieval during queries.

# 2.1 Create an S3 bucket, if needed

# Check if bucket exists, and if not create S3 bucket for KB data source

try:
    s3_client.head_bucket(Bucket=s3_bucket_name)
    print(f"Bucket '{s3_bucket_name}' already exists..")
except ClientError as e:
    print(f"Creating bucket: '{s3_bucket_name}'..")
    if aws_region == 'us-east-1':
        s3_client.create_bucket(Bucket=s3_bucket_name)
    else:
        s3_client.create_bucket(
            Bucket=s3_bucket_name,
            CreateBucketConfiguration={'LocationConstraint': aws_region}
        )

# 2.2 Download data and upload to S3

from urllib.request import urlretrieve

# URLs of shareholder letters to download
urls = [
    'https://s2.q4cdn.com/299287126/files/doc_financials/2023/ar/2022-Shareholder-Letter.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2022/ar/2021-Shareholder-Letter.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2021/ar/Amazon-2020-Shareholder-Letter-and-1997-Shareholder-Letter.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2020/ar/2019-Shareholder-Letter.pdf'
]

# Corresponding local file names
filenames = [
    'AMZN-2022-Shareholder-Letter.pdf',
    'AMZN-2021-Shareholder-Letter.pdf',
    'AMZN-2020-Shareholder-Letter.pdf',
    'AMZN-2019-Shareholder-Letter.pdf'
]

# Create local staging directory if it doesn't exist
os.makedirs(local_data_dir, exist_ok=True)

# Download each file and print confirmation
for url, filename in zip(urls, filenames):
    file_path = os.path.join(local_data_dir, filename)
    urlretrieve(url, file_path)
    print(f"Downloaded: '{filename}' to '{local_data_dir}'..")

for root, _, files in os.walk(local_data_dir):
    for file in files:
        full_path = os.path.join(root, file)
        s3_client.upload_file(full_path, s3_bucket_name, file)
        print(f"Uploaded: '{file}' to 's3://{s3_bucket_name}'..")

# 3 Steup AOSS Vector Index and COnfigure BKB Access Permissions

# In this section we'll create a vector index using Amazon OpenSearch Serverless (AOSS) and configure the encessary access permissions for the
# Bedrock Knowledge Base (BKB) that we'll set up later. AOSS provides a fully managed, serverless solution for running vector search workloads at billion-vector scale.
# It automatically handles resource scaling and eliminates the need for cluster managemen, while delivering low-laency, millisecond response times with pay-per-use pricing.

# While this example uses AOSS. It's worth noting that Bedrock Knowledge Bases alos supports other popular vector stores, including Amazon Aurora PostgreSQL with pgvector. Pinecone.
# Redis Enterprise Cloud and MongoDB among others

# 3.1 Create IAM Role with Necessary Permissions for Bedrock Knowledge Base

# Let's first create an IAM role with all the necessary policies and permissions to allow BKB to execute operations, such as invoking Bedrock FM's and reading data from an S3 bucket.
# We will use a helper function for this.

bedrock_kb_execution_role = utility.create_bedrock_execution_role(bucket_name=s3_bucket_name)
bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Role']['Arn']

print("Created KB execution role with ARN:", bedrock_kb_execution_role_arn)

# 3.2 Create AOSS Policies and Vector Collection

# Next we need to create and attach three key policies for securing and managing access to the AOSS collection: an encryption policy,
# a network access policy, and a data access policy. These policies ensure proper encryption, network security, and the necessary permissions for creating,
# reading, updating, and deleting collection items and indexes. This step is essential for configuring the OpenSeach collection to interact with BKB securely and 
# efficiently. We will use another helper function for this.

# In order to keep setup overhead at minimum, in this examples we allow public internet access,
# to the OpenSearch Serverless collection resource. However, for production environments we strongly suggest to leverage private connection between your VPC and Amazon OpenSearch
# Serverless resources via an VPC endpoint,

# Create AOSS policies for the new vector collection
aoss_encryption_policy, aoss_network_policy, aoss_access_policy = utility.create_policies_in_oss(
    vector_store_name=aoss_collection_name,
    aoss_client=aoss_client,
    bedrock_kb_execution_role_arn=bedrock_kb_execution_role_arn)

print("Created encryption policy with name:", aoss_encryption_policy['securityPolicyDetail']['name'])
print("Created network policy with name:", aoss_network_policy['securityPolicyDetail']['name'])
print("Created access policy with name:", aoss_access_policy['accessPolicyDetail']['name'])

# With all the necessary policies in place, let's proceed to actually create a new AOSS collection. 

# Request to create AOSS collection
aoss_collection = aoss_client.create_collection(name=aoss_collection_name, type='VECTORSEARCH')

# Wait until collection becomes active
print("Waiting until AOSS collection becomes active: ", end='')
while True:
    response = aoss_client.list_collections(collectionFilters={'name': aoss_collection_name})
    status = response['collectionSummaries'][0]['status']
    if status in ('ACTIVE', 'FAILED'):
        print(" done.")
        break
    print('█', end='', flush=True)
    time.sleep(5)

print("An AOSS collection created:", json.dumps(response['collectionSummaries'], indent=2))

# 3.2 Grant BKB Access to AOSS Data
# we create data access policy that grants BKB the necessary permissions to read from our AOSS collections.
# we then attach this policy to the bedrock execution role we created earlier, allowing BKB to securely access AOSS data
# when generating responses.

aoss_policy_arn = utility.create_oss_policy_attach_bedrock_execution_role(
    collection_id=aoss_collection['createCollectionDetail']['id'],
    bedrock_kb_execution_role=bedrock_kb_execution_role)

print("Waiting 60 sec for data access rules to be enforced: ", end='')
for _ in range(12):  # 12 * 5 sec = 60 sec
    print('█', end='', flush=True)
    time.sleep(5)
print(" done.")

print("Created and attached policy with ARN:", aoss_policy_arn)

# 3.3 Create an AOSS Vector Index
# Now that we have all necessary access permissions in place, we can create a vector index in the AOSS collection we created previously.

from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection

# Use default credential configuration for authentication
credentials = boto_session.get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    aws_region,
    'aoss',
    session_token=credentials.token)

# Construct AOSS endpoint host
host = f"{aoss_collection['createCollectionDetail']['id']}.{aws_region}.aoss.amazonaws.com"

# Build the OpenSearch client
os_client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=300
)

# Define the configuration for the AOSS vector index
index_definition = {
   "settings": {
      "index.knn": "true",
       "number_of_shards": 1,
       "knn.algo_param.ef_search": 512,
       "number_of_replicas": 0,
   },
   "mappings": {
      "properties": {
         "vector": {
            "type": "knn_vector",
            "dimension": embedding_model_dim,
             "method": {
                 "name": "hnsw",
                 "engine": "faiss",
                 "space_type": "l2"
             },
         },
         "text": {
            "type": "text"
         },
         "text-metadata": {
            "type": "text"
         }
      }
   }
}

# Create an OpenSearch index
response = os_client.indices.create(index=aoss_index_name, body=index_definition)

# Waiting for index creation to propagate
print("Waiting 30 sec for index update to propagate: ", end='')
for _ in range(6):  # 6 * 5 sec = 30 sec
    print('█', end='', flush=True)
    time.sleep(5)
print(" done.")

print("A new AOSS index created:", json.dumps(response, indent=2))

# 4. Configure Amazon Bedrock Knowledge Base and Synchronize it with Data Source

# In this section, we;ll create an Amazon Bedrock Knowledge Base (BKB) and connect it to the
# data that will be stored in our newly created AOSS vector index.
# 
# 4.1 Creat a Bedrock Knowledge base
# 
# 1. Knowledge Base involces providing two key configs
# 
# Storage Configuration tells Bedrock wher to store the generated vector embeddings
# by specifying the target vector store and provind the necessary connection detail (we use the AOSS vector index created earlier),
# 
# 2. Knowledge Base Config
# 
# Defines how Bedrock should generate vector embeddings from your data by specifying the embedding model to use
# (Titan Text Embeddings V2) in this sample, along with any additional settings required for handling multimodel content.

# Vector Storage Configuration
storage_config = {
    "type": "OPENSEARCH_SERVERLESS",
    "opensearchServerlessConfiguration": {
        "collectionArn": aoss_collection["createCollectionDetail"]['arn'],
        "vectorIndexName": aoss_index_name,
        "fieldMapping": {
            "vectorField": "vector",
            "textField": "text",
            "metadataField": "text-metadata"
        }
    }
}

# Knowledge Base Configuration
knowledge_base_config = {
    "type": "VECTOR",
    "vectorKnowledgeBaseConfiguration": {
        "embeddingModelArn": embedding_model_arn
    }
}

response = bedrock_agent_client.create_knowledge_base(
    name=bedrock_kb_name,
    description="Amazon shareholder letter knowledge base.",
    roleArn=bedrock_kb_execution_role_arn,
    knowledgeBaseConfiguration=knowledge_base_config,
    storageConfiguration=storage_config)

bedrock_kb_id = response['knowledgeBase']['knowledgeBaseId']

print("Waiting until BKB becomes active: ", end='')
while True:
    response = bedrock_agent_client.get_knowledge_base(knowledgeBaseId=bedrock_kb_id)
    if response['knowledgeBase']['status'] == 'ACTIVE':
        print(" done.")
        break
    print('█', end='', flush=True)
    time.sleep(5)

print("A new Bedrock Knowledge Base created with ID:", bedrock_kb_id)

# Call a Bedrock API to get info about the Knowledge Base:

response = bedrock_agent_client.get_knowledge_base(knowledgeBaseId=bedrock_kb_id)

print(json.dumps(response['knowledgeBase'], indent=2, default=str))

# 4.2 Connect BKB to a Data Source

# With the Knowledge Base in place, the next step is to connect it to a data source.

# 1. Create a data source for the Knowledge Base that will point to the location of our raw data
# 2. Define how that data should be processed and ingested into the vector store - for example by specifying a chunking
# configuration that controls how large each text fragment should be when generating vector embeddings for retrieval.

# Data Source Configuration
data_source_config = {
        "type": "S3",
        "s3Configuration":{
            "bucketArn": f"arn:aws:s3:::{s3_bucket_name}",
            # "inclusionPrefixes":["*.*"]   # you can use this if you want to create a KB using data within s3 prefixes.
        }
    }

# Vector Ingestion Configuration
vector_ingestion_config = {
        "chunkingConfiguration": {
            "chunkingStrategy": "FIXED_SIZE",
            "fixedSizeChunkingConfiguration": {
                "maxTokens": 512,
                "overlapPercentage": 20
            }
        }
    }

response = bedrock_agent_client.create_data_source(
    name=bedrock_kb_name,
    description="Amazon shareholder letter knowledge base.",
    knowledgeBaseId=bedrock_kb_id,
    dataSourceConfiguration=data_source_config,
    vectorIngestionConfiguration=vector_ingestion_config
)

bedrock_ds_id = response['dataSource']['dataSourceId']

print("A new BKB data source created with ID:", bedrock_ds_id)

# Bedrock API call to get info on the newly created BKB data source:

response = bedrock_agent_client.get_data_source(knowledgeBaseId=bedrock_kb_id, dataSourceId=bedrock_ds_id)

print(json.dumps(response['dataSource'], indent=2, default=str))

# 4.3 Synchronize BKB with Data Source

# Once the Knowledge Base and its data source are configured, we can start a fully - managed data ingestion job.

# Fully managed data ingestion workflow:
# Source data -> Text Extraction -> Chunking -> Embedding -> Indexing

# Start an ingestion job
response = bedrock_agent_client.start_ingestion_job(knowledgeBaseId=bedrock_kb_id, dataSourceId=bedrock_ds_id)

bedrock_job_id = response['ingestionJob']['ingestionJobId']

print("A new BKB ingestion job started with ID:", bedrock_job_id)

# Wait until ingestion job completes
print("Waiting until BKB ingestion job completes: ", end='')
while True:
    response = bedrock_agent_client.get_ingestion_job(
        knowledgeBaseId = bedrock_kb_id,
        dataSourceId = bedrock_ds_id,
        ingestionJobId = bedrock_job_id)
    if response['ingestionJob']['status'] == 'COMPLETE':
        print(" done.")
        break
    print('█', end='', flush=True)
    time.sleep(5)

print("The BKB ingestion job finished:", json.dumps(response['ingestionJob'], indent=2, default=str))

# In conclusion, we went through the process of creating an Amazon Bedrock Knowledge Base (BKB)
# and ingesting documnets to enable Retrieval Augemented Generation (RAG) capabilities.
# We started by setting up the environment, installing the required libraries, and initializing the necessary AWS
# service clients.
# THen, we created an Amazon S3 bucket to store unstructured data (PDF documents) and uploaded sample files.
# We proceeded by proviisioning an Amazon OpenSearch Serverless (AOSS) collection and index, configuring the appropirate 
# IAM roles and permissions, and granting access to the BKB. Finally, we created the BKB, connected it to the S3 data source,
# and sycnhronized the douments to generate vector emeddings, whcih were stored in AOSS.