import json
import boto3
import botocore
from IPython.display import display, Markdown
import time

# 1.2 Initial setup for clients, global variables and helper functions

# Initialize Bedrock client
session = boto3.session.Session()
region = session.region_name
bedrock = boto3.client(service_name='bedrock-runtime', region_name=region)

# Define model IDs that will be used in this module
MODELS = {
    "Claude 3.7 Sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "Claude 3.5 Sonnet": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Claude 3.5 Haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "Amazon Nova Pro": "us.amazon.nova-pro-v1:0",
    "Amazon Nova Micro": "us.amazon.nova-micro-v1:0",
    "Meta Llama 3.1 70B Instruct": "us.meta.llama3-1-70b-instruct-v1:0"
}

# Utility function to display model responses in a more readable format
def display_response(response, model_name=None):
    if model_name:
        display(Markdown(f"### Response from {model_name}"))
    display(Markdown(response))
    print("\n" + "-"*80 + "\n")

# Text Summarization with foundation models
text_to_summarize = """
AWS took all of that feedback from customers, and today we are excited to announce Amazon Bedrock, \
a new service that makes FMs from AI21 Labs, Anthropic, Stability AI, and Amazon accessible via an API. \
Bedrock is the easiest way for customers to build and scale generative AI-based applications using FMs, \
democratizing access for all builders. Bedrock will offer the ability to access a range of powerful FMs \
for text and images—including Amazons Titan FMs, which consist of two new LLMs we're also announcing \
today—through a scalable, reliable, and secure AWS managed service. With Bedrock's serverless experience, \
customers can easily find the right model for what they're trying to get done, get started quickly, privately \
customize FMs with their own data, and easily integrate and deploy them into their applications using the AWS \
tools and capabilities they are familiar with, without having to manage any infrastructure (including integrations \
with Amazon SageMaker ML features like Experiments to test different models and Pipelines to manage their FMs at scale).
"""

# 2.1 Text summarizations using the invoke model API

# Create prompt for summarization
prompt = f"""Please provide a summary of the following text. Do not add any information that is not mentioned in the text below.
<text>
{text_to_summarize}
</text>
"""

# Create request body for Claude 3.7 Sonnet
claude_body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,
    "temperature": 0.250,
    "top_p": 0.999,
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ],
})

# Send request to Claude 3.7 Sonnet
try:
    response = bedrock.invoke_model(
        modelId=MODELS["Claude 3.7 Sonnet"],
        body=claude_body,
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get('body').read())
    
    # Extract and display the response text
    claude_summary = response_body["content"][0]["text"]
    display_response(claude_summary, "Claude 3.7 Sonnet (Invoke Model API)")
    
except botocore.exceptions.ClientError as error:
    if error.response['Error']['Code'] == 'AccessDeniedException':
        print(f"\x1b[41m{error.response['Error']['Message']}\
            \nTo troubleshoot this issue please refer to the following resources.\
            \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
            \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
    else:
        raise error

