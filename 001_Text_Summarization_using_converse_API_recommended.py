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

# Create a converse request with our summarization task
converse_request = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "text": f"Please provide a concise summary of the following text in 2-3 sentences. Text to summarize: {text_to_summarize}"
                }
            ]
        }
    ],
    "inferenceConfig": {
        "temperature": 0.4,
        "topP": 0.9,
        "maxTokens": 500
    }
}

# Call Claude 3.7 Sonnet with Converse API
try:
    response = bedrock.converse(
        modelId=MODELS["Claude 3.7 Sonnet"],
        messages=converse_request["messages"],
        inferenceConfig=converse_request["inferenceConfig"]
    )
    
    # Extract the model's response
    claude_converse_response = response["output"]["message"]["content"][0]["text"]
    display_response(claude_converse_response, "Claude 3.7 Sonnet (Converse API)")
except botocore.exceptions.ClientError as error:
    if error.response['Error']['Code'] == 'AccessDeniedException':
        print(f"\x1b[41m{error.response['Error']['Code']}: {error.response['Error']['Message']}\x1b[0m")
        print("Please ensure you have the necessary permissions for Amazon Bedrock.")
    else:
        raise error
    
# Now, that we have used the Converse API, let's take some time to take a closer look. To use the Converse API, you use the Converse or ConverseStream (for streaming responses) operations to send messages to a model. 
# While, it is possible to use the existing base inference operations (InvokeModel or InvokeModelWithResponseStream) for conversation applications as well, we recommend using the Converse API as it provides consistent API, that works with all Amazon Bedrock models that support messages. 

# Easily switch between models
# One of the biggest advantages of the Converse APi is the ability to easily switch between models using the exact same request format.
# Let's compare summaries across different foundation models by looping over the model dictionary we defined above:

# call different models with the same converse request
results = {}    
for model_name, model_id in MODELS.items(): # looping over all models defined above
        try:
            start_time = time.time()
            response = bedrock.converse(
                modelId=model_id,
                messages=converse_request["messages"],
                inferenceConfig=converse_request["inferenceConfig"] if "inferenceConfig" in converse_request else None
            )
            end_time = time.time()
            
            # Extract the model's response using the correct structure
            model_response = response["output"]["message"]["content"][0]["text"]
            response_time = round(end_time - start_time, 2)
            
            results[model_name] = {
                "response": model_response,
                "time": response_time
            }
            
            print(f"✅ Successfully called {model_name} (took {response_time} seconds)")
            
        except Exception as e:
            print(f"❌ Error calling {model_name}: {str(e)}")
            results[model_name] = {
                "response": f"Error: {str(e)}",
                "time": None
            }

# Display results in a formatted way
for model_name, result in results.items():
    if "Error" not in result["response"]:
        display(Markdown(f"### {model_name} (took {result['time']} seconds)"))
        display(Markdown(result["response"]))
        print("-" * 80)

# 2.5 Cross Regional Inference in Amazon Bedrock

# Amazon Bedrock offers cross regional inference which automatically selects the optimal AWS Region within your geography to process your inference requests.
# Cross-Regional Inference offers higher throughput limits (up to 2x allocated quotas) and seamlessly manages traffic bursts by dynamically routing requests across multiple AWS regions.
# Enhancing application resilience during peak demand periods without additional routing or data transfer costs. Customers can control where there inference data flows by selecting from a pre-defined set of regions.
# Helping them comply with applicable data residency requirements and soveriegnty laws. Moreover, this capability prioritizes the connected bedrock API source region when possible, helping to minimize latency and improve responsivemness.
# As a result, customers can enhance their applications reliability performance, and efficiency. Please review the list of supported regions and models for inference profiles.

# Regular model invocation (standard region)
standard_response = bedrock.converse(
    modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",  # Standard model ID
    messages=converse_request["messages"]
)

# Cross-region inference (note the "us." prefix)
cris_response = bedrock.converse(
    modelId="us.anthropic.claude-3-5-sonnet-20240620-v1:0",  # Cross-region model ID with regional prefix
    messages=converse_request["messages"]
)

# Print responses
print("Standard response:", standard_response["output"]["message"]["content"][0]["text"])
print("Cross-region response:", cris_response["output"]["message"]["content"][0]["text"])

# 2.6 Multi-turn Conversations

# The Converse API makes multi-turn conversations simple.

# Example of a multi-turn conversation with Converse API
multi_turn_messages = [
    {
        "role": "user",
        "content": [{"text": f"Please summarize this text: {text_to_summarize}"}]
    },
    {
        "role": "assistant",
        "content": [{"text": results["Claude 3.7 Sonnet"]["response"]}]
    },
    {
        "role": "user",
        "content": [{"text": "Can you make this summary even shorter, just 1 sentence?"}]
    }
]

try:
    response = bedrock.converse(
        modelId=MODELS["Claude 3.7 Sonnet"],
        messages=multi_turn_messages,
        inferenceConfig={"temperature": 0.2, "maxTokens": 500}
    )
    
    # Extract the model's response using the correct structure
    follow_up_response = response["output"]["message"]["content"][0]["text"]
    display_response(follow_up_response, "Claude 3.7 Sonnet (Multi-turn conversation)")
    
except Exception as e:
    print(f"Error: {str(e)}")

# 2.7 Streaming Responses with ConverseStream API

# For longer generations, you might want to receive the content as it's being generated. The ConverseStream API supports streaming.
# Which allows you to process the response incrementally:

# Example of streaming with Converse API
def stream_converse(model_id, messages, inference_config=None):
    if inference_config is None:
        inference_config = {}
    
    print("Streaming response (chunks will appear as they are received):\n")
    print("-" * 80)
    
    full_response = ""
    
    try:
        response = bedrock.converse_stream(
            modelId=model_id,
            messages=messages,
            inferenceConfig=inference_config
        )
        response_stream = response.get('stream')
        if response_stream:
            for event in response_stream:

                if 'messageStart' in event:
                    print(f"\nRole: {event['messageStart']['role']}")

                if 'contentBlockDelta' in event:
                    print(event['contentBlockDelta']['delta']['text'], end="")

                if 'messageStop' in event:
                    print(f"\nStop reason: {event['messageStop']['stopReason']}")

                if 'metadata' in event:
                    metadata = event['metadata']
                    if 'usage' in metadata:
                        print("\nToken usage")
                        print(f"Input tokens: {metadata['usage']['inputTokens']}")
                        print(
                            f":Output tokens: {metadata['usage']['outputTokens']}")
                        print(f":Total tokens: {metadata['usage']['totalTokens']}")
                    if 'metrics' in event['metadata']:
                        print(
                            f"Latency: {metadata['metrics']['latencyMs']} milliseconds")

                
            print("\n" + "-" * 80)
        return full_response
    
    except Exception as e:
        print(f"Error in streaming: {str(e)}")
        return None
    
# Let's try streaming a longer summary
streaming_request = [
    {
        "role": "user",
        "content": [
            {
                "text": f"""Please provide a detailed summary of the following text, explaining its key points and implications:
                
                {text_to_summarize}
                
                Make your summary comprehensive but clear.
                """
            }
        ]
    }
]

# Only run this when you're ready to see streaming output
streamed_response = stream_converse(
    MODELS["Claude 3.7 Sonnet"], 
    streaming_request, 
    inference_config={"temperature": 0.4, "maxTokens": 1000}
)

# 4. FUnction Calling with Amazon Bedrock

# Modern LLMs like claude go beyond generating free-form text, they can also reason about when external tools or functions should be used to better answer user questions.
# This capability, known as function calling (or tool use), enables the model to decode which function to call, and what parameters to provide, but importantly, the model does not execute the function itself.

# Instead, the model reutrns a well-structured response (typically in JSON format) that describes the intended function call. It's then up to you application to detect this output, execute the requested function (such as calling an API or querying a database), and pass the result
# back to the model - allowing it to generate a final user-friendly response that incorporates real-world data.

# Function calling is especially useful when building LLM-powered applications that need acces to dynamc, external information - for examples, retrieving real-time waether data, which is exactly what we'll demonstrate in the "this" section.

def handle_function_calling(model_id, request, tool_config):
    try:
        # Step 1: Send initial request
        response = bedrock.converse(
            modelId=model_id,
            messages=request["messages"],
            inferenceConfig=request["inferenceConfig"],
            toolConfig=tool_config
        )
        
        # Check if the model wants to use a tool (check the correct response structure)
        content_blocks = response["output"]["message"]["content"]
        has_tool_use = any("toolUse" in block for block in content_blocks)
        
        if has_tool_use:
            # Find the toolUse block
            tool_use_block = next(block for block in content_blocks if "toolUse" in block)
            tool_use = tool_use_block["toolUse"]
            tool_name = tool_use["name"]
            tool_input = tool_use["input"]
            tool_use_id = tool_use["toolUseId"]
            
            # Step 2: Execute the tool
            if tool_name == "get_weather":
                tool_result = get_weather(tool_input["location"])
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}
            
            # Step 3: Send the tool result back to the model
            updated_messages = request["messages"] + [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": tool_use_id,
                                "name": tool_name,
                                "input": tool_input
                            }
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [
                                    {
                                        "json": tool_result
                                    }
                                ],
                                "status": "success"
                            }
                        }
                    ]
                }
            ]
            
            # Step 4: Get final response
            final_response = bedrock.converse(
                modelId=model_id,
                messages=updated_messages,
                inferenceConfig=request["inferenceConfig"],
                toolConfig=tool_config  
            )
            
            # Extract text from the correct response structure
            final_text = ""
            for block in final_response["output"]["message"]["content"]:
                if "text" in block:
                    final_text = block["text"]
                    break
            
            return {
                "tool_call": {"name": tool_name, "input": tool_input},
                "tool_result": tool_result,
                "final_response": final_text
            }
        else:
            # Model didn't use a tool, just return the text response
            text_response = ""
            for block in content_blocks:
                if "text" in block:
                    text_response = block["text"]
                    break
                    
            return {
                "final_response": text_response
            }
    
    except Exception as e:
        print(f"Error in function calling: {str(e)}")
        return {"error": str(e)}
    
function_result = handle_function_calling(
    MODELS["Claude 3.7 Sonnet"], 
    function_request,
    weather_tool
)

# Display the results
if "error" not in function_result:
    if "tool_call" in function_result:
        print(f"Tool Call: {function_result['tool_call']['name']}({function_result['tool_call']['input']})")
        print(f"Tool Result: {function_result['tool_result']}")
    
    display_response(function_result["final_response"], "Claude 3.7 Sonnet (Function Calling)")
else:
    print(f"Error: {function_result['error']}")



