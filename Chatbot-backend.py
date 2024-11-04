# Import the ConversationBufferMemory, ConversationChain, ChatBedrock (BedrockChat) Langchain Modules

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_aws import ChatBedrock

# Connect to Bedrock Service
def demo_chatbot(input_text):
    demo_llm = Chatbedrock(
        credentials_profile_name="default",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_kwargs={ "max_tokens_to_sample": 300,
    "temperature": 0.1,
    "top_p": 0.9,
    "stop_sequences": ["\n\nHuman:"]}
    )
    return demo_llm

# Setup The Conversation Buffer Memory
def demo_memory():
    llm_data=demo_chatbot()
    memory=ConversationSummaryBufferMemory(llm=llm_data, max_token_limit=300)
    return memory

# Setup the conversation Chain
def demo_conversation(input_text,memory):
    llm_chain_data=demo_chatbot()
    llm_conversation=ConversationChain(llm=llm_chain_data, memory=memory, verbose=True)

    # Invoke to send the prompt to the Foundational Model
    chat_reply=llm_conversation.invoke(input_text)
    return chat_reply['response']

