import streamlit as st
import replicate
import os
from transformers import AutoTokenizer

# App title
st.set_page_config(page_title="Streamlit Replicate Chatbot", page_icon="💬")

# Replicate Credentials
with st.sidebar:
    st.title('💬 Streamlit Replicate Chatbot')
    st.write('Create chatbots using various LLM models hosted at [Replicate](https://replicate.com/).')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your Replicate API token.', icon='⚠️')
            st.markdown("**Don't have an API token?** Head over to [Replicate](https://replicate.com) to sign up for one.")
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader("Models and parameters")
    model = st.selectbox("Select a model",("meta/meta-llama-3-70b-instruct", "mistralai/mistral-7b-instruct-v0.2", "google-deepmind/gemma-2b-it"), key="model")
    if model == "google-deepmind/gemma-2b-it":
        model = "google-deepmind/gemma-2b-it:dff94eaf770e1fc211e425a50b51baa8e4cac6c39ef074681f9e39d778773626"
    
 # Load FAQ data from a JSON file
faq_data = [
    {
        "question": "What is a script issue and how can it be identified?",
        "answer": "A script issue occurs when there is consistent failure. If one build fails while others pass, it is likely an intermittent issue. Changes in script functionality indicate a Jenkins issue."
    },
    {
        "question": "What does it mean if the cluster page is not loaded and no details are fetched?",
        "answer": "If the cluster page is not loaded and no details are fetched, it is a product issue (web-related)."
    },
    {
        "question": "What are the types of issues faced in Gateway Monitoring?",
        "answer": "Issues include: intermittent or timeout issues, assertion errors, elements not found, and backend or script-related problems."
    },
    {
        "question": "What is an intermittent or timeout issue?",
        "answer": "Intermittent or timeout issues occur due to delays in backend responses. Logs may show timeouts, assertion errors, or mismatched values."
    },
    {
        "question": "What do I do if the failure happens inconsistently?",
        "answer": "If failures happen inconsistently across runs without script or environment changes, delays in backend responses may be the cause. Debug by checking the backend for high server load, database latency, or network delays."
    },
    {
        "question": "How can I debug intermittent failures?",
        "answer": "Manually check if it works. Retrigger the cases to see if they pass. Ensure the service is operational. If failures persist across builds, it may be a script issue."
    },
    {
        "question": "What does a backend issue or timing issue mean?",
        "answer": "Backend issues involve missing or delayed data from the server. Timing issues occur when the script tries to access data before it is available due to asynchronous behavior."
    },
    {
        "question": "How can I debug backend or timing issues?",
        "answer": "Manually verify the failure. Check the script and retest if necessary. If failures persist consistently across builds, it points to a script issue."
    },
    {
        "question": "What is an assertion error and how can it be resolved?",
        "answer": "An assertion error occurs when expected and actual values do not match. Debug by reviewing failed test cases, ensuring test logic aligns with expected behavior, and checking the code for potential issues. Rerun the test after debugging."
    },
    {
        "question": "What is a script issue with a missing csrftoken.json file?",
        "answer": "A script issue arises if the csrftoken.json file is missing or inaccessible. This file is essential for authentication (e.g., CSRF token)."
    },
    {
        "question": "How do I debug missing csrftoken.json file issues?",
        "answer": "Check the specified path (/app/testcases/ui/) to verify the file exists. Ensure it contains valid data. If missing or invalid, regenerate the file."
    },
    {
        "question": "What to do if devices are offline and all test cases fail?",
        "answer": "Step 1: Check the device in the Classic Central page to confirm if the gateway is online. Step 2: Verify the license or subscription for the device. Step 3: Check the device inventory to ensure it is assigned to Central."
    }
]


# Store LLM-generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything."}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything."}]

st.sidebar.button('Clear chat history', on_click=clear_chat_history)

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to make sure we're not sending too much text
    text to the Model. Eventually we will replace this with ArcticTokenizer
    """
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

# Function for generating model response
def generate_response():
    prompt = []
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            prompt.append("<|im_start|>user\n" + dict_message["content"] + "<|im_end|>")
        else:
            prompt.append("<|im_start|>assistant\n" + dict_message["content"] + "<|im_end|>")
    
    prompt.append("<|im_start|>assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)
    
    if get_num_tokens(prompt_str) >= 3072:
        st.error("Conversation length too long. Please keep it under 3072 tokens.")
        st.button('Clear chat history', on_click=clear_chat_history, key="clear_chat_history")
        st.stop()

    for event in replicate.stream(model,
                           input={"prompt": prompt_str,
                                  "prompt_template": r"{prompt}",
                                  "temperature": temperature,
                                  "top_p": top_p,
                                  }):
        yield str(event)

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = generate_response()
        full_response = st.write_stream(response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
