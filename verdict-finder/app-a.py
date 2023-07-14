import os
import json
from config.apikey import googleapikey,openaiapikey
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, CouldNotRetrieveTranscript
from googleapiclient.discovery import build
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from product_scraper import scrape_product_data
from langchain.prompts import PromptTemplate
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)

# API KEY for google's youtube API
API_KEY = googleapikey



os.environ['OPENAI_API_KEY'] = openaiapikey
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import streamlit as st
import time

class Progress:
    """Progress bar and Status update."""
    def __init__(self, number_of_functions: int):
        self.n = number_of_functions
        self.bar = st.progress(0)
        self.progress = 1
        self.message = ""
        self.message_container = st.empty()

    def go(self, msg, function, *args, **kwargs):
        self.message += msg
        self.message_container.info(self.message)
        s = time.time()
        result = function(*args, **kwargs)
        self.message += f" [{time.time() - s:.2f}s]. "
        self.message_container.info(self.message)
        self.bar.progress(self.progress / self.n)
        self.progress += 1
        if self.progress > self.n:
            self.bar.empty()
        return result

def generate_transcript(video_id):
    try:
        transcript = ""
        tx = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        music_count = 0

        for segment in tx:
            # Ignoring bad transcripts
            if "[Music]" in segment['text']:
                music_count += 1
                if music_count > 5:
                    return
            transcript += segment['text']

        with open("transcript.txt", "a+") as opf:
            opf.seek(0)  # Move the file pointer to the beginning
            existing_data = opf.read()
            opf.seek(0)  # Move the file pointer to the beginning
            opf.truncate()  # Clear the file content
            opf.write(existing_data + transcript + "\n")  # Write the updated transcript

        return transcript
    except TranscriptsDisabled:
        print(f"Transcripts are disabled for Video ID: {video_id}. Skipping...")
    except CouldNotRetrieveTranscript:
        print(f"Transcript in English not available for Video ID: {video_id}. Skipping...")



def generate_video_ids(user_input):
    # Set up the YouTube Data API client
    api_key = API_KEY  # Replace with your own API key
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Search for videos based on user input
    search_response = youtube.search().list(
        q=user_input,
        part='id,snippet',
        maxResults=10,  # Retrieve the top 10 search results
        order='viewCount'  # Sort by view count in descending order
    ).execute()

    # Extract video IDs, URLs, and view counts from the search results
    video_ids = []
    for search_result in search_response.get('items', []):
        if search_result['id'].get('videoId'):
            video_id = search_result['id']['videoId']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            video_info = {
                'video_id': video_id,
                'video_url': video_url,
                'title': search_result['snippet']['title'],
                'thumbnail_url': search_result['snippet']['thumbnails']['high']['url'],
            }
            
            # Create a table with three columns, one for the thumbnails, one for the video details, and one for the checkboxes
            col1, col2, col3 = st.columns([1, 1, 1])
            # Add the thumbnail to the first column
            with col1:
                st.image(video_info['thumbnail_url'], width=100)
            # Add the video details to the second column
            with col2:
                st.write(f"Title: {video_info['title']}")
            # Add a checkbox to the third column with a label based on the video title
            with col3:
                video_selected = st.checkbox('.', key=f"{video_id}", value=video_id in st.session_state['video_ids'])
                if video_selected:
                    st.session_state['video_ids'].append(video_id)
                    print("selected video",video_id)
                else:
                    if video_id in st.session_state['video_ids']:
                        st.session_state['video_ids'].remove(video_id)
    
    # Return the selected video IDs
    return st.session_state['video_ids']

def get_text_chunks(raw_texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
    chunks = text_splitter.create_documents(raw_texts)
    print("Chunking Completed")
    return chunks


def get_vectorstore(chunks, existing_vectorstore=None):
    vectorstore_dir = "vectorstore_data"

    if not os.path.exists(vectorstore_dir):
        os.makedirs(vectorstore_dir)

    vectorstore_file = os.path.join(vectorstore_dir, "vectorstore.faiss")

    if not os.path.exists(vectorstore_file):
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
        if existing_vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            vectorstore = existing_vectorstore.append_documents(chunks, embeddings)
            vectorstore.save(vectorstore_file)
    else:
        vectorstore = FAISS.load(vectorstore_file)

    print("Embedding Completed")
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    coversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retreiver(),
        memory=memory
    )
    return coversation_chain

def merge_files():
    try:
        with open("transcript.txt", "r") as transcript_file, open("product_reviews.txt", "r") as review_file, open("combined_data.txt", "w") as combined_file:
            transcript_content = transcript_file.read()
            review_content = review_file.read()
            combined_file.write(transcript_content + "\n" + review_content)

        print("Files merged successfully.")
    except FileNotFoundError:
        print("One or more input files not found.")
    except IOError:
        print("An error occurred while merging the files.")


def compile_data():
    with open('combined_data.txt') as j:
        raw_text = j.read()

        chunks = get_text_chunks([raw_text])

        
        vectorstore = get_vectorstore(chunks)
        return vectorstore

def format_text_with_emoji(text, emoji):
    return f"{emoji} {text}"
def add_readme_to_sidebar():
    # Add content to the sidebar
    st.sidebar.markdown("# Verdictify")

    # Smaller version of the README content
    st.sidebar.markdown("🎯 Verdictify helps users make purchasing decisions for products by analyzing YouTube video transcripts and user-defined preferences.")

    st.sidebar.markdown("## Instructions")
    st.sidebar.markdown("1. Enter an Amazon product URL or product name. 🛍️")
    st.sidebar.markdown("2. Set your preferences for budget-friendliness, customer satisfaction, brand reputation, and innovation. 💡")
    st.sidebar.markdown("3. Select YouTube videos related to the product. 📺")
    st.sidebar.markdown("4. Click the 'Compile Data Set' button to generate transcripts and combine them with product reviews. 🔄")
    st.sidebar.markdown("5. View the final verdict and analysis provided by the app. ✔️")

    st.sidebar.markdown("Note: Provide the necessary API keys and dependencies as mentioned in the code comments.")

    st.sidebar.markdown("Enjoy using Verdictify to make informed purchasing decisions! 🎉")
# Streamlit app code
def main():
    # Title and description
    st.title("🎯 Verdictify: Your Decision Helper")
    st.markdown("Welcome to Verdictify! This site will help you decide whether to buy a product or not.")
    add_readme_to_sidebar()
    # Initialize session state
    if 'video_ids' not in st.session_state:
        st.session_state['video_ids'] = []

    # Initialize scores
    if 'scores' not in st.session_state:
        st.session_state['scores'] = {
            'budget_friendly': 0,
            'customer_satisfaction': 0,
            'innovation': 0
        }

    # Initialize product title
    product_title = ""

    # Input fields for product URL and name
    col1, col2 = st.columns(2)
    with col1:
        product = st.text_input("Enter Amazon Product URL")
    with col2:
        product_name = st.text_input("Enter Product Name")

    # Check if either the product URL or name is provided
    if product:
        product_title = st.session_state.get('product_title', "")
        if not product_title:
            product_title = scrape_product_data(product)
            st.session_state['product_title'] = product_title
    elif product_name:
        product_title = product_name

    with st.expander("What matters the most to you?"):
        # Select sliders for personalized input
        budget_friendly = st.select_slider(
            '💵 Budget Friendly',
            options=["doesn't matter", 'good to have', 'matters the most'],
            key=1
        )
        st.session_state['scores']['budget_friendly'] = 0 if budget_friendly == "doesn't matter" else 3 if budget_friendly == "good to have" else 5

        customer_satisfaction = st.select_slider(
            '😀 Customer Satisfaction',
            options=["doesn't matter", 'good to have', 'matters the most'],
            key=2
        )
        st.session_state['scores']['customer_satisfaction'] = 0 if customer_satisfaction == "doesn't matter" else 3 if customer_satisfaction == "good to have" else 5

        

        innovation = st.select_slider(
            '💡 Innovation',
            options=["doesn't matter", 'good to have', 'matters the most'],
            key=4
        )
        st.session_state['scores']['innovation'] = 0 if innovation == "doesn't matter" else 3 if innovation == "good to have" else 5

    with st.expander("Select YouTube Videos"):
        # Get video ids using the product title
        final_video_ids = generate_video_ids(product_title)

    if st.button("Verdictify!"):
        
        for video_id in final_video_ids:
            generate_transcript(video_id)
            print(f"Transcript for Video ID: {video_id} saved.")
        progress = Progress(number_of_functions=2)

        # Call functions with progress bar and status update
        # progress.go("Generating transcripts...", generate_transcript, final_video_ids)
        progress.go("AI Magic happening...", merge_files)
        vc = progress.go("More AI Magic happening...", compile_data)

        print("Initializing LLM.")

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        print("Initialized LLM. Building prompt")
        # context = " You are a helpful assistant"
        
        question = ""
        question = "customer_satisfaction " + str(st.session_state['scores']['customer_satisfaction'])
        question += " budget_friendly " + str(st.session_state['scores']['budget_friendly']) + " "
        question += " innovation " + str(st.session_state['scores']['innovation']) + " "
        print(question)

        template = """
            You are a master at suggesting products to customers. Take the context in the angluar brackets to do the following steps:
            step 1: Translate the context to english if it is not already in english.
            step 2: Extract relevant information from the context to give the feedback 
            to new customers who could be interesting in buying the product.
            step 3: Use the Scores {question} that are basically categories and weights chosen by the user (5 matters most, 3 Good to have,  0 doesn't matter). 
                    For example, when Budget-Friendly is 5, the user is concerned most about the budget friendliness of the product. 
                    For example. when customer_satisfaction is 5, the user is looking for more positive reviews of the product.
                    For example. when innovation is 5, the user is looking for special features and new technology advancements
            step 4: Calculate the final verdict by combinig the user preferences in the score from step 3
                    It has to be a definitve answer and then give the calculation/inference for that decision. 
            Step 5: Assign the sentiment -  positive, negative and netural based on the context.
            step 5: Format your response as a JSON object with 
                    "Summary of the product", "Item", "Brand", "Positive Reviews", "Negative Reviews", "Key Features", "Final Verdict" as the keys.                    
                    Summary of the product : Should contain the summary of the entire context in 100 characters.
                    Positive Reviews: Extract and display the positive sentiments in 1-2 sentences (from step 5)
                    Negative Reviews: Extract and display the negative sentiments in 1-2 sentences (from step 5)
                    Key Features: Should contain all the features describing the product in short phrases.
                    Final Verdict: Display the final verdict from step 4 in accordance with the user choices. 
            step 6: Display the final json object.            
            context: <{context}>
        """
        # Build prompt

        print("Initializing Prompt Template")
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        print("Running chain")
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vc.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        question = ""

        print("Getting the Result")
        result = qa_chain({"query": question})

        print(result["result"])
        print(type(result["result"]))
        

        output_dict = json.loads(result["result"])
    
        st.balloons()
        st.header("🎯 The Verdict")
        st.write(f"**Summary of the product:** {output_dict['Summary of the product']}")
        st.write(f"**Item:** {output_dict['Item']} 📱")
        st.write(f"**Brand:** {output_dict['Brand']} 🏷️")
        st.write(f"**Positive Reviews:** {output_dict['Positive Reviews']} 👍")
        st.write(f"**Negative Reviews:** {output_dict['Negative Reviews']} 👎")
        st.write(f"**Key Features:** {output_dict['Key Features']} 🔑")
        st.subheader("Final Verdict")
        st.markdown(f"<p style='font-size:20px;'>{format_text_with_emoji(output_dict['Final Verdict'], '✅')} ✔️</p>", unsafe_allow_html=True)
                

        # print("Extracting the response data")
        # response_data = result["result"]

        # # Present the response data using Streamlit
        # st.subheader("Verdictify Result")
        # st.write("Here is the Verdictify analysis for the product:")

        # # Sentiment and Anger
        # st.write(f"Sentiment: {response_data['Sentiment']}")
        # st.write(f"Anger: {response_data['Anger']}")

        # # Item and Brand
        # st.write(f"Item: {response_data['Item']}")
        # st.write(f"Brand: {response_data['Brand']}")

        # # Price
        # st.write(f"Price: {response_data['Price']}")

        # # Positive and Negative Reviews
        # st.subheader("Customer Reviews")
        # st.write("Positive Reviews:")
        # for review in response_data['Positive Reviews']:
        #     st.write(f"- {review}")
            
        # st.write("Negative Reviews:")
        # for review in response_data['Negative Reviews']:
        #     st.write(f"- {review}")

        # # Key Features and Product Fit
        # st.subheader("Product Details")
        # st.write("Key Features:")
        # for feature in response_data['Key Features']:
        #     st.write(f"- {feature}")
            
        # st.write(f"Product Fit: {response_data['product fit']}")

        # # Summary based on Scores and Final Verdict
        # st.subheader("Verdict and Summary")
        # st.write(f"Summary based on Scores: {response_data['Summary based on Scores']}")
        # st.write(f"Final Verdict: {response_data['Final Verdict']}")




    if st.button("Clear Screen"):
        st.experimental_rerun()



if __name__ == "__main__":
    main()
