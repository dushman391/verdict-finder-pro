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

# API KEY for google's youtube API
API_KEY = googleapikey

os.environ['OPENAI_API_KEY'] = openaiapikey
os.environ['TOKENIZERS_PARALLELISM'] = 'false'



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
        maxResults=15,  # Retrieve the top 20 search results
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


# Streamlit app code
def main():
    st.title("Verdict Finder Pro")
    product = st.text_input("Enter Amazon Product URL")
    product_title = ""
    if product:
        product_title = scrape_product_data(product)

    # Initialize session state
    if 'video_ids' not in st.session_state:
        st.session_state['video_ids'] = [] 

    with st.expander("Select YouTube Videos"):
        # Get video ids
        final_video_ids = []
        final_video_ids = generate_video_ids(product_title)

    if st.button("Compile Data Set"):
        for video_id in final_video_ids:
            generate_transcript(video_id)
            print(f"Transcript for Video ID: {video_id} saved.")

        merge_files()
        vc = compile_data()

        st.write("Initializing LLM.")

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        st.write("Initialized LLM. Building prompt")

        # Build prompt
        template = """Identify the following items from below context: 
        - Sentiment (positive or negative)
        - Is the reviewer expressing anger? (true or false)
        - Item purchased by reviewer
        - Company that made the item
        - consider the context and generate overall summary which should indicate wether product is worth buying, 
        indicate positive and negative points 
        - summarise negative reviews
        - total number of reviews

        The context is delimited with triple backticks. \
        Format your response as a JSON object with \
        "Sentiment", "Anger", "Item", "Brand", "Summary", "Cons", "number of reviews" as the keys.
        If the information isn't present, use "unknown" \
        as the value.
        Make your response as short as possible.
        Format the Anger value as a boolean.
        '''{context}'''
        """
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        # Run chain
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vc.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        question = ""

        result = qa_chain({"query": question})

        st.write(result["result"])

if __name__ == "__main__":
    main()
