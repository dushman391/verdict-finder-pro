from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, CouldNotRetrieveTranscript
from googleapiclient.discovery import build
from googleapiclient.discovery import build
import streamlit as st


class YouTubeTranscriptGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.video_ids = []

    def generate_transcript(self, video_id):
        try:
            transcript = ""
            tx = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            music_count = 0

            for segment in tx:
                if "[Music]" in segment['text']:
                    music_count += 1
                    if music_count > 5:
                        return None
                transcript += segment['text']

            with open("transcripts.txt", "a+") as opf:
                opf.seek(0)
                existing_data = opf.read()
                opf.seek(0)
                opf.truncate()
                opf.write(existing_data + transcript + "\n")

            return transcript
        except TranscriptsDisabled:
            print(f"Transcripts are disabled for Video ID: {video_id}. Skipping...")
        except CouldNotRetrieveTranscript:
            print(f"Transcript in English not available for Video ID: {video_id}. Skipping...")

    def generate_video_ids(self, user_input):
        search_response = self.youtube.search().list(
            q=user_input,
            part='id,snippet',
            maxResults=10,
            order='viewCount'
        ).execute()

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
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.image(video_info['thumbnail_url'], width=100)
                with col2:
                    st.write(f"Title: {video_info['title']}")
                with col3:
                    video_selected = st.checkbox('.', key=f"{video_id}", value=video_id in self.video_ids)
                    if video_selected:
                        self.video_ids.append(video_id)
                    else:
                        if video_id in self.video_ids:
                            self.video_ids.remove(video_id)

        return self.video_ids
