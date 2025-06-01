import streamlit as st
from rag_pipeline import *
st.title("ChatWithYT - Youtube Chatbot")
# take input from user using streamlit
url = st.text_input("Enter youtube url")
query = st.text_input("Enter your query")
if st.button("Process"):
    if not url or not query:
        st.warning("Please provide both YouTube URL and a query.")
    else:
        with st.spinner("Analyzing..."):
            transcript = process_youtube_video(url)
            print(transcript)
            if transcript:
                vector_store = create_vector_store(transcript)
                response = run_rag_pipeline(vector_store, query)
                print(response)
                st.success("Done!")
                st.markdown("### Response:")
                st.write(response)
            else:
                st.error("Transcript not available or disabled for this video.")