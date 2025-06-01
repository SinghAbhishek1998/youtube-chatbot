from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import urllib.parse as urlparse

load_dotenv()



# function to fetch video_id through youtube url link
def get_video_id(youtube_url):
    parsed = urlparse.urlparse(youtube_url)
    video_id = urlparse.parse_qs(parsed.query).get("v")
    return video_id[0] if video_id else None

#Indexing stage of RAG pipeline
#Youtube Transcript Api used for getting transcripts from youtube Video id with default language set here to english
def process_youtube_video(url):
  video_id = get_video_id(url)
  try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    return transcript
  except TranscriptsDisabled:
    return None


def create_vector_store(text):

  #splitter for chunking larger text into smaller documents 
  splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
  chunks = splitter.create_documents([text])

  #google generative AI embedding is used here for converting these document chunks into embeddings
  # so that it could be stored in vector database (FAISS here)
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)

  return vector_store

#function to merge multiple documents page content into one 
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

def run_rag_pipeline(vector_store,query):
  #Retrieval stage of RAG pipeline
  retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k": 4}) 
  #using similarity search type based on cosine similarity here other better options maybe MultiQuery Retriever or Contextual Compression Retriever


  #Augmentation
  llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.2)

  prompt = PromptTemplate(
      template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the Context is insufficient, just say you don't know.
        {context}
        Question: {question}
        Answer:
      """,
      input_variables= ['context','question']
  )

  parser = StrOutputParser()

  #create a parallel chain
  parallel_chain = RunnableParallel({
      'context': retriever | RunnableLambda(format_docs),
      'question': RunnablePassthrough()
  })


  final_chain = parallel_chain | prompt | llm | parser

  return final_chain.invoke(query)

