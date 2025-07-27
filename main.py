# app.py
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI
import os

# Load .env variables (for OpenAI key)
load_dotenv()

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Prompt templates
positive_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a thank you note for this positive feedback: {feedback}."),
])

negative_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a response addressing this negative feedback: {feedback}."),
])

neutral_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a request for more details for this neutral feedback: {feedback}."),
])

escalate_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Generate a message to escalate this feedback to a human agent: {feedback}."),
])

classification_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
])

# Define routing logic
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
)

# Combine classification and branching
classification_chain = classification_template | model | StrOutputParser()
chain = classification_chain | branches

# Streamlit UI
st.title("Feedback Response Generator")

feedback = st.text_area("Enter your feedback", height=150)

if st.button("Generate Response"):
    if feedback.strip():
        with st.spinner("Processing..."):
            response = chain.invoke({"feedback": feedback})
            st.subheader("Generated Response")
            st.write(response)
    else:
        st.warning("Please enter feedback first.")
