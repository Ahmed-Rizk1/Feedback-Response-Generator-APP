﻿# Feedback-Response-Generator-APP
# Feedback Response Generator

This project is a simple Streamlit web app that:
- Accepts user feedback as input
- Uses a language model (GPT-4o) to classify the sentiment as positive, negative, neutral, or escalate
- Generates an appropriate AI response based on the sentiment

It uses [LangChain](https://www.langchain.com/), [OpenAI](https://platform.openai.com/), and [Streamlit](https://streamlit.io/).

---

## Demo

![Screenshot](screenshot.png) <!-- optional, add image if available -->

---

## Features

- Sentiment classification using GPT-4o
- Dynamic prompt templates per feedback type
- Response generation via LangChain runnable pipeline
- Lightweight web UI using Streamlit

---

## Installation

1. Clone the repo

```bash
git clone https://github.com/Ahmed-Rizk1/Feedback-Response-Generator-APP
cd feedback-response-generator
