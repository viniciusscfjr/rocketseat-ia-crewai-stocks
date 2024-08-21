"""
This module sets up a Streamlit application to analyze stock prices and market news.
It uses various agents and tasks to fetch stock price data, analyze news, and generate reports.
"""

# IMPORT DAS LIBS
import os
from datetime import datetime
from crewai import Agent, Task, Process, Crew

import yfinance as yf

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

import pandas as pd


# CRIANDO YAHOO FINANCE TOOL
def fetch_stock_price(ticket: str) -> pd.DataFrame:
    """
    Fetches the stock price data for a given stock ticker from Yahoo Finance.

    Args:
        ticket (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.).

    Returns:
        pd.DataFrame: A DataFrame containing the stock price data for the past year.
    """
    stock = yf.download(ticket, period="1y")
    return stock


yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="This tool fetches stock prices from Yahoo Finance API for {ticket} in the last year.",
    func=fetch_stock_price,
)

# IMPORTANDO OPENAI LLM - GPT
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])

# CRIANDO AGENTE - STOCK PRICE ANALYST
stockPriceAnalyst = Agent(
    role="Senior Stock Price Analyst",
    goal="Find the {ticket} stock price in the last year and analyze it.",
    backstory="You're a highly experienced in analyzing the price of an specific stock and make predictions about it's future price.",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=False,
    tools=[yahoo_finance_tool],
)

getStockPrice = Task(
    description="Analyze the stock price history of {ticket} and create a trend analyses of up, down or sideways",
    expected_output=""" Specify the current trend of the stock price of {ticket} - up, down or sideways.
    eg. stock = 'AAPL, price UP'
    """,
    agent=stockPriceAnalyst,
)

# IMPORTANDO A TOOL DE SEARCH
search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)

# CRIANDO AGENTE - NEWS ANALYST
newsAnalyst = Agent(
    role="Stock News Analyst",
    goal="""
    Create a short summary of the market news related to the stock {ticket} company.
    Specify the current trend of the stock price of {ticket} - up, down or sideways with the news context.
    For each requested stock asset, specify a number between 0 and 100, where 0 means extreme fear and 100 means extreme greed.
    """,
    backstory="""
    You're a highly experienced in analyzing the market trends and news related to an specific stock and
    make predictions about it's future price and have tracked assets for more than 10 years.
    You're also master level analyst in the traditional market and have deep understanding of human psychology.
    You understand news, their titles and information, but you look at those with a healthy dose of skepticism,
    as you know that news can be biased and that the market is driven by human emotions.
    You also consider the source of news article.
    """,
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=False,
    tools=[search_tool],
)

getNews = Task(
    description="""
    Analyze the news related to stock {ticket} and create a summary of the market news related to the stock {ticket}.
    Use the search tool to find the news.

    The current date is {current_date}.

    Compose the results into a helpful report for the user.
    """,
    expected_output="""A summary of the overall market and one sentence summary for each request asset.
    Include a fear/greed score for each asset based on the news. Use the format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    """,
    agent=newsAnalyst,
)

# CRIANDO AGENTE - STOCK ANALYST WRITE
stockPriceAnalystWrite = Agent(
    role="Senior Stock Analyst Writer",
    goal="""Write an insightful compelling and informative 3 paragraph long newsletter based on the stock report and price trend.""",
    backstory="""You're widely accepted as the best stock analyst in the market.
    You understand complex concepts and create compelling stories and narratives that resonate with wider audiences.
    You have a deep understanding of human psychology and how it affects the market.
    You have a strong track record of making accurate predictions and have been featured in major financial publications.
    You have a large following on social media and are known for your ability to explain complex concepts in simple terms.
    You understand macro factors and combine multiple theories.
    eg. cycle theory, fundamental analysis, technical analysis, etc.
    You're able to hold multiple opinions when analyzing a stock and can see both sides of an argument.
    """,
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True,
)

writeAnalysis = Task(
    description="""
    Use the stock price trend and the stock news report to create an analysis and write the newsletter about
    the {ticket} company that is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
    Include the previous analysis of stock trend and news summary.
    """,
    expected_output="""An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner.
    It should contain the following information:
    - 3 bullets executive summary.
    - Introduction - set the overall picture and spike up the interest.
    - Main part - provides the details of the analysis, including the news summary and fear/greed scores.
    - Summary -  Wrap up the key facts and concrete future trend prediction - up, down or sideways.
    """,
    agent=stockPriceAnalystWrite,
    context=[getStockPrice, getNews],
)

# CRIAR GRUPO DE AGENTES DE IA
crew = Crew(
    agents=[stockPriceAnalyst, newsAnalyst, stockPriceAnalystWrite],
    tasks=[getStockPrice, getNews, writeAnalysis],
    verbose=True,
    process=Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15,
)

# CRIAR A APLICAÇÃO
with st.sidebar:
    st.header("Enter the Stock to Research")

    with st.form(key="research_form"):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label="Run Research")

if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results = crew.kickoff(inputs={"ticket": topic, "current_date": datetime.now()})

        st.subheader("Results of research:")
        st.write(results["final_output"])
