
# Asset Portfolio Management and Investment Portfolio Optimization using ML and RL

This repository represents work for two related projects:

1. **Asset Portfolio Management using Deep Reinforcement Learning (DRL)**

    - **Objective**: This task aims to develop a model-based Deep Reinforcement Learning algorithm to implement Tactical Asset Allocation (TAA) of a portfolio. The goal is to capture short to medium-term market trends and anomalies to optimize portfolio performance while increasing risk-adjusted returns.

    - **Project Layout**: The project is divided into six parts, each explained in detail in corresponding Jupyter notebooks:

        1. Loading and Installing Relevant Python Libraries
        2. Downloading Data for Analysis
        3. Feature Engineering and Data Preprocessing
        4. Uniform Weights and Maximum Sharpe Strategies
        5. Deep Reinforcement Learning Portfolios

    - **Tech Stack**: Python, FinRL Library, PyPortfolioOpt, Ta-Lib, Pyfolio
  
![RL.png](https://github.com/neelgandhi108/Financial-Stock-Recommendation-System/raw/main/Assets/RL.png)


![RLPart2.png](https://github.com/neelgandhi108/Financial-Stock-Recommendation-System/raw/main/Assets/RLPart2.png)



2. **FinanceFormer Transformer-based Model for Investment Portfolio Optimization**

    - **Objective**: This project focuses on identifying high-quality stocks with high relative returns in a portfolio of 300 stocks. It constructs leading indicators (features) to predict future price changes and optimize portfolio returns.

    - **Project Components**:

        - Importing necessary libraries
        - Loading the data
        - Exploratory Data Analysis (EDA)
        - Creating subplots
        - Data preparation and dataset definition for modeling
        - Defining a FinanceFormer model and a training loop

    - **Tech Stack**: Python, PyTorch, Pandas, NumPy, Matplotlib, TQDM
  
![Transformer.jpg](https://github.com/neelgandhi108/Financial-Stock-Recommendation-System/raw/main/Assets/Transformer.jpg)

![Transformer.png](https://github.com/neelgandhi108/Financial-Stock-Recommendation-System/raw/main/Assets/Transformer.png)


## Hybrid Recommendation System 
![Stockhybrid.jpg](https://github.com/neelgandhi108/Financial-Stock-Recommendation-System/raw/main/Assets/Stockhybrid.jpg)

![AppPhoto.png](https://github.com/neelgandhi108/Financial-Stock-Recommendation-System/raw/main/Assets/AppPhoto.png)

![Flask.png](https://github.com/neelgandhi108/Financial-Stock-Recommendation-System/raw/main/Assets/Flask.png)

Incorporating recommendation system concepts into this combined project can enhance its functionality. Here are some additional concepts:

### Item-Based Collaborative Filtering using RL

Implement item-based collaborative filtering using reinforcement learning techniques to recommend assets or stocks based on their historical performance and user preferences.

### Content-Based Filtering using Transformers

Leverage transformer-based models to perform content-based filtering for investment assets. Transformers can analyze textual information, news sentiment, and other data sources to make personalized recommendations.

### Hybrid Recommendation Systems with Implicit Feedback

Combine collaborative filtering and content-based filtering approaches to build a hybrid recommendation system that handles implicit feedback. This approach can provide personalized investment recommendations.

### Dealing with Cold Start Problem

Addressed the cold start problem by implementing techniques to provide recommendations for new or less-traded stocks in a large-scale portfolio, ensuring users receive relevant suggestions from the beginning.

### Real-Time Recommendations

Implement real-time recommendation capabilities by integrating streaming data processing and updating recommendations dynamically based on market changes.

## Requirements

- Python 3.x
- PyTorch 1.9 or later
- Pandas
- NumPy
- Matplotlib
- TQDM

## [Asset-Portfolio-Management-using-Deep-Reinforcement-Learning in Depth](https://github.com/neelgandhi108/Financial-Stock-Recommendation-System/tree/main#asset-portfolio-management-using-deep-reinforcement-learning-)

## [Introduction](https://github.com/neelgandhi108/Financial-Stock-Recommendation-System/tree/main#introduction)

This repository represents work for the Worldquant University Capstone Project titled: Asset Portfolio Management using Deep Reinforcement Learning (DRL). The work presented explores the use of Deep Reinforcement Learning in dynamically allocating assets in a portfolio in order to solve the Tactical Asset Allocation (TAA) problem.

## [Objective](https://github.com/neelgandhi108/Financial-Stock-Recommendation-System/tree/main#objective)

The objective of the project is to develop a model based Deep Reinforcement Learning Algorithms that will implement a Tactical Asset Allocation of a portfolio. The project aims to address the TAA problem by accurately capturing short to medium term market trends and anomalies in order to allocate the assets in a portfolio so as to optimize its performance by increasing the risk adjusted returns.

## [Project Layout](https://github.com/neelgandhi108/Financial-Stock-Recommendation-System/tree/main#project-layout)

The project is broken down in six parts in each of the attached notebooks as follows:

**1.**  **Loading and Installing the Relevant Python Libraries**. The notebook gives the necessary libraries that are used for the implementation of the project. The FinRL library is utilized to implement our Deep Reinforcement Learning algorithms that are based on the Stable Baselines environments. In this notebook we clone the github repository for the FinRL library ([https://github.com/AI4Finance-LLC/FinRL-Library](https://github.com/AI4Finance-LLC/FinRL-Library)).

Other relevant libraries include the PyPortfolioOpt librariy which make use of in implementing the Maximum Sharpe allocation strategy which is used as comparison against the performance of the DRL models. We also make use of the Ta-Lib to add technical indicators to our input data. The backtesting and evaluation of models is implemented using the Pyfolio library.

**2.**  **Downloading the Data for the Analysis**. We make use of the YahooDownloader API in the FinRL library to get data for the 30 Dow Jones Industrial Average (Dow JIA) for the period 01 Jan 2009 to 31 December 2020. The data is checked for any missing values and for any indicators with missing data for the period specified to ensure that all the tickers have equal number of data points for the analysis. After cleaning up, the data is saved in the csv format for use in the analysis.

**3.**  **Feature Engineering and Data Preprocessing**. Feature Engineering and Data Preprocessing is performed on our dataset. We first load the saved csv file in a dataframe and then add technical indicators. A total of ten technical indicators have been added from the categories of Volume, Volatility, Trend and Momentum indicators. Additionally, we add covariance matrices with a one-year lookback to our input dataset. After adding the indicators, the data is split into train and test data in the ration of 80/20 respectively. The processed data is stored in dataframes for use in the analysis.

**4.**  **Uniform Weights and Maximum Sharpe Strategies**. Two strategies are developed for comparison with the DRL models. We first build a naïve model which is based on equal weight allocation. Then a model based on optimization of the risk adjusted returns (the Maximum Sharpe) is built.

The two portfolios are stored in data frames for Backtesting and benchmarking against the DRL models.

**5.**  **Deep Reinforcement Learning Portfolios**. Three DRL models are developed by making use of the FinRL Library which is based on the Stable Baselines environment. We present an implementation that explores the use of a number of technical indicators as inputs for solving the TTA problem.

The three models which we implement are based on the Stable Baselines algorithms for Actor Critic (A2C), Proximal Policy Optimization (PPO) and Deep Deterministic Policy Gradient (DDPG) algorithms.

The models implemented based on three algorithms are stored in dataframes for benchmarking and backtesting.

# [FinanceFormer Transformer based Model for Investment Portfolio-Optimization in Depth](https://github.com/neelgandhi108/FinanceFormer#financeformer-transformer-based-model-for-investment-portfolio-optimization)

## [Description](https://github.com/neelgandhi108/FinanceFormer#description)

The task aims to identify high-quality stocks with high relative returns in a portfolio of 300 stocks. The approach taken is to construct leading indicators (features) that can serve as signals of future price changes and are used as a basis for future predictions. The dataset contains 300 features for each of the 300 investments and the problem to be predicted is the relative return size of different investment targets on each day (time_id). The evaluation metric is the Pearson correlation coefficient, where a higher value indicates a more accurate prediction of the relative return size of all investment targets in the entire investment portfolio.


### Thank you for exploring my GitHub repository

Feel free to explore the code and notebooks in this repository to gain insights into asset portfolio management and investment portfolio optimization using machine learning and reinforcement learning techniques.












