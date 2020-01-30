# Machine Learning Engineer Nanodegree
## Capstone Proposal
Arnab Karmakar  
Januray 26th, 2020

## Proposal

### Domain Background

Stock market is one of the most competitive financial markets and traders need to compute the financial workloads with low latency and high throughput. In the past, people were using the traditional store and process method to calculate the heavy financial workloads efficiently. However to achieve low latency and high throughput, data-centers were forced to be physically located close to the data sources, instead of other more economically beneficial locations. This is the main reason, the data-streaming model was developed and it can process large amount of data more efficiently. It was shown in studies that using data streaming we can solve the options pricing and risk assessment problems using traditional methods, for example Japanese candlesticks, Monte-Carlo models, Binomial models, with low latency and high throughput. However instead of using those traditional methods, we approached the problems using machine learning techniques. We tried to revolutionize the way people address data processing problems in stock market by predicting the behaviour of the stocks. In fact, if we can predict how the stock will behave in the short-term future we can queue up our transactions earlier and be faster than everyone else. In theory, this allows us to maximize our profit without having the need to be physically located close to the data sources.

We examined two models.
**Model 1** - We used a complete random model using a random number genertor. If the generated number is greater than .5, we buy a stock and if it is less than .5 we sell. We close our position just before the exchange closes for the day.
**Model 2** - We use multiple machine learning algorithms to get the final decision. For example, we will use classification techniques to classify stocks into different buckets (For us, it is 3). Then we train 3 neural networks for these 3 buckets and take position in the market based on confidence levels of predicted values from these networks. Each of those models are applied on real stock market data and checked whether it could return profit.

### Problem Statement

For the concept of this thesis we tried to predict the price of the stock in the short term future and decide whether is better to buy, sell or hold our stocks. There is no strict definition of short term future. It can be any interval from nanoseconds until a few days. We decided that we will use 1-day interval as our prediction time. As the stock price depends on the time, time interval is a parameter that had to be decided. We think that 1-day can be a good representation of short term future. Also using a constant time interval simplifies the problem significantly. The main objective is to maximize the profit by trying to increase the capital. Through the years the economists investigated many different methods to try and find an optimal way to predict the movement of a stock. Some of them are Japanese candlesticks, Monte Carlo Models, Binomial Models, Black-Scholes formula and more. However instead of using those traditional methods we approached the problem of predicting stock prices using machine learning techniques. Then we tried to come up with an optimal trading strategy to maximize the potential profit. The main idea is to model a stock trading into 1-day intervals and using historical information of the stock. we tried to predict the stock price after 1 day. Also we tried to train and test our model on historical stock data collected during the period of November 2010 to June 2019.

### Datasets and Inputs

There are two types of data we have to download from external sources, the list of stocks we want to analyze and then prices for each stock in the given time period for analysis. [DataHub](https://datahub.io/core/nyse-other-listings) provides API to download list of tickers(instruments). Secondly, yahoo finance provides free stock price data. We are going to use pandas_datareader package to download Yahoo finance data. The dataset will provide us Open, High, Low, Close, Adj. Close and Volume data.

Once we have data downloaded from the external sources, we are going to calculate different technical indicator values using the library [talib](https://pypi.org/project/tablib/). We will then choose our window for price change calculation. If we consider N as our window, we will first calculate changes for past 1, 2, 3, ..., N-1 days changes. Following table elaborates it further.

*Downloaded stock price data*

| Date          | Open    | High    | Low     | Close   | Adj. Close | Volume     |
| --------------- | --------- | --------- | --------- | --------- | ------------ | ------------ |
| Dec 13, 2019	| 361.05	| 365.21	| 354.64	| 358.39	| 358.39	| 6,570,900 |
| Dec 16, 2019	| 362.55	| 383.61	| 362.5	| 381.5	| 381.5	| 18,174,200 |
| Dec 17, 2019	| 378.99	| 385.5	| 375.9	| 378.99	| 378.99	| 8,496,800 |
| Dec 18, 2019	| 380.63	| 395.22	| 380.58	| 393.15	| 393.15	| 14,121,000 |
| Dec 19, 2019	| 397.32	| 406.85	| 396.5	| 404.04	| 404.04	| 18,107,100 |


*With N = 2, we generate changes data*

| Date  | Open  | High  | Low  | Close  | Adj. Close | Volume | N - 1 | N - 2 |
|-------|-------|-------|------|--------|------------|--------|-------|-------|
| Dec 13, 2019	| 361.05	| 365.21	| 354.64	| 358.39	| 358.39	| 6,570,900	| 	|  | 
| Dec 16, 2019	| 362.55	| 383.61	| 362.5	| 381.5	| 381.5	| 18,174,200	| 23.11	|  | 
| Dec 17, 2019	| 378.99	| 385.5	  | 375.9	| 378.99	| 378.99	| 8,496,800	| -2.51	| 20.6 |
| Dec 18, 2019	| 380.63	| 395.22	| 380.58	| 393.15	| 393.15	| 14,121,000	| 14.16	| 11.65 |
| Dec 19, 2019	| 397.32	| 406.85	| 396.5	| 404.04	| 404.04	| 18,107,100	| 10.89	| 25.05|
| Dec 20, 2019	| 410.29	| 413	| 400.19	| 405.59	| 405.59	| 14,752,700	| 1.55	| 12.44 |


*With technical indicators calculated (for example Simple Moving Average, SMA)*

| Date  | Open  | High  | Low  | Close  | Adj. Close | Volume | N - 1 | N - 2 | SMA (9) |
|-------|-------|-------|------|--------|------------|--------|-------|-------|----------|
| Dec 13, 2019	| 361.05	| 365.21	| 354.64	| 358.39	| 358.39	| 6,570,900		| 	|  |  | 
| Dec 16, 2019	| 362.55	| 383.61	| 362.5	| 381.5	| 381.5	| 18,174,200	| 23.11		| | |
| Dec 17, 2019	| 378.99	| 385.5	| 375.9	| 378.99	| 378.99	| 8,496,800	| -2.51	| 20.6	| | 
| Dec 18, 2019	| 380.63	| 395.22	| 380.58	| 393.15	| 393.15	| 14,121,000	| 14.16	| 11.65	|  | 
| Dec 19, 2019	| 397.32	| 406.85	| 396.5	| 404.04	| 404.04	| 18,107,100	| 10.89	| 25.05	|  | 
| Dec 20, 2019	| 410.29	| 413	| 400.19	| 405.59	| 405.59	| 14,752,700	| 1.55	| 12.44	|  | 
| Dec 23, 2019	| 411.78	| 422.01	| 410	| 419.22	| 419.22	| 13,319,600	| 13.63	| 15.18	 |  | 
| Dec 24, 2019	| 418.36	| 425.47	| 412.69	| 425.25	| 425.25	| 8,054,700	| 6.03	| 19.66	|  | 
| Dec 26, 2019	| 427.91	| 433.48	| 426.35	| 430.94	| 430.94	| 10,633,900	| 5.69	| 11.72	| 399.6744444 |
| Dec 27, 2019	| 435	| 435.31	| 426.11	| 430.38	| 430.38	| 9,945,700	| -0.56	| 5.13	| 407.6733333 |
| Dec 30, 2019	| 428.79	| 429	| 409.26	| 414.7	| 414.7	| 12,586,400	| -15.68	| -16.24	| 411.3622222 |


*Our label is going to be the next day change*

| Date          | Open  | High  | Low  | Close  | Adj. Close | Volume | N - 1 | N - 2 | SMA (14) | Label |
|-------        |-------|-------|------|--------|------------|--------|-------|-------|----------|-------|
| Dec 13, 2019	| 361.05	| 365.21	| 354.64	| 358.39	| 358.39	| 6,570,900		| 	|  |  |  |
| Dec 16, 2019	| 362.55	| 383.61	| 362.5	| 381.5	| 381.5	| 18,174,200	| 23.11		| | | -2.51 |
| Dec 17, 2019	| 378.99	| 385.5	| 375.9	| 378.99	| 378.99	| 8,496,800	| -2.51	| 20.6	| |  14.16	|
| Dec 18, 2019	| 380.63	| 395.22	| 380.58	| 393.15	| 393.15	| 14,121,000	| 14.16	| 11.65	|  |  10.89	|
| Dec 19, 2019	| 397.32	| 406.85	| 396.5	| 404.04	| 404.04	| 18,107,100	| 10.89	| 25.05	|  |  1.55	|
| Dec 20, 2019	| 410.29	| 413	| 400.19	| 405.59	| 405.59	| 14,752,700	| 1.55	| 12.44	|  |  13.63	|
| Dec 23, 2019	| 411.78	| 422.01	| 410	| 419.22	| 419.22	| 13,319,600	| 13.63	| 15.18	 |  |  6.03	|
| Dec 24, 2019	| 418.36	| 425.47	| 412.69	| 425.25	| 425.25	| 8,054,700	| 6.03	| 19.66	|  |  5.69	|
| Dec 26, 2019	| 427.91	| 433.48	| 426.35	| 430.94	| 430.94	| 10,633,900	| 5.69	| 11.72	| 399.6744444 | -0.56	|
| Dec 27, 2019	| 435	| 435.31	| 426.11	| 430.38	| 430.38	| 9,945,700	| -0.56	| 5.13	| 407.6733333 | -16.24	|
| Dec 30, 2019	| 428.79	| 429	| 409.26	| 414.7	| 414.7	| 12,586,400	| -15.68	| -16.24	| 411.362222 |

We are going to normalize our data using sklearn's MinMaxScaler.

### Solution Statement

The number stocks are more than 3000. We are going to discard stocks whose volume is less than 1,000,0000.
We reach our end result in two steps.

**Step 1** - We will remove the label column and use all the normalized features and pass through different classification models. For example, we could use k-means algorithm to divide the data into 3 buckets. Later, we would analyze the clusters to understand how well the data have been categorized. We will have to do some parameter tuning to reach to our desired value.

**Step 2** - Once data has been categoried, we are going to run 3 recurrent neural network for the 3 clusters. We are going to use following features - 

    - open
    - high
    - low
    - close
    - adj. close
    - volumne
    - Triangular Moving Average
    - SAR                  
    - MACD                 
    - RSI                  
    - STOCH                
    - AD                   
    - ATR     
    - N  - x (changes each day compared one of the previous days going back upto N days)

During prediction, data will be passed to all the 3 networks. Decision will be made in following manner

- Cluster 1 RNN predicted value with highest confidence - Take Buy position
- Cluster 2 RNN predicted value with highest confidence - Take Sell position
- Cluster 3 RNN predicted value with highest confidence - Skip trading
- If there is a tie - Skip trading

### Benchmark Model

Trading is a probability game. Our first model is using a random number generator to generate a number between 0 and 1 and taking position based on the value. This is similar to flipping a coin. The goal is to prove that a systemic approach (in this case using machine learning models) yields higher returns as compared to taking position completely randomly.

If the value of random number generator is less .5, we will short the stock and close the position before market closes for the day. Similarly, if the generated number is greater than .5, we will buy the the stock. The success of this approach is going to be based on how many positive were closed with profit. This data is going to be our benchmark to compare with second modal.

### Evaluation Metrics

The evaluation metrics for both the models are simple. We calculate how many positions were closed profitably and divide the number with total positions taken.

    Accuracy = Profitable positions / Total number of positions

The highest accuracy model is the winning model.

We do not have to consider trading cost because as of Jan 2020, most of the large brokers have removed their round trip trading cost for stocks.

### Project Design
_(approx. 1 page)_



In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

### References

[1] Stock Market prediction using Artiﬁcial Neural Networks, RAFAEL KONSTANTINOU

[2] Predicting Stocks with Machine Learning, Magnus Olden

