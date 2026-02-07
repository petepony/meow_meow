{
 "cells": [
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "d133d459-e417-4ea9-9626-f396d2ab5c34",
   "metadata": {},
   "source": [
    "# Optiver Realized Volatility Prediction\n",
    "\n",
    "Volatility is a measurement of how varied the returns of a given security or market index are over time. It is measured from the standard deviation of the securities returns over set time.\n",
    "\n",
    "The aim of the project was to predict, to the best of our ability the volatility of given stocks over the span of `10` minutes, from the market snapshots over previous `10`. The stocks are masked by their `stock_id`.\n",
    "\n",
    "This project is based on the following kaggle competition with $100k total prize pool:\\\n",
    "https://www.kaggle.com/competitions/optiver-realized-volatility-prediction"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "68bdcf14-83e7-45d4-87c9-55a2051a2fe5",
   "metadata": {},
   "source": [
    "# The Goal\n",
    "\n",
    "As the competition had ended four years ago we could not test our results through the website.\\\n",
    "We had decided to introduce a `85 : 15 | train : test` split\n",
    "\n",
    "The best scores in the competition were around `0.195`.\\\n",
    "\n",
    "There, unfortunately, was a way to reorder `time_id`'s chronologically, which gave a significant advantage by allowing inference from the future.\n",
    "We decided to not apply such methods as they conflicted with the whole purpose of this project. \n",
    "\n",
    "On a side note, the task of inferring the actual order of deliberately scrambled `time_id`'s is interesting in its own right. It seems reasonable to assume that the feature vectors lie on a 1-dimensional manifold in some very high-dimensional space. Applying a more sophisticated non-linear dimensionality reduction, like `UMAP`, clearly showed this. Then, ordering the data by the x-axis in the $2d$ embedding yielded the correct timeline.)\n",
    "\n",
    "The best scores without such methods hover around `0.21`.\\\n",
    "Many employ advanced machine learning techniques with models trained over days on powerful GPU's.\n",
    "\n",
    "At the start of the project we decided we'd conisder it a great success if we ended up in the middle of the leaderboard.\n",
    "\n",
    "- `Q1` score: `0.22322`\n",
    "- `Q2` score: `0.22755`\n",
    "- `Q3` score: `0.24188`"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a4651259-4828-40af-b0ab-213241f684a1",
   "metadata": {},
   "source": [
    "# Dataset Files Overview\n",
    "\n",
    "#### 1. `book_train.parquet`\n",
    "A parquet file partitioned by `stock_id`. It contains order book data showing the most competitive buy and sell orders entered into the market. The top two levels of the order book are included:\n",
    "- **First level**: More competitive in price, receiving execution priority.\n",
    "- **Second level**: Less competitive in price, with lower execution priority.\n",
    "\n",
    "##### Key Columns:\n",
    "- `stock_id`: ID code for the stock. Note that not all stock IDs appear in every time bucket.\n",
    "- `time_id`: ID code for the time bucket. Time IDs are consistent across all stocks, though not necessarily sequential.\n",
    "- `seconds_in_bucket`: Number of seconds from the start of the bucket, always starting at 0.\n",
    "- `bid_price[1/2]`: Normalized prices for the most (1) and second most (2) competitive buy levels.\n",
    "- `ask_price[1/2]`: Normalized prices for the most (1) and second most (2) competitive sell levels.\n",
    "- `bid_size[1/2]`: Number of shares on the most (1) and second most (2) competitive buy levels.\n",
    "- `ask_size[1/2]`: Number of shares on the most (1) and second most (2) competitive sell levels.\n",
    "\n",
    "#### 2. `trade_train.parquet`\n",
    "A parquet file partitioned by `stock_id`. It contains data on executed trades. Since trade data is more sparse than order book updates, there are usually fewer rows.\n",
    "\n",
    "##### Key Columns:\n",
    "- `stock_id`: Same as in `book_train.parquet`.\n",
    "- `time_id`: Same as in `book_train.parquet`.\n",
    "- `seconds_in_bucket`: Same as in `book_train.parquet`, though this field may not always start from 0 due to trade sparsity.\n",
    "- `price`: Average price of executed trades during one second, normalized and weighted by the number of shares traded in each transaction.\n",
    "- `size`: Total number of shares traded in that second.\n",
    "- `order_count`: Number of unique trade orders executed in that second.\n",
    "\n",
    "#### 3. `train.csv`\n",
    "This file contains the ground truth values for training.\n",
    "\n",
    "##### Key Columns:\n",
    "- `stock_id`: Same as above, but loaded as an integer column in this CSV format.\n",
    "- `time_id`: Same as above.\n",
    "- `target`: The realized volatility over the 10-minute window following the feature data for the same `stock_id` and `time_id`. There is no overlap between feature data and target data.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "673ce761-ac82-459b-9db7-075af073d2d9",
   "metadata": {},
   "source": [
    "# Processing the Data\n",
    "\n",
    "- The `book_train` dataset was ridiculously length `~167M` rows.\n",
    "- The `trade_train` dataset was still heavy `~38M` rows\n",
    "- There were `~429K` pairs of`stock_id`, `time_id` in the `train` dataset.\n",
    "\n",
    "First thing we did was we merged the trades df by adding the book snapshot immediately before and after this trade happened.\\\n",
    "This allowed us to work on the smaller dataframe while still using crucial information from the book.\n",
    "\n",
    "The notebook used was `data_process.ipynb`. The kernel crashed multiple times during this lengthy operation,\n",
    "my laptop's `32GB` of ram were not enough. We managed to fix the issue by increasing my `swap file` memory to `50GB`."
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "4b8e5e2a-83d7-40b7-a649-22c0fac033bd",
   "metadata": {},
   "source": [
    "# Data Cleanup\n",
    "\n",
    "Having this dataset in `data_cleanup.ipynb` we removed rows with missing values in `trades` and the ones for which the merge couldn't be completed. The ones for which there was no book record before the trade happened.\n",
    "\n",
    "We had computed the volatility in the previous 10 minutes from this data as we believed it will be the single most predictive feature aggregating it into the `train` dataset for which we will later develop models.\n",
    "\n",
    "We also aggregated into it following features: \n",
    "- Current spread\n",
    "- Current mid\n",
    "- The direction of the trade\n",
    "- Maximal time difference between book updates\n",
    "- Sum of all order sizes in the last 10 minutes"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7fdc096f-f3e9-448b-bd21-95b204d709d4",
   "metadata": {},
   "source": [
    "# Baseline Models\n",
    "\n",
    "We decided to code following models:\n",
    "\n",
    "- Linear Regression `...` | Loss: `0.36642` | Issues: It's a linear regression, no embedding,\n",
    "- XGBRegressor `3000,3`| Loss: `0.29148` | Issues: None! I love XGB, easy embeddings and simple to use\n",
    "- Simple NN `64,3`| Loss: `0.30379` | Issues: No embedding yet, but high hope with more features\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ff659dfa-ea9d-4d15-9013-8ec5962d6661",
   "metadata": {},
   "source": [
    "# Feature Engineering\n",
    "\n",
    "Using educated trial and error methods we had ended up with the following 19 predictive features:\n",
    "- `stock_id`, `time_id` as indices\n",
    "\n",
    "For a particular pair stock_id, time_id's we aggregated the `trades` dataframe to include:\n",
    "- Mean, std, skew and kurt of spread of particular stock. Intuitively with high spread the market is uncertain, hence more volatile.\n",
    "- average number of trades when at least one happened in the given second, the standard deviation of previous, indicating how fast market reacts\n",
    "- mean of logarithm of time between trades - how often the stock trades,\n",
    "- log of the traded total volume (we had to recover original prices to do that)\n",
    "- skew and kurtosis of the order book\n",
    "- When trades had only one direction we had defined a product shift as the volume of these trades. We looked at mean of it and std\n",
    "- Defined shift ratio as the ratio of product shift to total trade volume, to see what percent of volume the market is certain of future price\n",
    "- Defined trade impact as the mean of how much a book sizes changed when a trade happened\n",
    "\n",
    "Additionally to implement broader market movements we added:\n",
    "- mean sigma of all other stocks, weighted by log volume\n",
    "- The trend in the market volume by time, had to fit linear regression over all trades, without intercept second with\n",
    "- The latter model's `R^2` score, we found that the easier the market is to predict the higher the volatility will be"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8d1adf9a-2579-4b59-8a47-01bd03402d84",
   "metadata": {},
   "source": [
    "# Interesting Plots\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "33c4b5dc-b9f9-4612-a3f3-afe69da25239",
   "metadata": {},
   "source": [
    "# Final Model\n",
    "\n",
    "Considering the finickiness of financial data, characterised by a high noise-to-signal ratio and the proclivity to overfitting, we had a few options for our model architecture. Simple neural networks, random forests and boost-based models failed to significantly improve our predictions upon the baseline yielded by the realized volatility in the previous timeframe. Isolating the highly predictive `stock_id` feature was also problematic as it significantly reduced the training set. The models also proved prone to overfitting, and not expressive enough to capture the convoluted relationships in the data. To address these issues, we decided on a neural network with residual blocks to help the model separate the signal from noise. We also applied __layer norm__ to stabilise the training. This model showed a marked improvement, as adding $6$ basic features reduced the RMSPE from $0.36$ to $0.25$. Adding a simple embedding of the categorical `stock_id` feature further reduced it to $0.25$. Scaling the model slightly allowed the relatively few features found through careful engineering to achieve RMSPE at $0.23$."
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "36cb5f0d-9b55-4b91-bf2e-78dfce06ab8c",
   "metadata": {},
   "source": [
    "## Alternative approaches\n",
    "\n",
    "Training a recurrent neural network or a transformer yields rather poor performance. This is probably because such models are very data-hungry and, in the case of financial data, require careful hyperparameter-tuning. However, building an ensemble model encompassing a GRU model trained on raw tick data, a convolutional neural network and a residual neural network might further improve our baseline model. "
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "id": "436be967-c8c4-4d8a-a4cc-b7526e79e2af",
   "metadata": {},
   "source": [
    " # The Result\n",
    "\n",
    "We ended up in!:\n",
    "- top `46.8503937%` of all competitors\n",
    "- while using merely `70%` of available data\n",
    "- and only `19` model features\n",
    "- in a few days without touching the university servers\n",
    "\n",
    "### This we consider a great success!!"
   ]
  },
  {
   "cell_type": "code",
   "outputs": [],
   "source": [],
   "metadata": {
    "collapsed": false
   },
   "id": "b72e9acc575b5fd1"
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
