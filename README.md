# Recommender Systems
Note: Without the dumped npy files the code will take a lot of time to process the data and build the relevant matrices.
Implementation and comparision of various techniques of building recommender systems, such as:
* Collaborative Filtering
* SVD (Using Dense Matrices)
* CUR (Using Sparse Matrices)


## Dataset
We use the [Movielens 1M](https://grouplens.org/datasets/movielens/1m/) movie ratings dataset to train and test the various models. The datasets contain around 1 million anonymous ratings of approximately 3,900 movies made by 6,000 MovieLens users who joined MovieLens in 2000.

## How to run
1. Clone this repo / click "Download as Zip" and extract the files.
2. Download and extract the dataset (see [Dataset](#dataset) section) into a new folder called `dataset`.
3. To test train the data, Run train_test_split.
4. Then Run the "preprocess.py" file .
5. To run the recommender for the specific technique, run its respective file. 

## Contributors
* [Amogh Saxena](2017B4A71731H)
* [Nikhil Mohan Krishna]( 2018A7PS0496H)
* [Simran Sahni](2017B5A70856H)
