# Machine-Learning
Project: Stock Price prediction

Install

This project requires Python and the following Python libraries installed:

NumPy
Pandas
matplotlib
scikit-learn
You will also need to have software installed to run and execute a Jupyter Notebook.

If you do not have Python installed yet, it is highly recommended that you install the Anaconda distribution of Python, which already has the above packages and more included.
What is LSTM?

It is a special type of RNN, capable of learning long-term dependencies.

"Long short-term memory (LSTM) units are units of a recurrent neural network (RNN). An RNN composed of LSTM units is often called an LSTM network. A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell"

Long Short Term Memory (LSTM) is a type of deep learning model that is mostly used for analysis of sequential data (time series data prediction).

->>
Stock prices are downloaded from finance.yahoo.com. Disneyland (DIS) Stock Price CSV file.
	-Closed value (column[5]) is used in the network, LSTM Code
-Values are normalized in range (0,1).
-Datasets are splitted into train and test sets, 50% test data, 50% training data.
-Keras-Tensorflow is used for implementation.
-LSTM network consists of 25 hidden neurons, and 1 output layer (1 dense layer).
-LSTM network features input: 1 layer, output: 1 layer , hidden: 25 neurons, optimizer:adam, dropout:0.1, timestep:300, batchsize:300, epochs:10 (features -can be further optimized).
-Root mean squared errors are calculated.
-Output files: lstm_results (consists of prediction and actual values), plot file (actual and prediction values).
