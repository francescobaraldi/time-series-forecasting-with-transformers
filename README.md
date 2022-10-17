# Financial time series forecasting with transformers

## Introduction

The aim of the project is to investigate the use of Transformer model in financial time series forecasting and evaluate its performance against a recurrent neural network, in particulat the LSTM model. Two version of the Transformer has been used with two different approaches for the forecasting:
- **Transformer Decoder**: based on the GPT-3 structure, only the decoder part of the original architecture is used, working with a self-supervised approach, the model is fed with an input time series of length L and it has to forecast the next value, giving in output a time series of length L, which has to be the input shifted to right by 1;
- **Transformer**: in this case the whole original architecture is used with the classical approach of the transformer, the encoder is fed with a time series of length L, while the decoder is fed with a time series of lenght T which is the target shifted to left by 1. The model gives in output a time series of length T.

The project has been developed as part of the course "School in AI: Deep Learning, Vision and Language for Industry".

## Dataset

The dataset used is taken by Yahoo Finance and it contains the *Open*, *High*, *Low*, *Close*, *Adj Close* and *Volume* values of S&P 500 index for each days from 29-01-1993 to 06-10-2022. For the goal of this project only the *Close* feature is used in order to obtain **univariate time series**.

## Training
In the ```experiment.py``` file the training has been performed, for each model a set of values for each hyper parameter has been tested together and the results of the training is saved in the specific directories.
For each model:
- batch size = 64;
- input time series length = 90;
- trainset split = 70 %
- testset split = 30 %
- loss function = mean squared error
- optimizer = Adam

### Transformer decoder
- learning rate = 1e-05
- number of decoders = 1
- dimension of the model (feature size) = 128
- number of heads = 8
- dropout = 0
- dimension of the fully connected inside the decoder = 256
- positional encoding = classical sinusoidal encoding
### Transformer
- output time series length = 30
- learning rate = 1e-05
- number of encoders = 1
- number of decoders = 1
- dimension of the model (feature size) = 128
- number of heads = 8
- dropout = 0
- dimension of the fully connected inside the decoder = 256
- positional encoding = classical sinusoidal encoding

### LSTM
- learning rate = 1e-05
- number of layers = 2
- dimension of the hidden states = 64
- dropout = 0

## Results
TODO...

## Usage

Clone the repository:
```git clone https://github.com/francescobaraldi/time-series-forecasting-with-transformers.git```

Install the dependencies:
```pip install -r requirements.txt```

Then it is possible to execute the main file which runs the three models tested in order to forecast the next 30 days closing price of the S&P 500 index:
```python main.py```
The results will be plotted and saved in the directory ```inference_results/``` in png format, one image for each model tested.

It is also possible to experiment with the models, in the ```experiment.py``` file there is the code to train the models and it's possible to try with different hyper parameters, different lengths of the input time series and output time series. The notebook version of the file allows to run the code in [Google Colab](https://colab.research.google.com/).

## References

- [Attention Is All You Need - Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin](https://arxiv.org/abs/1706.03762)
