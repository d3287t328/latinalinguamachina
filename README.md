 Hi this is my starting out point for getting started with building generative AI resources natively in classical Latin. The website at https://www.latinalinguamachina.com/ I have a newsletter in Latin and we are building a chatgot clone for Latin learners and a social network.`

For now this repository is going to be a grab bag of tools for Large Language Models in classical Latin I will be building. 

I hope to build or contribute towards a Large Language Model specifically made for Classical Latin. I expect to insert all of the available classical literature and vocabulary from open source sources. 

= Project Goals =

1. PDF chat bot service based on https://github.com/mayooear/gpt4-pdf-chatbot-langchain
2. Latin wikipedia chatbot based on https://twitter.com/StephanSturges/status/1651567091051372549
3. Dedicated web client
4. Under consideration: Mightynetworks chatbot.

All software in this repository is released by Latina Lingua Machina LLC under the GPL 3.0 unless otherwise stated.

= Scripts Explaners =

data_preprocessing.py: A script to clean, tokenize, and preprocess your text data for optimal input into your language model.

dataset_splitter.py: A script to split your preprocessed data into training, validation, and testing sets.

vocabulary_builder.py: A script to create and save the vocabulary of unique tokens (words, subwords, or characters) from your dataset.

token_embedding.py: A script to create and manage token embeddings (e.g., using word2vec, GloVe, or FastText) for input into your language model.

model_architecture.py: A script to define the architecture of your large language model, such as the number of layers, hidden units, and attention mechanisms.

training_script.py: A script to train your large language model using your preprocessed dataset, monitoring validation loss and applying techniques such as learning rate scheduling, gradient clipping, and early stopping.

evaluation_metrics.py: A script to compute evaluation metrics, such as perplexity and BLEU score, on your test dataset to assess the performance of your large language model.

model_checkpointing.py: A script to save and load model checkpoints during training to prevent data loss in case of crashes or to resume training at a later point.

text_generation.py: A script to generate text using your trained large language model, allowing you to experiment with different decoding strategies like greedy search, beam search, or top-k sampling.

fine_tuning.py: A script to fine-tune your large language model on domain-specific or task-specific datasets to improve its performance on specialized tasks or adapt it to new data distributions.

These scripts will help you to build and train Large Language Models.
