## Topic Modelling
LDA, which stands for Latent Dirichlet Allocation, is not typically used for text classification directly but is a probabilistic model often employed for topic modeling in natural language processing (NLP). However, it can indirectly assist in text classification tasks as part of a larger pipeline.

Here's an overview of LDA and how it can relate to text classification:

1. **Latent Dirichlet Allocation (LDA):** LDA is a generative statistical model used for topic modeling. It assumes that documents are mixtures of topics and that words in a document are attributable to the document's topics. LDA tries to uncover these topics and their distribution in a corpus of text documents. It's unsupervised, meaning you don't need labeled data to apply LDA.

2. **Text Classification:** Text classification, on the other hand, is the task of assigning predefined categories or labels to text documents. This can be binary (e.g., spam or not spam) or multi-class (e.g., classifying news articles into different topics like sports, politics, or entertainment).

How LDA can relate to text classification:

- **Feature Generation:** LDA can be used to generate topic-based features for your text documents. Each document can be represented as a distribution over topics, and these distributions can serve as features for a subsequent text classification task. For instance, you can calculate the topic distribution for each document using LDA and then use this distribution as input features for a classification algorithm like logistic regression or a neural network.

- **Dimensionality Reduction:** LDA can help in reducing the dimensionality of the text data. By representing documents as topic distributions, you can reduce the number of features, making it computationally more efficient and potentially improving classification performance.

- **Understanding Text Content:** LDA can provide insights into the underlying themes or topics within your text corpus. This understanding can be valuable for feature engineering or for making informed decisions about how to structure your text classification problem.

In practice, a common approach is to use LDA to generate topic-based features and then combine these features with other traditional text features (e.g., TF-IDF, word embeddings) for text classification tasks. The combination of these different features can often lead to improved classification performance.
## LDA without pipeline

The code performs various text preprocessing and topic modeling tasks on a dataset using Python libraries like Gensim, spaCy, scikit-learn, and pyLDAvis. Here's a summary of the entire code:

1. **Importing Libraries:** The code begins by importing necessary libraries, including Gensim, spaCy, scikit-learn, and pyLDAvis, for text processing, topic modeling, and visualization.

2. **Loading the Dataset:**
   - It loads the 20 Newsgroups dataset, which is a collection of newsgroup documents organized into categories.

3. **Text Preprocessing:**
   - The code preprocesses the text data in the following steps:
     - Removing email addresses from the text.
     - Converting text to lowercase.
     - Removing stopwords (common words like "the," "and," etc.).
     - Removing punctuation.
     - Tokenizing the text into words.
     - Lemmatizing the words, keeping only certain parts of speech (NOUN, ADJ, VERB, ADV).
     - Forming bigrams and trigrams (two- and three-word phrases) to capture meaningful phrases.

4. **Topic Modeling:**
   - The code then proceeds to perform topic modeling using Latent Dirichlet Allocation (LDA) on the preprocessed text data.
   - It creates a dictionary and a corpus of text data.
   - The LDA model is trained with a specified number of topics, and topic distributions are computed for each document.

5. **Visualizing Topics:**
   - It uses pyLDAvis to create an interactive visualization of the topics extracted by the LDA model.
   - The visualization helps in understanding the topics and their relationships.

6. **Evaluating the Number of Topics:**
   - The code includes a function to compute coherence scores for LDA models with different numbers of topics.
   - It then plots the coherence scores to help determine the optimal number of topics.

7. **Training a Classifier:**
   - After determining the number of topics, the code uses scikit-learn to split the data into training and testing sets.
   - It trains a Logistic Regression classifier on the training data.

8. **Model Evaluation:**
   - The code predicts labels for the testing data and calculates the accuracy score of the classifier.

9. **Adding Topic Features:**
   - It combines the topic distribution features with the original text and class labels in a Pandas DataFrame.
