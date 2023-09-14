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

The code Latent_Dirichlet_Allocation.ipynb performs various text preprocessing and topic modeling tasks on a dataset using Python libraries like Gensim, spaCy, scikit-learn, and pyLDAvis. Here's a summary of the entire code:

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

## LDA with pipeline

The file LDA_Pipeline.ipynb contains a comprehensive text analysis and classification pipeline that can be used for various natural language processing (NLP) tasks. The pipeline is built using Python and leverages popular libraries such as scikit-learn, Gensim, NLTK, and spaCy.

1. **Data Loading and Preprocessing**: 
 - The code includes data loading and preprocessing steps to prepare text data for analysis. It demonstrates techniques for handling datasets and cleaning text data.

2. **Topic Modeling with Latent Dirichlet Allocation (LDA)**: 
 - The repository provides a custom transformer class for fitting an LDA topic model to text data. It extracts meaningful topics from text documents.

3. **Machine Learning Pipeline**: 
 - The code constructs a scikit-learn pipeline that automates the entire workflow, from data preprocessing to modeling. It includes custom transformers for text processing and classification models.

4. **Train-Test Split**: 
 - The repository demonstrates how to split the dataset into training and testing subsets while preserving class distribution. This is crucial for evaluating model performance.

5. **Model Training and Evaluation**: 
 - The pipeline trains a machine learning model (e.g., Support Vector Classifier, SVC) on the preprocessed text data and evaluates its performance using accuracy metrics.


