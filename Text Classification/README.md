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

## All the code files were executed on Google Colab.
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

1. **Data Loading and Preprocessing**: The code includes data loading and preprocessing steps to prepare text data for analysis. It demonstrates techniques for handling datasets and cleaning text data.

2. **Topic Modeling with Latent Dirichlet Allocation (LDA)**: The code provides a custom transformer class for fitting an LDA topic model to text data. It extracts meaningful topics from text documents.

3. **Machine Learning Pipeline**: The code constructs a scikit-learn pipeline that automates the entire workflow, from data preprocessing to modeling. It includes custom transformers for text processing and classification models.

4. **Train-Test Split**: The code demonstrates how to split the dataset into training and testing subsets while preserving class distribution. This is crucial for evaluating model performance.

5. **Model Training and Evaluation**: The pipeline trains a machine learning model (e.g., Support Vector Classifier, SVC) on the preprocessed text data and evaluates its performance using accuracy metrics.

## LDA with Semi Supervised Learning
The code LDANAiveBayesEM.ipynb cover various aspects of text data processing, classification using Naive Bayes, and topic modeling using LDA. They also include cross-validation to assess model performance and evaluate the impact of the number of labeled documents on classification performance.

1. **Data Preprocessing and Cleaning**: The code begins by preparing the text data, including removing punctuation, sentence tokenization, and lemmatization to obtain cleaner and more structured text data.

2. **Text Vectorization**: The cleaned text data is converted into numerical features using the TF-IDF vectorization method, resulting in TF-IDF representations for both the training and test datasets.

3. **Data Splitting**: The training data is divided into two sets: a labeled dataset (`X_l_combined`, `y_l`) and an unlabeled dataset (`X_u_combined`, `y_u`). This split is crucial for subsequent semi-supervised learning.

4. **Cross-Validation for Naive Bayes Classifier**: Cross-validation is performed using labeled data with a traditional Multinomial Naive Bayes classifier (`MultinomialNB`). Various metrics such as accuracy, precision, recall, and F1-score are recorded for different quantities of labeled documents.

5. **Cross-Validation for Semi-Supervised EM Naive Bayes Classifier**: Similar to the previous step, cross-validation is conducted, but this time employing a semi-supervised EM-based Naive Bayes classifier (`Semi_EM_MultinomialNB`). This classifier utilizes both labeled and unlabeled data during training.

6. **Test Data Evaluation**: The performance of both the traditional Naive Bayes classifier and the semi-supervised EM Naive Bayes classifier on the test data is assessed. Metrics like accuracy, precision, recall, and F1-score are calculated for different numbers of labeled documents.

7. **Topic Modeling with LDA**: Latent Dirichlet Allocation (LDA) topic modeling is applied to the text data, both in the training and test datasets. This produces topic weights as features.

8. **Conversion of LDA Topic Weights to DataFrames**: The LDA topic weights assigned to documents are transformed into DataFrames (`train_features_tfidf` and `test_features_tfidf`) to represent the topic features of the text data.


Certainly, here's the summary of the code snippets along with the associated graphs:

Data Preprocessing and Cleaning: Text data is preprocessed by removing punctuation, sentence tokenization, and lemmatization to obtain cleaner and more structured text data.

Text Vectorization: The cleaned text data is converted into numerical features using the TF-IDF vectorization method, resulting in TF-IDF representations for both the training and test datasets.

Data Splitting: The training data is divided into two sets: a labeled dataset (X_l_combined, y_l) and an unlabeled dataset (X_u_combined, y_u). This split is crucial for subsequent semi-supervised learning.

Cross-Validation for Naive Bayes Classifier: Cross-validation is performed using labeled data with a traditional Multinomial Naive Bayes classifier (MultinomialNB). Various metrics such as accuracy, precision, recall, and F1-score are recorded for different quantities of labeled documents. A line plot and error bars are used to visualize the results.

Cross-Validation for Semi-Supervised EM Naive Bayes Classifier: Similar to the previous step, cross-validation is conducted, but this time employing a semi-supervised EM-based Naive Bayes classifier (Semi_EM_MultinomialNB). This classifier utilizes both labeled and unlabeled data during training. Results are visualized with a line plot and error bars.

Test Data Evaluation: The performance of both the traditional Naive Bayes classifier and the semi-supervised EM Naive Bayes classifier on the test data is assessed. Metrics like accuracy, precision, recall, and F1-score are calculated for different numbers of labeled documents. Line plots are used to visualize the changes in performance.

Topic Modeling with LDA: Latent Dirichlet Allocation (LDA) topic modeling is applied to the text data, both in the training and test datasets. This produces topic weights as features.

Conversion of LDA Topic Weights to DataFrames: The LDA topic weights assigned to documents are transformed into DataFrames (train_features_tfidf and test_features_tfidf) to represent the topic features of the text data.

Additionally, there are various graphs for visualizing the results, including line plots showing changes in accuracy, precision, recall, F1-score, and more with varying quantities of labeled documents. Heatmaps are used to visualize confusion matrices, and there are graphs for topic modeling results as well.

<object data="Poster.pdf" width="1000" height="1000" type='application/pdf'/>


