# Sentiment Analysis

Sentiment analysis, also known as opinion mining, is a natural language processing (NLP) technique that determines the emotional tone expressed in text data. Its main goal is to classify text as positive, negative, or neutral based on the sentiment it conveys.

In sentiment analysis:

- **Positive sentiment** indicates positive emotions like happiness or approval.
- **Negative sentiment** indicates negative emotions like anger or disappointment.
- **Neutral sentiment** suggests a lack of strong sentiment.

Sentiment analysis finds applications in:

- Analyzing product reviews to gauge customer satisfaction.
- Monitoring social media for public sentiment.
- Assessing customer feedback from surveys or support interactions.
- Managing brand reputation online.
- Informing financial market analysis.

Challenges include language nuances, context, and sarcasm. Various tools and libraries, like NLTK, TextBlob, and machine learning frameworks, aid in sentiment analysis by identifying sentiment in text data. It helps organizations understand public opinion and customer sentiment.

## EDA.ipynb

This file contains a Python script for analyzing and visualizing social media data. The code is designed to work with a dataset, containing social media-related information, which is loaded from a CSV file named 'TheSocialDilemma.csv'.

The code is organized into several parts, each serving a specific purpose:

1. **Importing Libraries**: The script begins by importing various Python libraries. These libraries include NumPy, Pandas, Matplotlib, Plotly, NLTK, Seaborn, and more. These libraries are essential for data manipulation, analysis, and visualization.

2. **Loading Data**: The dataset is read into a Pandas DataFrame called 'social' using the 'pd.read_csv()' function. This DataFrame is the foundation for subsequent data analysis.

3. **Preprocessing and Visualization**:
   
   a. **Date and Time Processing**:
      - The 'user_created' and 'date' columns are processed to ensure they are in datetime format. This conversion is essential for date and time-based analysis.
      - Date components, specifically 'day' and 'time,' are extracted from the 'date' column. These components will be useful for further analysis.

   b. **Visualizing Unique Users per Day**:
      - Data is grouped to count the number of unique users per day.
      - A bar plot is created using Seaborn to visualize the distribution of unique users over different days. This helps identify trends in user activity.

   c. **Visualizing Tweet Distribution over Hours**:
      - Data is grouped to count the number of tweets per hour.
      - A scatter plot is created using Seaborn to visualize the distribution of tweets over different hours of the day. This allows users to understand when tweets are most active.

4. **Date Filtering**: A specific date range is applied to filter the data. This date filtering is performed on the 'user_created' column, ensuring that only data within the specified date range is considered for analysis.

5. **Further Data Grouping**: The filtered data is further grouped based on 'user_name,' 'Sentiment,' and 'user_created.' The code calculates the count of occurrences within each group. This grouping provides insights into user sentiment over time.

6. **Final Output**: The resulting grouped data is printed to the console to provide users with a summary of the analysis. Additionally, the code sets a figure size, although it does not generate any additional plots in the provided code snippets.

The code is designed to be informative and suitable for analyzing social media data. Users can adapt and extend it for their specific data analysis and visualization needs.

## SentimentAnalysis.ipynb

1. **Code for Loading Libraries and Data**:
   - The code initializes various Python libraries for data manipulation, natural language processing, and machine learning.
   - It loads a CSV file named 'TheSocialDilemma.csv' into a Pandas DataFrame called 'data' for further analysis.

2. **Data Preprocessing**:
   - Text data in the 'text' column of the 'data' DataFrame undergoes preprocessing steps, including lowercase conversion, URL removal, elimination of Twitter handles, punctuation removal, and cleaning of emails and extra whitespace.
   - The cleaned text is stored back in the 'data' DataFrame for subsequent tasks.

3. **Tokenization and Lemmatization**:
   - Tokenization and lemmatization of the preprocessed text data are performed using the spaCy library.
   - Stop words are eliminated, and lemmatization occurs for nouns, adjectives, verbs, and adverbs.
   - The resulting tokenized and lemmatized text is saved in the 'data' DataFrame.

4. **Creating a New DataFrame**:
   - A new DataFrame ('df') is created, which includes the detokenized and lemmatized text ('tweet') and retains the 'Sentiment' column from the original data.

5. **Data Splitting**:
   - The 'df' DataFrame is split into training and testing sets by employing the `train_test_split` function.

6. **Multinomial Naive Bayes Classifier Training**:
   - Training of a Multinomial Naive Bayes classifier is executed on the TF-IDF transformed training data using the scikit-learn library.

7. **Hyperparameter Tuning and Evaluation**:
   - Hyperparameter tuning is conducted via Grid Search while evaluating the classifier's performance using metrics such as accuracy, precision, recall, F1-score, and the confusion matrix.

8. **Feature Selection and Evaluation**:
   - Feature selection is accomplished through the chi-squared (chi2) test to select the most informative features.
   - Subsequently, a Multinomial Naive Bayes classifier is trained on the chosen features, and its performance is assessed using accuracy.

9. **TF-IDF Vectorization and Dimension Reduction**:
   - TF-IDF features are generated from the text data with the help of scikit-learn's `TfidfVectorizer`.
   - Feature count is restricted to a maximum of 100,000, and the n-gram range is set to (1, 3).

10. **Confusion Matrix Visualization**:
    - Visualization of the Multinomial Naive Bayes classifier's confusion matrix is carried out to assess its performance on the test set. Two versions, normalized and unnormalized, are displayed.

11. **Summary Plot**:
    - A plot is created to compare the validation set accuracy of two distinct feature extraction methods: trigram TF-IDF vectorization and TF-IDF vectorization with dimensions reduced using chi-squared feature selection.
    - The plot offers insights into the influence of feature count and feature selection on model performance.
