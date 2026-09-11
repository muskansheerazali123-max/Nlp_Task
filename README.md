**Student name:Muskan Sheeraz Ali
Roll num:BSAI/2k24/67
Assignment of NLP
Accademic year 2026
Submitted to MR RAJESH KUMAR**
                                **Natural Language Processing Lab 04**
                        **Bag of Words Matrix Construction & Document Search Engine**

This project implements two fundamental Natural Language Processing (NLP) tasks using Python and popular machine learning libraries.

Tasks Included
Task 1: Bag of Words (BoW) Matrix Construction using CountVectorizer
Task 2: Document Search Engine and Relevance Ranking using Cosine Similarity
Project Overview

The purpose of this lab is to demonstrate how textual data can be converted into numerical representations and how those representations can be used to compare documents with a search query.

The project uses:

Pandas for structured data presentation
NumPy for numerical operations
Scikit-learn for text vectorization and similarity calculation
Technologies Used
Technology	Purpose
Python 3	Programming language
Pandas	Data manipulation and DataFrame creation
NumPy	Numerical operations and ranking
Scikit-learn	NLP vectorization and cosine similarity
Python Libraries
pandas
numpy
scikit-learn

Installation
1. Clone the Repository
git clone https://github.com/muskansheerazali123-max/Nlp_Task.git


Navigate to the project directory:

cd Nlp_Task

2. Install Required Libraries

Run:

python -m pip install pandas numpy scikit-learn


You can verify the installations with:

python -c "import pandas, numpy, sklearn; print('All libraries installed successfully')"

Project Structure
Nlp_Task/
│
├── nlp_task.py
├── README.md
└── .gitignore

Task 1: Bag of Words Matrix Construction
Objective

The first task converts a collection of customer reviews into a numerical Bag of Words (BoW) matrix.

The following sample reviews are used:

The product performance is amazing and fast
The service was fast and performance was great
Terrible customer service and bad performance

Processing Steps
Create the input corpus.
Initialize CountVectorizer with English stop-word removal.
Fit the vectorizer to the corpus.
Transform the text into a numerical matrix.
Extract the vocabulary using get_feature_names_out().
Convert the sparse matrix into a Pandas DataFrame.
Display the resulting term-frequency matrix.
Implementation
vectorizer = CountVectorizer(stop_words='english')
bow_matrix = vectorizer.fit_transform(corpus)

feature_names = vectorizer.get_feature_names_out()

df_bow = pd.DataFrame(
    bow_matrix.toarray(),
    columns=feature_names,
    index=[f"Review {i+1}" for i in range(len(corpus))]
)

What the BoW Matrix Represents

Each row represents a document/review, while each column represents a unique word.

The values indicate how many times each word occurs in the corresponding document.

For example:

             amazing  bad  fast  performance  service
Review 1         1     0     1       1           0
Review 2         0     0     1       1           1
Review 3         0     1     0       1           1


English stop words such as the, is, and, and was are removed by CountVectorizer(stop_words='english').

Task 2: Document Search Engine & Relevance Ranking
Objective

The second task implements a simple document search engine.

The system:

Stores a collection of documents.
Converts documents into numerical vectors.
Converts a user query into a vector.
Calculates cosine similarity between the query and every document.
Ranks documents according to their relevance.
Sample Documents
Doc 1: Machine learning algorithms analyze structured data effectively

Doc 2: Deep learning and neural networks excel at processing unstructured data

Doc 3: Natural language processing helps computers understand human language

Doc 4: Python is widely used for machine learning and data science

Search Query
machine learning algorithms for data

Vectorization

The project uses CountVectorizer:

vectorizer = CountVectorizer()
vectorizer.fit(documents)

doc_vectors = vectorizer.transform(documents).toarray()
query_vector = vectorizer.transform(query).toarray()


The documents and search query are therefore represented as numerical vectors.

Cosine Similarity

Cosine similarity measures how similar two vectors are.

The calculation is performed using:

similarity_scores = cosine_similarity(
    query_vector,
    doc_vectors
)[0]


The resulting scores are used to determine document relevance.

A higher cosine similarity score indicates that a document is more similar to the search query.

Relevance Ranking

The documents are sorted from the highest similarity score to the lowest:

ranking_indices = np.argsort(similarity_scores)[::-1]


The results are then displayed in a table containing:

Rank
Document ID
Cosine Similarity Score
Document Text

Example format:

Rank   Doc ID   Cosine Score    Document Text
---------------------------------------------------------------
1      Doc 1    0.xxxx          Machine learning algorithms...
2      Doc 4    0.xxxx          Python is widely used...
3      Doc 2    0.xxxx          Deep learning and neural...
4      Doc 3    0.xxxx          Natural language processing...


The exact scores are calculated automatically when the program runs.

How to Run the Project

From the project directory, execute:

python nlp_task.py


Alternatively:

python -u nlp_task.py


The program will execute both tasks sequentially.

Program Output

The program displays:

Task 1
Input customer reviews
Extracted vocabulary
Number of unique features
Bag of Words term-frequency matrix
Task 2
Document collection
Search query
Vocabulary size
Query vector shape
Document vector shape
Cosine similarity scores
Ranked search results

At the end of successful execution:

Execution completed successfully.

Main Functions
run_task_1()

Performs Bag of Words matrix construction.

Returns:

vectorizer
bow_matrix
df_bow


Where:

vectorizer is the fitted CountVectorizer
bow_matrix is the sparse BoW matrix
df_bow is the Pandas DataFrame representation
run_task_2()

Performs document search and relevance ranking.

Returns:

vectorizer
similarity_scores
df_ranking


Where:

vectorizer is the fitted CountVectorizer
similarity_scores contains cosine similarity values
df_ranking contains the ranked search results
Concepts Demonstrated

This lab demonstrates several important NLP concepts:

Natural Language Processing
Text preprocessing
Stop-word removal
Vocabulary extraction
Bag of Words
Term-frequency representation
Text vectorization
Document representation
Query representation
Cosine similarity
Information retrieval
Document ranking
Search relevance
Dependencies

The project requires the following Python packages:

numpy
pandas
scikit-learn


Install them using:

python -m pip install numpy pandas scikit-learn

Author

NLP Lab 04

Institute of Mathematics & Computer Science
University of Sindh

License

This project is created for educational and academic purposes.# Nlp_Task
