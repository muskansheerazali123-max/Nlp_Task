import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def run_task_1():
    print("=" * 80)
    print("TASK 1: BAG OF WORDS MATRIX CONSTRUCTION (CUSTOMER REVIEWS DATASET)")
    print("=" * 80)

    # Sample Input Text
    corpus = [
        "The product performance is amazing and fast",
        "The service was fast and performance was great",
        "Terrible customer service and bad performance"
    ]

    print("\n[Step 1] Input Corpus:")
    for idx, doc in enumerate(corpus, 1):
        print(f"  Review {idx}: \"{doc}\"")

    # 1. Instantiate CountVectorizer(stop_words='english') and fit-transform the corpus
    vectorizer = CountVectorizer(stop_words='english')
    bow_matrix = vectorizer.fit_transform(corpus)

    # 2. Extract the vocabulary list using get_feature_names_out()
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n[Step 2] Extracted Vocabulary ({len(feature_names)} unique features):")
    print(f"  {list(feature_names)}")

    # 3. Convert the transformed sparse matrix into a Pandas DataFrame for structured presentation
    df_bow = pd.DataFrame(
        bow_matrix.toarray(),
        columns=feature_names,
        index=[f"Review {i+1}" for i in range(len(corpus))]
    )

    print("\n[Step 3] Bag of Words (BoW) Term-Frequency Matrix:")
    print(df_bow.to_string())

    return vectorizer, bow_matrix, df_bow

                    # TASK 2: Document Search Engine & Relevance Ranking

def run_task_2():
    print("\n" + "=" * 80)
    print("TASK 2: DOCUMENT SEARCH ENGINE & RELEVANCE RANKING")
    print("=" * 80)

    # Sample Input Documents & Query
    documents = [
        "Machine learning algorithms analyze structured data effectively",
        "Deep learning and neural networks excel at processing unstructured data",
        "Natural language processing helps computers understand human language",
        "Python is widely used for machine learning and data science"
    ]

    query = ["machine learning algorithms for data"]

    print("\n[Step 1] Document Collection:")
    for idx, doc in enumerate(documents, 1):
        print(f"  Doc {idx}: \"{doc}\"")
    print(f"\n  Search Query: \"{query[0]}\"")

    # 1. Fit CountVectorizer on documents
    vectorizer = CountVectorizer()
    vectorizer.fit(documents)

    # 2. Transform both documents and query into numerical vector arrays
    doc_vectors = vectorizer.transform(documents).toarray()
    query_vector = vectorizer.transform(query).toarray()

    print(f"\n[Step 2] Vocabulary Size: {len(vectorizer.get_feature_names_out())} unique tokens")
    print(f"  Query Vector shape: {query_vector.shape}")
    print(f"  Doc Vectors shape:   {doc_vectors.shape}")

    # 3. Compute pairwise cosine similarity using cosine_similarity(query_vector, doc_vectors)
    similarity_scores = cosine_similarity(query_vector, doc_vectors)[0]

    # 4. Display ranked documents from highest score to lowest score
    ranking_indices = np.argsort(similarity_scores)[::-1]

    ranking_data = []
    print("\n[Step 4] Relevance Ranking Results (Highest to Lowest):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Doc ID':<8} {'Cosine Score':<15} {'Document Text'}")
    print("-" * 80)

    for rank, idx in enumerate(ranking_indices, 1):
        score = similarity_scores[idx]
        doc_text = documents[idx]
        doc_id = f"Doc {idx + 1}"
        ranking_data.append({
            "Rank": rank,
            "Document ID": doc_id,
            "Cosine Similarity": round(score, 4),
            "Document Text": doc_text
        })
        print(f"{rank:<6} {doc_id:<8} {score:<15.4f} \"{doc_text}\"")
    print("-" * 80)

    df_ranking = pd.DataFrame(ranking_data)
    return vectorizer, similarity_scores, df_ranking


if __name__ == "__main__":
    print("=" * 80)
    print("NATURAL LANGUAGE PROCESSING LAB 04 - EXECUTION")
    print("Institute of Mathematics & Computer Science, University of Sindh")
    # print("=" * 80)

    run_task_1()
    run_task_2()

    print("\nExecution completed successfully.")