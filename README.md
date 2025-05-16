# 🔍 Search Engine App

A powerful, educational **search engine mobile app** built with **Flutter**, demonstrating core concepts of **Information Retrieval (IR)**. This project integrates classic IR algorithms with a clean, modular Flutter architecture to simulate how real-world search engines work—right on your phone.

---

## 🚀 Features

### ✅ Core Search Techniques

* **Text Preprocessing**

  * **Tokenization**: Breaks text into individual tokens
  * **Stop Word Removal**: Filters out common, non-informative words
  * **Case Folding**: Converts all text to lowercase
  * **Stemming**: Simplifies words to their root form (e.g., *running → run*)
  * **Lemmatization**: Maps words to their dictionary form (e.g., *children → child*)

* **Search Methods**

  * **Document Term Incidence**: Simple boolean retrieval (AND operation)
  * **Inverted Index**: Efficient keyword-based search using term-document mapping
  * **TF-IDF + Cosine Similarity**: Calculates document relevance and ranks results based on similarity to the query

### 🧠 Additional Algorithms

* **Soundex**: Phonetic algorithm for matching similar-sounding words
* **Jaccard Coefficient**: Measures set similarity for fuzzy matching

---

## 🧱 App Architecture

Follows a clean, modular structure:

* `lib/ui`: UI components (widgets for results, filters, and navigation)
* `lib/models`: Data classes for documents, queries, and results
* `lib/services`: `SearchService` for coordinating all search operations
* `lib/utils`: Text processing and similarity algorithms

---

## 🖼️ UI Highlights

* Switch between **search methods** (Boolean, Inverted Index, TF-IDF)
* View **ranked results** with title, snippet, and relevance score
* **Related Results** section shows similar content using Jaccard or Soundex
* Built-in **info panels** that explain each search strategy

---

## 📱 Screenshots

![Landing (2)](https://github.com/user-attachments/assets/0c442a65-2764-477d-a61e-cdfde5777c35)

