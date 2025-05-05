
# 🚀 SearchPro – Intelligent Search Engine App

<p align="center">
  <img src="lib/img/icon.png" alt="SearchPro Logo" width="200"/>
</p>

## 📘 Overview

**SearchPro** is a powerful and intelligent search engine application developed using **Flutter**. It is designed to deliver lightning-fast, highly relevant results through a combination of advanced search algorithms and natural language processing (NLP) techniques. Whether you're searching through **200 documents across fields such as programming, AI, data science, cybersecurity, biotechnology, medicine, engineering, marketing, or energy**, SearchPro provides accurate and efficient results with a modern, responsive user interface.

---

## 🔍 Key Features

### 🧠 Core Search Capabilities

* **Multi-Algorithm Search Engine**:

  * **TF-IDF**: Evaluates term importance in context
  * **BM25**: Enhances result ranking beyond TF-IDF
  * **Inverted Index**: Enables fast full-text lookups
  * **Vector Space Model** with **Cosine Similarity**: Determines document similarity and ranks results accordingly

* **Natural Language Processing**:

  * Tokenization, Lemmatization, and Stemming
  * Named Entity Recognition (NER)
  * Part-of-Speech Tagging
  * Sentiment Analysis
  * Stop Word Filtering

* **Advanced Query Processing**:

  * Query Expansion with synonyms
  * Real-time Spell Correction
  * Autocomplete Suggestions
  * Query Intent Recognition

---

### 🎨 User Experience

* Clean, intuitive UI with animated onboarding
* Responsive design for all screen sizes and orientations
* Adaptive theming (Dark/Light Mode)
* Gesture-based navigation for smooth interactions

---

### ⚙️ Technical Features

* **Offline Functionality**: Core search works without internet
* **Smart Caching**: Stores frequent queries for instant responses
* **Low Memory & Battery Use**: Optimized for performance on all devices

---

## 🏗️ Search Engine Architecture

1. **Query Processing Layer**

   * Parses and normalizes input
   * Applies query expansion and correction

2. **Indexing Layer**

   * Maintains an inverted index of all documents
   * Manages tokenization, updates, and optimization

3. **Ranking Layer**

   * Computes TF-IDF and BM25 scores
   * Ranks and sorts results by relevance using vector similarity

4. **Caching Layer**

   * Implements LRU (Least Recently Used) caching
   * Speeds up repeated and popular searches

---

## 📖 Text Preprocessing Workflow

Before executing a search, the text undergoes:

1. **Tokenization** – Breaking text into individual words
2. **Stop Word Removal** – Filtering out common, low-value terms
3. **Case Folding** – Standardizing all text to lowercase
4. **Stemming/Lemmatization** – Reducing words to their base forms

Example:
*“Writing”, “writer”, and “wrote” → “write”*

---

## 📱 How to Use SearchPro

1. Launch the app to view the welcome/onboarding guide
2. Enter a search term in the input bar
3. Choose your preferred search algorithm (e.g., TF-IDF, Inverted Index)
4. Review results sorted by relevance
5. Tap on a result to view document details
6. Toggle dark/light mode as needed

---

## 🧪 Algorithms in Action

* **Document Term Incidence**: Simple word matching
* **Inverted Index**: Fast retrieval via word-to-document mapping
* **TF-IDF + Cosine Similarity**: Sophisticated ranking based on word significance and document similarity

---

## 📥 Download the App

Get the latest release here:
👉 [Download SearchPro](https://drive.google.com/file/d/1vOPlcQ8bAxit7kejCj8o4PxOR20cQ5hH/view?usp=drivesdk)

---

## 🌐 Technologies Used

* **Flutter** (UI Framework)
* **Dart** (Programming Language)
* **NLP Techniques** (Text Preprocessing, NER, Sentiment Analysis)
* **Search Algorithms** (TF-IDF, BM25, Cosine Similarity)

---
