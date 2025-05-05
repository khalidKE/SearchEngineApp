# SearchPro

<p align="center">
  <img src="lib/img/icon.png" alt="SearchPro Logo" width="200"/>
</p>

## Overview

SearchPro is a modern search engine application built with Flutter that leverages advanced algorithms and natural language processing to deliver accurate and efficient search results. The app features a sleek dark-themed UI with intuitive onboarding screens to introduce users to its powerful capabilities.

## Features

- **Advanced Search Algorithms**: Utilizes TF-IDF, Inverted Index, and Cosine Similarity for accurate search results
- **Natural Language Processing**: Implements tokenization, lemmatization, and semantic analysis for context-aware searches
- **Adaptive User Interface**: Seamlessly switches between dark and light modes for optimal viewing in any environment
- **Responsive Design**: Optimized for various screen sizes and orientations
- **Smooth Animations**: Enhanced user experience with fluid transitions and Lottie animations



---

## 💡 Search Engine App Overview

### General Description

Alright, so this app is a simple yet powerful **search engine** built using **Flutter**, designed to work smoothly and beautifully on mobile devices. It uses advanced search techniques under the hood.

---

### 🔑 Key Features

#### 1. **Beautiful and Simple UI**:

* Clean and easy-to-use search screen
* Supports both light and dark modes
* Welcome/onboarding screen explaining how to use the app

#### 2. **Search Functionality**:

* Fast and effective search
* Results displayed in order of relevance
* Filters for different types of search
* Shows trending searches

---

### 🧠 Algorithms Used

The app uses **three main search algorithms**:

#### 1. **Document Term Incidence**

* Basic method:

  * Takes the search term
  * Scans all documents
  * Finds documents that exactly contain the word
  * Returns those documents
* Example: Searching for “Egypt” returns all documents containing exactly that word.

#### 2. **Inverted Index**

* Faster and more efficient:

  * Builds an index of all words in the documents
  * Each word links to the list of documents it's found in
  * Searching becomes a quick lookup in that index
* Like a book index—no need to read the whole book to find something.

#### 3. **TF-IDF with Cosine Similarity**

* Most advanced method:

  * Calculates how important a word is in a document (TF)
  * Considers how rare the word is across all documents (IDF)
  * Converts both documents and search queries into mathematical vectors
  * Uses **Cosine Similarity** to calculate how similar each document is to the search
  * Returns and ranks results based on similarity score
* This gives more **accurate** and **relevant** search results.

---

### 🛠️ Text Processing Steps (Before Search)

1. **Tokenization**: Splits text into separate words

   * E.g., "Welcome to Egypt" → \["Welcome", "to", "Egypt"]
2. **Stop Words Removal**: Removes common unimportant words like "to", "from", "in", etc.
3. **Case Folding**: Converts all letters to lowercase to avoid mismatch
4. **Stemming**: Reduces words to their root forms

   * E.g., "writing", "writer", "wrote" → "write"

---

### 📱 How to Use the App

1. **Open the app** → You'll see a welcome and onboarding screen
2. **Enter a search term** in the search bar
3. **Select the search method** from the filters below the search bar
4. **View results** ranked by relevance
5. **Tap on a result** to view more details
6. **Toggle light/dark mode** from the top right button

---
### you can download app https://drive.google.com/file/d/1vOPlcQ8bAxit7kejCj8o4PxOR20cQ5hH/view?usp=drivesdk

This app brings advanced information retrieval techniques into a **simple and user-friendly experience**. The algorithms used are similar to those found in major search engines like Google—but of course, at a smaller scale and simpler implementation.


