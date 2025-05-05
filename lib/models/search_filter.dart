enum SearchFilterType {
  documentTermIncidence,
  invertedIndex,
  tfIdf,
}

class SearchFilter {
  final SearchFilterType type;
  final String name;
  final String description;

  SearchFilter({
    required this.type,
    required this.name,
    required this.description,
  });

  static List<SearchFilter> getAllFilters() {
    return [
      SearchFilter(
        type: SearchFilterType.documentTermIncidence,
        name: 'Document Term Incidence',
        description: 'Basic boolean search using term presence in documents',
      ),
      SearchFilter(
        type: SearchFilterType.invertedIndex,
        name: 'Inverted Index',
        description: 'Efficient keyword-based document retrieval',
      ),
      SearchFilter(
        type: SearchFilterType.tfIdf,
        name: 'TF-IDF with Cosine Similarity',
        description: 'Relevance-based ranking using term frequency and inverse document frequency',
      ),
    ];
  }
}
