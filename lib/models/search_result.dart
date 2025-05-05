class SearchResult {
  final String id;
  final String title;
  final String url;
  final String snippet;
  final String content;
  final List<String> keywords;
  final double relevanceScore;
  final String lastUpdated;
  final String source;

  SearchResult({
    required this.id,
    required this.title,
    required this.url,
    required this.snippet,
    required this.content,
    required this.keywords,
    required this.relevanceScore,
    required this.lastUpdated,
    required this.source,
  });
}
