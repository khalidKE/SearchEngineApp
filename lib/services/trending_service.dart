import 'package:flutter/material.dart';
import 'package:flutter_search_app/models/trending_search.dart';

class TrendingService {
  List<TrendingSearch> getTrendingSearches() {
    // In a real app, this would fetch from an API or local database
    return [
      TrendingSearch(
        query: 'information retrieval',
        searchCount: 1245,
        icon: Icons.trending_up,
        timestamp: DateTime.now().subtract(const Duration(hours: 2)),
      ),
      TrendingSearch(
        query: 'text processing',
        searchCount: 987,
        icon: Icons.text_fields,
        timestamp: DateTime.now().subtract(const Duration(hours: 4)),
      ),
      TrendingSearch(
        query: 'search algorithms',
        searchCount: 856,
        icon: Icons.search,
        timestamp: DateTime.now().subtract(const Duration(hours: 6)),
      ),
      TrendingSearch(
        query: 'stemming techniques',
        searchCount: 743,
        icon: Icons.auto_awesome,
        timestamp: DateTime.now().subtract(const Duration(hours: 8)),
      ),
      TrendingSearch(
        query: 'phonetic matching',
        searchCount: 621,
        icon: Icons.record_voice_over,
        timestamp: DateTime.now().subtract(const Duration(hours: 10)),
      ),
      TrendingSearch(
        query: 'fuzzy search',
        searchCount: 589,
        icon: Icons.blur_on,
        timestamp: DateTime.now().subtract(const Duration(hours: 12)),
      ),
    ];
  }
}
