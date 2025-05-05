import 'package:flutter/material.dart';
import 'package:flutter_search_app/models/search_filter.dart';
import 'package:flutter_search_app/models/search_result.dart';
import 'package:flutter_search_app/services/search_service.dart';

class SearchProvider extends ChangeNotifier {
  final SearchService _searchService = SearchService();
  
  List<SearchResult> _searchResults = [];
  bool _isSearching = false;
  SearchFilterType _currentFilter = SearchFilterType.tfIdf;
  
  List<SearchResult> get searchResults => _searchResults;
  bool get isSearching => _isSearching;
  SearchFilterType get currentFilter => _currentFilter;
  
  void setSearchFilter(SearchFilterType filterType) {
    _currentFilter = filterType;
    notifyListeners();
  }
  
  Future<void> search(String query) async {
    if (query.isEmpty) {
      _searchResults = [];
      notifyListeners();
      return;
    }
    
    _isSearching = true;
    notifyListeners();
    
    try {
      switch (_currentFilter) {
        case SearchFilterType.documentTermIncidence:
          _searchResults = await _searchService.searchWithDocumentTermIncidence(query);
          break;
        case SearchFilterType.invertedIndex:
          _searchResults = await _searchService.searchWithInvertedIndex(query);
          break;
        case SearchFilterType.tfIdf:
          _searchResults = await _searchService.searchWithTfIdf(query);
          break;
      }
    } catch (e) {
      _searchResults = [];
      print('Search error: $e');
    } finally {
      _isSearching = false;
      notifyListeners();
    }
  }
}
