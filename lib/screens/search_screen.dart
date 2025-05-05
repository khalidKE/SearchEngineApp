import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter_search_app/providers/search_provider.dart';
import 'package:flutter_search_app/providers/theme_provider.dart';
import 'package:flutter_search_app/providers/onboarding_provider.dart';
import 'package:flutter_search_app/widgets/search_bar_widget.dart';
import 'package:flutter_search_app/widgets/search_results_widget.dart';
import 'package:flutter_search_app/widgets/search_filters_widget.dart';
import 'package:flutter_search_app/widgets/search_info_widget.dart';

class SearchScreen extends StatefulWidget {
  const SearchScreen({super.key});

  @override
  State<SearchScreen> createState() => _SearchScreenState();
}

class _SearchScreenState extends State<SearchScreen> {
  final TextEditingController _searchController = TextEditingController();
  bool _showFilters = false;

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final searchProvider = Provider.of<SearchProvider>(context);
    final themeProvider = Provider.of<ThemeProvider>(context);
    final onboardingProvider = Provider.of<OnboardingProvider>(context);
    final isSearching = searchProvider.isSearching;
    final hasResults = searchProvider.searchResults.isNotEmpty;
    final hasQuery = _searchController.text.isNotEmpty;
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('Search Engine'),
        actions: [
          IconButton(
            icon: Icon(
              themeProvider.isDarkMode ? Icons.light_mode : Icons.dark_mode,
            ),
            onPressed: () {
              themeProvider.toggleTheme();
            },
          ),
          PopupMenuButton<String>(
            onSelected: (value) {
              if (value == 'reset_onboarding') {
                onboardingProvider.resetOnboarding();
                Navigator.of(context).pushNamedAndRemoveUntil('/', (route) => false);
              }
            },
            itemBuilder: (context) => [
              const PopupMenuItem<String>(
                value: 'reset_onboarding',
                child: Text('Reset Onboarding'),
              ),
            ],
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: Row(
                children: [
                  Expanded(
                    child: SearchBarWidget(
                      controller: _searchController,
                      onSearch: (query) {
                        searchProvider.search(query);
                      },
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton(
                    icon: Icon(
                      _showFilters ? Icons.filter_list_off : Icons.filter_list,
                      color: Theme.of(context).colorScheme.primary,
                    ),
                    onPressed: () {
                      setState(() {
                        _showFilters = !_showFilters;
                      });
                    },
                  ),
                ],
              ),
            ),
            
            if (_showFilters)
              SearchFiltersWidget(
                onFilterChanged: (filterType) {
                  searchProvider.setSearchFilter(filterType);
                  if (_searchController.text.isNotEmpty) {
                    searchProvider.search(_searchController.text);
                  }
                },
                selectedFilter: searchProvider.currentFilter,
              ),
              
            Expanded(
              child: isSearching
                  ? const Center(child: CircularProgressIndicator())
                  : hasResults
                      ? SearchResultsWidget(results: searchProvider.searchResults)
                      : hasQuery
                          ? const Center(child: Text('No results found'))
                          : const SearchInfoWidget(),
            ),
          ],
        ),
      ),
    );
  }
}
