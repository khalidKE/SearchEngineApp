import 'package:flutter/material.dart';
import 'package:flutter_search_app/models/search_filter.dart';

class SearchFiltersWidget extends StatelessWidget {
  final Function(SearchFilterType) onFilterChanged;
  final SearchFilterType selectedFilter;

  const SearchFiltersWidget({
    super.key,
    required this.onFilterChanged,
    required this.selectedFilter,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 50,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: ListView(
        scrollDirection: Axis.horizontal,
        children: [
          _buildFilterChip(
            context,
            SearchFilterType.documentTermIncidence,
            'Document Term Incidence',
            Icons.check_box_outline_blank,
          ),
          _buildFilterChip(
            context,
            SearchFilterType.invertedIndex,
            'Inverted Index',
            Icons.list_alt,
          ),
          _buildFilterChip(
            context,
            SearchFilterType.tfIdf,
            'TF-IDF',
            Icons.analytics_outlined,
          ),
        ],
      ),
    );
  }

  Widget _buildFilterChip(
    BuildContext context,
    SearchFilterType type,
    String label,
    IconData icon,
  ) {
    final isSelected = selectedFilter == type;
    
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: FilterChip(
        label: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 16,
              color: isSelected
                  ? Colors.white
                  : Theme.of(context).colorScheme.primary,
            ),
            const SizedBox(width: 4),
            Text(label),
          ],
        ),
        selected: isSelected,
        onSelected: (selected) {
          if (selected) {
            onFilterChanged(type);
          }
        },
        backgroundColor: Theme.of(context).colorScheme.surface,
        selectedColor: Theme.of(context).colorScheme.primary,
        checkmarkColor: Colors.white,
        labelStyle: TextStyle(
          color: isSelected
              ? Colors.white
              : Theme.of(context).colorScheme.onSurface,
        ),
      ),
    );
  }
}
