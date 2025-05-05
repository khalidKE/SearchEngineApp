import 'package:flutter/material.dart';
import 'package:flutter_search_app/models/trending_search.dart';
import 'package:flutter_search_app/services/trending_service.dart';

class TrendingSearchesWidget extends StatelessWidget {
  final Function(String) onTrendingSelected;

  const TrendingSearchesWidget({super.key, required this.onTrendingSelected});

  @override
  Widget build(BuildContext context) {
    final trendingService = TrendingService();
    final trendingSearches = trendingService.getTrendingSearches();

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Trending Searches',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 16),
          Expanded(
            child: GridView.builder(
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                childAspectRatio: 1.5,
                crossAxisSpacing: 16,
                mainAxisSpacing: 16,
              ),
              itemCount: trendingSearches.length,
              itemBuilder: (context, index) {
                final trend = trendingSearches[index];
                return TrendingCard(
                  trend: trend,
                  index: index,
                  onTap: () => onTrendingSelected(trend.query),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

class TrendingCard extends StatelessWidget {
  final TrendingSearch trend;
  final int index;
  final VoidCallback onTap;

  const TrendingCard({
    super.key,
    required this.trend,
    required this.index,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Card(
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Theme.of(context).colorScheme.primary.withOpacity(0.7),
                Theme.of(context).colorScheme.primary,
              ],
            ),
          ),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(trend.icon, color: Colors.white, size: 20),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        '#${index + 1}',
                        style: const TextStyle(
                          color: Colors.white70,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    Text(
                      '${trend.searchCount}',
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                const Spacer(),
                Text(
                  trend.query,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
