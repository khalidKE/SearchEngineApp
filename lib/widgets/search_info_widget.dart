import 'package:flutter/material.dart';

class SearchInfoWidget extends StatelessWidget {
  const SearchInfoWidget({super.key});

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildInfoCard(context, 'Preprocessing Steps', [
            'Tokenization: Breaking text into individual tokens',
            'Stop Word Removal: Filtering out common words',
            'Case Folding: Converting text to lowercase',
            'Lemmatization: Reducing words to their base dictionary form',
            'Stemming: Reducing words to their root form',
          ], Icons.text_fields),
          const SizedBox(height: 16),
          _buildInfoCard(context, 'Search Methods', [
            'Document Term Incidence: Basic boolean search',
            'Inverted Index: Efficient keyword-based retrieval',
            'TF-IDF with Cosine Similarity: Relevance-based ranking',
          ], Icons.search),
          const SizedBox(height: 16),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.info_outline,
                        color: Theme.of(context).colorScheme.primary,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'How to Use',
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    '1. Enter your search query in the search bar\n'
                    '2. Select a search method using the filter button\n'
                    '3. View search results ranked by relevance\n'
                    '4. Tap on any result to see detailed information',
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoCard(
    BuildContext context,
    String title,
    List<String> items,
    IconData icon,
  ) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: Theme.of(context).colorScheme.primary),
                const SizedBox(width: 8),
                Text(title, style: Theme.of(context).textTheme.titleLarge),
              ],
            ),
            const SizedBox(height: 8),
            ...items.map(
              (item) => Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [const Text('• '), Expanded(child: Text(item))],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
