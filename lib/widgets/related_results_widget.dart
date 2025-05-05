import 'package:flutter/material.dart';
import 'package:flutter_search_app/models/search_result.dart';
import 'package:flutter_search_app/screens/result_detail_screen.dart';
import 'package:flutter_search_app/services/search_service.dart';

class RelatedResultsWidget extends StatelessWidget {
  final SearchResult currentResult;

  const RelatedResultsWidget({
    super.key,
    required this.currentResult,
  });

  @override
  Widget build(BuildContext context) {
    final searchService = SearchService();
    final relatedResults = searchService.getRelatedResults(currentResult);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Related Results',
          style: Theme.of(context).textTheme.titleLarge,
        ),
        const SizedBox(height: 16),
        ListView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          itemCount: relatedResults.length,
          itemBuilder: (context, index) {
            final result = relatedResults[index];
            return ListTile(
              title: Text(
                result.title,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
              subtitle: Text(
                result.snippet,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
              leading: CircleAvatar(
                backgroundColor: Theme.of(context).colorScheme.primary,
                child: Text(
                  result.title.substring(0, 1),
                  style: const TextStyle(color: Colors.white),
                ),
              ),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => ResultDetailScreen(result: result),
                  ),
                );
              },
            );
          },
        ),
      ],
    );
  }
}
