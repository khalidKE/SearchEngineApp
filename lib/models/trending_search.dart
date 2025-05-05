import 'package:flutter/material.dart';

class TrendingSearch {
  final String query;
  final int searchCount;
  final IconData icon;
  final DateTime timestamp;

  TrendingSearch({
    required this.query,
    required this.searchCount,
    required this.icon,
    required this.timestamp,
  });
}
