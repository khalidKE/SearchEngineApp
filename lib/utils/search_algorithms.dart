
class SearchAlgorithms {
  // Tokenization
  List<String> tokenize(String text) {
    // Simple tokenization by splitting on whitespace and handling basic punctuation
    return text
        .replaceAll('.', ' ')
        .replaceAll(',', ' ')
        .replaceAll('!', ' ')
        .replaceAll('?', ' ')
        .replaceAll('(', ' ')
        .replaceAll(')', ' ')
        .split(RegExp(r'\s+'))
        .where((token) => token.isNotEmpty)
        .toList();
  }

  // Stop Words Removal
  List<String> removeStopWords(List<String> tokens) {
    final stopWords = {
      'a',
      'an',
      'the',
      'and',
      'or',
      'but',
      'is',
      'are',
      'was',
      'were',
      'in',
      'on',
      'at',
      'to',
      'for',
      'with',
      'by',
      'about',
      'of',
      'as',
      'from',
      'that',
      'this',
      'these',
      'those',
      'it',
      'its',
      'which',
      'who',
      'whom',
      'whose',
      'where',
      'when',
      'why',
      'how',
      'what',
      'be',
      'been',
      'being',
      'have',
      'has',
      'had',
      'do',
      'does',
      'did',
      'can',
      'could',
      'will',
      'would',
      'shall',
      'should',
      'may',
      'might',
      'must',
    };
    return tokens
        .where((token) => !stopWords.contains(token.toLowerCase()))
        .toList();
  }

  // Normalization (Case Folding)
  List<String> normalize(List<String> tokens) {
    return tokens.map((token) => token.toLowerCase()).toList();
  }

  // Stemming (simplified Porter stemmer-like algorithm)
  String stemWord(String word) {
    String result = word.toLowerCase();

    // Handle some common endings
    if (result.endsWith('ing')) {
      // Special case for 'ing' ending
      if (result.length > 4) {
        result = result.substring(0, result.length - 3);
        if (result.endsWith('n')) {
          result = result.substring(0, result.length - 1);
        }
      }
    } else if (result.endsWith('ed')) {
      // Special case for 'ed' ending
      if (result.length > 3) {
        result = result.substring(0, result.length - 2);
      }
    } else if (result.endsWith('s') &&
        !result.endsWith('ss') &&
        !result.endsWith('us') &&
        !result.endsWith('is')) {
      // Special case for plural 's' ending
      if (result.length > 2) {
        result = result.substring(0, result.length - 1);
      }
    } else if (result.endsWith('ly')) {
      // Special case for 'ly' ending
      if (result.length > 3) {
        result = result.substring(0, result.length - 2);
      }
    } else if (result.endsWith('ment')) {
      // Special case for 'ment' ending
      if (result.length > 5) {
        result = result.substring(0, result.length - 4);
      }
    } else if (result.endsWith('ness')) {
      // Special case for 'ness' ending
      if (result.length > 5) {
        result = result.substring(0, result.length - 4);
      }
    } else if (result.endsWith('tion')) {
      // Special case for 'tion' ending
      if (result.length > 5) {
        result = result.substring(0, result.length - 4) + 't';
      }
    } else if (result.endsWith('ies')) {
      // Special case for 'ies' ending (e.g., 'categories' -> 'categori')
      if (result.length > 4) {
        result = result.substring(0, result.length - 3) + 'i';
      }
    } else if (result.endsWith('es')) {
      // Special case for 'es' ending
      if (result.length > 3) {
        result = result.substring(0, result.length - 2);
      }
    }

    return result;
  }

  // Lemmatization (simplified version - in a real app, you'd use a dictionary)
  String lemmatizeWord(String word) {
    // This is a very simplified lemmatization
    // In a real app, you would use a proper NLP library with a dictionary
    String lowerWord = word.toLowerCase();

    // Some common irregular forms
    Map<String, String> irregulars = {
      'am': 'be',
      'is': 'be',
      'are': 'be',
      'was': 'be',
      'were': 'be',
      'has': 'have',
      'had': 'have',
      'does': 'do',
      'did': 'do',
      'better': 'good',
      'best': 'good',
      'worse': 'bad',
      'worst': 'bad',
      'children': 'child',
      'men': 'man',
      'women': 'woman',
      'people': 'person',
      'mice': 'mouse',
      'geese': 'goose',
      'feet': 'foot',
      'teeth': 'tooth',
      'leaves': 'leaf',
      'lives': 'life',
      'knives': 'knife',
    };

    if (irregulars.containsKey(lowerWord)) {
      return irregulars[lowerWord]!;
    }

    // Handle regular forms
    if (lowerWord.endsWith('s') &&
        !lowerWord.endsWith('ss') &&
        !lowerWord.endsWith('us') &&
        !lowerWord.endsWith('is')) {
      return lowerWord.substring(0, lowerWord.length - 1);
    } else if (lowerWord.endsWith('ies')) {
      return lowerWord.substring(0, lowerWord.length - 3) + 'y';
    } else if (lowerWord.endsWith('es')) {
      return lowerWord.substring(0, lowerWord.length - 2);
    } else if (lowerWord.endsWith('ing')) {
      // Try to restore the 'e' if it was removed
      if (lowerWord.length > 4) {
        String stem = lowerWord.substring(0, lowerWord.length - 3);
        // Check if the stem ends with a consonant
        if (stem.isNotEmpty && !'aeiou'.contains(stem[stem.length - 1])) {
          return stem + 'e';
        }
        return stem;
      }
    } else if (lowerWord.endsWith('ed')) {
      // Try to restore the 'e' if it was removed
      if (lowerWord.length > 3) {
        String stem = lowerWord.substring(0, lowerWord.length - 2);
        // Check if the stem ends with a consonant
        if (stem.isNotEmpty && !'aeiou'.contains(stem[stem.length - 1])) {
          return stem + 'e';
        }
        return stem;
      }
    }

    return lowerWord;
  }

  // Adding the missing methods

  // Soundex Algorithm
  String getSoundex(String word) {
    if (word.isEmpty) return '0000';

    // Convert to uppercase
    word = word.toUpperCase();

    // Keep first letter
    String code = word[0];

    // Map for converting letters to Soundex digits
    Map<String, String> soundexMap = {
      'B': '1',
      'F': '1',
      'P': '1',
      'V': '1',
      'C': '2',
      'G': '2',
      'J': '2',
      'K': '2',
      'Q': '2',
      'S': '2',
      'X': '2',
      'Z': '2',
      'D': '3',
      'T': '3',
      'L': '4',
      'M': '5',
      'N': '5',
      'R': '6',
    };

    // Replace letters with digits
    String previousDigit = '';
    for (int i = 1; i < word.length; i++) {
      String letter = word[i];
      String digit = soundexMap[letter] ?? '0';

      // Skip vowels and 'H', 'W', 'Y'
      if (['A', 'E', 'I', 'O', 'U', 'H', 'W', 'Y'].contains(letter)) {
        continue;
      }

      // Skip consecutive duplicate digits
      if (digit != previousDigit) {
        code += digit;
        previousDigit = digit;
      }
    }

    // Pad with zeros and truncate to 4 characters
    code = code.padRight(4, '0').substring(0, 4);

    return code;
  }

  // Jaccard Coefficient
  double getJaccardCoefficient(String s1, String s2) {
    if (s1.isEmpty && s2.isEmpty) return 1.0;
    if (s1.isEmpty || s2.isEmpty) return 0.0;

    // Get bigrams
    Set<String> bigrams1 = _getBigrams(s1);
    Set<String> bigrams2 = _getBigrams(s2);

    // Calculate intersection and union
    Set<String> intersection = bigrams1.intersection(bigrams2);
    Set<String> union = bigrams1.union(bigrams2);

    // Calculate Jaccard coefficient
    return intersection.length / union.length;
  }

  Set<String> _getBigrams(String s) {
    if (s.length < 2) return {s};

    Set<String> bigrams = {};
    for (int i = 0; i < s.length - 1; i++) {
      bigrams.add(s.substring(i, i + 2));
    }
    return bigrams;
  }
}
