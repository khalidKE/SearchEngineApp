import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class OnboardingProvider extends ChangeNotifier {
  bool _hasCompletedOnboarding;
  
  OnboardingProvider(this._hasCompletedOnboarding);
  
  bool get hasCompletedOnboarding => _hasCompletedOnboarding;
  
  Future<void> completeOnboarding() async {
    _hasCompletedOnboarding = true;
    notifyListeners();
    
    // Save to shared preferences
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('hasCompletedOnboarding', true);
  }
  
  Future<void> resetOnboarding() async {
    _hasCompletedOnboarding = false;
    notifyListeners();
    
    // Save to shared preferences
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('hasCompletedOnboarding', false);
  }
}
