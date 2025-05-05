import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter_search_app/providers/onboarding_provider.dart';
import 'package:flutter_search_app/screens/intro_screen.dart';
import 'package:flutter_search_app/screens/search_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<double> _scaleAnimation;

  @override
  void initState() {
    super.initState();

    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _animationController,
        curve: const Interval(0.0, 0.5, curve: Curves.easeIn),
      ),
    );

    _scaleAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(
        parent: _animationController,
        curve: const Interval(0.0, 0.5, curve: Curves.easeOut),
      ),
    );

    _animationController.forward();

    // Navigate to the next screen after animation completes
    Future.delayed(const Duration(milliseconds: 2500), () {
      _navigateToNextScreen();
    });
  }

  void _navigateToNextScreen() {
    final onboardingProvider = Provider.of<OnboardingProvider>(
      context,
      listen: false,
    );

    if (onboardingProvider.hasCompletedOnboarding) {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const SearchScreen()),
      );
    } else {
      Navigator.of(
        context,
      ).pushReplacement(MaterialPageRoute(builder: (_) => const IntroScreen()));
    }
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isDarkMode = Theme.of(context).brightness == Brightness.dark;
    final screenSize = MediaQuery.of(context).size;
    final isSmallScreen = screenSize.width < 600;
    final isTablet = screenSize.width >= 600 && screenSize.width < 1024;

    // Responsive sizing
    final containerSize =
        isSmallScreen
            ? screenSize.width * 0.3
            : isTablet
            ? screenSize.width * 0.25
            : screenSize.width * 0.2;
    final iconSize =
        isSmallScreen
            ? 50.0
            : isTablet
            ? 60.0
            : 70.0;
    final padding =
        isSmallScreen
            ? 16.0
            : isTablet
            ? 24.0
            : 32.0;

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors:
                isDarkMode
                    ? [const Color(0xFF1A1A2E), const Color(0xFF16213E)]
                    : [const Color(0xFFE9F1FF), const Color(0xFFD6E4FF)],
          ),
        ),
        child: Center(
          child: AnimatedBuilder(
            animation: _animationController,
            builder: (context, child) {
              return Opacity(
                opacity: _fadeAnimation.value,
                child: Transform.scale(
                  scale: _scaleAnimation.value,
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        width: containerSize,
                        height: containerSize,
                        decoration: BoxDecoration(
                          color: Theme.of(context).colorScheme.primary,
                          borderRadius: BorderRadius.circular(
                            containerSize * 0.25,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: Theme.of(
                                context,
                              ).colorScheme.primary.withOpacity(0.3),
                              blurRadius: 20,
                              offset: const Offset(0, 10),
                            ),
                          ],
                        ),
                        child: Icon(
                          Icons.search,
                          size: iconSize,
                          color: Colors.white,
                        ),
                      ),
                      SizedBox(height: padding * 1.5),
                      Text(
                        'Search Engine',
                        style: Theme.of(
                          context,
                        ).textTheme.headlineMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: Theme.of(context).colorScheme.onBackground,
                          fontSize:
                              isSmallScreen
                                  ? 24
                                  : isTablet
                                  ? 28
                                  : 32,
                        ),
                      ),
                      SizedBox(height: padding / 2),
                      Text(
                        'Discover information efficiently',
                        style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                          color: Theme.of(
                            context,
                          ).colorScheme.onBackground.withOpacity(0.7),
                          fontSize:
                              isSmallScreen
                                  ? 16
                                  : isTablet
                                  ? 18
                                  : 20,
                        ),
                      ),
                    ],
                  ),
                ),
              );
            },
          ),
        ),
      ),
    );
  }
}
