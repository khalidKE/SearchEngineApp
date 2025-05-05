import 'package:flutter/material.dart';
import 'package:flutter_search_app/screens/onboarding_screen.dart';
import 'package:lottie/lottie.dart';

class IntroScreen extends StatefulWidget {
  const IntroScreen({super.key});

  @override
  State<IntroScreen> createState() => _IntroScreenState();
}

class _IntroScreenState extends State<IntroScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<double> _slideAnimation;

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
        curve: const Interval(0.2, 0.8, curve: Curves.easeIn),
      ),
    );

    _slideAnimation = Tween<double>(begin: 50.0, end: 0.0).animate(
      CurvedAnimation(
        parent: _animationController,
        curve: const Interval(0.2, 0.8, curve: Curves.easeOut),
      ),
    );

    _animationController.forward();
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

    // Responsive padding and font scaling
    final padding =
        isSmallScreen
            ? 16.0
            : isTablet
            ? 32.0
            : 48.0;
    final lottieSize =
        isSmallScreen
            ? screenSize.width * 0.6
            : isTablet
            ? screenSize.width * 0.5
            : screenSize.width * 0.4;
    final buttonWidth =
        isSmallScreen
            ? screenSize.width * 0.9
            : isTablet
            ? screenSize.width * 0.6
            : screenSize.width * 0.4;

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
        child: SafeArea(
          child: LayoutBuilder(
            builder: (context, constraints) {
              return SingleChildScrollView(
                child: ConstrainedBox(
                  constraints: BoxConstraints(minHeight: constraints.maxHeight),
                  child: IntrinsicHeight(
                    child: Column(
                      children: [
                        Expanded(
                          child: AnimatedBuilder(
                            animation: _animationController,
                            builder: (context, child) {
                              return Opacity(
                                opacity: _fadeAnimation.value,
                                child: Transform.translate(
                                  offset: Offset(0, _slideAnimation.value),
                                  child: Padding(
                                    padding: EdgeInsets.all(padding),
                                    child: Column(
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: [
                                        SizedBox(
                                          height: lottieSize,
                                          width: lottieSize,
                                          child: LottieBuilder.asset(
                                            'lib/img/hello.json',
                                            fit: BoxFit.contain,
                                          ),
                                        ),
                                        SizedBox(height: padding),
                                        Text(
                                          'Welcome to Search Engine',
                                          style: Theme.of(
                                            context,
                                          ).textTheme.headlineMedium?.copyWith(
                                            fontWeight: FontWeight.bold,
                                            color:
                                                Theme.of(
                                                  context,
                                                ).colorScheme.onBackground,
                                            fontSize:
                                                isSmallScreen
                                                    ? 24
                                                    : isTablet
                                                    ? 32
                                                    : 36,
                                          ),
                                          textAlign: TextAlign.center,
                                        ),
                                        SizedBox(height: padding / 2),
                                        Text(
                                          'A powerful search engine that implements advanced information retrieval algorithms to help you find exactly what you\'re looking for.',
                                          style: Theme.of(
                                            context,
                                          ).textTheme.bodyLarge?.copyWith(
                                            color: Theme.of(context)
                                                .colorScheme
                                                .onBackground
                                                .withOpacity(0.7),
                                            height: 1.5,
                                            fontSize:
                                                isSmallScreen
                                                    ? 16
                                                    : isTablet
                                                    ? 18
                                                    : 20,
                                          ),
                                          textAlign: TextAlign.center,
                                        ),
                                        SizedBox(height: padding),
                                        Text(
                                          'Discover how it works',
                                          style: Theme.of(
                                            context,
                                          ).textTheme.titleMedium?.copyWith(
                                            color:
                                                Theme.of(
                                                  context,
                                                ).colorScheme.primary,
                                            fontWeight: FontWeight.bold,
                                            fontSize:
                                                isSmallScreen
                                                    ? 18
                                                    : isTablet
                                                    ? 20
                                                    : 22,
                                          ),
                                        ),
                                        SizedBox(height: padding / 2),
                                        Icon(
                                          Icons.keyboard_arrow_down,
                                          color:
                                              Theme.of(
                                                context,
                                              ).colorScheme.primary,
                                          size:
                                              isSmallScreen
                                                  ? 24
                                                  : isTablet
                                                  ? 28
                                                  : 32,
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              );
                            },
                          ),
                        ),
                        Padding(
                          padding: EdgeInsets.all(padding),
                          child: SizedBox(
                            width: buttonWidth,
                            height: isSmallScreen ? 48 : 56,
                            child: ElevatedButton(
                              onPressed: () {
                                Navigator.of(context).pushReplacement(
                                  MaterialPageRoute(
                                    builder: (_) => const OnboardingScreen(),
                                  ),
                                );
                              },
                              style: ElevatedButton.styleFrom(
                                backgroundColor:
                                    Theme.of(context).colorScheme.primary,
                                foregroundColor: Colors.white,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(16),
                                ),
                                elevation: 0,
                              ),
                              child: Text(
                                'Get Started',
                                style: TextStyle(
                                  fontSize: isSmallScreen ? 14 : 16,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
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
