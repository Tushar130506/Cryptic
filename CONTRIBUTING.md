# Contributing to Cryptic

Thanks for your interest in contributing. This project focuses on research-grade backtesting and statistical validity.

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-change`
3. Make changes with tests or notes on validation
4. Run: `python main.py`
5. Submit a pull request with a clear description

## Style Guidelines

- Use clear, descriptive names
- Avoid large feature sets without evidence of IC improvement
- Keep changes reproducible (fixed random seeds)
- Do not add regime-specific logic unless it improves IC and is statistically validated

## What We Accept

- New features with IC validation
- Bug fixes and performance improvements
- Documentation improvements
- Additional asset tests

## What We Avoid

- Unvalidated model complexity
- Overfitted regimes or proprietary data
- Hidden dependencies or unclear changes

## Reporting Issues

Use GitHub Issues with:

- Steps to reproduce
- Expected vs actual behavior
- Logs and environment details
