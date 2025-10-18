# Contributing to Balorg AI

Thank you for your interest in contributing to Balorg AI! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Cutting-edge-Balorg-AI-.git
   cd Cutting-edge-Balorg-AI-
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

4. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Making Changes

1. Make your changes in your feature branch
2. Write or update tests for your changes
3. Ensure all tests pass:
   ```bash
   pytest tests/
   ```

4. Update documentation if needed

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions small and focused
- Use type hints where appropriate

### Testing

- Write tests for all new functionality
- Ensure tests are descriptive and cover edge cases
- Run the full test suite before submitting:
  ```bash
  pytest tests/ -v
  ```

### Commit Messages

- Use clear and descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Keep the first line under 50 characters
- Add a detailed description if needed

Example:
```
Add batch normalization layer

- Implement BatchNormLayer class
- Add forward and backward pass
- Include tests for training and inference modes
```

## Pull Request Process

1. Update the README.md if you're adding new features
2. Add your changes to CHANGELOG.md if applicable
3. Ensure all tests pass
4. Push to your fork and submit a pull request
5. Wait for review and address any feedback

### Pull Request Checklist

- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Commit messages are clear
- [ ] No unnecessary files included

## Types of Contributions

### Bug Reports

When filing a bug report, please include:
- A clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python version and environment details

### Feature Requests

When suggesting a feature:
- Explain the use case
- Describe the proposed solution
- Consider alternatives
- Be open to discussion

### Code Contributions

We welcome:
- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test improvements

## Areas for Contribution

### High Priority

1. **Optimizers**: Implement Adam, SGD, RMSprop, etc.
2. **Loss Functions**: Implement various loss functions
3. **Convolutional Layers**: Add CNN support
4. **Recurrent Layers**: Add LSTM/GRU support
5. **Model Serialization**: Complete save/load functionality

### Medium Priority

1. **GPU Support**: Add CUDA integration
2. **Data Loaders**: Add support for more datasets
3. **Visualization**: Training visualization tools
4. **Metrics**: More evaluation metrics

### Documentation

1. More examples and tutorials
2. API documentation
3. Performance benchmarks
4. Architecture diagrams

## Community

- Be respectful and constructive
- Help others when you can
- Share your knowledge
- Give credit where it's due

## Questions?

If you have questions about contributing, please:
1. Check existing issues and discussions
2. Open a new issue with the "question" label
3. Be specific about what you need help with

Thank you for contributing to Balorg AI!
