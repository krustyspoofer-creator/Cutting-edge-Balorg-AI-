# Contributing to Balorg AI

Thank you for your interest in contributing to Balorg AI! This document provides guidelines and instructions for contributing to the project.

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/krustyspoofer-creator/Cutting-edge-Balorg-AI-.git
   cd Cutting-edge-Balorg-AI-
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write descriptive docstrings for all public functions and classes
- Keep functions focused and modular

### Testing

- Write tests for new features
- Ensure all tests pass before submitting
- Run tests with: `pytest tests/`
- Check coverage: `pytest --cov=balorg_ai tests/`

### Committing Changes

- Write clear, descriptive commit messages
- Reference issues in commits when applicable
- Keep commits focused on a single change

Example commit message:
```
Add nucleus sampling to text generation

- Implement top-p (nucleus) sampling
- Add parameter validation
- Update documentation
- Add unit tests

Fixes #123
```

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- Python version and OS
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages and stack traces
- Minimal code example if possible

### Feature Requests

For feature requests, please describe:
- The problem you're trying to solve
- Your proposed solution
- Alternative solutions considered
- Any relevant examples or references

### Code Contributions

Areas where contributions are welcome:
- New model architectures
- Training optimizations
- Inference improvements
- Documentation enhancements
- Bug fixes
- Test coverage improvements
- Example scripts

## Pull Request Process

1. **Update documentation** - Update README and docstrings as needed
2. **Add tests** - Include tests for new functionality
3. **Run tests** - Ensure all tests pass
4. **Update CHANGELOG** - Add entry describing your changes
5. **Submit PR** - Provide a clear description of changes

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests pass
```

## Code Review Process

- Maintainers will review PRs within a few days
- Address review feedback promptly
- Be open to suggestions and discussion
- Changes may be requested before merging

## Questions?

- Open an issue for questions
- Tag issues with `question` label
- Check existing issues first

## License

By contributing, you agree that your contributions will be licensed under the CC0 1.0 Universal License.

Thank you for contributing to Balorg AI! 🎉
