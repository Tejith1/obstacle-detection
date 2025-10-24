# 🤝 Contributing to Drone Obstacle Detection System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Git
- Basic knowledge of computer vision and PyTorch

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/drone-obstacle-detection.git
cd drone-obstacle-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install flask pytest black flake8

# Verify installation
python webcam_detect.py
```

## 📋 How to Contribute

### 1. 🐛 Reporting Bugs
- Use the [GitHub Issues](https://github.com/yourusername/drone-obstacle-detection/issues) page
- Include detailed description and steps to reproduce
- Provide system information (OS, Python version, hardware)
- Include error messages and logs

### 2. 💡 Suggesting Features
- Open a [Feature Request](https://github.com/yourusername/drone-obstacle-detection/issues/new)
- Describe the feature and its benefits
- Provide use cases and examples
- Consider implementation complexity

### 3. 🔧 Code Contributions

#### Branch Naming
- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

#### Pull Request Process
1. **Fork** the repository
2. **Create** a feature branch from `main`
3. **Make** your changes with clear commit messages
4. **Test** your changes thoroughly
5. **Update** documentation if needed
6. **Submit** a pull request

#### Code Style
```bash
# Format code with Black
black *.py

# Check linting with flake8
flake8 *.py --max-line-length=88

# Run tests
pytest tests/
```

## 🎯 Development Areas

### High Priority
- [ ] **Model Performance**: Improve detection accuracy
- [ ] **Drone Integration**: Phase 2 hardware integration
- [ ] **Real-time Optimization**: Performance improvements
- [ ] **Testing**: Unit tests and integration tests

### Medium Priority
- [ ] **Web Interface**: Enhanced dashboard features
- [ ] **Documentation**: Video tutorials and examples
- [ ] **Mobile Support**: Responsive web interface
- [ ] **Configuration**: GUI-based settings management

### Low Priority
- [ ] **Docker Support**: Containerization
- [ ] **Cloud Integration**: Remote monitoring
- [ ] **Multi-camera**: Support for multiple camera feeds
- [ ] **Analytics**: Historical data and reporting

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_navigation.py

# Run with coverage
pytest --cov=. tests/
```

### Writing Tests
```python
# tests/test_navigation.py
import pytest
from drone_navigation import DroneNavigator

def test_clear_path():
    navigator = DroneNavigator()
    result = navigator.analyze_obstacles([])
    assert result['status'] == 'CLEAR_PATH'
    assert result['direction'] == 'FORWARD'

def test_obstacle_detection():
    navigator = DroneNavigator()
    # Mock detection in center zone
    detections = [[300, 200, 400, 300, 0.8, 0]]
    result = navigator.analyze_obstacles(detections)
    assert result['status'] != 'CLEAR_PATH'
```

### Test Coverage Areas
- [ ] Navigation logic
- [ ] Detection processing
- [ ] Web API endpoints
- [ ] Class mapping utilities
- [ ] Performance benchmarks

## 📚 Documentation Standards

### Code Documentation
```python
def analyze_obstacles(self, detections):
    """
    Analyze obstacle positions and determine safe navigation direction.
    
    Args:
        detections (list): List of [x1, y1, x2, y2, conf, class] detections
        
    Returns:
        dict: Navigation recommendations with status, direction, and confidence
        
    Example:
        >>> navigator = DroneNavigator()
        >>> result = navigator.analyze_obstacles(detections)
        >>> print(result['direction'])  # 'LEFT'
    """
```

### README Updates
- Keep installation instructions current
- Update feature lists when adding functionality
- Include screenshots for visual features
- Maintain performance benchmarks

### API Documentation
- Document all endpoints in `docs/API.md`
- Include request/response examples
- Update parameter references
- Add integration examples

## 🔍 Code Review Guidelines

### For Contributors
- **Self-review** your code before submitting
- **Test thoroughly** on different systems
- **Write clear** commit messages
- **Keep changes focused** on single issues

### For Reviewers
- **Be constructive** and helpful
- **Test the changes** locally
- **Check documentation** updates
- **Verify performance** impact

## 🏷️ Release Process

### Version Numbering
- **Major** (1.0.0): Breaking changes, major features
- **Minor** (0.1.0): New features, backwards compatible
- **Patch** (0.0.1): Bug fixes, small improvements

### Release Checklist
- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Test on multiple platforms
- [ ] Update documentation
- [ ] Create GitHub release
- [ ] Tag the release

## 🎨 Design Principles

### Code Quality
- **Readability**: Clear, self-documenting code
- **Modularity**: Separate concerns and responsibilities
- **Performance**: Optimize for real-time processing
- **Reliability**: Handle errors gracefully

### User Experience
- **Simplicity**: Easy to install and use
- **Feedback**: Clear status and error messages
- **Flexibility**: Configurable parameters
- **Documentation**: Comprehensive guides

## 🆘 Getting Help

### Community Support
- [GitHub Discussions](https://github.com/yourusername/drone-obstacle-detection/discussions)
- [Issues Page](https://github.com/yourusername/drone-obstacle-detection/issues)

### Development Questions
- Check existing issues and discussions
- Provide minimal reproducible examples
- Include relevant system information
- Be specific about the problem

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

Thank you for helping make this project better! 🚀