## Description

Describe your changes clearly and concisely. Include the motivation and context for the change.

## Type of Change

<!-- Check all that apply -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would break existing functionality)
- [ ] Documentation update
- [ ] Hardware/RTL change
- [ ] CI/CD or build system change

## How Has This Been Tested?

Describe the tests you ran to verify your changes. Provide instructions so reviewers can reproduce.

```bash
# Example:
pytest tests/ -v
```

## Checklist

- [ ] Tests pass (`pytest tests/ tests_enhanced/ -v`)
- [ ] Lint passes (`ruff check src/ tests/ tests_enhanced/`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG.md updated (if applicable)
- [ ] No existing files were modified unintentionally
