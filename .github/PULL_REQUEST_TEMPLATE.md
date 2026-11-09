## Description

Please provide a concise summary of the proposed changes, the motivation behind them, and how they improve the repository.

---

## Related Issue

Link the issue this pull request addresses (if applicable):
- Closes #
- Fixes #
- Relates to #

---

## Type of Change

Please select the relevant option(s):
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Documentation update (improving guides, README, or docstrings)
- [ ] Performance optimization (improving inference or visualization speed)
- [ ] Code refactoring (structure, cleaning, or style improvements without logic changes)
- [ ] Upstream patch update (changes to `custom_changes/` or build configuration)

---

## Changes Made

- Detailed bullet point 1
- Detailed bullet point 2
- Detailed bullet point 3

---

## Testing Performed

Describe the testing performed to verify that the changes work as intended. Specify the hardware, operating system, and commands executed.

- **Environment**:
  - OS: [e.g., Ubuntu 22.04 LTS, Windows 11, Google Colab]
  - Python: [e.g., 3.10.22]
  - PyTorch: [e.g., 2.1.2+cu118]
  - CUDA / GPU: [e.g., CUDA 11.8 / NVIDIA Tesla T4]

- **Commands Executed**:
  ```bash
  # Example verification commands
  python3 -m py_compile <modified_file>.py
  python setup_demo.py
  python visualize_demo.py
  ```

- **Observations / Results**:
  - Describe the output or confirm successful execution without runtime errors.

---

## Documentation Updates

- [ ] Updated `README.md` if user-facing commands or architectures changed.
- [ ] Updated `custom_changes/README.md` if modifications touch `custom_changes/`.
- [ ] Added or updated Python script docstrings and inline comments.

---

## Contributor Checklist

- [ ] My code adheres to the project's coding standards and PEP 8 guidelines.
- [ ] I have not introduced hardcoded machine-specific paths (e.g., `/home/username/` or `C:\Users\username\`).
- [ ] All new and modified Python files pass syntax compilation (`python3 -m py_compile <file>`).
- [ ] I have verified that any relative file paths and imports remain functional.
- [ ] If modifying upstream patches, changes are mirrored in `custom_changes/`.
- [ ] No sensitive credentials, private API keys, or large binary datasets are committed.
