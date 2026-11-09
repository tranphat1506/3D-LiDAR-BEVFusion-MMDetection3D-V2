# Contributing to 3D-LiDAR-BEVFusion-MMDetection3D-V2

Thank you for your interest in contributing to this project. This repository provides a dedicated, modernized framework for 3D LiDAR Object Detection using BEVFusion on the nuScenes dataset.

Contributions from the community help keep this project reliable, efficient, and accessible to researchers, engineers, and students in autonomous driving.

---

## 1. Contribution Principles

- **Reproducibility**: Changes must preserve reproducible behavior across local environments (Linux, Windows) and cloud platforms (Google Colab).
- **Transparency**: Clear explanations of changes, test logs, and environment configurations must accompany every contribution.
- **Academic and Engineering Quality**: Code should be well-commented, strictly conform to PEP 8, and use clean, professional documentation without emojis.

---

## 2. Ways to Contribute

You can contribute to this project in several ways:

1. **Bug Reports**: Identify and report runtime errors, installation mismatches, or unexpected visualization behavior using the GitHub Issue tracker.
2. **Feature Requests**: Propose enhancements such as new sensor fusion modes, alternative 3D detection heads, or optimized data loaders.
3. **Documentation**: Improve installation guides, clarify mathematical formulations, refine docstrings, or document new deployment environments.
4. **Code Improvements**: Refactor existing modules, enhance error handling, or simplify pipeline execution.
5. **Performance Optimization**: Accelerate point cloud preprocessing, improve GPU batch inference throughput, or optimize video rendering speed.

---

## 3. Prerequisites and Development Environment

To ensure compatibility with the pre-compiled CUDA operations and sparse convolution libraries, your development environment should match the verified specifications:

- **Operating System**: Ubuntu 22.04 LTS (recommended) or Windows 10/11
- **Python**: 3.10 (managed via Pixi, Conda, or venv)
- **PyTorch**: 2.1.2 with CUDA 11.8 (`torch==2.1.2+cu118 torchvision==0.16.2+cu118`)
- **MMCV**: 2.1.0 (built with CUDA 11.8)
- **MMEngine**: 0.10.7
- **MMDetection3D**: 1.4.0 (`dev-1.x` branch)
- **Key Libraries**: `numpy==1.26.4`, `pandas==2.2.2`, `opencv-python`, `open3d>=0.17.0`, `matplotlib`

Before developing, review the installation steps in [README.md](README.md) and understand the upstream patching mechanism detailed in [custom_changes/README.md](custom_changes/README.md).

---

## 4. Fork-and-Pull-Request Workflow

We follow standard GitHub fork-and-pull-request conventions:

1. **Fork the Repository**: Create your personal fork on GitHub.
2. **Clone Locally**:
   ```bash
   git clone https://github.com/<your-username>/3D-LiDAR-BEVFusion-MMDetection3D-V2.git
   cd 3D-LiDAR-BEVFusion-MMDetection3D-V2
   ```
3. **Configure Upstream Remote**:
   ```bash
   git remote add upstream https://github.com/<original-owner>/3D-LiDAR-BEVFusion-MMDetection3D-V2.git
   ```
4. **Create a Feature Branch**: Always branch from `main` using descriptive branch names:
   ```bash
   git checkout -b feature/optimize-point-cloud-loader
   ```
5. **Implement Changes**: Develop your changes, adhering to the coding standards below.
6. **Test Locally**: Thoroughly execute manual verification (see Section 8).
7. **Commit Changes**: Use Conventional Commits formatting (see Section 6).
8. **Push to Fork**:
   ```bash
   git push origin feature/optimize-point-cloud-loader
   ```
9. **Open a Pull Request**: Submit your pull request against the `main` branch of the upstream repository using the provided template.

---

## 5. Branch Naming Conventions

Use clear prefixes to indicate the intent of your branch:

- `feature/<name>`: New features, capabilities, or pipeline enhancements (e.g., `feature/point-cloud-sampling`)
- `bugfix/<name>`: Non-urgent bug fixes and error resolutions (e.g., `bugfix/fix-matrix-dimension-swap`)
- `hotfix/<name>`: Critical fixes resolving breaking crashes or build failures (e.g., `hotfix/cxxabi-linker-error`)
- `docs/<name>`: Documentation, README, or docstring updates (e.g., `docs/update-colab-instructions`)

---

## 6. Commit Message Conventions

We recommend following the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` Adds a new feature or functionality
- `fix:` Fixes a bug or resolves an error
- `docs:` Changes to documentation, guides, or markdown files
- `refactor:` Code refactoring that neither fixes a bug nor adds a feature
- `test:` Adding or updating manual verification scripts
- `chore:` Maintenance tasks, gitignore adjustments, or dependency version updates

### Example Commit Messages:
```
feat: add multi-sweep lidar point cloud accumulator
fix: correct coordinate axis mapping in bev video renderer
docs: clarify nvcc compiler compatibility flags in custom_changes
refactor: streamline argument parsing in bev_inferencer
```

---

## 7. Coding Guidelines

When contributing Python code to this repository:

1. **PEP 8 Compliance**: Follow PEP 8 style recommendations for formatting, variable naming, and indentation (4 spaces).
2. **No Hardcoded Machine Paths**: Never use user-specific absolute paths (such as `C:\Users\username\` or `/home/username/`). Always use relative paths computed dynamically:
   ```python
   base_dir = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_dir, "data", "nuscenes")
   ```
3. **Coordinate System Consistency**:
   - nuScenes uses the **RFU (Right-Forward-Up)** coordinate system:
     - $X$: Right (Lateral)
     - $Y$: Forward (Longitudinal direction of vehicle travel)
     - $Z$: Up (Vertical)
   - Any BEV projection must respect this convention: vehicle heading points upward on the vertical axis ($v = cy - Y \times \text{scale}$).
4. **Preserve Fault-Tolerant Patterns**: When modifying inference scripts, retain the self-healing fallback mechanisms that catch `box_mode_3d` postprocessing exceptions and call `inferencer.model.test_step()` directly.
5. **Upstream Modifications**: Any bug fixes or changes that affect files inside `external/mmdetection3d/` must be mirrored in `custom_changes/` and documented in `custom_changes/README.md`.
6. **No Emojis in Documentation**: Maintain an academic, professional style in all Markdown documentation, docstrings, and commit messages.

---

## 8. Testing and Validation Requirements

> **Note on Automated Testing**:  
> This repository currently does not have an automated testing suite (such as `pytest` or a continuous integration runner). Therefore, all contributors are required to validate changes manually using the procedures below before submitting a pull request.

### Mandatory Verification Steps:

1. **Syntax Compilation**: Ensure all modified files compile without syntax or import errors:
   ```bash
   python3 -m py_compile <modified_file>.py
   ```
2. **Quick Demo Verification**: Verify that basic 10-frame inference executes properly:
   ```bash
   python setup_demo.py
   python visualize_demo.py
   ```
   Confirm that prediction JSON files are generated in `demo/outputs/preds/` and images are saved in `demo/outputs/vis/`.
3. **Batch Inference Check**: If modifying core inference logic, run a test run on nuScenes data:
   ```bash
   python run_inference_all.py
   ```
4. **Video and Visualization Check**: If modifying visualization logic, confirm output validity:
   ```bash
   python create_full_nuscenes_video.py
   ```
   Check that generated video files play correctly and bounding box orientations accurately match camera views.

Include the commands you ran and your environment specifications in the Pull Request description.

---

## 9. Pull Request Requirements and Review Process

When opening a Pull Request:

1. **Use the Template**: Complete all sections of the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).
2. **Link Related Issues**: Explicitly cite any related issues (e.g., `Fixes #12`).
3. **Provide Verification Proof**: Document the hardware, OS, and commands used to verify the code.
4. **Keep PRs Focused**: Avoid bundling unrelated changes into a single pull request. Separate refactoring from new features.
5. **Address Feedback**: Maintainers will review your submission and may suggest modifications. Please respond constructively to feedback.

---

## 10. Reporting Problems and Requesting Help

If you encounter issues or have questions:

- **Bug Reports**: Open an issue using the [Bug Report form](.github/ISSUE_TEMPLATE/bug_report.yml).
- **Feature Proposals**: Open an issue using the [Feature Request form](.github/ISSUE_TEMPLATE/feature_request.yml).
- **General Questions**: Post your inquiry on the repository's GitHub Issues or Discussions section (if enabled).

Please search existing issues before opening a new one to avoid duplicate tickets.
