# Security Policy

## 1. Supported Versions

Because this is an open-source research and educational project, formal long-term support (LTS) releases are not maintained. Security updates and bug fixes are applied directly to the active development branch:

| Branch | Supported | Notes |
| :--- | :--- | :--- |
| `main` | Yes | Active development and latest validated patches |
| Historical forks / untracked branches | No | Please rebase or cherry-pick fixes onto `main` |

Contributors and users are strongly encouraged to keep their local clones synchronized with the latest commits on the `main` branch.

---

## 2. Reporting a Vulnerability

If you discover a security vulnerability, security flaw, or potential exploit in this repository, **please do not disclose it publicly** via GitHub Issues, Discussions, or public pull requests.

### Preferred Reporting Method

Please use **GitHub Private Vulnerability Reporting**:
1. Navigate to the main page of the repository on GitHub.
2. Click on the **Security** tab.
3. Select **Report a vulnerability** under the "Private vulnerability reporting" section.
4. Fill in the required details and submit the advisory draft.

This ensures the vulnerability is communicated privately to project maintainers and can be remediated before public disclosure.

If private vulnerability reporting is not accessible or enabled on this repository, please contact the repository owner privately through their verified GitHub profile contact details.

---

## 3. Information to Include in a Report

To help maintainers evaluate and resolve the issue quickly, please include:

- **Vulnerability Description**: A clear and comprehensive summary of the vulnerability.
- **Affected Components**: Specify the exact scripts, configuration files, or CUDA operator compilation files involved.
- **Attack Vector & Severity**: How an attacker could exploit the issue and the potential impact (e.g., arbitrary code execution during inference, local privilege escalation, or path traversal).
- **Proof of Concept (PoC)**: Minimal, step-by-step instructions or sample scripts required to reproduce the behavior.
- **Proposed Fix or Workaround**: Any remediation suggestions or code patches, if you have identified them.

---

## 4. Remediation Process

Upon receiving a private security report:

1. **Investigation**: Maintainers will review the report and attempt to reproduce the reported behavior in an isolated environment.
2. **Coordination**: If verified, maintainers will work on a patch in a private branch or advisory.
3. **Disclosure & Release**: Once resolved, the fix will be committed to `main`, and a security advisory or release note will credit the reporter (unless anonymity is requested).

---

## 5. Security Best Practices for Machine Learning & Datasets

Users and contributors must adhere to standard data and model security hygiene:

- **Never Commit Sensitive Credentials**: Do not commit API tokens, cloud storage keys, or SSH credentials to this repository.
- **Sanitize File Paths**: Do not hardcode local operating system credentials or internal corporate network paths.
- **Verify Untrusted Checkpoints**: Machine learning model weights (`.pth` files) serialized with `torch.save` use Python's `pickle` format by default, which can execute arbitrary code upon deserialization. Only load model weights from verified and trusted sources (such as official OpenMMLab distribution mirrors).
- **Protect Private Data**: Ensure proprietary camera images, LiDAR sweeps, or sensitive location metadata are not committed into public version control.
