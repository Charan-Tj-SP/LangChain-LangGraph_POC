# 🚀 DeepAgents Installation Guide

Welcome to the official developer guide. Select your preferred package manager from the tab layout below to view the exact install command.

---

### 📦 Choose Your Environment

=== "pip"

    Use the native Python package installer to download the package from PyPI:

    ```bash
    pip install deepagents
    ```

=== "uv"

    Use the modern, fast rust-based environment tool to install the package:

    ```bash
    uv pip install deepagents
    ```

=== "Poetry"

    If you manage your project using Poetry dependencies, run this inside your project root:

    ```bash
    poetry add deepagents
    ```

---

### 🔍 Quick Verification
Once your chosen installation finishes, verify the package setup by running this quick test script in your environment:

```python
import deepagents
print("DeepAgents successfully installed! Version:", deepagents.__version__)
```

