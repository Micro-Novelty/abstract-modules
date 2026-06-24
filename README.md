# Abstract-modules 
[=] Introduction:
abstract-modules repository provides the New AbstractIntegratedModule library dedicated for the latest versions.
This repository also Contains Python Wheels for:
 - x86_64, aarch64 architecture (musllinux v1.2+ and manylinux v2.17+), supports python 3.10, 3.11, 3.12 only.
 - macOS (v10.9+) architecture, supports python 3.10, 3.11, 3.12 only.
   
[=] Short Description of AbstractIntegratedModule library:
- Development Stage on PyPi: 0.7.6 Official Release.
- Author and Maintainer: Micro-Novelty and EpsitronNet-bot.
- library Source-Code is Open-sourced with MIT License.
- Purpose: Specifically Designed for providing Non-LLM AI Agent Framework for edge Devices, Optimized for ARM64 architecture.
  
- Main Library installation:
  ```bash
   pip install AbstractIntegratedModule
   python -m install AbstractIntegratedModule
   ```

- Library installation if you dont have aarch64 setup, you can download the correct wheel for your setup in this repository or by using pip:
  - ✨ use pip for downloading the correct wheels for your setup:
  - ```
    pip install abstractintegratedmodule --find-links https://github.com/Micro-Novelty/abstract-modules/releases  --break-system-packages
    ```
    
- Github Link (for Visiting and cloning the Main Repo)
- https://github.com/Micro-Novelty/IntegratedPipeline-Specialized-Non-LLM-AI-Agent-Framework

- The AbstractIntegratedModule library also includes precompiled binaries for:
- aarch64 manylinux (accepts version 2.17+) architecture, accepts python version 3.10, 3.11 only.
- aarch64 musllinux (accepts version 1.2+) architecture, accepts python version 3.10, 3.11 only.
- Windows 64 bit architecture (only python 3.13 only)
- tar.gz / WinRAR file for multi-platform support.

[=] Note:
- If your setup has matched the required hardware version and python version, you could directly install AbstractIntegratedModule library via pip install

- Proven Capabilities:

  - The library has been thoroughly tested in Multiple Environments from Windows to ARM64 Environment. The library is now Robust for Wider use and Deployment.
  - Proven Works on ARM64 Environment, Training and Prediction works efficient on Docker ARM64 environment with QEMU, good parallelizing behavior is guaranteed.
  - P2P Works efficiently in ARM64 Docker + QEMU, No conflicting socket and all prediction works efficiently.
  - AWE setup Proven Efficient on Hard-uncontrolled dataset such as Activity Recognition from the given Database.
  - LSTM is Optimized efficiently for scarce data with AWE method.
  - Robust Advanced prediction capabilities proven effective on ARM64 Using MLP + LSTM Architectures.
  - Transformer Optimized using Cython, to reduce Memory overhead and Reduce CPU Usage, With Reduced Training Time.
- Changelog:
    v0.7.6:
    [=] New features:
       - Adding Optimization and refinements for parsing JSON in Rust.
       - Ensure robustness in calculating Samples similarity to prevent misleading answers in different environment.

