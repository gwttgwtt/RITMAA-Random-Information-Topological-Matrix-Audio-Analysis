# RITMAA: Random Information Topological Matrix Audio Analysis

[![Signal Processing](https://img.shields.io/badge/Journal-Signal%20Processing%20%28Elsevier%29-blue.svg)](https://www.journals.elsevier.com/signal-processing)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

**RITMAA** is a deterministic, high-performance computing framework designed for macrostructural recovery, topological alignment, and compositional lineage reconstruction within high-entropy audio signal systems. 

Unlike traditional Music Information Retrieval (MIR) toolkits that rely on shallow acoustic similarity or black-box deep learning embeddings, **RITMAA** operates strictly on information-theoretic observables combined with Random Matrix Theory (RMT) and elastic time-warping in Skorokhod spaces. The system is specifically engineered to bypass industrial mastering noise (multi-band brick-wall compression, transients) and isolate the underlying invariant geometric skeleton ("compositional germ") of entire discographies or large-scale musical corpora.

---

## 🛠 Core Computational Architecture

The framework is divided into three distinct operational layers, allowing for non-linear analysis and information-driven signal synthesis:

### 1. Vectorized Information Space Mapping
* **Shannon Distribution Entropy ($H$):** Measures macrostructural structural uncertainty over adaptive segments ($256 \le N_{seg} \le 65536$).
* **Gradient-Based Fisher Information ($F_{max}$):** Tracks micro-temporal boundary volatility, percussive attacks, and vocal flow transients.
* **Permutation Entropy over RMS Envelopes ($\text{PE}_{RMS}$):** Acts as an amplitude demodulator, extracting syntactic phrasing complexity and speech-rhythm flow directly from the acoustic envelope while normalizing for mastering gain variance.

### 2. Dual Marchenko–Pastur Eigen-Decomposition
* **Global Mode:** Builds an $M \times T$ (songs × segments) observation space, projecting the cross-track empirical covariance matrix against the analytical RMT noise bound:
    $$\lambda_+ = \sigma^2(1 + \sqrt{M/T})^2$$
    Eigenvalues exceeding $\lambda_+$ are decoupled as deterministic structural archetypes, stripping away the producer's acoustic signature.
* **Vertical Packet Mode:** Computes localized micro-RMT thresholds across all tracks simultaneously at a single timestamp, mapping the continuous time-evolution of collective structural events.

### 3. Elastic Resynthesis and Automated Deviation Radar
* **Information-Driven Audio Filtering:** Uses the reconstructed clean matrix $X_{rec}$ to dynamically scale segmental gain, outputting audio artifacts filtered by collective archetype presence.
* **MP Archetype Collage:** Executes maximum-likelihood tracking across the structural manifold ($t_{winner} = \arg\max_i [X_{rec}(i,t)]$) to synthesize a single continuous generative composition using physical segments exclusively from the dominant tracks.
* **Skorokhod Discrepancy Warping:** Computes micro-temporal synchronization offsets ($\tau$) against an architectural blueprint. A non-linear **Hysteresis Deviation Detector** maps the structural drift to throw an automated radar-style alarm whenever a track breaks free from the collective archetype.

---

## 🚀 Repository Directory Structure

* `view_Sound_MP_CPU_1_MAIN_EXPORT.py`: Main GUI Application. Handles bulk audio extraction, dual-mode Marchenko–Pastur decomposition, CSV/PDF academic report generation, and triggers the audio filtering / collage engines.
* `skorokhod_viewer.py`: Elastic Warping Visualizer. Loads exported `.npz` structural matrices, executes Numba-accelerated Sinkhorn optimal transport, tracks Scale-Stabilized Soliton Coherence Indices ($SCI$), and plots Korolyuk phase portraits.
* `Entropy_Tester.py`: Vectorized Single-File Explorer. A standalone, high-speed validator featuring the newly integrated 9-row Shannon-Fisher-Permutation matrix stack.

---

## 📦 Installation

Ensure you have a clean Python installation (preferably inside a virtual environment).

```bash
# Clone the repository
git clone [https://github.com/gwttgwtt/RITMAA-Random-Information-Topological-Matrix-Audio-Analysis.git](https://github.com/gwttgwtt/RITMAA-Random-Information-Topological-Matrix-Audio-Analysis.git)
cd RITMAA-Random-Information-Topological-Matrix-Audio-Analysis

# Install dependencies
pip install numpy scipy pandas soundfile matplotlib
