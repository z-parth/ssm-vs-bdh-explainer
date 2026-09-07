# SSM vs. BDH-GPU: Sequence Recall Under Noise

**Live Demo:** https://ssm-vs-bdh-explainer-77xthpahljef8e9qemm2sf.streamlit.app/

## Core Claim
 When exposed to an increasing stream of distractor noise tokens, a continuous-time SSM's fixed hidden state decays associative recall accuracy faster than BDH-GPU's additive synaptic weight matrix.

## Project Overview
This repository was built for the DataForge 2026 Pathway Track. It provides a localized, interactive sandbox to test how different sequence modeling architectures handle associative recall when flooded with distractor tokens.

**Disclaimer:** This is a 16-dimensional mathematical toy model built for educational purposes. It isolates the Hebbian/sparse memory mechanics to demonstrate a specific concept. It is not the official Pathway BDH or BDH-CQ production model.

## Running Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/z-parth/ssm-vs-bdh-explainer.git
   cd ssm-vs-bdh-explainer
   ```
2. Set up the environment:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
3. Install dependencies and run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Repository Structure
- `app.py`: Streamlit frontend and interactive sandbox.
- `src/ssm_engine.py`: Simulates continuous-time discretized SSM updates 
- `src/bdh_engine.py`: Simulates low-rank linear attention via sparse Hebbian associative writes 
- `src/benchmark.py`: Test harness for injecting noise and calculating cosine similarity.

## References & Disclosures

**References**
1. **Kosowski, A., Uznański, P., et al. (2025).** *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.* arXiv:2509.26507. 
2. **Jelassi, S., et al. (2024).** *Repeat After Me: Transformers are Better than State Space Models at Copying.* arXiv:2402.01032. 
3. **Eyuboglu, S., Arora, S., et al. (2023).** *Zoology: Measuring and Improving Recall in Efficient Language Models.* arXiv:2312.04927. 
4. **Gu, A., & Dao, T. (2023).** *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752. 

**Disclosures**
- **AI Assistance:** LLMs (Claude/Gemini) were used to generate Streamlit UI boilerplate and layout formatting. All core math, matrix operations, and claims were manually written and verified.
- **Data/Weights:** No proprietary pretrained weights are used. The benchmark runs entirely on synthetically generated random uniform vectors.