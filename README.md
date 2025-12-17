# Auditing Occupational Gender Bias in Large Language Models

**A quantitative analysis of gender stereotypes in AI-generated text using statistical methods and NLP techniques**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HPC: Alpine](https://img.shields.io/badge/HPC-Alpine-green.svg)](https://www.colorado.edu/rc/alpine)

---

## Overview

This project investigates whether large language models perpetuate occupational gender stereotypes when generating text about traditionally male-dominated (engineering, construction) versus female-dominated (nursing, caregiving) professions. Using Meta's **Llama-3.1-8B-Instruct** model and the **BOLD dataset**, we analyzed 1,525 model-generated completions through three complementary bias detection methods.

### Key Finding
When the model uses gendered language (4.3% of cases), it exhibits **significant stereotypical bias**: male-dominated professions are **24.65× more likely** to be described with masculine terms compared to female-dominated professions (*p* < 0.001).

Large language models are increasingly deployed in:
- **Automated job description generation**
- **Resume screening and ranking systems**
- **Career counseling chatbots**
- **Educational content creation**

Even infrequent gender bias in these applications can reinforce occupational segregation and limit opportunities for underrepresented groups.

### Methods Implemented

1. **Lexicon-Based Gender Analysis**
   - Custom gender word lexicons (13 male terms, 13 female terms)
   - Per-prompt word counting with statistical aggregation
   - Handles zero-inflated distributions common in bias research

2. **Pointwise Mutual Information (PMI) Co-occurrence Analysis**
   - Measures contextual association strength between gender and occupation terms
   - 5-word sliding window for capturing local discourse patterns
   - Reveals *how* gender and profession are linked, not just *if* they appear

3. **Comprehensive Statistical Testing**
   - Chi-square test of independence (χ² = 62.86, *p* < 0.001)
   - Welch's t-test for unequal variances
   - Cohen's d effect size calculation
   - Odds ratio with 95% confidence intervals
   - Designed for unbalanced datasets (1,172 vs. 353 samples)

### Infrastructure

- **HPC Platform**: University of Colorado Boulder's Alpine cluster (NSF Award #2201538)
- **GPU**: NVIDIA A100 (40GB VRAM)
- **Model**: Meta Llama-3.1-8B-Instruct (8 billion parameters, float16 precision)
- **Dataset**: BOLD (Bias in Open-Ended Language Generation) - 1,525 profession prompts across 224 unique occupations

---

## Results

| Metric | Male-Dominated Professions | Female-Dominated Professions |
|--------|---------------------------|------------------------------|
| **Total gendered words** | 64 male, 22 female | 7 male, 64 female |
| **Gender ratio (M:F)** | 2.9:1 | 1:9.1 |
| **Completions with gendered language** | 2.6% | 9.9% |
| **Odds Ratio** | 24.65× more likely to use male words [95% CI: 10.07-60.37] |

**Statistical Significance**: 
- Chi-square test: *p* = 2.22 × 10⁻¹⁵ (highly significant)
- PMI comparison: *p* = 0.018 (significant)

**Interpretation**: Most completions are gender-neutral, but when gender appears, it strongly reinforces stereotypes.

---

### Prerequisites
```bash
# Python 3.10+ with CUDA-enabled PyTorch
pip install torch transformers accelerate pandas numpy scipy matplotlib seaborn
```

### Execution
```bash
# On HPC with GPU
python scripts/final_bias_analysis.py

# Runtime: ~20-40 minutes for full dataset
# Outputs saved to results/ directory
```

## Author

**Aidan Coughlin**  
https://www.linkedin.com/in/aidan-coughlin/ | aico7866@colorado.edu 

*Conducted as part of INFO 4615 at University of Colorado Boulder*

---

## Acknowledgments

- **Alpine HPC**: University of Colorado Boulder Research Computing (NSF Award #2201538)
- **BOLD Dataset**: Amazon Science & University of Washington
- **Llama Model**: Meta AI
- **Course Instruction**: Lex Beattie

---

## Contact

Interested in discussing this project, bias detection methodologies, or AI fairness research? Feel free to reach out!

- **GitHub Issues**: For technical questions or code improvements
- **Email**: [aico7866@colorado.edu]
- **LinkedIn**: https://www.linkedin.com/in/aidan-coughlin/

---

*Last Updated: December 2025*
