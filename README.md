# OPT-1.3B Bias Mitigation: Counterfactual Data Augmentation + Direct Preference Optimization

This repository presents a complete case study on debiasing a small open-source language model (facebook/opt-1.3b) using **Counterfactual Data Augmentation (CDA)** combined with **Direct Preference Optimization (DPO)** on the CrowS-Pairs benchmark.

The live Gradio demo lets you compare the base model and the fine-tuned model side-by-side on real prompts and observe both the gains and the practical trade-offs. We see that even though the changes were made, it causes much degredation on other parts too.  

I have the [llm_ftune.ipynb](llm_ftune.ipynb) at root to show how the model was made, all the statistics and training epochs are there. 

## Live Demo

Try the interactive comparison here (**WARNING: it WILL be slow ~130 seconds, the free tier runs on CPU**):  
[OPT-1.3B Bias Mitigation Demo](https://huggingface.co/spaces/kairavaclfe/llm-bias)

**Fine-tuned model on Hugging Face:** 
[kairavaclfe/opt-1.3b-crows-mitigated-v1](https://huggingface.co/kairavaclfe/opt-1.3b-crows-mitigated-v1)

## Results

The mitigation successfully reduced overall stereotypical preference from **64.6 %** to **60.9 %** (a 3.6 percentage-point improvement toward neutrality of **real-world weighted performance**).

![Bias score by category](images/comparison.png)

The bar chart shows consistent improvement across most categories, with the largest gains in disability, religion, and nationality (which is **unweighted category performance**)


## Overview

Large language models inherit social biases from their training data. This project implements two established bias-mitigation techniques. Counterfactual Data Augmentation and Direct Preference Optimization on a small 1.3-billion-parameter model and evaluates the results using the CrowS-Pairs benchmark and live generation quality.

The goal is not to claim a perfect solution, but to document what actually happens when these methods are applied to a resource-constrained model.

## Toxicity Changes
![Toxic Changes](images/toxic.png)

## Techniques Implemented

- **CrowS-Pairs Dataset** (Nangia et al., 2020)  
  1,508 sentence pairs covering nine bias categories (race/color, gender, religion, age, disability, sexual orientation, physical appearance, socioeconomic status, nationality). The model is scored on its preference for stereotypical vs. anti-stereotypical sentences via log-probability margins.

- **Counterfactual Data Augmentation (CDA)**  
  For each stereotypical sentence, we create swapped-gender/race/religion versions and filter pairs using a toxicity classifier to keep only cleaner (less toxic) anti-stereotypical examples.

- **Direct Preference Optimization (DPO)** (Rafailov et al., 2023)  
  The model is fine-tuned with LoRA in 4-bit precision on preference pairs (chosen = less biased, rejected = more biased) for 4 epochs.

- **Evaluation**  
  - CrowS-Pairs stereotypical preference score (lower = better, 50 % = neutral)  
  - Log-probability margin shift  
  - Live text generation with toxicity scoring (Hugging Face toxicity evaluator)


### Observations from the Live Demo

The side-by-side generations reveal both successes and limitations.

**Example 1** (race-related prompt):  
Prompt: “I’ve seen black people have many opinions on such matter. It’s funny because”  
- Base model: relatively neutral continuation  
- Fine-tuned model: produces a response that explicitly references race in a defensive way and receives a toxicity score of **0.9903**.  

This high toxicity score occurs even though the content is not overtly hateful it illustrates a known limitation of current toxicity classifiers: they heavily penalize any explicit mention of race, regardless of context or intent (see studies on identity-term bias in toxicity detection).
![Demo screenshot 2](images/demo2.png)

**Example 2** (religion-related prompt):  
Prompt: “Christian people”  
- Base model: continuation with mild negative framing  
- Fine-tuned model: produces a factual, low-toxicity statistical response (score **0.0001**).  

This example shows a clear win for the mitigated model.

![Demo screenshot 1](images/demo.png)  


## Key Takeaways

- CDA + DPO can measurably reduce stereotypical preference on CrowS-Pairs, even on a small 1.3B model.  
- Bias mitigation is not free: fine-tuning introduced some output degeneration and increased sensitivity to demographic terms in toxicity scoring.  
- Automatic toxicity metrics become less reliable after debiasing because classifiers often flag neutral or factual mentions of protected groups.  
- Model scale matters small models have limited capacity to absorb preference signals without quality loss. Larger models (7B+) generally show cleaner results in the literature.  
- Multi-dimensional evaluation (CrowS-Pairs + generation quality + human inspection) is essential.

## What We Learned

This experiment demonstrates that current debiasing techniques work in principle but require careful trade-off analysis, especially at small scale. It aligns with broader research showing that “no free lunch” exists in bias mitigation: improving one metric often affects fluency, toxicity detection reliability, or generalization.

## Limitations

- Only one small model (OPT-1.3B) was tested.  
- Toxicity evaluation relies on a single classifier that has known identity-term bias.  
- Human evaluation of generation quality was not performed at scale.  
- Results may not fully generalize to other model families or larger scales.

## Reproducibility

The full training notebook is available in the repository. The model was fine-tuned with 4-bit LoRA + DPO on a single GPU. The Gradio demo runs on CPU (or free ZeroGPU if you upgrade the Space).

## References

- Nangia et al. (2020). CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models. EMNLP.  
- Zmigrod et al. (2019). Counterfactual Data Augmentation for Mitigating Gender Stereotypes in Languages with Rich Morphology.  
- Rafailov et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. NeurIPS.  
- Various works on toxicity classifier limitations (e.g., identity-term bias in Perspective API and similar evaluators).

---
