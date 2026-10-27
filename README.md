# Dataset Size vs. Repetition in LoRA Fine-Tuning: A Fixed-Budget Study

A controlled experiment on a common piece of LoRA advice, "use a small, curated dataset." Fifteen SDXL LoRAs were trained in a full 5 × 3 grid (dataset size × batch size) with the **total training budget fixed at 10,000 samples per run**. The question is a practical one: with a fixed amount of training, is it better to show the model a few images many times or many images a few times?

Hong & Choo (2025) name this trade-off between batch size, epochs and repeats under a fixed budget as an open question in their LoRA study; this experiment tests it directly.

## Key results

- **Smaller datasets score higher on CLIP similarity to held-out reference images.** The mean rises from 0.848 (400 images, 25 epochs) to 0.878 (25 images, 400 epochs). Across the 15 runs, r = +0.83 with log(epochs), permutation p < 0.001. The 25-vs-400 gap is about 6× the run-to-run spread within a dataset size, and the trend survives dropping either extreme group.
- **The top-scoring run did not look best.** Run 14 (25 images) ranked first of 15 on both CLIP and LPIPS similarity, yet showed visible artifacts that a 400-image run did not. This is one unblinded comparison, but it is a concrete case of automated metrics and visual quality disagreeing.
- **Batch size (1, 2, 4) had no measurable effect** at matched repetitions per image: r = 0.00 with CLIP similarity, and adding batch size to the regression explains no additional variance.
- **Open question:** *why* small datasets score higher. Under a fixed budget, epochs = 10,000 ÷ dataset size, so fewer images always means more repetitions. This design cannot separate the two explanations (see [What this design can and can't show](#what-this-design-can-and-cant-show)).

## What this design can and can't show

Total training = dataset size × epochs, so any experiment has to tie two of the three together:

| Design | Held fixed | Tied together |
|---|---|---|
| **This study** | total training | dataset size ↔ epochs |
| Same dataset, train longer | dataset size | epochs ↔ total training |
| Same epochs, more images | epochs | dataset size ↔ total training |

This study answers the practitioner's question: **given a fixed budget, how should it be split?** It cannot say whether the CLIP gain comes from repetition (each image seen more often) or from dataset size itself (a smaller, more uniform target). Every small-dataset run is also a high-repetition run.

Batch size, by contrast, varies independently. Within each dataset size, epochs are constant while batch size and step count change 4×, so the batch-size result is a clean comparison.

## Experimental design

| Data size | Epochs | Batch 4 | Batch 2 | Batch 1 |
|---:|---:|---|---|---|
| 400 | 25 | run 1 (2,500 steps) | run 2 (5,000) | run 3 (10,000) |
| 200 | 50 | run 4 | run 5 | run 6 |
| 100 | 100 | run 7 | run 8 | run 9 |
| 50 | 200 | run 10 | run 11 | run 12 |
| 25 | 400 | run 13 | run 14 | run 15 |

Step counts are the same in every row: 2,500 / 5,000 / 10,000 for batch 4 / 2 / 1.

### Data

- **Subject:** one anime character (Kiss-Shot Acerola-Orion Heart-Under-Blade, *Monogatari*), chosen because base SDXL renders her poorly.
- **Training sets:** 400 hand-curated images, randomly halved to 200, 100, 50 and 25. Each smaller set is a subset of the larger ones.
- **Reference set:** 20 images of the character that appear in **none** of the training sets, so no configuration gets an advantage from having trained on them.
- Images were collected from public fan-art and screenshot sources and are not redistributed because of copyright.

### Training

All LoRAs were trained on Illustrious-XL v0.1 with CivitAI's LoRA trainer (a Kohya `sd-scripts` fork), using identical settings apart from dataset, batch size and steps:

| Setting | Value | Setting | Value |
|---|---|---|---|
| Network dim / alpha | 32 / 16 | Optimizer | Adafactor |
| UNet LR | 1.2e-4 | Text encoder LR | 5e-5 |
| LR schedule | cosine with restarts (3 cycles) | Min-SNR gamma | 5 |
| Resolution | 2047 (trainer setting; bucketing on) | Clip skip | 2 |
| Caption shuffle / keep tokens | yes / 3 | Flip augmentation | yes |

The learning rate was the same for every batch size, so the batch-size result applies at a fixed learning rate.

### Generation

For each LoRA: 5 prompts × 8 seeds = 40 images at 1024 × 1024, 30 sampling steps, CFG 7.0, with a shared negative prompt. Seeds are identical across runs (base seed 1337 + image index), so every LoRA is evaluated on the same 40 noise samples. Prompts are in [`prompts/test_prompts.json`](prompts/test_prompts.json).

### Metrics

| Metric | Column | Definition |
|---|---|---|
| CLIP similarity | `clip_identity` | For each generated image, cosine similarity (OpenCLIP ViT-H/14, LAION-2B) to its most similar reference image, averaged over the 40 images. Higher = more similar. This is whole-image similarity: outfit, pose and style count, not only character identity. |
| LPIPS distance | `lpips_identity` | For each generated image, LPIPS (VGG) distance to its closest reference image, averaged over the 40 images. Lower = more similar. |
| Diversity | `diversity_lpips` | Mean LPIPS between 120 randomly sampled pairs of generated images from the same run. |
| FID | `fid_vs_refs` | Clean-FID between the 40 generations and the 20 references. **Not meaningful at this sample size** (it estimates a 2,048-dimensional covariance from 20 images); reported for completeness only. |
| Overfitting proxy | `overfit_ratio` | Share of generations with CLIP cosine > 0.35 to any of 64 sampled training images. **Saturated at 1.0 for every run**, so it gives no signal. |

## Results

Full results: [`data/lora_metrics.csv`](data/lora_metrics.csv).

| Run | Data | Batch | Epochs | CLIP sim. ↑ | LPIPS dist. ↓ | Diversity | FID |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 400 | 4 | 25 | 0.847 | 0.527 | 0.546 | 191.7 |
| 2 | 400 | 2 | 25 | 0.843 | 0.523 | 0.529 | 207.7 |
| 3 | 400 | 1 | 25 | 0.853 | 0.524 | 0.548 | 198.3 |
| 4 | 200 | 4 | 50 | 0.865 | 0.509 | 0.497 | 193.5 |
| 5 | 200 | 2 | 50 | 0.858 | 0.508 | 0.520 | 201.0 |
| 6 | 200 | 1 | 50 | 0.855 | 0.509 | 0.524 | 191.9 |
| 7 | 100 | 4 | 100 | 0.859 | 0.516 | 0.532 | 199.4 |
| 8 | 100 | 2 | 100 | 0.853 | 0.519 | 0.536 | 210.3 |
| 9 | 100 | 1 | 100 | 0.859 | 0.519 | 0.523 | 208.0 |
| 10 | 50 | 4 | 200 | 0.858 | 0.537 | 0.564 | 189.1 |
| 11 | 50 | 2 | 200 | 0.860 | 0.518 | 0.538 | 192.5 |
| 12 | 50 | 1 | 200 | 0.868 | 0.527 | 0.560 | 189.5 |
| 13 | 25 | 4 | 400 | 0.876 | 0.529 | 0.566 | 194.5 |
| 14 | 25 | 2 | 400 | **0.886** | **0.505** | 0.540 | 189.1 |
| 15 | 25 | 1 | 400 | 0.872 | 0.521 | 0.580 | 201.6 |

**Means by dataset size**

| Data | Epochs | CLIP sim. ↑ | LPIPS dist. ↓ | Diversity |
|---:|---:|---:|---:|---:|
| 400 | 25 | 0.848 | 0.525 | 0.541 |
| 200 | 50 | 0.860 | 0.509 | 0.513 |
| 100 | 100 | 0.857 | 0.518 | 0.530 |
| 50 | 200 | 0.862 | 0.527 | 0.554 |
| 25 | 400 | 0.878 | 0.518 | 0.562 |

**Means by batch size**

| Batch | CLIP sim. ↑ | LPIPS dist. ↓ | Diversity |
|---:|---:|---:|---:|
| 1 | 0.861 | 0.520 | 0.547 |
| 2 | 0.860 | 0.515 | 0.533 |
| 4 | 0.861 | 0.523 | 0.541 |

### Analysis

All numbers below are reproduced by [`analysis/robustness_checks.py`](analysis/robustness_checks.py).

**Correlations across the 15 runs**

| Metric | vs. log(epochs) | vs. log(batch size) |
|---|---:|---:|
| CLIP similarity | **+0.83** | 0.00 |
| LPIPS distance | +0.11 | +0.16 |
| Diversity | +0.57 | −0.11 |

Since log(epochs) = log(10,000) − log(dataset size), each correlation with log(dataset size) is the same value with the sign flipped. The ±0.83 is one result, not two.

**The CLIP effect is robust.**

- Permutation test (20,000 shuffles): p < 0.001.
- The gap between the 25- and 400-image group means (0.030) is about 6× the mean within-group SD (0.005).
- Dropping the 25-image runs leaves r = +0.68; dropping the 400-image runs leaves r = +0.73.
- The trend is not perfectly smooth (the 100-image mean sits slightly below the 200-image mean); most of the change is at the two ends.

**Batch size adds nothing.**

| CLIP model | R² | Adjusted R² |
|---|---:|---:|
| log(epochs) only | 0.690 | 0.667 |
| log(data) × log(batch), with interaction | 0.690 | 0.606 |

The three-predictor model explains no more variance than epochs alone; its adjusted R² is lower only because of the penalty for extra predictors. Because "epochs only" contains the same information as "dataset size only," this shows that batch size doesn't help. It does not distinguish epochs from dataset size. For LPIPS and FID, neither model explains meaningful variance.

**Other metrics.** LPIPS distance to the references shows no trend with dataset size (R² = 0.01). Smaller-dataset runs get closer to the references in CLIP embedding space without getting closer perceptually. Diversity rises somewhat with epochs (r = +0.57), but that is hard to interpret: LPIPS between generations rises both with healthy variation and with artifacts. With the overfitting proxy saturated, **this study has no working measure of memorization.**

### Visual inspection

Run 14 (25 images, 400 epochs) led on CLIP and LPIPS similarity. Compared side by side with Run 3 (400 images, 25 epochs; the best-scoring 400-image run) on the same prompts and seeds, Run 14 showed edge ringing and oversharpening, anatomical inconsistencies, repeated features and loss of fine detail that Run 3 did not.

This is one pair of runs, judged by the author without blinding, and the two runs also differ in batch size and step count. It shows that a model can lead on these similarity metrics and still look worse, but not how often that happens. Side-by-side images are on the [Hugging Face model card](https://huggingface.co/KS-AO-HUB/lora-dataset-size-vs-epoch-exposure).

## Limitations

- Dataset size and epochs are perfectly confounded by design (see above).
- One subject, one base model, one training budget.
- One run per configuration; no seed replicates.
- 15 runs at 5 dataset sizes; the permutation test treats runs as independent, which is approximate.
- Fixed learning rate across batch sizes; only batch sizes 1–4.
- CLIP similarity measures whole-image similarity, not character identity alone.
- Visual comparison is qualitative, unblinded, and covers one pair of runs.
- FID is not meaningful at 20 references; the overfitting proxy saturated.

## Hypotheses and next experiments

These are working hypotheses, **not** results of this study.

1. **Small datasets score higher because repetition overtrains them**, pushing outputs toward surface features CLIP rewards while fine detail degrades. *Test:* evaluate checkpoints from a single run at increasing epochs, with the dataset fixed.
2. **The main advantage of a large dataset is that it can be trained longer before overfitting.** *Test:* train a small and a large dataset past the same epoch count and compare when each degrades.
3. **Repetitions per image is the main lever.** *Test:* train different dataset sizes to the same epoch count and compare.
4. **CLIP similarity can't reliably pick the best LoRA.** *Test:* blinded forced-choice comparisons across all 15 runs, compared against the CLIP ranking.
5. **Rising CLIP similarity with flat LPIPS distance signals overtraining.** *Test:* check whether that pattern predicts the runs people rate worse in test 4.

Other useful extensions: seed replicates; larger batch sizes with tuned learning rates; a continuous similarity-to-training-set measure; other subjects, styles and base models.

## Related work

- **Lee & Lee (2026)** study batch size in LLM LoRA fine-tuning under a fixed-sample protocol and find accuracy roughly flat at small batch sizes, degrading above a critical size that shrinks with dataset size. The null result here for batch 1–4 is consistent with that, though their setting (LLMs, 25K+ examples, tuned learning rates) differs.
- **Muennighoff et al. (2023)** find that for LM pretraining at fixed compute, repeating data up to about 4 epochs is nearly as good as new data, after which its value drops quickly.
- **Hernandez et al. (2022)** find that repeating a small fraction of data many times at fixed total training sharply degrades LM performance, and link this to memorization.
- **Gu et al. (2025)** find that memorization in diffusion models rises as the training set shrinks and as training runs longer.
- **Hong & Choo (2025)** tested batch sizes 1, 2, 4 and 8 for SD 1.5 LoRAs on architectural massing images and recommend 4. Their CLIP scores differed only slightly across batch sizes (0.208–0.212, computed on one hand-selected image per prompt), and the choice rested largely on visual judgment, so it does not conflict with the null result here. They list as a limitation that they "did not analyze in detail the changes in model performance according to combinations of hyperparameter settings for Batch Size, Epoch, and Repeats," and suggest varying batch size and repeats with the training budget fixed. That is close to the design used here (they propose fixing training steps; this study fixes total samples).

## Repository contents

| Path | Contents |
|---|---|
| `data/lora_metrics.csv` | Per-run results for all 15 LoRAs |
| `prompts/test_prompts.json` | The 5 evaluation prompts |
| `experimental_notebook/generate_results_table.ipynb` | Generation and metric pipeline (Colab). The LoRAs were trained on CivitAI and loaded from Drive; the notebook's Kohya training cell was not used for these runs. |
| `Analysis.ipynb` | Regressions, partial-effect plots and heatmaps |
| `analysis/robustness_checks.py` | Reproduces every summary statistic in this README from the CSV |
| `requirements.txt` | Core package versions |
| `environment/colab_pip_freeze.txt` | Full record of the Colab environment |

### Reproducing

The analysis needs only the CSV:

```bash
pip install numpy pandas
python analysis/robustness_checks.py
```

The generation and metric pipeline needs the trained LoRAs, the training images and the reference set, which are not redistributed. With your own data in the same folder layout, `generate_results_table.ipynb` runs end to end on a single A100 in Colab.

## References

- Gu, X., Du, C., Pang, T., Li, C., Lin, M. & Wang, Y. (2025). On Memorization in Diffusion Models. *TMLR*. arXiv:2310.02664.
- Hernandez, D. et al. (2022). Scaling Laws and Interpretability of Learning from Repeated Data. arXiv:2205.10487.
- Hong, S.M. & Choo, S. (2025). Systematic Parameter Optimization for LoRA-Based Architectural Massing Generation Using Diffusion Models. *Buildings*, 15(19), 3477.
- Hu, E.J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Lee, S. & Lee, J. (2026). Beware of the Batch Size: Hyperparameter Bias in Evaluating LoRA. arXiv:2602.09492.
- Muennighoff, N. et al. (2023). Scaling Data-Constrained Language Models. NeurIPS 2023. arXiv:2305.16264.
- Parmar, G., Zhang, R. & Zhu, J.-Y. (2022). On Aliased Resizing and Surprising Subtleties in GAN Evaluation. CVPR 2022.
- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.
- Zhang, R., Isola, P., Efros, A.A., Shechtman, E. & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. CVPR 2018.

## License

Code and results: MIT (see [LICENSE](LICENSE)). Training and reference images are not included.
