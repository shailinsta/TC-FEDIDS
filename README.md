# TC-FedIDS: Trajectory-Aware Federated Intrusion Detection with Environment-Invariant Aggregation and Empirical Byzantine Robustness under Differential Privacy

**Authors:** Saumya Mishra¹, Megha Rathi², Ahmad Raza Khan³, Shailendra Mishra⁴*

¹ Department of Computer Science and Engineering, MNNIT Allahabad, India  
² Department of Computer Science and Engineering, Jaypee Institute of Information Technology, Noida, India  
³ Department of IT, College of Computer and Information Sciences, Majmaah University, Al-Majmaah 11952, Saudi Arabia  
⁴ Department of Computer Engineering, College of Computer and Information Sciences, Majmaah University, Al-Majmaah 11952, Saudi Arabia  

\* Corresponding author: s.mishra@mu.edu.sa

---

## Abstract

Federated learning enables collaborative training of neural intrusion-detection models without centralising raw network traffic. In practical federated intrusion detection (FL-IDS), heterogeneous multi-source NetFlow data, adaptive Byzantine updates, privacy noise and severe class imbalance can jointly degrade reliability. Most existing defences rely on single-round update geometry or fixed robust aggregation rules. They rarely examine the interaction between multi-round client behaviour, environment-level distribution shifts, Byzantine perturbations and differential privacy.

**TC-FedIDS** is a trajectory-aware federated intrusion-detection framework that combines:

1. **Multi-round update-trajectory extractor** — direction persistence, spectral entropy, second-order acceleration and norm variance of client updates, fused with environment-alignment and validation-residual signals into an exponential-moving-average trust weight.
2. **Environment-invariant GroupDRO-style aggregation** — server-side fusion cast as an optimisation problem with a Huber regulariser centred at the coordinate-wise median and solved by FISTA.
3. **Empirical Byzantine robustness radius** — quantifies the largest evaluated Byzantine fraction for which minority-class recall remains above a prescribed threshold under a given Rényi differential-privacy budget.

## Key Results

| Setting | Metric | TC-FedIDS | Best Baseline |
|---------|--------|-----------|---------------|
| Clean (0% Byzantine) | Balanced Accuracy | 0.914 ± 0.005 | FedAvg 0.927 ± 0.001 |
| Clean | Minority Recall | 0.947 ± 0.010 | Trimmed Mean 0.969 ± 0.003 |
| 30% ALIE | Minority Recall | **0.960 ± 0.011** | FedAvg / Median ≈ 0.949 |
| Ablation (full vs median) | Minority Recall gain | **+1.1 pp** over median | — |
| Overhead | Sec/round | 49.93 | ≈ 48.5 (≈ 3% overhead) |
| Byzantine fraction ≤ 40% | Minority Recall | ≥ 0.91 | — |

**Ablation insight:** Trajectory trust used alone does **not** improve minority recall over coordinate-wise median. The gain appears only when trajectory information is combined with the GroupDRO–FISTA stage (joint mechanism).

## Repository Structure

```
TC-FedIDS/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── code/
│   └── tc_fedids.py          # Full experimental pipeline
├── figures/
│   ├── fig1a_clean_bacc.png
│   ├── fig1b_clean_minrec.png
│   ├── fig2a_alie_bacc.png
│   ├── fig2b_alie_minrec.png
│   ├── fig3_ablation_minrec.png
│   ├── fig4_overhead.png
│   ├── fig5_dp_sweep.png
│   ├── fig6_empirical_byzantine_robustness.png
│   └── fig7_attack_types.png
├── tables/
│   ├── table1_clean_baseline.csv
│   ├── table2_alie_comparison.csv
│   ├── table2b_wilcoxon.csv
│   ├── table3_ablation.csv
│   ├── table4_overhead.csv
│   ├── table5_dp.csv
│   ├── table6_byzantine_robustness.csv
│   ├── table7_attack_types.csv
│   ├── table8_dataset_characteristics.csv
│   └── paper_ready_tables_figures_plan.csv
├── paper/
│   └── TC-FedIDS_Draft_Paper.docx
└── docs/
    └── GitHub_and_DOI_Setup_Instructions.pdf
```

## Datasets

Experiments use four real flow-level NetFlow sources under a 28-client Dirichlet (α = 0.4) federation:

| Source | Mode | n | Attack rate | Feature dim |
|--------|------|---|-------------|-------------|
| NF-CSE-CIC-IDS2018 | full | 50 000 | 0.065 | 10 |
| NF-ToN-IoT | full | 50 000 | 0.828 | 10 |
| UNSW-NB15 | harm5 | 50 000 | 0.549 | 5 |
| CIDDS-001 | harm5 | 50 000 | 0.015 | 5 |

Place the dataset files under a local `Dataset/` directory next to the script (see code comments for expected layout). The pipeline does **not** silently substitute synthetic data when real files are missing.

## Requirements

```bash
pip install -r requirements.txt
```

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CUDA recommended)
- NumPy, Pandas, SciPy, scikit-learn, Matplotlib, PyArrow

## Reproducing Experiments

```bash
cd code
python tc_fedids.py
```

The script produces all eight tables and seven figures with checkpointing. Completed tables are cached and skipped on subsequent runs. To force recomputation of a single table:

```python
run_or_load("table3", _compute_table3, force=True)
```

**Configuration used in the paper**

- Seeds: 11, 22, 33, 44, 55  
- Communication rounds: 30  
- Local epochs: 2  
- Batch size: 512  
- Model: Residual MLP (77 122 parameters)  
- Trajectory history: 5 rounds  
- GroupDRO–FISTA: max 12 iterations, initialised from coordinate-wise median  

## Citation

If you use this code or the associated results, please cite:

```bibtex
@article{mishra2025tcfedids,
  title   = {TC-FedIDS: Trajectory-Aware Federated Intrusion Detection with Environment-Invariant Aggregation and Empirical Byzantine Robustness under Differential Privacy},
  author  = {Mishra, Saumya and Rathi, Megha and Khan, Ahmad Raza and Mishra, Shailendra},
  year    = {2025},
  note    = {Preprint / under review}
}
```

## License

This project is released under the MIT License (see `LICENSE`).

## Acknowledgements

The authors thank the maintainers of the NF-CSE-CIC-IDS2018, NF-ToN-IoT, UNSW-NB15 and CIDDS-001 datasets.
