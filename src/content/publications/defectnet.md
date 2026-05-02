---
title: "A Foundation Model for Non-Destructive Defect Identification from Vibrational Spectra"
authors: "Mouyang Cheng†,*, Chu-Liang Fu†, Bowen Yu†, Eunbi Rha, Abhijatmedhi Chotrattanapituk, Douglas L. Abernathy, Yongqiang Cheng, and Mingda Li*"
authorNote: "† Equal contribution. * Corresponding author."
image: "/images/publications/DefectNet.png"
imageAlt: "DefectNet workflow demonstration"
summary: "DefectNet predicts the chemical identity and concentration of substitutional point defects directly from vibrational spectra, establishing vibrational spectroscopy as a non-destructive probe for defect quantification."
arxivDate: 2025-05-31
publicationDate: 2026-03-30
venue: "Matter"
status: "Published"
arxivUrl: "https://arxiv.org/abs/2506.00725"
paperUrl: "https://www.cell.com/matter/abstract/S2590-2385(26)00091-3"
highlighted: true
highlightOrder: 1
---

Defects are ubiquitous in solids and strongly influence materials' mechanical and functional properties. However, non-destructive characterization and quantification of defects, especially when multiple types coexist, remain a long-standing challenge.

Here we introduce DefectNet, a foundation machine learning model that predicts the chemical identity and concentration of substitutional point defects with multiple coexisting elements directly from vibrational spectra, specifically phonon density-of-states (PDoS). Trained on over 16,000 simulated spectra from 2,000 semiconductors, DefectNet employs a tailored attention mechanism to identify up to six distinct defect elements at concentrations ranging from 0.2% to 25%.

The model generalizes well to unseen crystals across 56 elements and can be fine-tuned on experimental data. Validation using inelastic scattering measurements of SiGe alloys and MgB2 superconductor demonstrates its accuracy and transferability. Our work establishes vibrational spectroscopy as a viable, non-destructive probe for point defect quantification in bulk materials, and highlights the promise of foundation models in data-driven defect engineering.
