---
title: "Enhancing Materials Discovery with Valence Constrained Design in Generative Modeling"
authors: "Mouyang Cheng†,*, Weiliang Luo†, Hao Tang†, Bowen Yu, Yongqiang Cheng, Weiwei Xie, Ju Li, Heather J. Kulik, and Mingda Li*"
authorNote: "† Equal contribution. * Corresponding author."
image: "/images/publications/CrysVCD.png"
imageAlt: "CrysVCD workflow demonstration"
summary: "CrysVCD integrates chemical valence constraints into a generative materials pipeline, improving chemical validity while supporting conditional discovery of stable functional materials."
arxivDate: 2025-07-26
status: "In review"
arxivUrl: "https://arxiv.org/abs/2507.19799"
paperUrl: "https://arxiv.org/abs/2507.19799"
highlighted: true
highlightOrder: 2
---

Diffusion-based deep generative models have emerged as powerful tools for inverse materials design. Yet many existing approaches overlook essential chemical constraints such as oxidation state balance, which can lead to chemically invalid structures.

Here we introduce CrysVCD (Crystal generator with Valence-Constrained Design), a modular framework that integrates chemical rules directly into the generative process. CrysVCD first employs a transformer-based elemental language model to generate valence-balanced compositions, followed by a diffusion model to generate crystal structures. The valence constraint enables orders-of-magnitude more efficient chemical valence checking compared to pure data-driven approaches with post-screening.

When fine-tuned on stability metrics, CrysVCD achieves 85% thermodynamic stability and 68% phonon stability. Moreover, CrysVCD supports conditional generation of functional materials, enabling discovery of candidates such as high thermal conductivity semiconductors and high-k dielectric compounds. Designed as a general-purpose plugin, CrysVCD can be integrated into diverse generative pipelines to promote chemical validity.
