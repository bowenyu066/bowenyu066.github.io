---
title: "Enhancing Materials Discovery with Valence Constrained Design in Generative Modeling"
authors: "Mouyang Cheng†,*, Weiliang Luo†, Hao Tang†, Bowen Yu†, Yongqiang Cheng, Weiwei Xie, Ju Li, Heather J. Kulik, and Mingda Li*"
authorNote: "† Equal contribution. * Corresponding author."
image: "/images/publications/CrysVCD.png"
imageAlt: "CrysVCD workflow demonstration"
summary: "Integrating chemical valence constraints into the generative materials pipeline"
arxivDate: 2025-07-26
publicationDate: 2026-03-30
venue: "Nature Computational Science"
status: "Published"
arxivUrl: "https://arxiv.org/abs/2507.19799"
paperUrl: "https://www.nature.com/articles/s43588-026-01037-2"
highlighted: false
---

Diffusion-based deep generative models have emerged as powerful tools for inverse materials design. Yet many existing approaches overlook essential chemical constraints such as oxidation state balance, which can lead to chemically invalid structures. Here we introduce CrysVCD (Crystal generator with Valence-Constrained Design), a modular framework that integrates chemical rules directly into the generative process. CrysVCD first employs a transformer-based elemental language model to generate valence-balanced compositions, followed by a diffusion model to generate crystal structures. The valence constraint enables orders-of-magnitude more efficient chemical valence checking compared to pure data-driven approaches with post-screening. When fine-tuned on stability metrics, CrysVCD achieves 85% thermodynamic stability and 68% phonon stability. Moreover, CrysVCD supports conditional generation of functional materials, enabling discovery of candidates such as high thermal conductivity semiconductors and high-k dielectric compounds. Designed as a general-purpose plugin, CrysVCD can be integrated into diverse generative pipelines to promote chemical validity.
