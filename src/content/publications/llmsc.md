---
title: "Can LLMs extract scientific consensus? A case study in high-temperature superconductivity"
authors: "Mouyang Cheng†,*, Wenhao He†, Zhuotao Jin†, Bowen Yu, Ju Li, Boris Kozinsky, Yao Wang, Pavel Volkov, Liangzi Deng, Ching-Wu Chu, Xiao-Gang Wen, and Mingda Li*"
authorNote: "† Equal contribution. * Corresponding author."
image: "/images/publications/LLMSC.png"
imageAlt: "CrysVCD workflow demonstration"
summary: "Robust, interpretable scientific consensus and its evolution from large, contested literatures can be extracted through LLM, by structuring ~18,000 high-temperature-superconductivity (HTS) papers into a knowledge graph of mechanisms, materials, evidence, and citations"
arxivDate: 2026-05-26
status: "In review"
arxivUrl: "https://arxiv.org/abs/2606.07570"
highlighted: false
---

Scientific knowledge is increasingly dispersed across vast and heterogeneous scientific literature, where important claims are often implicit, evolving, and internally debated. While large language models (LLMs) have shown impressive performance in information extraction and summarization, their ability to recover latent scientific consensus remains unclear. Here, we investigate this problem in the context of high-temperature superconductivity (HTS), a long-standing and highly debated topic in condensed matter physics, as a challenging testbed. Using near 18,000 highly-cited publications over the past seven decades, we construct a structured knowledge graph linking competing superconducting mechanisms, material families, evidential modalities, and citation relations. We find that LLM-extracted representations recover coherent and physically interpretable structures, including family-dependent mechanism profiles, evidence-specific correlations, and citation-mediated temporal evolution of scientific beliefs. Ablation studies on LLM further show that the global structure remains robust across prompting, decoding, and model variations. Our results suggest that LLMs can indeed serve as scalable tools for deciphering scientific knowledge in domains characterized by competing interpretations and evolving knowledge.
