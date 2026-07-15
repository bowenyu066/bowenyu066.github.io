---
title: "Reinforcement learning-guided optimization of critical current in high-temperature superconductors"
authors: "Mouyang Cheng†,*, Qiwei Wan†, Bowen Yu†, Eunbi Rha, Michael J. Landry, and Mingda Li*"
authorNote: "† Equal contribution. * Corresponding author."
image: "/images/publications/RLJC.png"
imageAlt: "RL-guided critical current workflow demonstration"
summary: "Combining reinforcement learning (RL) with time-dependent Ginzburg-Landau (TDGL) simulations to autonomously optimize defect configurations for high-temperature superconductors"
arxivDate: 2025-10-25
status: "In review"
arxivUrl: "https://arxiv.org/abs/2510.22424"
paperUrl: "https://arxiv.org/abs/2510.22424"
highlighted: false
---

High-temperature superconductors are essential for next-generation energy and quantum technologies, yet their performance is often limited by the critical current density (Jc), which is strongly influenced by microstructural defects. Optimizing Jc through defect engineering is challenging due to the complex interplay of defect type, density, and spatial correlation. Here we present an integrated workflow that combines reinforcement learning (RL) with time-dependent Ginzburg-Landau (TDGL) simulations to autonomously identify optimal defect configurations that maximize Jc. In our framework, TDGL simulations generate current-voltage characteristics to evaluate Jc, which serves as the reward signal that guides the RL agent to iteratively refine defect configurations. We find that the agent discovers optimal defect densities and correlations in two-dimensional thin-film geometries, enhancing vortex pinning and Jc relative to the pristine thin-film, approaching 60% of the theoretical depairing limit with up to 15-fold enhancement compared to random initialization. This RL-driven approach provides a scalable strategy for defect engineering, with broad implications for advancing HTS applications in fusion magnets, particle accelerators, and other high-field technologies.
