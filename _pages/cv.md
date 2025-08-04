---
layout: archive
title: "Curriculum Vitae"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}
<div><br></div>
[Download the one-page PDF version here](https://bowenyu066.github.io/files/cv.pdf){: .btn .btn--primary}

## Education

* **B.S.** in **Massachusetts Institute of Technology** (Cambridge, MA), *Aug. 2024 – Present*
  * *Major*: Physics (Course 8), Artificial Intelligence and Decision Making (Course 6-4)
  * *Selected Coursework*: Machine Learning, Introduction to Deep Learning, Fundamentals of Programming, Design and Analysis of Algorithms, Low-level Programming in C and Assembly, Computational Architecture, Linear Algebra and Optimization, Quantum Physics I, II \& III, Physics of Solids, Statistical Mechanics II, Quantum Field Theory I
* **B.S.** in **Peking University** (Beijing, China), *Sep. 2023 – Jul. 2024*
  * *Major*: Physics
  * *Selected Coursework*: Introduction to Computation, Data Structure and Algorithms, Classical Mechanics, Quantum Mechanics, Thermodynamics and Statistical Mechanics, Quantum Statistical Physics, Lie Groups and Lie Algebras
  * Transfer to MIT after one year of study

## Achievements

* **International Physics Olympiad (IPhO) Gold Medalist**, *Jul. 2023*
  * Represented China at the [53rd International Physics Olympiad](https://international-physics-olympiad2023-tokyo.jp/) (one of 5 team members)
  * Ranked [1st place](https://ipho-unofficial.org/timeline/2023/individual) in the world and awarded a Gold Medal
* **Chinese Physics Olympiad (CPhO) Gold Medalist**, *Dec. 2021 & Oct. 2022*
  * Awarded the Gold Medal in the finals of the [39th Chinese Physics Olympiad](https://cpho.pku.edu.cn/info/1095/1281.htm) and selected for the national team.
  * Awarded the Gold Medal in the finals of the [38th Chinese Physics Olympiad](https://cpho.pku.edu.cn/info/1086/1270.htm).


## Working Experiences

* **AI & Material Science Research Assistant @ MIT**, *Jan. 2025 — Present*
  * [Quantum Measurement Group](https://qm.mit.edu), Massachusetts Institute of Technology
  * Extensively apply various machine learning force field (MLFF) models, especially MACE, to predict relevant physical properties of materials ( \\(\Gamma\\)-phonon, heat conductivity, etc.) while preserving accuracy
  * Trained a foundational model for non-destructive defect identification from vibrational spectra, specifically phonon density-of-states (PDoS)
  * Supervisor: Professor [Mingda Li](https://web.mit.edu/nse/people/faculty/mli.html)
* **AI Workload Deployment Intern @ Intel**, *Jun. 2025 — Jul. 2025*
  * Conducted in-depth research on low-level GPU memory layouts, including [CuTe layout](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/index.html) ([NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)) and [linear layout](https://arxiv.org/abs/2505.23819v1) ([OpenAI Triton](https://github.com/triton-lang/triton))
  * Explored NVIDIA’s recently released [CuTeDSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) framework in depth, analyzing its lowering process to the backend using operations in the [CUTLASS library](https://github.com/NVIDIA/cutlass) as a case study
  * Supervisor: Fangwen Fu, Xiaodong Qiu, Ivan Luo

## Publications

### 2025

**1. A Foundation Model for Non-Destructive Defect Identification from Vibrational Spectra** <br>
  *Mouyang Cheng<sup>†,*</sup>, Chu-Liang Fu<sup>†</sup>, <b><u>Bowen Yu</u></b><sup>†</sup>, Eunbi Rha, Abhijatmedhi Chotrattanapituk, Douglas L Abernathy, Yongqiang Cheng, and Mingda Li<sup>*</sup>* <br>
  <span style="color: gray; font-size: 12px;">
  <sup>†</sup>These authors contributed equally.
  <sup>*</sup>Corresponding author.
  </span>
  - In review, paper available at: [arXiv:2506.00725](https://arxiv.org/pdf/2506.00725)
  ![DefectNet workflow demonstration]({% include base_path %}/images/publications/DefectNet.png)

**2. Enhancing Materials Discovery with Valence Constrained Design in Generative Modeling** <br>
  *Mouyang Cheng<sup>†,*</sup>, Weiliang Luo<sup>†</sup>, Hao Tang<sup>†</sup>, <b><u>Bowen Yu</u></b>, Yongqiang Cheng, Weiwei Xie, Ju Li, Heather J. Kulik, and Mingda Li<sup>*</sup>* <br>
  <span style="color: gray; font-size: 12px;">
  <sup>†</sup>These authors contributed equally.
  <sup>*</sup>Corresponding author.
  </span>
  - In review, paper available at: [arXiv:2507.19799](https://arxiv.org/abs/2507.19799)
  ![CrysVCD workflow demonstration]({% include base_path %}/images/publications/CrysVCD.png)

## Projects

- **MIT Statistical Physics II (8.08) Course Project**, *Jan. 2025*
  - Simulate the non-equilibrium relaxation and the phase transition of the 2D Ising model, using both discrete- and continuous-time simulation methods
  - Code available at: [https://github.com/bowenyu066/ising-kmc](https://github.com/bowenyu066/ising-kmc)

## Skills

* **Languages**: English (*fluent*), Chinese (*native*), Cantonese (*basic*), Spanish (*basic*)
* **Programming Languages**: C/C++, Python, Bluespec, RISC-V Assembly, LaTeX
*	**Libraries**: PyTorch, NumPy, Matplotlib, ASE, PyMatgen, Matformer, MACE, Triton, CUTLASS
*	**Technologies**: Linux, Git, GitHub, Docker, SSH, VSCode, Jupyter Notebook

## Hobbies

* **Long-distance running**:
  As of now, I have completed 1 half marathon, with the following personal bests:
  * Half marathon: 1:48:56 (*2023 Wuhan Marathon*)
* **Table tennis**
* **Movies**
  
<!-- Talks
======
  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html  %}
  {% endfor %}</ul> -->
  
<!-- Teaching
======
  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul> -->
  
<!-- Service and leadership
======
* Currently signed in to 43 different slack teams -->
