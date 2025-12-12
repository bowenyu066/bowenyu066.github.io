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

* **B.S.**, Physics & AI in **Massachusetts Institute of Technology** *(Cambridge, MA, United States)*, *Aug. 2024 – Jun. 2027 (Expected)*
  * *Major*: Physics (Course 8), Artificial Intelligence and Decision Making (Course 6-4)
  * *GPA*: 5.00/5.00 (as of Jun. 2025)
  * *Selected Coursework*: Machine Learning, Deep Learning, Natural Language Processing, Design and Analysis of Algorithms, Computational Architecture, Probability and Statistics, Quantum Physics I, II & III, Physics of Solids, Statistical Mechanics II, Quantum Field Theory I
* **B.S.**, Physics in **Peking University** *(Beijing, China)*, *Sep. 2023 – Jul. 2024*
  * *Major*: Physics
  * *GPA*: 3.91/4.00
  * *Selected Coursework*: Introduction to Computation, Data Structure and Algorithms, Classical Mechanics, Quantum Mechanics, Thermodynamics and Statistical Mechanics, Quantum Statistical Physics, Lie Groups and Lie Algebras
  * Transfer to MIT after one year of study
* **High School** in **No.1 Middle School Affiliated to Central China Normal University** *(Wuhan, China)*, *Sep. 2020 – Jul. 2023*

## Working Experiences

* **AI & Material Science Research Assistant @ MIT** *(Cambridge, MA, United States)*, *Jan. 2025 — Present*
  * Extensively leveraging various AI techniques in material and defect engineering, including convolutional neural networks, reinforcement learning methods, generative models, and machine learning force fields (MLFF)
  * Co-authored three papers in machine learning-based defect engineering and generative model-powered valence-informed material discovery
  * Supervisor: Professor [Mingda Li](https://web.mit.edu/nse/people/faculty/mli.html)
* **Large Language Model Data Researcher @ ByteDance** *(Beijing, China)*, *Aug. 2025 — Sep. 2025*
  * Served as the Seed LLM data partner, led the design of benchmarks comprised of 500+ physics problems, from university to PhD level
  * Contributed to ongoing internal projects aimed at improving model reasoning fidelity on symbolic tasks
  * Co-led the development of a test-time scaling (TTS) pipeline that enabled gold-medal performance of AI models on IPhO 2025 theoretical problems
* **AI Workload Deployment Intern @ Intel** *(Shanghai, China)*, *Jun. 2025 — Jul. 2025*
  * Researched and authored internal tutorials on GPU memory layouts, including CuTe layout and linear layout
  * Analyzed the CuTeDSL lowering process to backend kernels using GEMM as a case study, informing optimizations for AI workload deployment
  * Authored internal technical notes adopted by the CuTeDSL framework team for performance reference

## Achievements

* **International Physics Olympiad (IPhO) Gold Medalist**, *Jul. 2023*
  * Represented China at the [53rd International Physics Olympiad](https://international-physics-olympiad2023-tokyo.jp/) (one of 5 team members)
  * Ranked [1st place](https://ipho-unofficial.org/timeline/2023/individual) in the world and awarded a Gold Medal
* **Chinese Physics Olympiad (CPhO) Gold Medalist**, *Dec. 2021 & Oct. 2022*
  * Awarded the Gold Medal in the finals of the [39th Chinese Physics Olympiad](https://cpho.pku.edu.cn/info/1095/1281.htm) and selected for the national team.
  * Awarded the Gold Medal in the finals of the [38th Chinese Physics Olympiad](https://cpho.pku.edu.cn/info/1086/1270.htm).

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

**3. Reinforcement learning-guided optimization of critical current in high-temperature superconductors** <br>
  *Mouyang Cheng<sup>†,*</sup>, Qiwei Wan<sup>†</sup>, <b><u>Bowen Yu</u></b><sup>†</sup>, Eunbi Rha, Michael J. Landry, and Mingda Li<sup>*</sup>* <br>
  <span style="color: gray; font-size: 12px;">
  <sup>†</sup>These authors contributed equally.
  <sup>*</sup>Corresponding author.
  </span>
  - In review, paper available at: [arXiv:2510.22424](https://arxiv.org/abs/2510.22424)
  ![RL for JC workflow demonstration]({% include base_path %}/images/publications/RLJC.png)

## Projects

- **PaperPlay (HackMIT 2025)**, *Sep. 2025*
  - Working in a group of four, developed a web platform that converts hand-drawn Mario-style levels into playable platformers in less than 5 minutes by leveraging OpenCV-based image recognition and deployed via Modal
  - Won **2nd place** in the Modal sponsor track
  - Demo available at: [https://demo-description.vercel.app/](https://demo-description.vercel.app/); Code available at: [https://github.com/HACKMIT-2025](https://github.com/HACKMIT-2025)
- **Representation Efficiency in Neural Reasoning (MIT 6.7960 Deep Learning)**, *Oct. 2025 — Dec. 2025*
  - In a team of three, designed and executed a multilingual evaluation of mathematical reasoning on MMATH and GSM8K, comparing frontier APIs (ChatGPT-5.1, Gemini-2.5-Flash, DeepSeek-V3.2) and open-source 8B models (Qwen3-8B, Llama-3.1-8B-Instruct) across English, Chinese, Spanish, and Thai
  - Performed detailed token-length and character-length analyses under five tokenizers to quantify an **Encoder Gap** between *intrinsic* (character-level) and *realized* (token-level) representation density, showing up to 5-10% token savings in well-aligned multilingual setups
  * Implemented LoRA fine-tuning of Llama-3.1-8B on Chinese GSM8K, improving Chinese accuracy by 8.8 percentage points without degrading English performance, demonstrating that representation bottlenecks are largely learnable rather than architectural
  * Project repo available at: [https://github.com/bowenyu066/language-shapes-reasoning/](https://github.com/bowenyu066/language-shapes-reasoning/); blog post available at: [https://bowenyu066.github.io/posts/notes/multilingual-math-reasoning/](https://bowenyu066.github.io/posts/notes/multilingual-math-reasoning/)
  

## Skills

* **Languages**: English (*fluent*), Chinese (*native*), Cantonese (*basic*), Spanish (*basic*)
* **Programming Languages**: C/C++, Python, Bluespec, RISC-V Assembly, LaTeX
*	**Libraries**: PyTorch, NumPy, Matplotlib, ASE, PyMatgen, Matformer, MACE, Triton, CUTLASS

## Hobbies

* **Long-distance running**:
  As of Dec. 2025, I have completed 1 half marathon, with the following personal bests:
  * Half marathon: 1:48:56 (*2023 Wuhan Marathon*)
* **Table tennis**
* **Movies**
* *More to explore...*

---

Last updated: December 12, 2025
  
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
