---
title: "A Foundation Model for Non-Destructive Defect Identification from Vibrational Spectra"
collection: publications
category: manuscripts
permalink: /publication/DefectNet
excerpt: "Defects are ubiquitous in solids and strongly influence materials' mechanical and functional properties. However, non-destructive characterization and quantification of defects, especially when multiple types coexist, remain a long-standing challenge. Here we introduce DefectNet, a foundation machine learning model that predicts the chemical identity and concentration of substitutional point defects with multiple coexisting elements directly from vibrational spectra, specifically phonon density-of-states (PDoS). Trained on over 16,000 simulated spectra from 2,000 semiconductors, DefectNet employs a tailored attention mechanism to identify up to six distinct defect elements at concentrations ranging from 0.2% to 25%. The model generalizes well to unseen crystals across 56 elements and can be fine-tuned on experimental data. Validation using inelastic scattering measurements of SiGe alloys and MgB2 superconductor demonstrates its accuracy and transferability. Our work establishes vibrational spectroscopy as a viable, non-destructive probe for point defect quantification in bulk materials, and highlights the promise of foundation models in data-driven defect engineering."
date: 2025-05-31
venue: 'arXiv'
# slidesurl: 'http://academicpages.github.io/files/slides1.pdf'
paperurl: 'https://arxiv.org/pdf/2506.00725'
# bibtexurl: 'http://academicpages.github.io/files/bibtex1.bib'
# citation: 'Your Name, You. (2009). &quot;Paper Title Number 1.&quot; <i>Journal 1</i>. 1(1).'
---
![DefectNet workflow demonstration]({% include base_path %}/images/publications/DefectNet.png)

Defects are ubiquitous in solids and strongly influence materials' mechanical and functional properties. However, non-destructive characterization and quantification of defects, especially when multiple types coexist, remain a long-standing challenge. Here we introduce DefectNet, a foundation machine learning model that predicts the chemical identity and concentration of substitutional point defects with multiple coexisting elements directly from vibrational spectra, specifically phonon density-of-states (PDoS). Trained on over 16,000 simulated spectra from 2,000 semiconductors, DefectNet employs a tailored attention mechanism to identify up to six distinct defect elements at concentrations ranging from 0.2% to 25%. The model generalizes well to unseen crystals across 56 elements and can be fine-tuned on experimental data. Validation using inelastic scattering measurements of SiGe alloys and MgB2 superconductor demonstrates its accuracy and transferability. Our work establishes vibrational spectroscopy as a viable, non-destructive probe for point defect quantification in bulk materials, and highlights the promise of foundation models in data-driven defect engineering.

{% if page.paperurl %}
[Download the paper here]({{ page.paperurl }}){: .btn .btn--primary}
{% endif %}