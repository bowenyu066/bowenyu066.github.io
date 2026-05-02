---
title: "PaperPlay"
date: 2025-09-13
excerpt: "A HackMIT 2025 project that turns hand-drawn Mario-style levels into playable platformers using computer vision and a web frontend."
tags:
  - HackMIT
  - Computer Vision
  - Game
  - TypeScript
image: "/images/projects/paperplay/overall.png"
highlighted: false
links:
  - label: "GitHub Org"
    url: "https://github.com/HACKMIT-2025"
  - label: "Demo"
    url: "https://demo-description.vercel.app"
  - label: "LinkedIn"
    url: "https://www.linkedin.com/feed/update/urn:li:activity:7373551842288644096/"
showHeroImage: false
---

![PaperPlay Demo](/images/projects/paperplay/overall.png)
![PaperPlay](/images/projects/paperplay/img2.jpeg)

PaperPlay was a HackMIT 2025 project built by a team of four (me, Yuanbo Pang, Zehua Wang and John Francis Aradan; shout out to my wonderful teammates) in less than 24 hours.

PaperPlay lets anyone turn a hand-drawn map into a fully playable Mario-style game in less than a minute. The system automatically detects terrain, coins, obstacles, and enemies, then converts it into a working platformer you can play instantly on the web. The project was inspired by the idea of making game design more accessible and fun, allowing users to bring their creative visions to life without needing to learn complex game development tools.

The project was built upon the seamless infrastructure provided by Modal, which supports instant one-click remote deployment and powerful GPU auto-scaling for heavy image-to-level inference, delivering over 10x acceleration compared to local GPU computation. OpenCV was utilized for image processing and feature detection, while the game logic and frontend were implemented using TypeScript and React.

The project won **second place** in the Modal sponsor track at **HackMIT 2025**! Check out the [demo](https://demo-description.vercel.app) today to give it a try!

![Award Ceremony](/images/projects/paperplay/img1.jpeg)
