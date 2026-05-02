---
title: "Fireroad.ai"
date: 2026-04-25
excerpt: "An AI-powered course-planning prototype for MIT students, combining a Fireroad-style planner with a tool-calling academic advising chat agent."
image: "/images/projects/fireroad.ai/overall.png"
highlighted: true
highlightOrder: 1
tags:
  - Hackathon
  - React
  - AI Agent
  - MIT
links:
  - label: "GitHub"
    url: "https://github.com/bowenyu066/Fireroad.ai"
  - label: "Demo"
    url: "https://fireroad-ai-lime.vercel.app"
showHeroImage: false
---

![Fireroad.ai](/images/projects/fireroad.ai/overall.png)

[Fireroad.ai](https://fireroad-ai-lime.vercel.app) is an experimental AI course planner for MIT students. The prototype combines a visual semester planner with an advising AI agent that can search the MIT course catalog, summarize schedules, check requirements, and propose validated changes to a student's active-semester plan.

The current implementation uses a static React frontend with a small Node/Express backend. The backend keeps model API keys server-side, exposes chat and course-planning routes, and supports a tool-calling workflow so the assistant can inspect course data before answering.

The project won **MIT CSAIL Agentic AI Hackathon 2026** in the Agents for MIT Track! MIT students can sign up and get started with the current prototype at this [link](https://fireroad-ai-lime.vercel.app). Shout out to my wonderful teammates (Kangyang Zhou, Yifan Kang, Chunji Wang, and Dianne Cao) for the fun collab and to the organizers for putting on a great event!

![Team Photo](/images/projects/fireroad.ai/team.png)
