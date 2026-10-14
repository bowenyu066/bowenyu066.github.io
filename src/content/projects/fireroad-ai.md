---
title: "Fireroad.ai (MIT CSAIL Agentic AI Hackathon 2026)"
date: 2026-04-25
excerpt: "An AI-powered course-planning prototype for MIT students, combining a Fireroad-style planner with a tool-calling academic advising chat agent."
image: "/images/projects/fireroad.ai/overall.png"
highlighted: true
highlightOrder: 3
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

Planning courses at MIT is surprisingly ... complicated. You can ask ChatGPT "Hey Chat what courses should I take next semester", but it might not know your full academic history, what requirements you've already fulfilled, or even which subjects are being offered next semester. Of course, you could manually organize all the relevant context and provide it for ChatGPT, but ... at that point, you've already done half the planning yourself.

It's not that we don't have any good tools; in fact, tools like Fireroad and CourseRoad contain plenty of useful information, but planning still means jumping between course catalogs, degree requirements, prerequisites, evaluations, and schedules, piecing everything together on your own. **So we asked: why can't an AI agent understand both your academic background and your school's rules, and just help you figure out what to take?**

That's the idea behind **Fireroad.ai**: rather than building another chatbot that gives generic course recommendations, we wanted an AI academic advisor that could actually work with your academic context. It should know what you've already taken, understand what remains for your degree, look up real course information, and reason about your preferences and constraints. More importantly, it shouldn't just tell you what to do. It should be able to act on its recommendations, *directly inside your course planner*.

We built an experimental prototype that brings all these pieces together. Students can provide academic context through onboarding, including searchable transcript PDFs. The agent can consult MIT course data, explore current and historical offerings, check degree requirements, and propose concrete changes to an interactive semester plan. In this way, we hope to close the loop between understanding a student's situation, reasoning about their options, and actually updating their plan, without forcing them to manually bridge every piece of information.

The project won the **Agents for MIT Track at the MIT CSAIL Agentic AI Hackathon 2026**! MIT students can sign up and get started with the current prototype at this [link](https://fireroad-ai-lime.vercel.app). Shout out to my wonderful teammates (Kangyang Zhou, Yifan Kang, Chunji Wang, and Dianne Cao) for the fun collab and to the organizers for putting on a great event!

![Team Photo](/images/projects/fireroad.ai/team.jpeg)
