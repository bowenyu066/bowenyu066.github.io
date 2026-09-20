---
title: "Sparkie (HackMIT 2026)"
date: 2026-09-20
excerpt: "An AI meeting agent that listens, speaks, and GETS WORK DONE in video calls."
image: "/images/projects/sparkie/real-demo.png"
imageAlt: "Sparkie workspace with meeting context, background tasks, and a generated report"
showHeroImage: false
highlighted: true
highlightOrder: 1
tags:
  - AI Agent
  - Voice
  - Zoom
  - Python
links:
  - label: "GitHub"
    url: "https://github.com/Hope7Happiness/sparkie"
  - label: "Website"
    url: "https://hope7happiness.github.io/sparkie/"
  - label: "Demo"
    url: "https://youtu.be/-kOeUK9_kIs"
  - label: "Presentation"
    url: "https://bowenyu066.github.io/sparkie/"
draft: false
---

![Sparkie workspace with task progress and a generated report](/images/projects/sparkie/real-demo.png)

AI agents are good tools, but they still live in a separate chat window. Why not let ChatGPT join our Zoom meetings, act as a real teammate, listen to the discussion, and get work done while we keep talking?

This is what we have built, [Sparkie](https://hope7happiness.github.io/sparkie/); it closes the loop between conversation and action. Participants can ask Sparkie to act on anything they hope to do. It can answer questions about the discussion, research an idea, write a document, code on projects, then bring the result back through voice and a shared workspace. The conversation itself became the input to the work. (Check out our cool [demo video](https://youtu.be/-kOeUK9_kIs) for a real meeting example!)

![Workflow](/images/projects/sparkie/workflow.png)

The above diagram shows the workflow of Sparkie. In our design, we separate the voice conversation from background execution.

- The voice agent is powered by GPT Realtime-2.1 and handles low-latency spoken responses and semantic turn detection. In parallel with it, a Deepgram transcription worker continually transcribes participant tracks and stores them in a shared workspace.
- GPT Realtime-2.1 will delegate tasks to a backend agent if the task requires tool usage or file access. In our setup, this agent can be either a fast model (SWE 1.6 Fast) to handle simple tasks or a deep reasoning model (GPT 6 Astra, GPT 5.6 Sol, etc.) to handle complicated tasks.
- The Backend agent has access to the full transcription from Deepgram, and is free to use any tools or even call more subagents to complete the task. Whenever it finishes, the result will be sent back to the voice agent, which will then present it to the meeting participants.
- To avoid waking up the voice agent for every single turn, we use a fast model (Gemini 3.5 Flash) to decide whether a turn calls for Sparkie to respond within < 1 second. In this way, Sparkie can retain full access to all recent conversation context, but only respond when necessary.

Sparkie is a hackathon project built with the joint effort from [@Hope7Happiness](https://github.com/Hope7Happiness) and [@YIFANK](https://github.com/YIFANK) at HackMIT 2026. Our next goal is to make Sparkie an even more proactive and useful teammate, capable of taking initiative even when not explicitly asked. We also hope to extend Sparkie to support more meeting platforms and integrate with more tools. Contact us if you are interested in collaborating on this project!

![Group photo of the Sparkie team](/images/projects/sparkie/group-photo.jpeg)
