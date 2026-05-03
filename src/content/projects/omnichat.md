---
title: "OmniChat"
date: 2026-01-27
excerpt: "A native macOS SwiftUI app that brings ChatGPT, Claude, and Gemini into one local-first interface with shared memory, attachments, and workspace context."
image: "/images/projects/omnichat/overall.png"
showHeroImage: false
highlighted: false
tags:
  - Swift
  - macOS
  - AI
  - Local-first
links:
  - label: "GitHub"
    url: "https://github.com/bowenyu066/OmniChat"
  - label: "Demo"
    url: "https://bowenyu066.github.io/OmniChat/"
draft: false
---

![OmniChat](/images/projects/omnichat/overall.png)

OmniChat is a native macOS app for using multiple frontier model providers from one interface. It unifies OpenAI, Anthropic, and Google models while keeping conversations, memories, API keys, and workspace context local to the user's Mac.

The project grew out of a very practical annoyance: switching between chat products fragments context. OmniChat treats models as interchangeable guests and keeps the user's state as the stable center. It supports persistent memories across conversations, drag-and-drop multimodal attachments, streaming Markdown and LaTeX rendering, local codebase indexing for coding context, and macOS-native storage/security patterns such as SwiftData and Keychain.

The repo is public and actively evolving as a personal tool and product experiment. Check it out at this [link](https://github.com/bowenyu066/OmniChat)!
