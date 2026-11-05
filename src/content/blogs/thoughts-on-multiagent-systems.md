---
title: Some Thoughts on Scalable Multi-Agent Systems
date: 2026-11-4 21:11:13 -0500
permalink: /posts/notes/thoughts-on-multiagent-systems
excerpt: "Personal thoughts on how scalable multiagent systems should be built."
image: "/images/posts/thoughts-on-multiagent-systems/overview.png"
imageAlt: "Overview"
tags: 
    - LLM
    - Agent
    - Multi-Agent System
categories: 
    - Notes
show: true
highlighted: true
highlightOrder: 1
---

Put 1024 agents into the same environment and ask them to do one thing well together (or, many things). How should they coordinate?

This is the question I have been thinking about lately. Looking back at our recent experiments, I find that the ways of getting many agents to work together fall roughly into two design philosophies, which I will call Method A and Method B:

- **A**: give the agents a **coordination protocol** that builds in **many human priors and structural constraints**, so that they can start working together quickly and immediately beat a single agent or a simpler and less constrained protocol (for example, putting every agent into one big group chat);
- **B**: define only the **goal** and the **environment**, **constrain and intervene** as little as possible in how the agents act, and let them **keep learning** from the environment, so that structure *grows* (emerges) on its own.

Below are my thoughts on these two philosophies.

## Method A: hand them a coordination protocol

Over its long history, human society has distilled a great deal of experience about organization. From families and companies to nations and international bodies, there is always some structure and some set of rules. This is the philosophy behind Method A: human society is, in some sense, a multi-agent system, and the experience and rules it has worked out for efficient cooperation might carry over to LLM multi-agent systems, directly or with small changes.

Starting from this idea, several design lines branch out. Early on, with only a few agents, people assign each agent a role and responsibilities and design a workflow of, say, verifier, leader and worker, in which they cooperate. As the number of agents grows, assigning roles by hand stops being practical, and designers instead specify a higher-level structure (a harness). Here there are two lines of thought, agent-centralized and decentralized. The first may want agents to build something like a company, with a single coordinator, or a deeper tree with departments. The second has agents work as peers around a shared workspace, claiming tasks and each contributing on top of the others' results, much like an open-source community or a blockchain (for example, ScaleMAS[^1] and Agensh[^2]).

Which structure is better is an open question, just as some human companies are flat and others are heavily hierarchical. A thousand agents on the same collaboration network can easily run into heavy conflict and inefficient cooperation, while a structure that is too hierarchical can block the flow of information and slow down decisions. But either way, the goal here is essentially *one-shot*: let the agents start working under some architecture, and show that under this harness many agents not only cooperate effectively but also carry over to many other tasks, doing better than multi-agent methods under other architectures. Of course, finding such a structure takes the designers many rounds of iteration, but as long as the structure works across many tasks, that cost is paid once, and spread over all its uses it becomes negligible.

The most common criticism of Method A is that it may go against ***the bitter lesson***[^3], that hand-injected priors are not necessarily optimal, and we might as well let the agents just learn by themselves. Beyond that, I see two more concrete problems. First, human organizational experience was designed around human bottlenecks, such as limited communication bandwidth and the high cost of building trust. LLM bottlenecks are different: limited context, information lost to compaction, and errors that are highly correlated across agents. Majority voting makes sense for a group of people, but several answers written by the same model tend to make the same mistakes, so a vote can end up outvoting the correct minority. Second, agents do not necessarily follow a structure that is written into the prompt. We have seen many cases where a root node leave hundreds of messages unread and do hands-on work by itself, making the structure no longer carry the effect it was meant to have. How to balance enforced structure against the agents' own freedom is therefore a question worth studying within Method A.

## Method B: give only a goal and an environment

Since the bitter lesson has come up, a natural next thought is: what if, instead of prescribing how agents should cooperate, we let them learn to cooperate on their own? That is the philosophy of Method B: define only the goal and the environment, constrain the agents' actions as little as possible, and expect them to learn continually from the environment and gradually work out how to cooperate. Seen this way, it is not an out-of-the-box system; it is closer to a continual-learning framework.

I think the fundamental philosophical difference between A and B is whether there is an oracle. Human society has many companies, laws and hierarchies, but no outside being ever stood above us and said "this is how you should organize." All these institutions and structures were worked out by humans themselves over a long evolution. So in the Method B we hope for, whatever structure eventually appears should likewise be negotiated and learned by the agents themselves: how to divide the work, how to merge results, how to pass experience on to others. In an ideal Method B, the fewer priors the better, and every lesson is left for the agents to discover. For example, the environment does not tell agents to "leave a trace of your work"; instead, the agents find out that leaving traces makes things go better, pass this on, and in the end all of them learn to work that way.

That said, the bitter lesson only partly applies here. Sutton's original point was that general methods that leverage *computation* eventually beat *hand-coded* knowledge[^3]. In Method B, the model weights are not updated, and all the "learning" happens in context. Still, many people believe that because these models went through long pre-training, they already carry a great deal of human experience; they simply have not been trained on these specific tasks, so through in-context learning or test-time training they can find better ways on their own. This is actually a testable claim: given more time, does their cooperation improve? Whether this approach works is still an open question. Do agents have any capacity for reflection at the level of the organization? Existing research shows that LLM self-correction is unreliable without external feedback and works reasonably well only when verifiable external feedback is available. Even if they can reflect, how long does it take to go from not knowing how to knowing how? That may differ a lot from task to task. If a task takes ten or twenty hours under Method B, and many episodes, before any useful lesson emerges, while Method A can do it right away, then Method A is the better choice. We also cannot ignore the limited context window of LLMs. When a task is long and the context is small, an agent's earlier experience may be compacted away before it has had a chance to reflect. Then again, after a dozen or so rounds it might realize by itself that "I should write things down." If that really happens, it is itself a sign that Method B is working.

## Where A ends and B begins

If Method B also depends on an environment, where is the line between A and B? By *human prior*, I mean structural experience about cooperation, not the environment itself. When we compare A and B, both face the same environment and the same problem; the only difference is whether the way of cooperating is fixed in advance or formed by the agents themselves. Designing the environment and the problem matters a lot too, but that is a separate matter and is not what we discuss here.

To draw the line concretely, I would put it this way: a capability the environment makes available counts as environment, and only a process that is mandatory counts as Method A. Take a shared repository. If agents can use it or not as they like, it is part of the environment; if every change must be merged through it, as in ScaleMAS[^1], it is Method A. This way Method B does not have to take away even the ability to write files just to be "prior-free," and leaving traces of work can only be an available capability in B, not a requirement. By the same standard, approaches such as having the agents "automatically search for a structure" also belong to Method A, because we force them to follow that procedure, like a "board of directors" in human society. I should stress that this does not make such approaches bad: a board that pushes everyone to set up chains of cooperation may well be an effective way to cooperate in its own right.

## What kind of environment lets B happen

If we want to run Method B experiments, how should the task be designed? First, the goal has to be clear enough, and the signal fast and clear enough. Our resources are limited; we cannot afford one vague signal per hour, which might take thousands of iterations to produce a good result, while the context window makes the agents forget the earlier iterations.

Second, the task should ideally be continuous rather than a series of separate episodes. Human society has no episodes; people can reflect and adjust at any moment. Tasks should keep flowing in, and the agents should be able to keep reflecting and improving. This also means the tasks must be similar enough that a way of cooperating learned on the 50th task is still useful on the 51st. Otherwise the agents keep reflecting but are always chasing a moving target. Human society could accumulate institutions partly because its environment was relatively stable, so one generation's lessons were still useful to the next.

Human society offers a few more lessons. The signal for human learning at the level of organization is actually poor: wars, famines and bankruptcies take decades to play out and are mixed up with countless other factors. Humans learned anyway, through enormous parallelism and a very long time, and through selection and imitation between groups: organizations that did badly were weeded out, and practices that worked were seen and copied by others. We cannot afford that price. Conversely, the areas where humans learn quickly, such as markets, scientific experiments and sports, have signals that are not only fast and clear but also local and attributable: each person can see what their own approach led to, and can also see how others are doing.

This is a reminder for environment design. If 1024 agents share one global score, then even if that score is fast and clear, no single agent knows whether its own way of cooperating is good or bad. For organizational learning to happen, the signal may need to be broken down to the local level, and agents' successes and failures need to be visible to one another. If we want something even closer to human society, the environment may need several groups that can observe, compete with and imitate each other. That is still a property of the environment, not an oracle.

As for incentives, my view is that they must be clear enough, or the agents will not know which direction to iterate in. But incentives may work on LLMs differently than on people. The core of the human incentive problem is the gap between individual interest and collective goals; homogeneous agents share the same prompt and have no such conflict to begin with. For them, what matters more may be whether there is enough attributable feedback to tell them which approaches work. I have not fully thought this through yet.

## Summary

Although we spent much of this post elaborating on the differences between Method A and Method B, I want to stress that they are not mutually exclusive. In fact, it would be interesting to view Method A as a certain type of "pre-training" and Method B as "post-training", simply without updating the model weights. Good structures designed in Method A can be used as effective starting points for the exploration phase of Method B, and Method B can, in return, help us understand which priors in Method A are truly necessary.

[^1]: https://arxiv.org/abs/2610.16120
[^2]: https://arxiv.org/abs/2609.26781
[^3]: https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf
