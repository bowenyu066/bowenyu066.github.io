# Bowen Yu Personal Website

A small Astro static site. Content changes are Markdown-only; design changes live in a few obvious template/CSS files.

## Local development

```bash
npm install
npm run dev
```

Open the URL Astro prints (usually `http://localhost:4321`).

Build and preview the production version:

```bash
npm run build
npm run preview
```

## Folder structure

```text
src/
  content/
    publications/   # one Markdown file per paper
    blogs/          # one Markdown file per blog post
    projects/       # one Markdown file per project
  pages/            # routes
  components/       # PublicationItem, ContentCard, Comments, Icon
  layouts/          # BaseLayout (header, footer, theme toggle)
  styles/site.css   # the entire visual design
public/
  files/cv.pdf      # the CV the nav links to
  images/           # publication, blog, profile, favicon assets
```

## Add a publication

Create `src/content/publications/<slug>.md`:

```markdown
---
title: "Paper Title"
authors: "First Author, Bowen Yu, Last Author"
authorNote: "† Equal contribution. * Corresponding author."
image: "/images/publications/my-paper.png"
imageAlt: "Short image description"
summary: "One or two sentences shown on publication lists."
arxivDate: 2026-01-15
publicationDate: 2026-04-20      # omit if not yet published
venue: "Journal or Conference"   # omit if not yet published
status: "Published"               # default "In review"
arxivUrl: "https://arxiv.org/abs/..."
paperUrl: "https://..."
highlighted: true
highlightOrder: 1
---

Detailed abstract or notes go here.
```

Rules:

- `arxivDate` is required and orders the full publication list (newest first).
- If only on arXiv, omit `publicationDate` and `venue`; the list shows `In review`.
- If published, set `publicationDate` and `venue`; both dates show.
- `highlighted: true` puts the paper in **Selected**; `highlightOrder` sets the order.

## Add a blog post

Create `src/content/blogs/<slug>.md`:

```markdown
---
title: "Post Title"
date: 2026-02-01
excerpt: "Short summary shown on the index."
image: "/images/posts/folder/cover.png"
imageAlt: "Cover image description"
tags:
  - Computer Science
  - Physics
---

Markdown body.
```

Posts are ordered by `date`, newest first. Drop images under `public/images/posts/...`.

## Add a project

Create `src/content/projects/<slug>.md`:

```markdown
---
title: "Project Name"
date: 2026-03-01
excerpt: "Short project summary."
image: "/images/projects/cover.png"
imageAlt: "Project screenshot"
tags:
  - AI
links:
  - label: "GitHub"
    url: "https://github.com/..."
draft: false
---

Project writeup.
```

`draft: true` hides the project from the index until you flip it.

## CV

Replace `public/files/cv.pdf` when the CV changes. The nav opens it directly in a new tab.

## Comments and theme

Giscus comments live in `src/components/Comments.astro`. The light/dark toggle is in `src/layouts/BaseLayout.astro` and stores the choice in `localStorage`.

## Icons

All icons (socials, theme toggle, paper/arXiv link prefixes, arrows) live as inline SVGs in `src/components/Icon.astro`. To swap or add an icon, edit that file and reference it as `<Icon name="..." />`.

## Deployment

`.github/workflows/deploy.yml` builds with npm and pushes `dist/` to GitHub Pages. In repo settings, set Pages source to GitHub Actions.
