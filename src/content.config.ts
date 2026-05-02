import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const publications = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/publications' }),
  schema: z.object({
    title: z.string(),
    authors: z.string(),
    authorNote: z.string().optional(),
    image: z.string(),
    imageAlt: z.string().optional(),
    summary: z.string(),
    arxivDate: z.coerce.date(),
    publicationDate: z.coerce.date().optional(),
    venue: z.string().optional(),
    status: z.string().default('In review'),
    arxivUrl: z.string().url().optional(),
    paperUrl: z.string().url().optional(),
    highlighted: z.boolean().default(false),
    highlightOrder: z.number().optional(),
  }),
});

const blogs = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/blogs' }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    excerpt: z.string(),
    image: z.string().optional(),
    imageAlt: z.string().optional(),
    tags: z.array(z.string()).optional(),
    categories: z.array(z.string()).optional(),
  }),
});

const projects = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    excerpt: z.string(),
    image: z.string().optional(),
    imageAlt: z.string().optional(),
    tags: z.array(z.string()).optional(),
    links: z.array(z.object({
      label: z.string(),
      url: z.string().url(),
    })).optional(),
    draft: z.boolean().default(false),
  }),
});

export const collections = { publications, blogs, projects };
