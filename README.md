# chrisadas.github.io

Personal blog. Jekyll + [Minima](https://github.com/jekyll/minima) theme,
hosted on GitHub Pages at <https://chrisadas.github.io>.

## Writing a post

Add a Markdown file to `_posts/` named `YYYY-MM-DD-title.md` with front matter:

```yaml
---
layout: post
title: "Your title"
date: 2026-06-22 10:00:00 +0000
categories: general
---
```

Commit it (browser or git). GitHub rebuilds the site automatically within ~1 min.

## Pages

- `index.md` — homepage (lists posts)
- `about.md` — about page
- `_config.yml` — site title, theme, plugins

## Local preview (optional)

Requires Ruby + Bundler:

```bash
bundle install
bundle exec jekyll serve   # http://localhost:4000
```
