# Workbooks

Interactive pages published with GitHub Pages.

- Each page lives in `pages/<slug>/` as `index.html` (self-contained) or `index.md` (rendered automatically).
- On every push to `main`, `.github/workflows/deploy.yml` runs `scripts/build.py`, which builds `_site/`, generates the root index, and deploys.
- Live at: https://reporter6320.github.io/workbooks/

## Add a page
1. Create `pages/my-page/index.html` or `index.md`.
2. Commit and push to `main`.

## One-time setup
Settings → Pages → Build and deployment → Source: **GitHub Actions**.
