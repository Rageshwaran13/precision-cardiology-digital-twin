# Publish this project to GitHub

## 1. Create an empty GitHub repository

Recommended repository name: `precision-cardiology-digital-twin`

Choose **Public** only if you are ready to share the code, slide deck, and bundled assets with everyone. If you are unsure about asset permissions or team approval, choose **Private** first. Create the repository without adding a README, license, or `.gitignore` because this project already contains its own README and ignore rules.

## 2. Open a terminal in the extracted project folder

Confirm that the current directory contains `README.md`, `src/`, `dashboard/`, and `requirements.txt`.

```bash
git init -b main
git add .
git status
git commit -m "Initial commit: Precision Cardiology Digital Twin"
git remote add origin https://github.com/YOUR-USERNAME/precision-cardiology-digital-twin.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username and use the repository URL shown by GitHub. GitHub may prompt you to authenticate.

Before committing, run `git status` and check that no generated CSV datasets, trained model binaries, `.env` files, tokens, or other sensitive files are staged. The repository's `.gitignore` excludes the generated data and model outputs by default.

## 3. Make the repository easy to discover

Suggested GitHub description:

> Synthetic-data cardiac digital twin demo combining EHR-style records, wearable time series, gradient-boosted risk modeling, and a Streamlit 3D heart dashboard.

Suggested topics: `digital-twin`, `cardiology`, `streamlit`, `machine-learning`, `healthcare-ai`, `wearable-data`, `synthetic-data`, `python`.

## 4. After publishing

Open the repository page and verify the README preview, presentation links, and folder structure. Review `THIRD_PARTY_ASSETS.md` and confirm asset permissions before changing the repository to Public. Add a project license only after the team agrees on the terms.
