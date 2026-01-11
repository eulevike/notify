# GitHub Pages Setup Guide

This guide will help you enable GitHub Pages for the Stock Monitor web UI.

## Step 1: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings** tab
3. Scroll down to the "Code and automation" section and click on **Pages**
4. Under "Build and deployment", configure:
   - **Source**: Deploy from a branch
   - **Branch**: `main` (or your default branch)
   - **Folder**: `/docs`
5. Click **Save**

## Step 2: Create GitHub Personal Access Token (PAT)

The web UI needs a GitHub token to manage your repository (add/remove tickers, update settings, trigger workflows).

1. Go to https://github.com/settings/tokens
2. Click **Generate new token** → **Generate new token (classic)**
3. Configure the token:
   - **Name**: Stock Monitor UI
   - **Expiration**: Choose your preferred expiration (or no expiration)
   - **Scopes**: Check `repo` (Full control of private repositories)
4. Click **Generate token**
5. **Important**: Copy the token immediately (it starts with `ghp_`)

## Step 3: Configure the UI

1. Once GitHub Pages deploys (may take 1-2 minutes), visit:
   ```
   https://YOUR_USERNAME.github.io/notify/
   ```
   Replace `YOUR_USERNAME` with your GitHub username and `notify` with your repo name.

2. You'll see an authentication modal. Enter your GitHub PAT and click "Save Token".

3. The token is stored locally in your browser's localStorage only - it's never saved to the repository.

## Step 4: Update Repository Name in Config

If your repository is not `eulevike/notify`, update `docs/config.js`:

```javascript
const CONFIG = {
    repo: 'YOUR_USERNAME/YOUR_REPO_NAME',
    // ...
};
```

## Troubleshooting

### Page not found (404)
- Wait 1-2 minutes for GitHub Pages to deploy
- Check that you selected the `/docs` folder in Pages settings
- Verify the URL format: `https://username.github.io/repo-name/`

### Authentication fails
- Verify your PAT has the `repo` scope
- Check that the token hasn't expired
- Ensure your repository name in `config.js` matches your actual repo

### Workflows not triggering
- Go to Actions tab in your repository
- Ensure all workflows are enabled
- Check that workflows have "Allow read and write permissions" enabled:
  1. Settings → Actions → General
  2. Scroll to "Workflow permissions"
  3. Select "Read and write permissions"
  4. Click Save

### Artifact download fails
- The UI uses artifact download which may require additional setup
- For now, you can view results by:
  1. Going to the Actions tab
  2. Clicking on the latest workflow run
  3. Downloading artifacts manually

## Features

Once set up, you can:
- **Add/Remove tickers** from your watchlist
- **Configure settings** (chart visibility, explanations, refresh interval)
- **View last run results** with analysis summary
- **View annotated charts** from the latest run
- **Manually trigger** the stock monitor

## Security Notes

- The GitHub PAT is stored in browser localStorage only
- Never commit your PAT to the repository
- The UI communicates directly with GitHub API from your browser
- No backend server is required - everything runs on GitHub infrastructure
