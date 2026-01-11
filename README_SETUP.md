# GitHub Actions Setup Guide

## 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/stock-monitor.git
git push -u origin main
```

## 2. Add Secrets

Go to: https://github.com/YOUR_USERNAME/stock-monitor/settings/secrets/actions

Add these secrets:
- `ZAI_API_KEY` = your z.ai API key
- `NTFY_TOPIC` = notifylevi (or your topic)

## 3. Enable Actions

Go to: https://github.com/YOUR_USERNAME/stock-monitor/actions

Click "I understand my workflows, go ahead and enable them"

## 4. Done!

The workflow will run every hour at minute 2.

View logs at: https://github.com/YOUR_USERNAME/stock-monitor/actions
