// GitHub Configuration
// Update these values for your repository

const CONFIG = {
    // GitHub repository in format "username/repo-name"
    // Example: "eulevike/notify"
    repo: 'eulevike/notify',

    // GitHub API base URL
    apiBase: 'https://api.github.com',

    // LocalStorage key for GitHub token
    tokenKey: 'github_monitor_token'
};

// Get GitHub token from localStorage
function getGitHubToken() {
    return localStorage.getItem(CONFIG.tokenKey);
}

// Save GitHub token to localStorage
function setGitHubToken(token) {
    localStorage.setItem(CONFIG.tokenKey, token);
}

// Check if user is authenticated
function isAuthenticated() {
    return !!getGitHubToken();
}

// API Endpoints
function getApiEndpoints() {
    return {
        // Trigger workflow to get all data
        getData: (runId) => `${CONFIG.apiBase}/repos/${CONFIG.repo}/actions/workflows/api-get-data.yml/runs`,

        // Get workflow run status
        getRunStatus: (runId) => `${CONFIG.apiBase}/repos/${CONFIG.repo}/actions/runs/${runId}`,

        // Get workflow artifacts
        getArtifacts: (runId) => `${CONFIG.apiBase}/repos/${CONFIG.repo}/actions/runs/${runId}/artifacts`,

        // Trigger add ticker workflow
        addTicker: (ticker) => `${CONFIG.apiBase}/repos/${CONFIG.repo}/actions/workflows/api-add-ticker.yml/dispatches`,

        // Trigger remove ticker workflow
        removeTicker: (ticker) => `${CONFIG.apiBase}/repos/${CONFIG.repo}/actions/workflows/api-remove-ticker.yml/dispatches`,

        // Trigger update settings workflow
        updateSettings: () => `${CONFIG.apiBase}/repos/${CONFIG.repo}/actions/workflows/api-update-settings.yml/dispatches`,

        // Trigger stock monitor workflow
        runMonitor: () => `${CONFIG.apiBase}/repos/${CONFIG.repo}/actions/workflows/stock-monitor.yml/dispatches`
    };
}
