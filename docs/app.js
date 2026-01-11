// Application State
const state = {
    tickers: [],
    settings: {},
    lastRun: null,
    isLoading: false
};

// DOM Elements
const elements = {
    authModal: document.getElementById('authModal'),
    githubTokenInput: document.getElementById('githubToken'),
    saveTokenBtn: document.getElementById('saveTokenBtn'),
    status: document.getElementById('status'),
    newTicker: document.getElementById('newTicker'),
    addTickerBtn: document.getElementById('addTickerBtn'),
    tickersList: document.getElementById('tickersList'),
    showCharts: document.getElementById('showCharts'),
    showExplanation: document.getElementById('showExplanation'),
    refreshInterval: document.getElementById('refreshInterval'),
    saveSettingsBtn: document.getElementById('saveSettingsBtn'),
    lastRunSummary: document.getElementById('lastRunSummary'),
    lastRunResults: document.getElementById('lastRunResults'),
    runMonitorBtn: document.getElementById('runMonitorBtn'),
    refreshBtn: document.getElementById('refreshBtn'),
    chartsSection: document.getElementById('chartsSection'),
    chartsGallery: document.getElementById('chartsGallery'),
    chartModal: document.getElementById('chartModal'),
    modalChartImage: document.getElementById('modalChartImage'),
    closeModal: document.querySelector('.close')
};

// Initialize app
function init() {
    // Check authentication
    if (!isAuthenticated()) {
        showAuthModal();
        return;
    }

    // Set up event listeners
    setupEventListeners();

    // Load initial data
    loadAllData();

    // Start auto-refresh
    startAutoRefresh();
}

// Show/hide auth modal
function showAuthModal() {
    elements.authModal.style.display = 'flex';
}

function hideAuthModal() {
    elements.authModal.style.display = 'none';
}

// Setup event listeners
function setupEventListeners() {
    // Auth
    elements.saveTokenBtn.addEventListener('click', saveToken);

    // Ticker management
    elements.addTickerBtn.addEventListener('click', addTicker);
    elements.newTicker.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') addTicker();
    });

    // Settings
    elements.saveSettingsBtn.addEventListener('click', saveSettings);

    // Monitor actions
    elements.runMonitorBtn.addEventListener('click', runMonitor);
    elements.refreshBtn.addEventListener('click', () => loadAllData());

    // Chart modal
    elements.closeModal.addEventListener('click', () => {
        elements.chartModal.style.display = 'none';
    });

    elements.chartModal.addEventListener('click', (e) => {
        if (e.target === elements.chartModal) {
            elements.chartModal.style.display = 'none';
        }
    });
}

// Save GitHub token
async function saveToken() {
    const token = elements.githubTokenInput.value.trim();
    if (!token) {
        showError('Please enter a token');
        return;
    }

    setGitHubToken(token);
    hideAuthModal();
    updateStatus('Authenticating...');

    try {
        // Test token by fetching repo info
        const response = await fetch(`${CONFIG.apiBase}/repos/${CONFIG.repo}`, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/vnd.github.v3+json'
            }
        });

        if (response.ok) {
            updateStatus('Connected', 'success');
            loadAllData();
            startAutoRefresh();
        } else {
            showError('Invalid token');
            localStorage.removeItem(CONFIG.tokenKey);
            showAuthModal();
        }
    } catch (error) {
        showError('Connection failed');
        console.error(error);
    }
}

// Load all data from GitHub
async function loadAllData() {
    if (state.isLoading) return;
    state.isLoading = true;
    updateStatus('Loading...', 'loading');

    try {
        // Trigger get-data workflow
        const token = getGitHubToken();
        const endpoints = getApiEndpoints();

        const response = await fetch(endpoints.getData(), {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/vnd.github.v3+json'
            },
            body: JSON.stringify({ ref: 'main' })
        });

        if (!response.ok) throw new Error('Failed to trigger workflow');

        const workflowData = await response.json();
        const runId = workflowData.id;

        // Wait for workflow to complete and get artifacts
        const data = await waitForWorkflowCompletion(runId, token);

        if (data) {
            // Update state
            state.tickers = data.tickers || [];
            state.settings = data.settings || {};
            state.lastRun = data.last_run || null;

            // Render UI
            renderTickers();
            renderSettings();
            renderLastRun();

            updateStatus('Updated', 'success');
        }
    } catch (error) {
        console.error('Error loading data:', error);
        showError('Failed to load data');
    } finally {
        state.isLoading = false;
    }
}

// Wait for workflow completion and get artifacts
async function waitForWorkflowCompletion(runId, token) {
    const maxAttempts = 30;
    const endpoints = getApiEndpoints();

    for (let i = 0; i < maxAttempts; i++) {
        // Check run status
        const statusResponse = await fetch(endpoints.getRunStatus(runId), {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/vnd.github.v3+json'
            }
        });

        if (!statusResponse.ok) continue;

        const runData = await statusResponse.json();

        if (runData.status === 'completed') {
            // Get artifacts
            const artifactsResponse = await fetch(endpoints.getArtifacts(runId), {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Accept': 'application/vnd.github.v3+json'
                }
            });

            if (artifactsResponse.ok) {
                const artifactsData = await artifactsResponse.json();
                const artifact = artifactsData.artifacts?.find(a => a.name === 'api-data-response');

                if (artifact) {
                    // Download artifact
                    const downloadUrl = artifact.archive_download_url;
                    const downloadResponse = await fetch(downloadUrl, {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    });

                    if (downloadResponse.ok) {
                        const zipBlob = await downloadResponse.blob();
                        // For simplicity, we'll return null if we can't extract
                        // In production, you'd use JSZip to extract the JSON
                        return null;
                    }
                }
            }
            break;
        }

        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    return null;
}

// Add ticker
async function addTicker() {
    const ticker = elements.newTicker.value.trim().toUpperCase();
    if (!ticker) return;

    if (!/^[A-Z]{1,10}$/.test(ticker)) {
        showError('Invalid ticker format');
        return;
    }

    updateStatus('Adding ticker...', 'loading');

    try {
        const token = getGitHubToken();
        const response = await fetch(getApiEndpoints().addTicker(ticker), {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/vnd.github.v3+json'
            },
            body: JSON.stringify({
                ref: 'main',
                inputs: { ticker }
            })
        });

        if (response.ok) {
            elements.newTicker.value = '';
            updateStatus('Ticker added', 'success');
            setTimeout(() => loadAllData(), 2000);
        } else {
            const error = await response.json();
            showError(error.message || 'Failed to add ticker');
        }
    } catch (error) {
        showError('Failed to add ticker');
        console.error(error);
    }
}

// Remove ticker
async function removeTicker(ticker) {
    if (!confirm(`Remove ${ticker} from watchlist?`)) return;

    updateStatus('Removing ticker...', 'loading');

    try {
        const token = getGitHubToken();
        const response = await fetch(getApiEndpoints().removeTicker(ticker), {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/vnd.github.v3+json'
            },
            body: JSON.stringify({
                ref: 'main',
                inputs: { ticker }
            })
        });

        if (response.ok) {
            updateStatus('Ticker removed', 'success');
            setTimeout(() => loadAllData(), 2000);
        } else {
            showError('Failed to remove ticker');
        }
    } catch (error) {
        showError('Failed to remove ticker');
        console.error(error);
    }
}

// Save settings
async function saveSettings() {
    updateStatus('Saving settings...', 'loading');

    try {
        const token = getGitHubToken();
        const response = await fetch(getApiEndpoints().updateSettings(), {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/vnd.github.v3+json'
            },
            body: JSON.stringify({
                ref: 'main',
                inputs: {
                    show_charts: elements.showCharts.value,
                    show_explanation: elements.showExplanation.checked.toString(),
                    refresh_interval: elements.refreshInterval.value
                }
            })
        });

        if (response.ok) {
            updateStatus('Settings saved', 'success');
        } else {
            showError('Failed to save settings');
        }
    } catch (error) {
        showError('Failed to save settings');
        console.error(error);
    }
}

// Run monitor manually
async function runMonitor() {
    if (!confirm('Run stock monitor now?')) return;

    updateStatus('Starting monitor...', 'loading');

    try {
        const token = getGitHubToken();
        const response = await fetch(getApiEndpoints().runMonitor(), {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/vnd.github.v3+json'
            },
            body: JSON.stringify({ ref: 'main' })
        });

        if (response.ok) {
            updateStatus('Monitor started', 'success');
        } else {
            showError('Failed to start monitor');
        }
    } catch (error) {
        showError('Failed to start monitor');
        console.error(error);
    }
}

// Render tickers list
function renderTickers() {
    elements.tickersList.innerHTML = '';

    if (state.tickers.length === 0) {
        elements.tickersList.innerHTML = '<p class="empty-state">No tickers in watchlist</p>';
        return;
    }

    state.tickers.forEach(ticker => {
        const div = document.createElement('div');
        div.className = 'ticker-item';
        div.innerHTML = `
            <span>${ticker}</span>
            <button class="btn-remove" data-ticker="${ticker}">Remove</button>
        `;
        elements.tickersList.appendChild(div);

        div.querySelector('.btn-remove').addEventListener('click', () => removeTicker(ticker));
    });
}

// Render settings
function renderSettings() {
    elements.showCharts.value = state.settings.show_charts || 'buy_only';
    elements.showExplanation.checked = state.settings.show_explanation !== false;
    elements.refreshInterval.value = state.settings.refresh_interval || 30;
}

// Render last run results
function renderLastRun() {
    if (!state.lastRun) {
        elements.lastRunSummary.innerHTML = '<p class="empty-state">No run data available</p>';
        elements.lastRunResults.innerHTML = '';
        elements.chartsSection.style.display = 'none';
        return;
    }

    // Summary
    const date = new Date(state.lastRun.timestamp);
    elements.lastRunSummary.innerHTML = `
        <div class="summary-stats">
            <div class="stat">
                <span class="stat-label">Time</span>
                <span class="stat-value">${date.toLocaleString()}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Analyzed</span>
                <span class="stat-value">${state.lastRun.total_analyzed}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Buy Signals</span>
                <span class="stat-value ${state.lastRun.buy_signals > 0 ? 'buy' : ''}">${state.lastRun.buy_signals}</span>
            </div>
        </div>
    `;

    // Results table
    const results = state.lastRun.results || [];
    if (results.length === 0) {
        elements.lastRunResults.innerHTML = '<p class="empty-state">No results</p>';
        return;
    }

    let tableHtml = '<table><thead><tr><th>Ticker</th><th>Signal</th><th>Price</th><th>VWAP</th><th>Volume Ratio</th><th>Pattern</th></tr></thead><tbody>';

    results.forEach(result => {
        const signalClass = result.signal === 'BUY' ? 'signal-buy' : 'signal-hold';
        tableHtml += `
            <tr>
                <td>${result.ticker}</td>
                <td class="${signalClass}">${result.signal}</td>
                <td>$${result.price?.toFixed(2) || '-'}</td>
                <td>$${result.vwap?.toFixed(2) || '-'}</td>
                <td>${result.volume_ratio?.toFixed(1) || '-'}x</td>
                <td>${result.pattern || '-'}</td>
            </tr>
        `;
    });

    tableHtml += '</tbody></table>';
    elements.lastRunResults.innerHTML = tableHtml;

    // Charts gallery (if enabled)
    if (state.settings.show_charts === 'always' ||
        (state.settings.show_charts === 'buy_only' && state.lastRun.buy_signals > 0)) {
        renderCharts(results);
    }
}

// Render charts gallery
function renderCharts(results) {
    elements.chartsSection.style.display = 'block';
    elements.chartsGallery.innerHTML = '';

    const showBuyOnly = state.settings.show_charts === 'buy_only';
    const filteredResults = showBuyOnly
        ? results.filter(r => r.signal === 'BUY')
        : results;

    if (filteredResults.length === 0) {
        elements.chartsGallery.innerHTML = '<p class="empty-state">No charts to display</p>';
        return;
    }

    filteredResults.forEach(result => {
        if (result.chart_path) {
            const div = document.createElement('div');
            div.className = 'chart-item';
            div.innerHTML = `
                <div class="chart-header">
                    <span>${result.ticker}</span>
                    <span class="${result.signal === 'BUY' ? 'signal-buy' : 'signal-hold'}">${result.signal}</span>
                </div>
                <img src="charts/${result.chart_path}" alt="${result.ticker} chart" class="chart-thumb" onclick="showChart('${result.chart_path}')">
                ${state.settings.show_explanation && result.reasoning ? `<p class="chart-note">${result.reasoning}</p>` : ''}
            `;
            elements.chartsGallery.appendChild(div);
        }
    });
}

// Show chart in modal
function showChart(path) {
    // For GitHub Artifacts, we'd need to construct the artifact URL
    // This is a placeholder - actual implementation would fetch from artifact
    elements.modalChartImage.src = `charts/${path}`;
    elements.chartModal.style.display = 'flex';
}

// Update status indicator
function updateStatus(message, type = 'info') {
    elements.status.textContent = message;
    elements.status.className = `status status-${type}`;
}

// Show error message
function showError(message) {
    updateStatus(message, 'error');
    setTimeout(() => updateStatus('Error', 'error'), 3000);
}

// Auto-refresh
function startAutoRefresh() {
    const interval = (state.settings.refresh_interval || 30) * 1000;
    setInterval(() => {
        if (!state.isLoading) {
            loadAllData();
        }
    }, interval);
}

// Initialize on load
document.addEventListener('DOMContentLoaded', init);
