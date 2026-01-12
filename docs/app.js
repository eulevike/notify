// Application State
const state = {
    tickers: [],
    lastRun: null,
    isLoading: false,
    refreshInterval: 30000 // 30 seconds default
};

// DOM Elements
const elements = {
    status: document.getElementById('status'),
    tickersList: document.getElementById('tickersList'),
    lastRunSummary: document.getElementById('lastRunSummary'),
    lastRunResults: document.getElementById('lastRunResults'),
    refreshBtn: document.getElementById('refreshBtn'),
    runMonitorBtn: document.getElementById('runMonitorBtn'),
    chartsSection: document.getElementById('chartsSection'),
    chartsGallery: document.getElementById('chartsGallery')
};

// Raw GitHub content URLs
function getRawUrls() {
    const baseUrl = `https://raw.githubusercontent.com/${CONFIG.repo}/main`;
    return {
        tickers: `${baseUrl}/tickers.txt`,
        lastRun: `${baseUrl}/last_run.json`,
        lastRunLSE: `${baseUrl}/last_run_lse.json`,
        lastRunUS: `${baseUrl}/last_run_us.json`,
        lastRunMU: `${baseUrl}/last_run_mu.json`,
        settings: `${baseUrl}/settings.json`,
        workflows: `https://github.com/${CONFIG.repo}/actions`
    };
}

// Initialize app
function init() {
    setupEventListeners();
    loadAllData();
    startAutoRefresh();
}

// Setup event listeners
function setupEventListeners() {
    elements.refreshBtn.addEventListener('click', () => loadAllData());
    elements.runMonitorBtn.addEventListener('click', () => {
        window.open(getRawUrls().workflows, '_blank');
    });
}

// Load all data from GitHub (public, no auth needed)
async function loadAllData() {
    if (state.isLoading) return;
    state.isLoading = true;
    updateStatus('Loading...', 'loading');

    try {
        const urls = getRawUrls();

        // Fetch tickers
        const tickersResponse = await fetch(urls.tickers);
        if (tickersResponse.ok) {
            const tickersText = await tickersResponse.text();
            state.tickers = tickersText
                .split('\n')
                .map(line => line.trim().toUpperCase())
                .filter(line => line && !line.startsWith('#'));
        } else {
            state.tickers = [];
        }

        // Fetch all exchange last_run files and merge them
        const [lseResponse, usResponse, muResponse] = await Promise.all([
            fetch(urls.lastRunLSE),
            fetch(urls.lastRunUS),
            fetch(urls.lastRunMU)
        ]);

        const exchanges = [];
        if (lseResponse.ok) exchanges.push(await lseResponse.json());
        if (usResponse.ok) exchanges.push(await usResponse.json());
        if (muResponse.ok) exchanges.push(await muResponse.json());

        // Merge exchange results into a single view
        state.lastRun = mergeExchangeResults(exchanges);

        // Fetch settings
        const settingsResponse = await fetch(urls.settings);
        if (settingsResponse.ok) {
            const settings = await settingsResponse.json();
            state.refreshInterval = (settings.refresh_interval || 30) * 1000;
        }

        // Render UI
        renderTickers();
        renderLastRun();

        updateStatus('Updated', 'success');

    } catch (error) {
        console.error('Error loading data:', error);
        showError('Failed to load data');
    } finally {
        state.isLoading = false;
    }
}

// Merge results from multiple exchanges into a single view
function mergeExchangeResults(exchanges) {
    if (exchanges.length === 0) return null;

    // Filter out null/undefined exchanges
    const validExchanges = exchanges.filter(e => e !== null && e !== undefined);
    if (validExchanges.length === 0) return null;

    // Sort by timestamp (most recent first)
    validExchanges.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    // Use the most recent timestamp for display
    const merged = {
        timestamp: validExchanges[0].timestamp,
        status: 'completed',
        total_analyzed: 0,
        buy_signals: 0,
        results: [],
        exchanges: [] // Track which exchanges contributed
    };

    // Merge all results
    validExchanges.forEach(exchange => {
        if (exchange.results) {
            merged.total_analyzed += exchange.total_analyzed || 0;
            merged.buy_signals += exchange.buy_signals || 0;
            merged.results.push(...(exchange.results || []));

            // Track exchange info
            const exchangeName = getExchangeName(exchange);
            if (exchangeName && !merged.exchanges.includes(exchangeName)) {
                merged.exchanges.push(exchangeName);
            }
        }
    });

    // If all were skipped
    if (merged.total_analyzed === 0 && validExchanges.some(e => e.status === 'skipped')) {
        const skipped = validExchanges.find(e => e.status === 'skipped');
        return skipped;
    }

    return merged;
}

// Get exchange name from result data
function getExchangeName(exchange) {
    // Try to determine exchange from the data source
    if (exchange.results && exchange.results.length > 0) {
        const ticker = exchange.results[0].ticker || '';
        if (ticker.endsWith('.L')) return 'LSE';
        if (ticker.endsWith('.MU')) return 'Munich';
        return 'US';
    }
    return null;
}

// Get exchange name from ticker symbol
function getExchangeFromTicker(ticker) {
    if (ticker.endsWith('.L')) return 'LSE';
    if (ticker.endsWith('.MU')) return 'Munich';
    return 'US';
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
        div.innerHTML = `<span>${ticker}</span>`;
        elements.tickersList.appendChild(div);
    });
}

// Render last run results
function renderLastRun() {
    if (!state.lastRun) {
        elements.lastRunSummary.innerHTML = '<p class="empty-state">No run data available</p>';
        elements.lastRunResults.innerHTML = '';
        elements.chartsSection.style.display = 'none';
        return;
    }

    // Handle skipped runs
    if (state.lastRun.status === 'skipped') {
        const timestamp = state.lastRun.timestamp;
        const timeDisplay = timestamp ? new Date(timestamp).toLocaleString() : 'Unknown';
        elements.lastRunSummary.innerHTML = `
            <div class="summary-stats">
                <div class="stat">
                    <span class="stat-label">Last Check</span>
                    <span class="stat-value">${timeDisplay}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Status</span>
                    <span class="stat-value" style="color: var(--warning)">Skipped</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Reason</span>
                    <span class="stat-value">${state.lastRun.skip_reason || 'Market closed'}</span>
                </div>
            </div>
            <p class="center-note">
                Click <strong>Run on GitHub</strong> to run the monitor manually (bypasses market hours)
            </p>
        `;
        elements.lastRunResults.innerHTML = '';
        elements.chartsSection.style.display = 'none';
        return;
    }

    // Summary
    const timestamp = state.lastRun.timestamp;
    const timeDisplay = timestamp ? new Date(timestamp).toLocaleString() : 'No data yet';
    const exchanges = state.lastRun.exchanges || [];
    const exchangesDisplay = exchanges.length > 0 ? exchanges.join(', ') : 'All';

    elements.lastRunSummary.innerHTML = `
        <div class="summary-stats">
            <div class="stat">
                <span class="stat-label">Time</span>
                <span class="stat-value">${timeDisplay}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Exchanges</span>
                <span class="stat-value">${exchangesDisplay}</span>
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

    let tableHtml = '<table><thead><tr><th>Ticker</th><th>Exchange</th><th>Signal</th><th>Price</th><th>VWAP</th><th>Volume Ratio</th><th>Pattern</th></tr></thead><tbody>';

    results.forEach(result => {
        const signalClass = result.signal === 'BUY' ? 'signal-buy' : 'signal-hold';
        const exchange = getExchangeFromTicker(result.ticker);
        tableHtml += `
            <tr>
                <td>${result.ticker}</td>
                <td><span class="exchange-badge exchange-${exchange.toLowerCase()}">${exchange}</span></td>
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

    // Charts gallery
    if (state.lastRun.buy_signals > 0) {
        renderCharts(results);
    }
}

// Render charts gallery
function renderCharts(results) {
    elements.chartsSection.style.display = 'block';
    elements.chartsGallery.innerHTML = '';

    const buyResults = results.filter(r => r.signal === 'BUY' && r.chart_path);

    if (buyResults.length === 0) {
        elements.chartsGallery.innerHTML = '<p class="empty-state">No buy signal charts to display</p>';
        return;
    }

    buyResults.forEach(result => {
        // Use GitHub Actions artifact URL for charts
        const artifactUrl = `https://github.com/${CONFIG.repo}/actions/workflows/stock-monitor.yml`;

        const div = document.createElement('div');
        div.className = 'chart-item';
        div.innerHTML = `
            <div class="chart-header">
                <span>${result.ticker}</span>
                <span class="signal-buy">${result.signal}</span>
            </div>
            <div class="chart-placeholder">
                <p>📊 Chart available in GitHub Actions</p>
                <a href="${artifactUrl}" target="_blank" class="btn-link">View on GitHub</a>
            </div>
            ${result.reasoning ? `<p class="chart-note">${result.reasoning}</p>` : ''}
        `;
        elements.chartsGallery.appendChild(div);
    });
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
let refreshTimer;
function startAutoRefresh() {
    if (refreshTimer) clearInterval(refreshTimer);

    refreshTimer = setInterval(() => {
        if (!state.isLoading) {
            loadAllData();
        }
    }, state.refreshInterval);
}

// Initialize on load
document.addEventListener('DOMContentLoaded', init);
