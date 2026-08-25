/**
 * Bayesian Spam Guard - Interactive Web Application
 * Handles real-time prediction, word-level explainability, batch classification, and Chart.js analytics.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Global App State
  let distributionChartInstance = null;
  let confMatrixChartInstance = null;
  let keywordsChartInstance = null;
  let lastBatchResults = null;
  let typingTimer = null;

  // Preset Sample Messages
  const SAMPLES = {
    lottery: "CONGRATULATIONS! You've won a $1,000 Walmart Gift Card! Call 09061743811 now or visit www.prize-claim.co.uk to claim your reward instantly. T&Cs apply.",
    crypto: "URGENT: Your crypto wallet has earned 2.45 BTC in daily trading rewards. Click http://claim-crypto-bonus.net/payout to deposit funds to your bank account immediately.",
    bank: "URGENT SECURITY ALERT: We detected unauthorized login activity on your online bank account. Please verify your identity now at https://secure-bank-login-verify.com to avoid account suspension.",
    meeting: "Hi team, please find attached the agenda for our upcoming quarterly sprint planning meeting tomorrow at 10:00 AM. Let me know if you have additional topics to discuss.",
    delivery: "Your Amazon order #402-8912384 has been shipped and is scheduled for delivery today by 6:00 PM. Track your package on the official Amazon app.",
    family: "Hey! Are you still free for Sunday family dinner at mom's place? Let me know if you can bring some dessert. See you soon!"
  };

  // Demo 10-Item Batch
  const DEMO_BATCH = [
    "WINNER! As a valued customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim.",
    "Hey, are we still meeting for lunch at 1 PM today?",
    "URGENT! Your mobile number won 2000 bonus points. Reply CLAIM to 88066 now.",
    "Can you please review the attached PDF report before our sync meeting?",
    "Free entry in 2 a weekly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121.",
    "Thanks for helping out yesterday, really appreciate your support!",
    "Customer service alert: Your subscription expires today. Call now to renew with 50% discount.",
    "I'll be home a bit late tonight, traffic is bad.",
    "Double your cash prize today only! Ring 08718726215 immediately.",
    "Don't forget to submit your weekly timesheet by 5pm today."
  ];

  // DOM Elements
  const emailInput = document.getElementById('emailInput');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const clearBtn = document.getElementById('clearBtn');
  const liveModeToggle = document.getElementById('liveModeToggle');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnIcon = document.getElementById('btnIcon');
  const charCountBadge = document.getElementById('charCountBadge');

  const resultEmptyState = document.getElementById('resultEmptyState');
  const resultActiveState = document.getElementById('resultActiveState');
  const verdictBanner = document.getElementById('verdictBanner');
  const verdictTitle = document.getElementById('verdictTitle');
  const verdictIcon = document.getElementById('verdictIcon');
  const confidenceValue = document.getElementById('confidenceValue');
  const spamProbText = document.getElementById('spamProbText');
  const hamProbText = document.getElementById('hamProbText');
  const spamProgressBar = document.getElementById('spamProgressBar');
  const hamProgressBar = document.getElementById('hamProgressBar');
  const highlightedTokensContainer = document.getElementById('highlightedTokensContainer');
  const tokensCountBadge = document.getElementById('tokensCountBadge');
  const topInfluentialChips = document.getElementById('topInfluentialChips');
  const wordExplainTableBody = document.getElementById('wordExplainTableBody');

  // Batch Elements
  const batchFileInput = document.getElementById('batchFileInput');
  const csvDropZone = document.getElementById('csvDropZone');
  const batchTextInput = document.getElementById('batchTextInput');
  const processBatchBtn = document.getElementById('processBatchBtn');
  const batchBtnSpinner = document.getElementById('batchBtnSpinner');
  const exportCsvBtn = document.getElementById('exportCsvBtn');
  const downloadSampleCsvBtn = document.getElementById('downloadSampleCsvBtn');
  const loadDemoBatchBtn = document.getElementById('loadDemoBatchBtn');
  const batchTotalCount = document.getElementById('batchTotalCount');
  const batchSpamCount = document.getElementById('batchSpamCount');
  const batchHamCount = document.getElementById('batchHamCount');
  const batchAvgConf = document.getElementById('batchAvgConf');
  const batchTableBody = document.getElementById('batchTableBody');

  // Initialize
  fetchModelStats();
  setupEventListeners();

  /**
   * Set up all UI event listeners
   */
  function setupEventListeners() {
    // Analyze Button
    analyzeBtn.addEventListener('click', () => {
      analyzeCurrentMessage();
    });

    // Clear Button
    clearBtn.addEventListener('click', () => {
      emailInput.value = '';
      updateCharCounter();
      resetResultCard();
    });

    // Text Input Keyup / Input counter & Live prediction
    emailInput.addEventListener('input', () => {
      updateCharCounter();
      if (liveModeToggle.checked) {
        clearTimeout(typingTimer);
        typingTimer = setTimeout(() => {
          if (emailInput.value.trim().length > 3) {
            analyzeCurrentMessage(true);
          }
        }, 350);
      }
    });

    // Preset Sample Pills
    document.querySelectorAll('.btn-sample-pill').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const sampleKey = btn.getAttribute('data-sample');
        if (SAMPLES[sampleKey]) {
          emailInput.value = SAMPLES[sampleKey];
          updateCharCounter();
          analyzeCurrentMessage();
        }
      });
    });

    // Header Stats Button scroll
    const headerStatsBtn = document.getElementById('headerStatsBtn');
    if (headerStatsBtn) {
      headerStatsBtn.addEventListener('click', (e) => {
        e.preventDefault();
        const analyticsTab = document.getElementById('analytics-tab');
        if (analyticsTab) {
          bootstrap.Tab.getOrCreateInstance(analyticsTab).show();
        }
      });
    }

    // Batch Drag & Drop
    ['dragenter', 'dragover'].forEach(eventName => {
      csvDropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        csvDropZone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      csvDropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        csvDropZone.classList.remove('dragover');
      });
    });

    csvDropZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        handleFileUpload(files[0]);
      }
    });

    batchFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
      }
    });

    // Direct Batch Textarea Process
    processBatchBtn.addEventListener('click', () => {
      const text = batchTextInput.value.trim();
      if (!text) {
        alert('Please paste some text lines or upload a CSV file.');
        return;
      }
      const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
      if (lines.length === 0) return;
      processBatchPayload({ messages: lines });
    });

    // Demo Batch Button
    loadDemoBatchBtn.addEventListener('click', () => {
      batchTextInput.value = DEMO_BATCH.join('\n');
      processBatchPayload({ messages: DEMO_BATCH });
    });

    // Download Sample CSV Template
    downloadSampleCsvBtn.addEventListener('click', () => {
      downloadSampleCsv();
    });

    // Export Batch Results to CSV
    exportCsvBtn.addEventListener('click', () => {
      exportResultsToCsv();
    });
  }

  /**
   * Update character and word counter
   */
  function updateCharCounter() {
    const text = emailInput.value;
    const chars = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    charCountBadge.textContent = `${chars} chars • ${words} words`;
  }

  /**
   * Reset result card to placeholder state
   */
  function resetResultCard() {
    resultActiveState.classList.add('d-none');
    resultEmptyState.classList.remove('d-none');
    wordExplainTableBody.innerHTML = '<tr><td colspan="5" class="text-center text-muted-light py-3">Analyze an email to view word-level mathematical attribution.</td></tr>';
  }

  /**
   * Send single message to /predict endpoint
   */
  async function analyzeCurrentMessage(isSilent = false) {
    const text = emailInput.value.trim();
    if (!text) {
      if (!isSilent) alert('Please enter an email or message to analyze.');
      return;
    }

    if (!isSilent) {
      btnSpinner.classList.remove('d-none');
      btnIcon.classList.add('d-none');
      analyzeBtn.disabled = true;
    }

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Prediction failed');
      }

      const data = await res.json();
      renderPredictionResult(data);
    } catch (err) {
      console.error('Prediction error:', err);
      if (!isSilent) alert('Failed to classify message: ' + err.message);
    } finally {
      if (!isSilent) {
        btnSpinner.classList.add('d-none');
        btnIcon.classList.remove('d-none');
        analyzeBtn.disabled = false;
      }
    }
  }

  /**
   * Render single prediction result and word explainability
   */
  function renderPredictionResult(data) {
    resultEmptyState.classList.add('d-none');
    resultActiveState.classList.remove('d-none');
    resultActiveState.classList.add('d-flex');

    const isSpam = data.prediction === 'spam';
    const spamPct = (data.spam_probability * 100).toFixed(1);
    const hamPct = (data.ham_probability * 100).toFixed(1);
    const confPct = (data.confidence * 100).toFixed(1);

    // Update Verdict Banner
    verdictBanner.className = 'verdict-banner p-3 rounded-3 mb-3 d-flex align-items-center justify-content-between ' + 
      (isSpam ? 'verdict-spam' : 'verdict-ham');
    
    verdictTitle.textContent = isSpam ? 'SPAM DETECTED' : 'LEGITIMATE (HAM)';
    verdictTitle.style.color = isSpam ? 'var(--danger)' : 'var(--success)';
    
    verdictIcon.className = isSpam ? 'fa-solid fa-shield-virus' : 'fa-solid fa-circle-check';
    confidenceValue.textContent = `${confPct}%`;

    // Probability Bars
    spamProbText.textContent = `${spamPct}%`;
    hamProbText.textContent = `${hamPct}%`;
    spamProgressBar.style.width = `${spamPct}%`;
    hamProgressBar.style.width = `${hamPct}%`;

    // Highlighted Tokens Rendering
    highlightedTokensContainer.innerHTML = '';
    let recognizedTokens = 0;

    data.token_details.forEach(tok => {
      const span = document.createElement('span');
      span.className = `token-tag token-${tok.impact}`;
      span.textContent = tok.raw;
      
      if (tok.in_vocab && tok.score !== 0) {
        recognizedTokens++;
        const sign = tok.score > 0 ? '+' : '';
        span.title = `Word: "${tok.stem}" | Log-Odds (Δ): ${sign}${tok.score} | Bias: ${tok.impact.toUpperCase()}`;
      }
      highlightedTokensContainer.appendChild(span);
      highlightedTokensContainer.appendChild(document.createTextNode(' '));
    });

    tokensCountBadge.textContent = `${recognizedTokens} active features`;

    // Top Influential Chips
    topInfluentialChips.innerHTML = '';
    const topWords = (data.word_impacts || []).slice(0, 6);
    if (topWords.length === 0) {
      topInfluentialChips.innerHTML = '<span class="x-small text-muted-light">No strong vocabulary keywords detected.</span>';
    } else {
      topWords.forEach(w => {
        const chip = document.createElement('span');
        chip.className = `influential-chip chip-${w.impact}`;
        const icon = w.impact === 'spam' ? 'fa-triangle-exclamation text-danger' : 'fa-check text-success';
        const sign = w.score > 0 ? '+' : '';
        chip.innerHTML = `<i class="fa-solid ${icon}"></i> ${w.word} <span class="opacity-75 font-monospace">(${sign}${w.score})</span>`;
        topInfluentialChips.appendChild(chip);
      });
    }

    // Explainability Table Body
    renderExplainTable(data.word_impacts || []);
  }

  /**
   * Render word impact rows in explainability table
   */
  function renderExplainTable(impacts) {
    if (!impacts || impacts.length === 0) {
      wordExplainTableBody.innerHTML = '<tr><td colspan="5" class="text-center text-muted-light py-3">No matching vocabulary tokens found in this message.</td></tr>';
      return;
    }

    wordExplainTableBody.innerHTML = '';
    impacts.forEach(item => {
      const tr = document.createElement('tr');
      const isSpamBias = item.impact === 'spam';
      const isHamBias = item.impact === 'ham';
      const sign = item.score > 0 ? '+' : '';
      const badgeClass = isSpamBias ? 'bg-danger text-white' : (isHamBias ? 'bg-success text-white' : 'bg-secondary text-white');
      const barColor = isSpamBias ? 'bg-danger' : (isHamBias ? 'bg-success' : 'bg-secondary');
      
      // Calculate visual bar width normalized to max scale ~5.0
      const absScore = Math.abs(item.score);
      const barWidth = Math.min(100, Math.round((absScore / 5.0) * 100));

      tr.innerHTML = `
        <td class="fw-bold font-monospace text-white">${escapeHtml(item.word)}</td>
        <td><span class="badge bg-dark-glass">${item.count}</span></td>
        <td><span class="badge ${badgeClass}">${item.impact.toUpperCase()}</span></td>
        <td class="font-monospace fw-semibold ${isSpamBias ? 'text-danger' : (isHamBias ? 'text-success' : 'text-muted')}">${sign}${item.score}</td>
        <td style="min-width: 140px;">
          <div class="progress progress-glass" style="height: 6px;">
            <div class="progress-bar ${barColor}" style="width: ${barWidth}%"></div>
          </div>
        </td>
      `;
      wordExplainTableBody.appendChild(tr);
    });
  }

  /**
   * Batch File Upload Handler
   */
  function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    document.getElementById('dropZoneTitle').textContent = `Uploaded: ${file.name}`;
    processBatchPayload(formData, true);
  }

  /**
   * Process Batch Payload (/batch-predict)
   */
  async function processBatchPayload(payload, isFormData = false) {
    batchBtnSpinner.classList.remove('d-none');
    processBatchBtn.disabled = true;

    try {
      const options = {
        method: 'POST',
        body: isFormData ? payload : JSON.stringify(payload)
      };
      if (!isFormData) {
        options.headers = { 'Content-Type': 'application/json' };
      }

      const res = await fetch('/batch-predict', options);
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Batch processing failed');
      }

      const summary = await res.json();
      lastBatchResults = summary;
      renderBatchResults(summary);
    } catch (err) {
      console.error('Batch error:', err);
      alert('Batch classification failed: ' + err.message);
    } finally {
      batchBtnSpinner.classList.add('d-none');
      processBatchBtn.disabled = false;
    }
  }

  /**
   * Render Batch Classification Results
   */
  function renderBatchResults(summary) {
    batchTotalCount.textContent = summary.total;
    batchSpamCount.textContent = `${summary.spam_count} (${summary.spam_rate}%)`;
    batchHamCount.textContent = `${summary.ham_count} (${(100 - summary.spam_rate).toFixed(1)}%)`;
    batchAvgConf.textContent = `${(summary.avg_confidence * 100).toFixed(1)}%`;

    exportCsvBtn.classList.remove('d-none');

    batchTableBody.innerHTML = '';
    summary.results.forEach(row => {
      const tr = document.createElement('tr');
      const isSpam = row.prediction === 'spam';
      const badgeClass = isSpam ? 'bg-danger' : 'bg-success';
      const confPct = (row.confidence * 100).toFixed(1);
      const spamPct = (row.spam_probability * 100).toFixed(1);

      tr.innerHTML = `
        <td class="font-monospace text-muted">${row.id}</td>
        <td class="text-truncate" style="max-width: 250px;" title="${escapeHtml(row.full_message)}">${escapeHtml(row.message)}</td>
        <td><span class="badge ${badgeClass} text-uppercase">${row.prediction}</span></td>
        <td class="font-monospace ${isSpam ? 'text-danger fw-bold' : 'text-muted'}">${spamPct}%</td>
        <td class="font-monospace">${confPct}%</td>
      `;
      batchTableBody.appendChild(tr);
    });
  }

  /**
   * Download Sample CSV
   */
  function downloadSampleCsv() {
    const csvContent = "message\n" +
      "\"CONGRATULATIONS! You have won $1,000 Walmart prize. Call 09061743811 now to claim.\"\n" +
      "\"Hi team, attached is the presentation for our sync meeting at 2pm.\"\n" +
      "\"URGENT: Your account was locked due to suspicious activity. Verify identity now.\"\n" +
      "\"Hey, are we still on for lunch tomorrow afternoon?\"\n" +
      "\"Free entry into our monthly cash draw! Text WIN to 88066.\"\n" +
      "\"Package delivered: Your shipment was left at the front door.\"\n";

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', 'sample_spam_test.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  /**
   * Export Batch Predictions to CSV
   */
  function exportResultsToCsv() {
    if (!lastBatchResults || !lastBatchResults.results) return;

    let csv = "ID,Prediction,Spam_Probability,Ham_Probability,Confidence,Message\n";
    lastBatchResults.results.forEach(r => {
      const cleanMsg = `"${r.full_message.replace(/"/g, '""')}"`;
      csv += `${r.id},${r.prediction},${r.spam_probability},${r.ham_probability},${r.confidence},${cleanMsg}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `spam_batch_predictions_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  /**
   * Fetch Model Performance & Dataset Metrics from /stats
   */
  async function fetchModelStats() {
    try {
      const res = await fetch('/stats');
      if (!res.ok) throw new Error('Stats fetch failed');
      const stats = await res.json();
      
      populateStatCards(stats);
      renderDistributionChart(stats);
      renderConfusionMatrixChart(stats);
      renderKeywordsChart(stats);
    } catch (err) {
      console.error('Failed to load stats:', err);
    }
  }

  /**
   * Populate Metric Numbers
   */
  function populateStatCards(stats) {
    if (stats.accuracy) {
      const acc = (stats.accuracy * 100).toFixed(1) + '%';
      document.getElementById('statAccuracy').textContent = acc;
      document.getElementById('headerAccBadge').textContent = `${acc} Acc`;
    }
    if (stats.precision) {
      document.getElementById('statPrecision').textContent = (stats.precision * 100).toFixed(1) + '%';
    }
    if (stats.recall) {
      document.getElementById('statRecall').textContent = (stats.recall * 100).toFixed(1) + '%';
    }
    if (stats.f1_score) {
      document.getElementById('statF1').textContent = (stats.f1_score * 100).toFixed(1) + '%';
    }
    if (stats.total_records) {
      document.getElementById('statRecords').textContent = stats.total_records.toLocaleString();
    }
    if (stats.vocabulary_size) {
      document.getElementById('statVocab').textContent = stats.vocabulary_size.toLocaleString();
    }
    if (stats.ham_count && stats.spam_count) {
      const total = stats.ham_count + stats.spam_count;
      document.getElementById('distHamCount').textContent = `${stats.ham_count.toLocaleString()} (${((stats.ham_count / total) * 100).toFixed(1)}%)`;
      document.getElementById('distSpamCount').textContent = `${stats.spam_count.toLocaleString()} (${((stats.spam_count / total) * 100).toFixed(1)}%)`;
    }
  }

  /**
   * Chart 1: Dataset Distribution (Doughnut)
   */
  function renderDistributionChart(stats) {
    const ctx = document.getElementById('distChart');
    if (!ctx) return;

    if (distributionChartInstance) {
      distributionChartInstance.destroy();
    }

    const hamCount = stats.ham_count || 4825;
    const spamCount = stats.spam_count || 747;

    distributionChartInstance = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Legitimate (Ham)', 'Spam'],
        datasets: [{
          data: [hamCount, spamCount],
          backgroundColor: ['#10b981', '#ef4444'],
          borderColor: 'rgba(22, 28, 45, 0.9)',
          borderWidth: 3,
          hoverOffset: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '72%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 12 } }
          },
          tooltip: {
            callbacks: {
              label: (item) => {
                const total = hamCount + spamCount;
                const pct = ((item.raw / total) * 100).toFixed(1);
                return ` ${item.label}: ${item.raw.toLocaleString()} messages (${pct}%)`;
              }
            }
          }
        }
      }
    });
  }

  /**
   * Chart 2: Confusion Matrix Heatmap / Breakdown
   */
  function renderConfusionMatrixChart(stats) {
    const ctx = document.getElementById('confMatrixChart');
    if (!ctx) return;

    if (confMatrixChartInstance) {
      confMatrixChartInstance.destroy();
    }

    // Default or API confusion matrix: [[True Spam, False Ham], [False Spam, True Ham]]
    const matrix = stats.confusion_matrix || [[137, 12], [5, 961]];
    const trueSpam = matrix[0][0];
    const falseHam = matrix[0][1];
    const falseSpam = matrix[1][0];
    const trueHam = matrix[1][1];

    confMatrixChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['True Spam', 'True Ham', 'False Ham (Missed)', 'False Spam (Alarm)'],
        datasets: [{
          label: 'Test Set Predictions',
          data: [trueSpam, trueHam, falseHam, falseSpam],
          backgroundColor: [
            'rgba(16, 185, 129, 0.85)',
            'rgba(99, 102, 241, 0.85)',
            'rgba(245, 158, 11, 0.85)',
            'rgba(239, 68, 68, 0.85)'
          ],
          borderColor: [
            '#10b981',
            '#6366f1',
            '#f59e0b',
            '#ef4444'
          ],
          borderWidth: 1.5,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (item) => ` Count: ${item.raw} messages`
            }
          }
        },
        scales: {
          x: {
            ticks: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 11 } },
            grid: { display: false }
          },
          y: {
            ticks: { color: '#64748b' },
            grid: { color: 'rgba(255, 255, 255, 0.06)' }
          }
        }
      }
    });
  }

  /**
   * Chart 3: Top Predictive Keywords
   */
  function renderKeywordsChart(stats) {
    const ctx = document.getElementById('keywordsChart');
    if (!ctx) return;

    if (keywordsChartInstance) {
      keywordsChartInstance.destroy();
    }

    const topSpam = (stats.top_spam_keywords || []).slice(0, 8);
    const topHam = (stats.top_ham_keywords || []).slice(0, 8);

    const labels = [...topSpam.map(k => k.word), ...topHam.map(k => k.word)];
    const values = [...topSpam.map(k => k.score), ...topHam.map(k => -k.score)];
    const colors = [
      ...topSpam.map(() => 'rgba(239, 68, 68, 0.8)'),
      ...topHam.map(() => 'rgba(16, 185, 129, 0.8)')
    ];

    keywordsChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Log-Odds (Red = Spam, Green = Ham)',
          data: values,
          backgroundColor: colors,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (item) => {
                const isSpam = item.raw >= 0;
                return ` ${item.label}: ${item.raw > 0 ? '+' : ''}${item.raw.toFixed(2)} (${isSpam ? 'Spam bias' : 'Ham bias'})`;
              }
            }
          }
        },
        scales: {
          x: {
            ticks: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 10 } },
            grid: { color: 'rgba(255, 255, 255, 0.06)' }
          },
          y: {
            ticks: { color: '#f1f5f9', font: { family: 'JetBrains Mono', size: 11 } },
            grid: { display: false }
          }
        }
      }
    });
  }

  // Utility to prevent XSS
  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

});
