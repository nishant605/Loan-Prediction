// ============================================
// LoanAI — Performance Dashboard Logic
// ============================================

document.addEventListener('DOMContentLoaded', async () => {

    // ---- Chart.js Global Config ----
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
    Chart.defaults.font.family = "'Inter', system-ui, sans-serif";

    // ---- Navbar scroll ----
    const navbar = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 20);
    });

    // ---- Fetch metrics ----
    let data;
    try {
        const resp = await fetch('/static/metrics.json');
        data = await resp.json();
    } catch (e) {
        console.error('Failed to load metrics:', e);
        return;
    }

    renderModelInfo(data.model_info);
    renderMetricCards(data.scores);
    renderConfusionMatrix(data.confusion_matrix);
    renderROCCurve(data.roc_curve);
    renderPRCurve(data.pr_curve);
    renderFeatureImportance(data.feature_importance);
    renderCrossValidation(data.cross_validation);
    renderClassDistribution(data.class_distribution);
    renderCibilDistribution(data.cibil_distribution);
    renderPredictionProbability(data.prediction_distribution);
    renderMetricsTable(data.scores);

    // Animate metric bars after paint
    setTimeout(() => {
        document.querySelectorAll('.metric-bar-fill').forEach(el => {
            el.style.width = el.dataset.width;
        });
    }, 300);


    // =============================================
    //  RENDER FUNCTIONS
    // =============================================

    function renderModelInfo(info) {
        const bar = document.getElementById('model-info-bar');
        const tags = [
            { icon: '🤖', label: 'Algorithm', value: info.algorithm },
            { icon: '🌲', label: 'Estimators', value: info.n_estimators },
            { icon: '📏', label: 'Max Depth', value: info.max_depth },
            { icon: '📚', label: 'Learning Rate', value: info.learning_rate },
            { icon: '📊', label: 'Samples', value: info.total_samples.toLocaleString() },
            { icon: '🔢', label: 'Features', value: info.n_features },
        ];
        bar.innerHTML = tags.map(t => `
      <div class="model-info-tag">
        <span class="tag-icon">${t.icon}</span>
        ${t.label}: <span class="tag-value">${t.value}</span>
      </div>
    `).join('');
    }


    function renderMetricCards(scores) {
        const grid = document.getElementById('metrics-grid');
        const cards = [
            { key: 'accuracy', label: 'Accuracy', accent: 'blue', icon: '🎯' },
            { key: 'precision', label: 'Precision', accent: 'green', icon: '✅' },
            { key: 'recall', label: 'Recall', accent: 'amber', icon: '📡' },
            { key: 'f1_score', label: 'F1 Score', accent: 'purple', icon: '⚡' },
            { key: 'roc_auc', label: 'ROC AUC', accent: 'cyan', icon: '📈' },
            { key: 'specificity', label: 'Specificity', accent: 'blue', icon: '🛡️' },
            { key: 'mcc', label: 'MCC', accent: 'green', icon: '🧮' },
            { key: 'kappa', label: 'Cohen\'s Kappa', accent: 'amber', icon: '📐' },
            { key: 'log_loss', label: 'Log Loss', accent: 'rose', icon: '📉' },
            { key: 'avg_precision', label: 'Avg Precision', accent: 'purple', icon: '🏆' },
        ];

        grid.innerHTML = cards.map(c => {
            const val = scores[c.key];
            const display = c.key === 'log_loss' ? val.toFixed(4) : (val * 100).toFixed(2) + '%';
            const barWidth = c.key === 'log_loss' ? Math.max(0, (1 - val) * 100) : val * 100;
            return `
        <div class="metric-card accent-${c.accent}">
          <div class="metric-label">${c.icon} ${c.label}</div>
          <div class="metric-value">${display}</div>
          <div class="metric-bar">
            <div class="metric-bar-fill" data-width="${barWidth}%"></div>
          </div>
        </div>
      `;
        }).join('');
    }


    function renderConfusionMatrix(cm) {
        const container = document.getElementById('confusion-matrix');
        const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
        const total = tn + fp + fn + tp;

        container.innerHTML = `
      <div class="cm-grid">
        <div class="cm-corner"></div>
        <div class="cm-header">Pred: Rejected</div>
        <div class="cm-header">Pred: Approved</div>

        <div class="cm-row-header">Actual: Rejected</div>
        <div class="cm-cell tn">
          <span class="cm-value">${tn}</span>
          <span class="cm-label">True Neg (${(tn / total * 100).toFixed(1)}%)</span>
        </div>
        <div class="cm-cell fp">
          <span class="cm-value">${fp}</span>
          <span class="cm-label">False Pos (${(fp / total * 100).toFixed(1)}%)</span>
        </div>

        <div class="cm-row-header">Actual: Approved</div>
        <div class="cm-cell fn">
          <span class="cm-value">${fn}</span>
          <span class="cm-label">False Neg (${(fn / total * 100).toFixed(1)}%)</span>
        </div>
        <div class="cm-cell tp">
          <span class="cm-value">${tp}</span>
          <span class="cm-label">True Pos (${(tp / total * 100).toFixed(1)}%)</span>
        </div>
      </div>
      <div class="cm-axis-labels">
        <span class="cm-axis-label"><strong>Total:</strong> ${total.toLocaleString()} samples</span>
        <span class="cm-axis-label"><strong>Errors:</strong> ${fp + fn}</span>
      </div>
    `;
    }


    function renderROCCurve(roc) {
        document.getElementById('roc-auc-label').textContent = `AUC = ${roc.auc}`;

        const points = roc.fpr.map((f, i) => ({ x: f, y: roc.tpr[i] }));

        new Chart(document.getElementById('roc-chart'), {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'ROC Curve',
                        data: points,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.08)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        borderWidth: 2.5,
                    },
                    {
                        label: 'Random Classifier',
                        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                        borderColor: 'rgba(255,255,255,0.15)',
                        borderDash: [6, 4],
                        pointRadius: 0,
                        borderWidth: 1.5,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'False Positive Rate' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                    y: { min: 0, max: 1, title: { display: true, text: 'True Positive Rate' }, grid: { color: 'rgba(255,255,255,0.04)' } }
                },
                plugins: {
                    legend: { display: true, labels: { boxWidth: 14, padding: 14, usePointStyle: true, pointStyle: 'line' } },
                    tooltip: { mode: 'nearest', intersect: false }
                }
            }
        });
    }


    function renderPRCurve(pr) {
        document.getElementById('pr-auc-label').textContent = `AP = ${pr.avg_precision}`;

        const points = pr.recall.map((r, i) => ({ x: r, y: pr.precision[i] }));

        new Chart(document.getElementById('pr-chart'), {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Precision-Recall',
                    data: points,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.08)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 2.5,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Recall' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                    y: { min: 0, max: 1, title: { display: true, text: 'Precision' }, grid: { color: 'rgba(255,255,255,0.04)' } }
                },
                plugins: {
                    legend: { display: true, labels: { boxWidth: 14, padding: 14, usePointStyle: true, pointStyle: 'line' } },
                    tooltip: { mode: 'nearest', intersect: false }
                }
            }
        });
    }


    function renderFeatureImportance(fi) {
        // Top 12 features
        const top = 12;
        const names = fi.names.slice(0, top).reverse();
        const values = fi.values.slice(0, top).reverse();

        const colors = values.map((_, i) => {
            const ratio = i / (values.length - 1);
            return `hsl(${210 + ratio * 80}, 70%, ${55 + ratio * 10}%)`;
        });

        new Chart(document.getElementById('feature-chart'), {
            type: 'bar',
            data: {
                labels: names,
                datasets: [{
                    label: 'Importance',
                    data: values,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c.replace('70%', '80%')),
                    borderWidth: 1,
                    borderRadius: 4,
                    barThickness: 18,
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Importance Score' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                    y: { grid: { display: false }, ticks: { font: { size: 11, family: "'JetBrains Mono', monospace" } } }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => `Importance: ${ctx.parsed.x.toFixed(4)}`
                        }
                    }
                }
            }
        });
    }


    function renderCrossValidation(cv) {
        const metrics = ['accuracy', 'f1', 'precision', 'recall'];
        const colors = {
            accuracy: '#3b82f6',
            f1: '#8b5cf6',
            precision: '#10b981',
            recall: '#f59e0b',
        };

        const datasets = metrics.map(m => ({
            label: `${m.charAt(0).toUpperCase() + m.slice(1)} (μ=${(cv[m].mean * 100).toFixed(1)}%)`,
            data: cv[m].scores.map(s => +(s * 100).toFixed(2)),
            backgroundColor: colors[m] + '30',
            borderColor: colors[m],
            borderWidth: 2,
            pointRadius: 5,
            pointBackgroundColor: colors[m],
            pointBorderColor: '#111827',
            pointBorderWidth: 2,
            tension: 0.3,
        }));

        new Chart(document.getElementById('cv-chart'), {
            type: 'line',
            data: {
                labels: ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'],
                datasets,
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: 85, max: 100, title: { display: true, text: 'Score (%)' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                    x: { grid: { color: 'rgba(255,255,255,0.04)' } }
                },
                plugins: {
                    legend: { labels: { boxWidth: 14, padding: 14, usePointStyle: true, pointStyle: 'circle' } },
                    tooltip: { mode: 'index', intersect: false }
                },
                interaction: { mode: 'index', intersect: false }
            }
        });
    }


    function renderClassDistribution(cd) {
        new Chart(document.getElementById('class-chart'), {
            type: 'doughnut',
            data: {
                labels: cd.labels,
                datasets: [{
                    data: cd.counts,
                    backgroundColor: ['rgba(244, 63, 94, 0.7)', 'rgba(16, 185, 129, 0.7)'],
                    borderColor: ['#f43f5e', '#10b981'],
                    borderWidth: 2,
                    hoverOffset: 12,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '62%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { padding: 20, boxWidth: 14, usePointStyle: true, pointStyle: 'circle' }
                    },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                                const pct = (ctx.parsed / total * 100).toFixed(1);
                                return ` ${ctx.label}: ${ctx.parsed.toLocaleString()} (${pct}%)`;
                            }
                        }
                    }
                }
            }
        });
    }


    function renderCibilDistribution(cd) {
        new Chart(document.getElementById('cibil-chart'), {
            type: 'bar',
            data: {
                labels: cd.bins,
                datasets: [
                    {
                        label: 'Approved',
                        data: cd.approved,
                        backgroundColor: 'rgba(16, 185, 129, 0.6)',
                        borderColor: '#10b981',
                        borderWidth: 1,
                        borderRadius: 3,
                    },
                    {
                        label: 'Rejected',
                        data: cd.rejected,
                        backgroundColor: 'rgba(244, 63, 94, 0.6)',
                        borderColor: '#f43f5e',
                        borderWidth: 1,
                        borderRadius: 3,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'CIBIL Score Range' }, grid: { display: false }, ticks: { font: { size: 10 } } },
                    y: { title: { display: true, text: 'Count' }, grid: { color: 'rgba(255,255,255,0.04)' } }
                },
                plugins: {
                    legend: { labels: { boxWidth: 14, padding: 14, usePointStyle: true, pointStyle: 'rect' } },
                    tooltip: { mode: 'index', intersect: false }
                }
            }
        });
    }


    function renderPredictionProbability(pd) {
        new Chart(document.getElementById('prob-chart'), {
            type: 'bar',
            data: {
                labels: pd.bins,
                datasets: [
                    {
                        label: 'Actually Approved',
                        data: pd.approved,
                        backgroundColor: 'rgba(16, 185, 129, 0.55)',
                        borderColor: '#10b981',
                        borderWidth: 1,
                        borderRadius: 2,
                    },
                    {
                        label: 'Actually Rejected',
                        data: pd.rejected,
                        backgroundColor: 'rgba(244, 63, 94, 0.55)',
                        borderColor: '#f43f5e',
                        borderWidth: 1,
                        borderRadius: 2,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { stacked: true, title: { display: true, text: 'Predicted Probability' }, grid: { display: false }, ticks: { font: { size: 10 } } },
                    y: { stacked: true, title: { display: true, text: 'Count' }, grid: { color: 'rgba(255,255,255,0.04)' } }
                },
                plugins: {
                    legend: { labels: { boxWidth: 14, padding: 14, usePointStyle: true, pointStyle: 'rect' } },
                    tooltip: { mode: 'index', intersect: false }
                }
            }
        });
    }


    function renderMetricsTable(scores) {
        const container = document.getElementById('metrics-table');
        const rows = [
            { name: 'Accuracy', key: 'accuracy', desc: 'Percentage of correct predictions overall', color: '#3b82f6' },
            { name: 'Precision', key: 'precision', desc: 'Proportion of positive identifications that are correct', color: '#10b981' },
            { name: 'Recall', key: 'recall', desc: 'Proportion of actual positives correctly identified', color: '#f59e0b' },
            { name: 'F1 Score', key: 'f1_score', desc: 'Harmonic mean of precision and recall', color: '#8b5cf6' },
            { name: 'Specificity', key: 'specificity', desc: 'Proportion of actual negatives correctly identified', color: '#3b82f6' },
            { name: 'ROC AUC', key: 'roc_auc', desc: 'Area under the Receiver Operating Characteristic curve', color: '#06b6d4' },
            { name: 'Avg Precision', key: 'avg_precision', desc: 'Weighted mean of precisions at each threshold', color: '#8b5cf6' },
            { name: 'MCC', key: 'mcc', desc: 'Matthews Correlation Coefficient — balanced measure even with imbalanced data', color: '#10b981' },
            { name: "Cohen's Kappa", key: 'kappa', desc: 'Agreement between predictions and actuals, adjusted for chance', color: '#f59e0b' },
            { name: 'Log Loss', key: 'log_loss', desc: 'Logarithmic loss — lower is better (measures uncertainty)', color: '#f43f5e' },
        ];

        function scoreClass(val, key) {
            if (key === 'log_loss') return val < 0.1 ? 'score-excellent' : val < 0.3 ? 'score-good' : val < 0.5 ? 'score-fair' : 'score-poor';
            return val >= 0.95 ? 'score-excellent' : val >= 0.85 ? 'score-good' : val >= 0.7 ? 'score-fair' : 'score-poor';
        }

        container.innerHTML = `
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Score</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(r => {
            const val = scores[r.key];
            const display = r.key === 'log_loss' ? val.toFixed(4) : (val * 100).toFixed(2) + '%';
            return `
              <tr>
                <td>
                  <span class="metric-name">
                    <span class="dot" style="background:${r.color}"></span>
                    ${r.name}
                  </span>
                </td>
                <td class="metric-score ${scoreClass(val, r.key)}">${display}</td>
                <td class="metric-desc">${r.desc}</td>
              </tr>
            `;
        }).join('')}
        </tbody>
      </table>
    `;
    }

});
