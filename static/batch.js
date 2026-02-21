// ============================================
// LoanAI — Batch Prediction Logic
// ============================================

document.addEventListener('DOMContentLoaded', () => {

    // --- Element References ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const btnRemove = document.getElementById('btn-remove-file');
    const btnUpload = document.getElementById('btn-upload');
    const uploadSection = document.getElementById('upload-section');
    const resultsSection = document.getElementById('results-section');
    const batchSummary = document.getElementById('batch-summary');
    const resultsCount = document.getElementById('results-count');
    const tableHead = document.getElementById('table-head');
    const tableBody = document.getElementById('table-body');
    const paginationEl = document.getElementById('pagination');
    const btnDownloadExcel = document.getElementById('btn-download-excel');
    const btnDownloadPdf = document.getElementById('btn-download-pdf');
    const btnNewBatch = document.getElementById('btn-new-batch');
    const toast = document.getElementById('toast');
    const navbar = document.getElementById('navbar');

    // Step indicators
    const step1 = document.getElementById('step-1-indicator');
    const step2 = document.getElementById('step-2-indicator');
    const step3 = document.getElementById('step-3-indicator');
    const connectors = document.querySelectorAll('.step-connector');

    let selectedFile = null;
    let batchData = null;
    let currentFilter = 'all';
    let currentPage = 1;
    const ROWS_PER_PAGE = 25;

    // --- Navbar scroll ---
    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 20);
    });

    // --- Drag & Drop ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (!['csv', 'xlsx', 'xls'].includes(ext)) {
            showToast('⚠️', 'Unsupported file. Please upload CSV or Excel.');
            return;
        }

        selectedFile = file;
        dropZone.style.display = 'none';
        fileInfo.style.display = 'flex';
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        btnUpload.disabled = false;
    }

    btnRemove.addEventListener('click', () => {
        clearFile();
    });

    function clearFile() {
        selectedFile = null;
        fileInput.value = '';
        dropZone.style.display = 'block';
        fileInfo.style.display = 'none';
        btnUpload.disabled = true;
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    // --- Upload & Predict ---
    btnUpload.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show processing overlay
        const overlay = document.createElement('div');
        overlay.className = 'processing-overlay';
        overlay.id = 'processing-overlay';
        overlay.innerHTML = `
      <div class="proc-spinner"></div>
      <div class="proc-text">Processing ${selectedFile.name}...</div>
      <div class="proc-sub">Running predictions on all rows</div>
    `;
        document.body.appendChild(overlay);

        btnUpload.classList.add('loading');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const resp = await fetch('/batch/upload', {
                method: 'POST',
                body: formData,
            });

            const result = await resp.json();

            if (!resp.ok) {
                throw new Error(result.error || 'Upload failed');
            }

            batchData = result;
            showResults(result);

        } catch (err) {
            showToast('❌', err.message || 'Upload failed. Check your file format.');
        } finally {
            btnUpload.classList.remove('loading');
            const ov = document.getElementById('processing-overlay');
            if (ov) ov.remove();
        }
    });

    // --- Show Results ---
    function showResults(data) {
        // Update step indicators
        step1.classList.remove('active');
        step1.classList.add('completed');
        step2.classList.add('completed');
        step3.classList.add('active');
        connectors.forEach(c => c.classList.add('active'));

        // Hide upload, show results
        uploadSection.style.display = 'none';
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Summary cards
        const approvalRate = ((data.approved_count / data.total_rows) * 100).toFixed(1);
        batchSummary.innerHTML = `
      <div class="summary-card blue">
        <div class="card-icon">📄</div>
        <div class="card-value">${data.total_rows.toLocaleString()}</div>
        <div class="card-label">Total Rows</div>
      </div>
      <div class="summary-card green">
        <div class="card-icon">✅</div>
        <div class="card-value">${data.approved_count.toLocaleString()}</div>
        <div class="card-label">Approved</div>
      </div>
      <div class="summary-card red">
        <div class="card-icon">❌</div>
        <div class="card-value">${data.rejected_count.toLocaleString()}</div>
        <div class="card-label">Rejected</div>
      </div>
      <div class="summary-card purple">
        <div class="card-icon">📊</div>
        <div class="card-value">${approvalRate}%</div>
        <div class="card-label">Approval Rate</div>
      </div>
      <div class="summary-card amber">
        <div class="card-icon">🎯</div>
        <div class="card-value">${data.avg_confidence}%</div>
        <div class="card-label">Avg Confidence</div>
      </div>
    `;

        // Download links
        btnDownloadExcel.href = `/batch/download/excel/${data.batch_id}`;
        btnDownloadPdf.href = `/batch/download/pdf/${data.batch_id}`;

        // Results count
        resultsCount.textContent = `${data.total_rows} rows`;

        // Filter buttons
        document.getElementById('filter-all').textContent = `All (${data.total_rows})`;
        document.getElementById('filter-approved').textContent = `✅ Approved (${data.approved_count})`;
        document.getElementById('filter-rejected').textContent = `❌ Rejected (${data.rejected_count})`;

        // Render table
        currentFilter = 'all';
        currentPage = 1;
        renderTable();
    }

    // --- Render Table ---
    function renderTable() {
        if (!batchData) return;

        let rows = batchData.preview;

        // Apply filter
        if (currentFilter === 'approved') {
            rows = rows.filter(r => r.prediction === 'Approved');
        } else if (currentFilter === 'rejected') {
            rows = rows.filter(r => r.prediction === 'Rejected');
        }

        // Apply search
        const searchVal = document.getElementById('table-search')?.value?.toLowerCase() || '';
        if (searchVal) {
            rows = rows.filter(r =>
                Object.values(r).some(v => String(v).toLowerCase().includes(searchVal))
            );
        }

        // Pagination
        const totalPages = Math.ceil(rows.length / ROWS_PER_PAGE);
        currentPage = Math.min(currentPage, totalPages || 1);
        const start = (currentPage - 1) * ROWS_PER_PAGE;
        const pageRows = rows.slice(start, start + ROWS_PER_PAGE);

        // Header
        const headerCols = [
            '#', 'Deps', 'Education', 'Self Emp', 'Income', 'Loan Amt',
            'Term', 'CIBIL', 'Prediction', 'Confidence', 'Est. EMI'
        ];
        tableHead.innerHTML = `<tr>${headerCols.map(c => `<th>${c}</th>`).join('')}</tr>`;

        // Body
        tableBody.innerHTML = pageRows.map((row, idx) => {
            const isApproved = row.prediction === 'Approved';
            const badgeClass = isApproved ? 'approved' : 'rejected';
            const badgeIcon = isApproved ? '✅' : '❌';

            const conf = row.confidence || 0;
            const confClass = conf >= 90 ? 'high' : conf >= 70 ? 'medium' : 'low';

            return `
        <tr>
          <td class="row-num">${start + idx + 1}</td>
          <td>${row.no_of_dependents}</td>
          <td>${row.education}</td>
          <td>${row.self_employed}</td>
          <td class="money">${formatCurrency(row.income_annum)}</td>
          <td class="money">${formatCurrency(row.loan_amount)}</td>
          <td>${row.loan_term} yr</td>
          <td><strong>${row.cibil_score}</strong></td>
          <td><span class="badge ${badgeClass}">${badgeIcon} ${row.prediction}</span></td>
          <td>
            <div class="confidence-bar">
              <div class="bar-track">
                <div class="bar-fill ${confClass}" style="width: ${conf}%"></div>
              </div>
              <span class="bar-label">${conf}%</span>
            </div>
          </td>
          <td class="money">${formatCurrency(row.estimated_emi)}</td>
        </tr>
      `;
        }).join('');

        // Pagination
        renderPagination(totalPages);
    }

    function renderPagination(totalPages) {
        if (totalPages <= 1) {
            paginationEl.innerHTML = '';
            return;
        }

        let html = `<button class="page-btn" ${currentPage === 1 ? 'disabled' : ''} data-page="${currentPage - 1}">‹</button>`;

        for (let i = 1; i <= totalPages; i++) {
            if (totalPages > 7 && i > 3 && i < totalPages - 1 && Math.abs(i - currentPage) > 1) {
                if (i === 4 || i === totalPages - 2) html += `<span style="color:var(--text-muted);padding:0 6px;">…</span>`;
                continue;
            }
            html += `<button class="page-btn ${i === currentPage ? 'active' : ''}" data-page="${i}">${i}</button>`;
        }

        html += `<button class="page-btn" ${currentPage === totalPages ? 'disabled' : ''} data-page="${currentPage + 1}">›</button>`;

        paginationEl.innerHTML = html;

        paginationEl.querySelectorAll('.page-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const page = parseInt(btn.dataset.page);
                if (page >= 1 && page <= totalPages) {
                    currentPage = page;
                    renderTable();
                }
            });
        });
    }

    // --- Filter Buttons ---
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFilter = btn.dataset.filter;
            currentPage = 1;
            renderTable();
        });
    });

    // --- Search ---
    const searchInput = document.getElementById('table-search');
    if (searchInput) {
        let searchTimeout;
        searchInput.addEventListener('input', () => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                currentPage = 1;
                renderTable();
            }, 300);
        });
    }

    // --- New Batch ---
    btnNewBatch.addEventListener('click', () => {
        resultsSection.style.display = 'none';
        uploadSection.style.display = 'block';

        // Reset steps
        step1.classList.add('active');
        step1.classList.remove('completed');
        step2.classList.remove('active', 'completed');
        step3.classList.remove('active', 'completed');
        connectors.forEach(c => c.classList.remove('active'));

        clearFile();
        batchData = null;
        currentFilter = 'all';
        currentPage = 1;

        uploadSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });

    // --- Helpers ---
    function formatCurrency(amount) {
        if (amount == null || isNaN(amount)) return '-';
        if (amount >= 10000000) return '₹' + (amount / 10000000).toFixed(2) + ' Cr';
        if (amount >= 100000) return '₹' + (amount / 100000).toFixed(2) + ' L';
        return '₹' + Number(amount).toLocaleString('en-IN');
    }

    function showToast(icon, message) {
        const toastIcon = toast.querySelector('.toast-icon');
        const toastMsg = toast.querySelector('.toast-message');
        toastIcon.textContent = icon;
        toastMsg.textContent = message;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 4000);
    }

});
