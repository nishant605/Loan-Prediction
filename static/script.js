// ============================================
// LoanAI — Frontend Logic
// ============================================

document.addEventListener('DOMContentLoaded', () => {
  // --- Element References ---
  const form = document.getElementById('loan-form');
  const predictBtn = document.getElementById('predict-btn');
  const placeholderCard = document.getElementById('placeholder-card');
  const resultCard = document.getElementById('result-card');
  const resultStatus = document.getElementById('result-status');
  const statusIcon = document.getElementById('status-icon');
  const statusTitle = document.getElementById('status-title');
  const statusMessage = document.getElementById('status-message');
  const summaryGrid = document.getElementById('summary-grid');
  const meterScore = document.getElementById('meter-score');
  const meterFill = document.getElementById('meter-fill');
  const toast = document.getElementById('toast');
  const navbar = document.getElementById('navbar');

  // --- Range Slider Live Updates ---
  const sliders = [
    { input: 'dependents', display: 'dependents-val' },
    { input: 'loan-term', display: 'loan-term-val' },
    { input: 'cibil-score', display: 'cibil-val' },
  ];

  sliders.forEach(({ input, display }) => {
    const el = document.getElementById(input);
    const valEl = document.getElementById(display);
    if (el && valEl) {
      el.addEventListener('input', () => {
        valEl.textContent = el.value;
        // Update CIBIL label
        if (input === 'cibil-score') {
          updateCibilLabel(parseInt(el.value));
        }
      });
    }
  });

  function updateCibilLabel(score) {
    const label = document.getElementById('cibil-label');
    if (!label) return;
    if (score >= 750) {
      label.textContent = '— Excellent';
      label.style.color = '#10b981';
    } else if (score >= 650) {
      label.textContent = '— Good';
      label.style.color = '#3b82f6';
    } else if (score >= 550) {
      label.textContent = '— Fair';
      label.style.color = '#f59e0b';
    } else {
      label.textContent = '— Poor';
      label.style.color = '#f43f5e';
    }
  }

  // --- Navbar scroll effect ---
  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 20);
  });

  // --- Format currency ---
  function formatCurrency(amount) {
    if (amount >= 10000000) {
      return '₹' + (amount / 10000000).toFixed(2) + ' Cr';
    } else if (amount >= 100000) {
      return '₹' + (amount / 100000).toFixed(2) + ' L';
    }
    return '₹' + amount.toLocaleString('en-IN');
  }

  // --- Form Submission ---
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Validate
    const income = parseFloat(document.getElementById('income').value);
    const loanAmount = parseFloat(document.getElementById('loan-amount').value);
    const totalAssets = parseFloat(document.getElementById('total-assets').value);

    if (!income || income <= 0) {
      showToast('⚠️', 'Please enter a valid annual income.');
      return;
    }
    if (!loanAmount && loanAmount !== 0) {
      showToast('⚠️', 'Please enter the loan amount.');
      return;
    }
    if (!totalAssets || totalAssets <= 0) {
      showToast('⚠️', 'Please enter your total assets value.');
      return;
    }

    // Prepare data
    const payload = {
      dependents: parseInt(document.getElementById('dependents').value),
      education: document.getElementById('education').value,
      selfEmployed: document.getElementById('self-employed').value,
      income: income,
      loanAmount: loanAmount,
      loanTerm: parseInt(document.getElementById('loan-term').value),
      cibilScore: parseInt(document.getElementById('cibil-score').value),
      totalAssets: totalAssets,
    };

    // Loading state
    predictBtn.classList.add('loading');

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Prediction failed');
      }

      const result = await response.json();
      displayResult(result);

    } catch (error) {
      showToast('❌', error.message || 'Something went wrong. Please try again.');
    } finally {
      predictBtn.classList.remove('loading');
    }
  });

  // --- Display Result ---
  function displayResult(result) {
    // Hide placeholder, show result
    placeholderCard.style.display = 'none';
    resultCard.classList.add('visible');

    const approved = result.approved;
    const s = result.summary;

    // Status banner
    resultStatus.className = 'result-status ' + (approved ? 'approved' : 'rejected');
    statusIcon.textContent = approved ? '✅' : '❌';
    statusTitle.textContent = approved ? 'Loan Approved!' : 'Loan Rejected';
    statusMessage.textContent = approved
      ? 'Based on your profile, you are likely to be approved for this loan.'
      : 'Based on your profile, this loan application may be rejected. Consider improving your CIBIL score or adjusting the loan amount.';

    // Summary items
    const summaryItems = [
      { label: 'Annual Income', value: formatCurrency(s.annualIncome) },
      { label: 'Loan Amount', value: formatCurrency(s.loanAmount) },
      { label: 'Loan Term', value: s.loanTerm + ' years' },
      { label: 'Total Assets', value: formatCurrency(s.totalAssets) },
      { label: 'Est. EMI', value: formatCurrency(s.emi) + '/mo' },
      { label: 'Loan-to-Income', value: s.loanToIncome + 'x' },
      { label: 'Loan-to-Asset', value: s.loanToAsset + '%' },
      { label: 'Asset/Income', value: s.assetToIncome + 'x' },
    ];

    summaryGrid.innerHTML = summaryItems
      .map(
        (item) => `
        <div class="summary-item">
          <span class="item-label">${item.label}</span>
          <span class="item-value">${item.value}</span>
        </div>
      `
      )
      .join('');

    // CIBIL meter
    const cibil = s.cibilScore;
    meterScore.textContent = cibil;
    const pct = ((cibil - 300) / 600) * 100;

    // Determine color class
    let meterClass = 'poor';
    if (cibil >= 750) {
      meterClass = 'excellent';
      meterScore.style.color = '#10b981';
    } else if (cibil >= 650) {
      meterClass = 'good';
      meterScore.style.color = '#3b82f6';
    } else if (cibil >= 550) {
      meterClass = 'fair';
      meterScore.style.color = '#f59e0b';
    } else {
      meterScore.style.color = '#f43f5e';
    }

    meterFill.className = 'meter-fill ' + meterClass;
    setTimeout(() => {
      meterFill.style.width = pct + '%';
    }, 100);

    // Scroll to result
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Confetti if approved
    if (approved) {
      launchConfetti();
    }
  }

  // --- Toast ---
  function showToast(icon, message) {
    const toastIcon = toast.querySelector('.toast-icon');
    const toastMsg = toast.querySelector('.toast-message');
    toastIcon.textContent = icon;
    toastMsg.textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 4000);
  }

  // --- Confetti ---
  function launchConfetti() {
    const canvas = document.getElementById('confetti-canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#f43f5e'];
    const particles = [];

    for (let i = 0; i < 150; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height - canvas.height,
        w: Math.random() * 10 + 5,
        h: Math.random() * 6 + 3,
        color: colors[Math.floor(Math.random() * colors.length)],
        speed: Math.random() * 4 + 2,
        angle: Math.random() * Math.PI * 2,
        spin: (Math.random() - 0.5) * 0.2,
        drift: (Math.random() - 0.5) * 2,
        opacity: 1,
      });
    }

    let frame = 0;
    const maxFrames = 180;

    function animate() {
      if (frame >= maxFrames) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach((p) => {
        p.y += p.speed;
        p.x += p.drift;
        p.angle += p.spin;
        p.opacity = Math.max(0, 1 - frame / maxFrames);

        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.angle);
        ctx.globalAlpha = p.opacity;
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
        ctx.restore();
      });

      frame++;
      requestAnimationFrame(animate);
    }

    animate();
  }

  // --- Smooth Scroll for nav links ---
  document.querySelectorAll('.nav-links a').forEach((link) => {
    link.addEventListener('click', (e) => {
      const href = link.getAttribute('href');
      if (href.startsWith('#')) {
        e.preventDefault();
        const target = document.querySelector(href);
        if (target) {
          target.scrollIntoView({ behavior: 'smooth' });
        }
      }
    });
  });

  // --- Number Input formatting visual feedback ---
  document.querySelectorAll('input[type="number"]').forEach((input) => {
    input.addEventListener('blur', () => {
      if (input.value && parseFloat(input.value) > 0) {
        input.style.borderColor = 'rgba(16, 185, 129, 0.3)';
        setTimeout(() => {
          input.style.borderColor = '';
        }, 1000);
      }
    });
  });
});
