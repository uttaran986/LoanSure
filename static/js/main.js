// LoanBERT — main.js

// Animate stat values
document.querySelectorAll('.dstat-val, .stat-value').forEach(el => {
  const raw = el.textContent.trim();
  const num = parseFloat(raw.replace(/[^0-9.]/g, ''));
  const suffix = raw.replace(/[0-9.,]/g, '');
  if (isNaN(num) || num === 0) return;
  let start = 0;
  const steps = 40;
  const inc = num / steps;
  let count = 0;
  const iv = setInterval(() => {
    count++;
    start = Math.min(start + inc, num);
    el.textContent = (Number.isInteger(num) ? Math.round(start) : start.toFixed(1)) + suffix;
    if (count >= steps) clearInterval(iv);
  }, 20);
});

// Animate progress bars
window.addEventListener('load', () => {
  document.querySelectorAll('.conf-bar-fill, .conf-mini-fill').forEach(bar => {
    const w = bar.style.width;
    bar.style.width = '0';
    setTimeout(() => { bar.style.width = w; }, 200);
  });
});
