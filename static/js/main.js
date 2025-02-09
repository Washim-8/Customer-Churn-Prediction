/**
 * ChurnIQ – main.js
 * UI interactivity: form preview, scroll animations, submit loading state.
 */

document.addEventListener("DOMContentLoaded", () => {

  // ── Live Preview Bar ─────────────────────────────────────────────────────
  const fields = {
    tenure:          { id: "tenure",          out: "prev-tenure",   fmt: v => `${v} mo`  },
    MonthlyCharges:  { id: "MonthlyCharges",  out: "prev-monthly",  fmt: v => `$${parseFloat(v).toFixed(0)}`  },
    Contract:        { id: "Contract",        out: "prev-contract", fmt: v => v          },
    InternetService: { id: "InternetService", out: "prev-internet", fmt: v => v          },
  };

  Object.values(fields).forEach(({ id, out, fmt }) => {
    const el  = document.getElementById(id);
    const outEl = document.getElementById(out);
    if (el && outEl) {
      const update = () => { outEl.textContent = fmt(el.value); };
      el.addEventListener("input",  update);
      el.addEventListener("change", update);
      update();
    }
  });

  // ── Form Submit Loading State ─────────────────────────────────────────────
  const form     = document.getElementById("churnForm");
  const btnText  = document.getElementById("btnText");
  const btnLoader= document.getElementById("btnLoader");
  const submitBtn= document.getElementById("submitBtn");

  if (form) {
    form.addEventListener("submit", () => {
      if (btnText && btnLoader && submitBtn) {
        btnText.style.display  = "none";
        btnLoader.style.display= "inline";
        submitBtn.disabled     = true;
        submitBtn.style.opacity= "0.7";
      }
    });
  }

  // ── Auto-compute TotalCharges hint ───────────────────────────────────────
  const tenureEl  = document.getElementById("tenure");
  const monthlyEl = document.getElementById("MonthlyCharges");
  const totalEl   = document.getElementById("TotalCharges");

  function autoTotal() {
    if (tenureEl && monthlyEl && totalEl) {
      const estimate = (parseFloat(tenureEl.value) || 0) *
                       (parseFloat(monthlyEl.value) || 0);
      // Only overwrite if the field hasn't been manually edited
      if (!totalEl.dataset.manual) {
        totalEl.value = estimate.toFixed(2);
      }
    }
  }

  if (tenureEl && monthlyEl) {
    tenureEl.addEventListener("input",  autoTotal);
    monthlyEl.addEventListener("input", autoTotal);
    autoTotal();
  }
  if (totalEl) {
    totalEl.addEventListener("input", () => { totalEl.dataset.manual = "1"; });
  }

  // ── Scroll Reveal ─────────────────────────────────────────────────────────
  const revealEls = document.querySelectorAll(
    ".feature-card, .chart-card, .result-metric, .pipeline-step"
  );
  if ("IntersectionObserver" in window) {
    const obs = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.style.opacity   = "1";
          e.target.style.transform = "translateY(0)";
          obs.unobserve(e.target);
        }
      });
    }, { threshold: 0.1 });

    revealEls.forEach(el => {
      el.style.opacity   = "0";
      el.style.transform = "translateY(20px)";
      el.style.transition= "opacity .5s ease, transform .5s ease";
      obs.observe(el);
    });
  }

  // ── Animate AUC bars on scroll ──────────────────────────────────────────
  const aucBars = document.querySelectorAll(".auc-bar");
  if ("IntersectionObserver" in window && aucBars.length) {
    const barObs = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.style.width = e.target.style.width; // trigger transition
          barObs.unobserve(e.target);
        }
      });
    }, { threshold: 0.5 });
    aucBars.forEach(b => barObs.observe(b));
  }

});
