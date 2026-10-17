let steps = [], current = 0;
function showStep(index) {
  if (!steps.length) return;
  current = index % steps.length;
  const step = steps[current];
  document.querySelector('#step-image').src = `assets/${step.image}`;
  document.querySelector('#step-image').alt = step.title;
  document.querySelector('#step-title').textContent = step.title;
  document.querySelector('#step-description').textContent = step.description;
  document.querySelector('#step-count').textContent = `STEP ${current + 1} / ${steps.length}`;
  document.querySelectorAll('[data-step]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.step) === current)));
  document.querySelector('#next-step').textContent = current === steps.length - 1 ? 'Replay from start ↻' : 'Next step →';
}
fetch('assets/trajectory.json').then(response => { if (!response.ok) throw Error('Replay unavailable'); return response.json(); }).then(data => { steps = data.steps; showStep(0); }).catch(() => { document.querySelector('#step-description').textContent = 'The replay could not be loaded. Please refresh the page.'; document.querySelector('#next-step').disabled = true; });
document.querySelectorAll('[data-step]').forEach(button => button.addEventListener('click', () => showStep(Number(button.dataset.step))));
document.querySelector('#next-step').addEventListener('click', () => showStep(current + 1));
document.querySelector('#copy-citation').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent); document.querySelector('#copy-status').textContent = 'Citation copied.'; }
  catch { document.querySelector('#copy-status').textContent = 'Select the citation above to copy it.'; }
});
