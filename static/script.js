document.addEventListener('DOMContentLoaded', () => {
  const modeButtons = document.querySelectorAll('.mode-button');
  const textPanel = document.getElementById('text-panel');
  const urlPanel = document.getElementById('url-panel');
  const textInput = document.getElementById('text');
  const urlInput = document.getElementById('url');
  const clearButton = document.getElementById('clear-btn');

  if (!modeButtons.length) {
    return;
  }

  const swapPanel = (mode) => {
    const isTextMode = mode === 'text';
    textPanel.classList.toggle('active', isTextMode);
    urlPanel.classList.toggle('active', !isTextMode);

    modeButtons.forEach((button) => {
      const active = button.dataset.mode === mode;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });

    if (isTextMode) {
      textInput.focus();
    } else {
      urlInput.focus();
    }
  };

  modeButtons.forEach((button) => {
    button.addEventListener('click', () => swapPanel(button.dataset.mode));
  });

  clearButton?.addEventListener('click', () => {
    if (textInput && textInput.value) {
      textInput.value = '';
      textInput.focus();
    }

    if (urlInput && urlInput.value) {
      urlInput.value = '';
    }
  });
});
