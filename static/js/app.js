const dropArea = document.getElementById('drop-area');

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, preventDefaults, false);
  document.body.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

// Highlight drop area when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
  dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
  dropArea.style.borderColor = 'var(--primary)';
  dropArea.style.background = 'rgba(255, 255, 255, 0.08)';
}

function unhighlight(e) {
  dropArea.style.borderColor = 'var(--secondary)';
  dropArea.style.background = 'transparent';
}

// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
  var dt = e.dataTransfer;
  var files = dt.files;
  handleFiles(files);
}

function handleFiles(files) {
  if (files.length > 0) {
    uploadFile(files[0]);
  }
}

function uploadFile(file) {
  // UI transitions
  document.getElementById('upload-spinner').classList.remove('hidden');
  document.querySelector('.drop-area label').classList.add('hidden');
  document.getElementById('results-panel').classList.add('hidden');

  let formData = new FormData();
  formData.append('file', file);

  fetch('/upload', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById('upload-spinner').classList.add('hidden');
    document.querySelector('.drop-area label').classList.remove('hidden');

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    displayResults(data);
  })
  .catch(error => {
    document.getElementById('upload-spinner').classList.add('hidden');
    document.querySelector('.drop-area label').classList.remove('hidden');
    console.error('Error:', error);
    alert('An error occurred during upload.');
  });
}

function displayResults(data) {
  // Show the results panel
  const resultsPanel = document.getElementById('results-panel');
  resultsPanel.classList.remove('hidden');

  // Fill images
  document.getElementById('orig-img').src = data.original_url;
  document.getElementById('ela-img').src = data.ela_url;
  document.getElementById('noise-img').src = data.noise_url;

  // Build the verdict component
  const statusDiv = document.getElementById('result-status');
  const bar = document.getElementById('confidence-bar');
  const text = document.getElementById('confidence-text');

  if (data.is_forged) {
    statusDiv.innerHTML = '<span class="status-forged"><i class="fa-solid fa-triangle-exclamation"></i> FORGERY DETECTED</span>';
    bar.style.background = 'var(--danger)';
    bar.style.width = data.confidence + '%';
    text.innerText = `The AI is ${data.confidence}% confident this image has been manipulated.`;
  } else {
    statusDiv.innerHTML = '<span class="status-real"><i class="fa-solid fa-shield-check"></i> AUTHENTIC</span>';
    bar.style.background = 'var(--success)';
    bar.style.width = data.confidence + '%';
    text.innerText = `The AI is ${data.confidence}% confident this image is authentic.`;
  }

  // Populate Metadata
  const metaContainer = document.getElementById('metadata-container');
  metaContainer.innerHTML = '';
  if (data.metadata && Object.keys(data.metadata).length > 0) {
      for (const [key, value] of Object.entries(data.metadata)) {
          metaContainer.insertAdjacentHTML('beforeend', `
              <div class="meta-item">
                  <div class="meta-key">${key}</div>
                  <div class="meta-value">${value}</div>
              </div>
          `);
      }
  } else {
      metaContainer.innerHTML = '<p class="text-muted" style="grid-column: 1 / -1; text-align: center;">No metadata available for this image.</p>';
  }

  // Smooth scroll to results
  resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
