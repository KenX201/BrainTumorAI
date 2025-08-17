const $ = (q, root = document) => root.querySelector(q);

// ===== Theme handling =====
function setTheme(theme) {
    var root = document.documentElement;
    if (theme === 'light') root.setAttribute('data-theme', 'light');
    else root.removeAttribute('data-theme');
    localStorage.setItem('bt_theme', theme);
    var toggle = $('#theme-toggle');
    if (toggle) toggle.innerHTML = theme === 'light' ? '🌙 Dark' : '☀️ Light';
}

function initTheme() {
    var saved = localStorage.getItem('bt_theme');
    setTheme(saved === 'light' ? 'light' : 'dark');
    var toggle = $('#theme-toggle');
    if (toggle) {
        toggle.addEventListener('click', () => {
            const isLight = document.documentElement.getAttribute('data-theme') === 'light';
            setTheme(isLight ? 'dark' : 'light');
        });
    }
}

// ===== Upload & SPA behavior =====
function setupUploader() {
    var form = $('#upload-form');
    var fileInput = $('#file-input');
    var dz = $('#dropzone');
    var preview = $('#preview');
    var previewImg = $('#preview-img');
    var submitBtn = $('#submit-btn');
    var loading = $('#loading');

    if (!form || !fileInput) return;

    // drag-over styling
    ['dragenter', 'dragover'].forEach(evt =>
        dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dz.classList.add('dragover'); })
    );
    ['dragleave', 'drop'].forEach(evt =>
        dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dz.classList.remove('dragover'); })
    );
    dz.addEventListener('drop', e => {
        const f = e.dataTransfer.files?.[0];
        if (f) { fileInput.files = e.dataTransfer.files; showPreview(f); submitBtn.disabled = false; }
    });

    fileInput.addEventListener('change', e => {
        const f = e.target.files?.[0];
        if (f) { showPreview(f); submitBtn.disabled = false; }
    });

    function showPreview(file) {
        if (!file.type.match(/^image\//)) {
            toast('Please select an image file.');
            fileInput.value = "";
            preview.style.display = "none";
            submitBtn.disabled = true;
            return;
        }
        const url = URL.createObjectURL(file);
        previewImg.src = url;
        preview.style.display = "block";
    }

    // Handle submit via fetch (stay on one page)
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!fileInput.files.length) { toast('Choose an image first.'); return; }

        submitBtn.disabled = true;
        loading.style.display = 'flex';

        try {
            const fd = new FormData();
            fd.append('file', fileInput.files[0]);
            const res = await fetch('/predict', { method: 'POST', body: fd });
            const data = await res.json();

            if (!data.success) {
                toast(data.error || 'Prediction failed.');
                return;
            }
            renderResult(data);
            // Smooth scroll to results on small screens
            $('#result-card')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } catch (err) {
            console.error(err);
            toast('Something went wrong. Try again.');
        } finally {
            loading.style.display = 'none';
            submitBtn.disabled = false;
        }
    });
}

function renderResult(data) {
    // Elements
    const sub = $('#result-sub');
    const img = $('#result-img');
    const label = $('#result-label');
    const table = $('#probs-table');
    const tbody = $('#probs-body');
    const dlHeat = $('#download-heatmap');
    const dlOrig = $('#download-original');

    // Fill
    img.src = data.overlay_path;
    img.style.display = 'block';
    label.innerHTML = `<strong>Predicted type:</strong> ${data.label}`;
    label.style.display = 'block';
    sub.style.display = 'none';

    // Downloads
    dlHeat.href = data.overlay_path;
    dlHeat.download = `heatmap_${(data.original_path.split('/').pop() || 'result')}`;
    dlOrig.href = data.original_path;
    dlOrig.download = data.original_path.split('/').pop() || 'original';
    $('#download-actions').style.display = 'flex';

    // Probs table
    tbody.innerHTML = '';
    const entries = Object.entries(data.probs || {});
    // sort by percent desc
    entries.sort((a, b) => (b[1] || 0) - (a[1] || 0));
    for (const [cls, pct] of entries) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${cls}</td><td>${pct}</td>`;
        tbody.appendChild(tr);
    }
    table.style.display = entries.length ? 'table' : 'none';
}

// ===== small toast =====
function toast(msg) {
    const t = $('#toast'); if (!t) return;
    t.textContent = msg; t.style.display = 'block';
    setTimeout(() => t.style.display = 'none', 2400);
}

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    setupUploader();
});