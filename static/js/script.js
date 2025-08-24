document.addEventListener('DOMContentLoaded', function () {
    const uploadBox = document.getElementById('uploadBox');
    const fileInput = document.getElementById('fileInput');
    const selectedFile = document.getElementById('selectedFile');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const removeFile = document.getElementById('removeFile');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const originalImage = document.getElementById('originalImage');
    const heatmapImage = document.getElementById('heatmapImage');
    const diagnosisResult = document.getElementById('diagnosisResult');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');

    let currentFile = null;

    // Drag and drop functionality
    uploadBox.addEventListener('dragover', function (e) {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.add('dragover');
    });

    uploadBox.addEventListener('dragleave', function (e) {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.remove('dragover');
    });

    uploadBox.addEventListener('drop', function (e) {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.remove('dragover');

        if (e.dataTransfer.files && e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    uploadBox.addEventListener('click', function () {
        fileInput.click();
    });

    fileInput.addEventListener('change', function () {
        if (fileInput.files && fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });

    removeFile.addEventListener('click', function () {
        e.stopPropagation();
        resetFileInput();
    });

    analyzeBtn.addEventListener('click', function () {
        if (!currentFile) {
            alert('Please select an image first.');
            return;
        }

        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="loading"></span> Analyzing...';

        const formData = new FormData();
        formData.append('image', currentFile);

        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
            .then(async (response) => {
                let data;
                try {
                    data = await response.json();
                }
                catch {
                    throw new Error('Invalid JSON');
                }
                if (!response.ok || data.success === false) {
                    throw new Error(data?.error || `HTTP ${response.status}`);
                }
                return response.json();
            })
            .then(data => {

                // Display results
                originalImage.innerHTML = `<img src="${data.image}" alt="Uploaded MRI">`;
                heatmapImage.innerHTML = `<img src="${data.heatmap}" alt="Heatmap">`;

                // Display diagnosis with appropriate styling
                diagnosisResult.innerHTML = `
                <i class="fas fa-${getDiagnosisIcon(data.diagnosis)}"></i>
                <p class="${getDiagnosisClass(data.diagnosis)}">${data.diagnosis}</p>`;

                // Display confidence if available
                if (data.confidence) {
                    const confidencePercent = Math.round(data.confidence * 100);
                    confidenceValue.textContent = `${confidencePercent}%`;
                    confidenceFill.style.width = `${confidencePercent}%`;
                    confidenceBar.style.display = 'block';
                }

                analyzeBtn.innerHTML = 'Analysis Complete';
            })
            .catch(error => {
                console.error('Error:', error);
                diagnosisResult.innerHTML = `
                <i class="fas fa-exclamation-triangle error"></i>
                <p class="error">An error occurred during analysis. Please try again.</p>`;
                analyzeBtn.innerHTML = 'Try Again';
            })
            .finally(() => {
                analyzeBtn.disabled = false;
            });
    });

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert('Please select an image file.');
            return;
        }

        currentFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        selectedFile.style.display = 'flex';
        analyzeBtn.disabled = false;

        // Preview image
        const reader = new FileReader();
        reader.onload = function (e) {
            originalImage.innerHTML = `<img src="${e.target.result}" alt="Uploaded MRI">`;
        };
        reader.readAsDataURL(file);

        // Reset results
        heatmapImage.innerHTML = `
            <i class="fas fa-fire"></i>
            <p>Heatmap will appear here</p>
        `;
        diagnosisResult.innerHTML = `
            <i class="fas fa-stethoscope"></i>
            <p>Analysis result will appear here</p>
        `;
        confidenceBar.style.display = 'none';
    }

    function resetFileInput() {
        currentFile = null;
        fileInput.value = '';
        fileName.textContent = 'No file selected';
        fileSize.textContent = '-';
        selectedFile.style.display = 'none';
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = 'Analyze Image';

        // Reset previews
        originalImage.innerHTML = `
            <i class="fas fa-image"></i>
            <p>Original image will appear here</p>
        `;

        heatmapImage.innerHTML = `
            <i class="fas fa-fire"></i>
            <p>Heatmap will appear here</p>
        `;

        diagnosisResult.innerHTML = `
            <i class="fas fa-stethoscope"></i>
            <p>Analysis result will appear here</p>
        `;

        confidenceBar.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function getDiagnosisIcon(diagnosis) {
        if (diagnosis === 'No Tumor') return 'check-circle';
        if (diagnosis === 'Glioma') return 'exclamation-circle';
        if (diagnosis === 'Meningioma') return 'exclamation-triangle';
        if (diagnosis === 'Pituitary') return 'info-circle';
        return 'stethoscope';
    }

    function getDiagnosisClass(diagnosis) {
        if (diagnosis === 'No Tumor') return 'no-tumor';
        if (diagnosis === 'Glioma') return 'glioma';
        if (diagnosis === 'Meningioma') return 'meningioma';
        if (diagnosis === 'Pituitary') return 'pituitary';
        return '';
    }
});