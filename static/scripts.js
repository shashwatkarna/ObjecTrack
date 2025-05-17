document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');
    const fileInfo = document.getElementById('fileInfo');
    const fileNameDisplay = document.getElementById('fileName');
    const removeFileBtn = document.getElementById('removeFileBtn');
    const submitBtn = document.getElementById('submitBtn');

    // Preview & Results Elements
    const previewEmptyState = document.getElementById('previewEmptyState');
    const imageContainer = document.getElementById('imageContainer');
    const previewImage = document.getElementById('previewImage');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const resultsList = document.getElementById('resultsList');
    const detectionCountBadge = document.getElementById('detectionCountBadge');

    let currentFile = null;

    // ----- Drag & Drop Handlers -----

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropZone.classList.add('border-brand-500', 'bg-dark-800');
        dropZone.classList.remove('border-slate-600', 'bg-dark-800/50');
    }

    function unhighlight() {
        dropZone.classList.remove('border-brand-500', 'bg-dark-800');
        dropZone.classList.add('border-slate-600', 'bg-dark-800/50');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    }

    // Trigger file input on click
    dropZone.addEventListener('click', () => {
        if (!currentFile) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', function () {
        if (this.files && this.files[0]) {
            handleFile(this.files[0]);
        }
    });

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // prevent clicking dropZone
        resetUI();
    });

    // ----- File Handling & Preview UI -----

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        currentFile = file;
        fileNameDisplay.textContent = file.name;

        // Update UI State for file selected
        dropZone.classList.add('hidden');
        fileInfo.classList.remove('hidden');
        submitBtn.disabled = false;

        // Load image for preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            showImagePreview();
        };
        reader.readAsDataURL(file);
    }

    function showImagePreview() {
        previewEmptyState.classList.add('hidden');
        imageContainer.classList.remove('hidden');
        clearBoundingBoxes();
        clearResults();
    }

    function resetUI() {
        currentFile = null;
        fileInput.value = ''; // clear input

        // Form UI
        dropZone.classList.remove('hidden');
        fileInfo.classList.add('hidden');
        submitBtn.disabled = true;

        // Preview UI
        previewEmptyState.classList.remove('hidden');
        imageContainer.classList.add('hidden');
        previewImage.src = '';

        clearBoundingBoxes();
        clearResults();
    }

    function clearBoundingBoxes() {
        const boxes = document.querySelectorAll('.bounding-box');
        boxes.forEach(box => box.remove());
    }

    function clearResults() {
        resultsList.innerHTML = '';
        resultsList.classList.add('hidden');
        detectionCountBadge.classList.add('hidden');
    }

    // ----- Form Submission & Analysis -----

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentFile) return;

        // Prepare UI for loading
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="ph-bold ph-spinner animate-spin"></i> Analyzing...';
        loadingOverlay.classList.remove('hidden');
        clearBoundingBoxes();
        clearResults();

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const results = await response.json();
            if (results.error) {
                throw new Error(results.error);
            }
            handleDetections(results);

        } catch (error) {
            console.error('Error during prediction:', error);
            resultsList.innerHTML = `
                <div class="bg-red-900/20 border border-red-500/50 rounded-lg p-4 text-red-400 text-sm flex items-center gap-2">
                    <i class="ph-fill ph-warning-circle text-xl"></i>
                    Something went wrong while processing the image. Ensure the server is running.
                </div>
            `;
            resultsList.classList.remove('hidden');
        } finally {
            // Restore UI
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="ph-bold ph-magnifying-glass"></i> Detect Animals';
            loadingOverlay.classList.add('hidden');
        }
    });

    // ----- Display Results & Draw Boxes -----

    function handleDetections(detections) {
        if (!detections || detections.length === 0) {
            resultsList.innerHTML = `
                <div class="bg-slate-800/50 border border-slate-700/50 rounded-lg p-4 text-slate-400 text-sm flex items-center justify-center gap-2">
                    <i class="ph-fill ph-info text-xl text-slate-500"></i>
                    No animals detected in this image.
                </div>
            `;
            resultsList.classList.remove('hidden');
            detectionCountBadge.textContent = '0 Detected';
            detectionCountBadge.classList.remove('hidden');
            return;
        }

        // Update badge
        detectionCountBadge.textContent = `${detections.length} Detected`;
        detectionCountBadge.classList.remove('hidden');
        detectionCountBadge.classList.add('animate-pulse');
        setTimeout(() => detectionCountBadge.classList.remove('animate-pulse'), 1000);

        resultsList.innerHTML = ''; // Keep it clean
        resultsList.classList.remove('hidden');

        // Note: app.py uses transforms.Resize((640, 640))
        // So the coordinates returned by YOLO are relative to a 640x640 grid space.
        const modelTargetSize = 640;

        detections.forEach((det, index) => {
            // 1. Draw Visual Bounding Box

            // Calculate percentages based on the 640x640 model input
            const leftPct = (det.xmin / modelTargetSize) * 100;
            const topPct = (det.ymin / modelTargetSize) * 100;
            const widthPct = ((det.xmax - det.xmin) / modelTargetSize) * 100;
            const heightPct = ((det.ymax - det.ymin) / modelTargetSize) * 100;

            // Create box element
            const box = document.createElement('div');
            box.className = `bounding-box box-animate`;
            box.style.left = `${leftPct}%`;
            box.style.top = `${topPct}%`;
            box.style.width = `${widthPct}%`;
            box.style.height = `${heightPct}%`;
            box.style.animationDelay = `${index * 0.1}s`; // Staggered animation

            // Create Label for Box
            const label = document.createElement('div');
            label.className = 'bounding-box-label';
            // Capitalize first letter of class name
            const displayTitle = det.name.charAt(0).toUpperCase() + det.name.slice(1);
            label.textContent = `${displayTitle} ${Math.round(det.confidence * 100)}%`;
            box.appendChild(label);

            imageContainer.appendChild(box);

            // 2. Add to Results List Area
            const confidencePct = Math.round(det.confidence * 100);

            // Generate a color tag based on confidence
            let confColor = 'bg-brand-500';
            if (confidencePct < 50) confColor = 'bg-red-500';
            else if (confidencePct < 75) confColor = 'bg-yellow-500';

            const listItem = `
                <div class="bg-dark-800 border border-slate-700/50 hover:border-brand-500/50 rounded-lg p-3 flex items-center justify-between transition-colors shadow-sm">
                    <div class="flex items-center gap-3">
                        <div class="w-2 h-2 rounded-full ${confColor} shadow-[0_0_5px_rgba(34,197,94,0.5)]"></div>
                        <span class="font-semibold text-slate-200">${displayTitle}</span>
                    </div>
                    <div class="flex items-center gap-2 text-xs font-semibold bg-dark-900 px-2.5 py-1 rounded-md text-slate-300">
                        <i class="ph-bold ph-target"></i>
                        ${confidencePct}% Match
                    </div>
                </div>
            `;

            resultsList.insertAdjacentHTML('beforeend', listItem);
        });
    }

});
