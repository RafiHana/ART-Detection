const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadButton = document.getElementById('uploadButton');
const uploadContent = document.getElementById('uploadContent');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const removeButton = document.getElementById('removeButton');
const analyzeButton = document.getElementById('analyzeButton');
const analyzeText = document.getElementById('analyzeText');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultContainer = document.getElementById('resultContainer');
const resultIcon = document.getElementById('resultIcon');
const resultTitle = document.getElementById('resultTitle');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceText = document.getElementById('confidenceText');
const resultDescription = document.getElementById('resultDescription');

let selectedFile = null;

// API Configuration
const API_BASE_URL = window.location.origin;
const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;

uploadButton.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('click', (e) => {
    if (e.target === uploadArea || e.target.closest('#uploadContent')) {
        fileInput.click();
    }
});

fileInput.addEventListener('change', handleFileSelect);
removeButton.addEventListener('click', resetUpload);
analyzeButton.addEventListener('click', analyzeImage);

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        alert('File format not supported. Please use JPG, JPEG, or PNG.');
        return;
    }

    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        alert('File size too large. Maximum 10MB.');
        return;
    }

    selectedFile = file;
    displayPreview(file);
}

function displayPreview(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadContent.style.display = 'none';
        previewContainer.style.display = 'block';
        analyzeButton.style.display = 'flex';
        resultContainer.style.display = 'none';
    };
    
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    uploadContent.style.display = 'block';
    previewContainer.style.display = 'none';
    analyzeButton.style.display = 'none';
    resultContainer.style.display = 'none';
}

async function analyzeImage() {
    if (!selectedFile) return;

    analyzeButton.disabled = true;
    analyzeText.textContent = 'Analyzing...';
    loadingSpinner.style.display = 'block';
    resultContainer.style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(PREDICT_ENDPOINT, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Prediction failed');
        }

        const data = await response.json();
        displayResult(data);

    } catch (error) {
        console.error('Error:', error);
        alert(`An error occurred: ${error.message}. Please try again.`);
    } finally {
        analyzeButton.disabled = false;
        analyzeText.textContent = 'Analyze Image';
        loadingSpinner.style.display = 'none';
    }
}

function displayResult(data) {
    const { prediction, confidence, probabilities } = data;
    const confidencePercent = (parseFloat(confidence) * 100).toFixed(1);

    if (prediction === 'real') {
        resultIcon.className = 'result-icon real';
        resultIcon.innerHTML = '✓';
        resultTitle.textContent = 'Real Painting';
        resultDescription.textContent = 'This image is detected as an authentic human-made painting. The model recognizes texture characteristics and details consistent with traditional artwork.';
    } else {
        resultIcon.className = 'result-icon ai';
        resultIcon.innerHTML = '🤖';
        resultTitle.textContent = 'AI-Generated Painting';
        resultDescription.textContent = 'This image is detected as AI-generated artwork. The model recognizes patterns and characteristics commonly found in images produced by artificial intelligence.';
    }

    confidenceFill.style.width = `${confidencePercent}%`;
    confidenceText.textContent = `Confidence Level: ${confidencePercent}%`;

    // Add probability details
    if (probabilities) {
        const realProb = (probabilities.real * 100).toFixed(1);
        const aiProb = (probabilities.ai * 100).toFixed(1);
        resultDescription.textContent += ` (Real: ${realProb}%, AI: ${aiProb}%)`;
    }

    resultContainer.style.display = 'block';
    
    setTimeout(() => {
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

document.querySelectorAll('.why-use, .technology, .upload-section').forEach(section => {
    section.style.opacity = '0';
    section.style.transform = 'translateY(30px)';
    section.style.transition = 'all 0.6s ease-out';
    observer.observe(section);
});