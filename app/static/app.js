// Q-RetinaScreen: Advanced Clinical Tele-Ophthalmology Client-Side System
// Features: Multi-Theme Switcher, Interactive Saliency Viewer, Progressive Inference Steps, & EHR Clipboard.

// --------------------------------------------------------------------------
// 1. Theme Management System
// --------------------------------------------------------------------------
const THEMES = {
    "quantum-dark": { label: "Quantum Dark", icon: "🌙" },
    "clinical-light": { label: "Clinical Light", icon: "☀️" },
    "midnight-oled": { label: "Midnight OLED", icon: "🌌" },
    "nordic-emerald": { label: "Nordic Emerald", icon: "🌿" }
};

function initThemeSystem() {
    const currentTheme = localStorage.getItem("q_retina_theme") || "quantum-dark";
    applyTheme(currentTheme, false);

    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const themeDropdown = document.getElementById("themeDropdown");

    if (themeToggleBtn && themeDropdown) {
        themeToggleBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            themeDropdown.classList.toggle("hidden");
        });

        // Handle option clicks
        const options = themeDropdown.querySelectorAll(".theme-option");
        options.forEach(opt => {
            opt.addEventListener("click", (e) => {
                e.stopPropagation();
                const selectedTheme = opt.getAttribute("data-theme");
                applyTheme(selectedTheme, true);
                themeDropdown.classList.add("hidden");
            });
        });

        // Close dropdown when clicking outside
        document.addEventListener("click", (e) => {
            if (!themeToggleBtn.contains(e.target) && !themeDropdown.contains(e.target)) {
                themeDropdown.classList.add("hidden");
            }
        });

        // Close on Escape key
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape") {
                themeDropdown.classList.add("hidden");
            }
        });
    }
}

function applyTheme(themeKey, showFeedback = false) {
    if (!THEMES[themeKey]) themeKey = "quantum-dark";
    
    document.documentElement.setAttribute("data-theme", themeKey);
    localStorage.setItem("q_retina_theme", themeKey);

    const iconSpan = document.getElementById("themeIcon");
    const labelSpan = document.getElementById("themeLabel");
    if (iconSpan && labelSpan) {
        iconSpan.textContent = THEMES[themeKey].icon;
        labelSpan.textContent = THEMES[themeKey].label;
    }

    // Highlight active option in dropdown
    const options = document.querySelectorAll(".theme-option");
    options.forEach(opt => {
        if (opt.getAttribute("data-theme") === themeKey) {
            opt.classList.add("active-theme");
        } else {
            opt.classList.remove("active-theme");
        }
    });

    if (showFeedback) {
        showToast(`Theme switched to ${THEMES[themeKey].label}`, "info");
    }
}

// --------------------------------------------------------------------------
// 2. Global Toast Notification System
// --------------------------------------------------------------------------
function showToast(message, type = "info") {
    const container = document.getElementById("toastContainer");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    
    const icon = type === "success" ? "✅" : (type === "error" ? "⚠️" : "💡");
    toast.innerHTML = `<span class="toast-icon">${icon}</span> <span class="toast-text">${message}</span>`;
    
    container.appendChild(toast);
    setTimeout(() => toast.classList.add("show"), 10);

    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 300);
    }, 3200);
}

// --------------------------------------------------------------------------
// 3. Image Acquisition & File Upload Enhancements
// --------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
    initThemeSystem();

    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const previewContainer = document.getElementById("filePreviewContainer");
    const previewImage = document.getElementById("previewImage");
    const fileNameLabel = document.getElementById("fileNameLabel");
    const fileDimLabel = document.getElementById("fileDimLabel");
    const clearFileBtn = document.getElementById("clearFileBtn");
    const form = document.getElementById("screeningForm");
    const submitBtn = document.getElementById("submitScreenBtn");
    const processingSteps = document.getElementById("processingSteps");

    if (dropZone && fileInput) {
        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add("dragover");
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove("dragover");
            }, false);
        });

        dropZone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFiles(files[0]);
            }
        });

        fileInput.addEventListener("change", () => {
            if (fileInput.files.length > 0) {
                handleFiles(fileInput.files[0]);
            }
        });
    }

    if (clearFileBtn) {
        clearFileBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            if (fileInput) fileInput.value = "";
            const hiddenSample = document.getElementById("hiddenSampleName");
            if (hiddenSample) hiddenSample.remove();
            if (previewContainer) previewContainer.classList.add("hidden");
            if (fileNameLabel) fileNameLabel.textContent = "";
            if (fileDimLabel) fileDimLabel.textContent = "";
            showToast("Fundus image cleared", "info");
        });
    }

    function handleFiles(file) {
        if (!file.type.startsWith("image/")) {
            showToast("Please upload an image file (PNG, JPG, or TIFF).", "error");
            return;
        }

        const sizeKb = (file.size / 1024).toFixed(1);
        if (fileNameLabel) fileNameLabel.textContent = `${file.name} (${sizeKb} KB)`;

        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewImage) {
                previewImage.src = e.target.result;
                previewImage.onload = () => {
                    if (fileDimLabel) {
                        fileDimLabel.textContent = `Native Resolution: ${previewImage.naturalWidth} × ${previewImage.naturalHeight} px`;
                    }
                };
            }
            if (previewContainer) previewContainer.classList.remove("hidden");
        };
        reader.readAsDataURL(file);
    }

    // Progressive inference status visualizer
    if (form) {
        form.addEventListener("submit", () => {
            if (submitBtn) {
                submitBtn.disabled = true;
                const btnText = submitBtn.querySelector(".btn-text");
                if (btnText) btnText.textContent = "Screening in progress...";
            }
            if (processingSteps) {
                processingSteps.classList.remove("hidden");
                animateProcessingSteps();
            }
        });
    }

    // 1-Click EHR Clinical Note Copy
    const copyEhrBtn = document.getElementById("copyEhrBtn");
    const hiddenEhr = document.getElementById("hiddenEhrPayload");
    if (copyEhrBtn && hiddenEhr) {
        copyEhrBtn.addEventListener("click", () => {
            const noteText = hiddenEhr.innerText.trim();
            navigator.clipboard.writeText(noteText).then(() => {
                showToast("Clinical progress note copied to clipboard!", "success");
            }).catch(() => {
                showToast("Failed to copy automatically. Please select text manually.", "error");
            });
        });
    }
});

// Animate multi-stage inference progress
function animateProcessingSteps() {
    const progressFill = document.getElementById("progressBarFill");
    const statusText = document.getElementById("processingStatusText");
    const steps = [
        { id: "step1", pct: "30%", text: "Executing Quality Gate verification..." },
        { id: "step2", pct: "60%", text: "Extracting convolutional embeddings (EfficientNet-B0)..." },
        { id: "step3", pct: "85%", text: "Simulating PennyLane 4-Qubit Variational Quantum Circuit..." },
        { id: "step4", pct: "98%", text: "Generating Grad-CAM overlay & Eye Health analytics..." },
    ];

    let currentStep = 0;
    const interval = setInterval(() => {
        currentStep++;
        if (currentStep < steps.length) {
            const s = steps[currentStep];
            if (progressFill) progressFill.style.width = s.pct;
            if (statusText) statusText.textContent = s.text;
            
            const prevLi = document.getElementById(steps[currentStep - 1].id);
            if (prevLi) {
                prevLi.classList.remove("active");
                prevLi.classList.add("done");
            }
            const currLi = document.getElementById(s.id);
            if (currLi) currLi.classList.add("active");
        } else {
            clearInterval(interval);
        }
    }, 750);
}

// --------------------------------------------------------------------------
// 4. Sample Selection Helper
// --------------------------------------------------------------------------
function selectSample(sampleFilename) {
    const form = document.getElementById("screeningForm");
    if (!form) return;

    let sampleInput = document.getElementById("hiddenSampleName");
    if (!sampleInput) {
        sampleInput = document.createElement("input");
        sampleInput.type = "hidden";
        sampleInput.id = "hiddenSampleName";
        sampleInput.name = "sample_name";
        form.appendChild(sampleInput);
    }
    sampleInput.value = sampleFilename;

    const previewContainer = document.getElementById("filePreviewContainer");
    const fileNameLabel = document.getElementById("fileNameLabel");
    const fileDimLabel = document.getElementById("fileDimLabel");
    if (previewContainer && fileNameLabel) {
        fileNameLabel.textContent = `Demo Cohort: ${sampleFilename}`;
        if (fileDimLabel) fileDimLabel.textContent = "Pre-loaded Reference Digital Fundus Photo";
        previewContainer.classList.remove("hidden");
    }

    const submitBtn = document.getElementById("submitScreenBtn");
    if (submitBtn) {
        submitBtn.click();
    }
}

// --------------------------------------------------------------------------
// 5. Interactive Grad-CAM Viewer Controls (Report View)
// --------------------------------------------------------------------------
function setViewerMode(mode) {
    const container = document.getElementById("imageComparisonContainer");
    const origBox = document.getElementById("origImgBox");
    const camBox = document.getElementById("camImgBox");
    const origImg = document.getElementById("origImg");
    const gradCamImg = document.getElementById("gradCamImg");
    const opacitySlider = document.getElementById("opacitySliderContainer");

    const btnSplit = document.getElementById("btnModeSplit");
    const btnBlend = document.getElementById("btnModeBlend");
    const btnRedfree = document.getElementById("btnModeRedfree");

    [btnSplit, btnBlend, btnRedfree].forEach(b => { if (b) b.classList.remove("active"); });

    if (mode === "split") {
        if (btnSplit) btnSplit.classList.add("active");
        if (container) container.className = "image-comparison-container mode-split";
        if (origImg) origImg.style.filter = "none";
        if (gradCamImg) gradCamImg.style.opacity = "1";
        if (opacitySlider) opacitySlider.style.display = "flex";
    } else if (mode === "blend") {
        if (btnBlend) btnBlend.classList.add("active");
        if (container) container.className = "image-comparison-container mode-blend";
        if (origImg) origImg.style.filter = "none";
        const slider = document.getElementById("heatmapOpacitySlider");
        const val = slider ? slider.value : 75;
        updateHeatmapOpacity(val);
        if (opacitySlider) opacitySlider.style.display = "flex";
    } else if (mode === "redfree") {
        if (btnRedfree) btnRedfree.classList.add("active");
        if (container) container.className = "image-comparison-container mode-split";
        // Ophthalmic red-free filter (green channel contrast enhancement to highlight microaneurysms)
        if (origImg) origImg.style.filter = "grayscale(100%) brightness(1.1) contrast(1.4) sepia(100%) hue-rotate(85deg)";
        if (gradCamImg) gradCamImg.style.opacity = "1";
    }
}

function updateHeatmapOpacity(val) {
    const gradCamImg = document.getElementById("gradCamImg");
    const display = document.getElementById("opacityDisplay");
    if (display) display.textContent = `${val}%`;
    if (gradCamImg) {
        gradCamImg.style.opacity = val / 100;
    }
}
