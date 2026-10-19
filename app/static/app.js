// Q-RetinaScreen Client-Side Interactions

document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const previewContainer = document.getElementById("filePreviewContainer");
    const previewImage = document.getElementById("previewImage");
    const fileNameLabel = document.getElementById("fileNameLabel");
    const form = document.getElementById("screeningForm");
    const submitBtn = document.getElementById("submitScreenBtn");
    const spinner = document.getElementById("spinner");

    if (dropZone && fileInput) {
        // Drag & Drop events
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

        fileInput.addEventListener("change", (e) => {
            if (fileInput.files.length > 0) {
                handleFiles(fileInput.files[0]);
            }
        });
    }

    function handleFiles(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please select an image file (PNG, JPG, or TIFF).");
            return;
        }

        fileNameLabel.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainer.classList.remove("hidden");
        };
        reader.readAsDataURL(file);
    }

    if (form) {
        form.addEventListener("submit", () => {
            if (spinner && submitBtn) {
                spinner.classList.remove("hidden");
                submitBtn.disabled = true;
                const btnText = submitBtn.querySelector(".btn-text");
                if (btnText) {
                    btnText.textContent = "Processing Quality Gate & Quantum Circuit...";
                }
            }
        });
    }
});

// Function to select sample fundus image from demo library
function selectSample(sampleFilename) {
    const form = document.getElementById("screeningForm");
    if (!form) return;

    // Create or update hidden input for sample_name
    let sampleInput = document.getElementById("hiddenSampleName");
    if (!sampleInput) {
        sampleInput = document.createElement("input");
        sampleInput.type = "hidden";
        sampleInput.id = "hiddenSampleName";
        sampleInput.name = "sample_name";
        form.appendChild(sampleInput);
    }

    sampleInput.value = sampleFilename;

    // Update preview UI
    const previewContainer = document.getElementById("filePreviewContainer");
    const fileNameLabel = document.getElementById("fileNameLabel");
    if (previewContainer && fileNameLabel) {
        fileNameLabel.textContent = `Selected Demo Sample: ${sampleFilename}`;
        previewContainer.classList.remove("hidden");
    }

    // Submit form automatically
    const submitBtn = document.getElementById("submitScreenBtn");
    if (submitBtn) {
        submitBtn.click();
    }
}
