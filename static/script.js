// Elements (they only exist on the home page)
const fileInput = document.getElementById("resume");
const fileName = document.getElementById("file-name");
const dropzone = document.getElementById("dropzone");
const form = document.getElementById("analyze-form");
const button = document.getElementById("analyze-btn");
const btnText = document.getElementById("btn-text");
const spinner = document.getElementById("spinner");
const jd = document.getElementById("job_description");
const counter = document.getElementById("char-count");

// Show the selected file name
function showFileName() {
    if (fileInput.files.length > 0) {
        fileName.textContent = "Selected: " + fileInput.files[0].name;
        fileName.classList.add("selected");
    } else {
        fileName.textContent = "No file selected (PDF, max 5 MB)";
        fileName.classList.remove("selected");
    }
}

if (fileInput) {
    fileInput.addEventListener("change", showFileName);
}

// Drag and drop highlight
if (dropzone) {
    ["dragenter", "dragover"].forEach(function (evt) {
        dropzone.addEventListener(evt, function (e) {
            e.preventDefault();
            dropzone.classList.add("drag");
        });
    });
    ["dragleave", "drop"].forEach(function (evt) {
        dropzone.addEventListener(evt, function (e) {
            e.preventDefault();
            dropzone.classList.remove("drag");
        });
    });
    dropzone.addEventListener("drop", function (e) {
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            showFileName();
        }
    });
}

// Character counter
function updateCounter() {
    counter.textContent = jd.value.length;
}
if (jd && counter) {
    jd.addEventListener("input", updateCounter);
    updateCounter();
}

// Loading state while the server analyzes
if (form) {
    form.addEventListener("submit", function () {
        button.disabled = true;
        btnText.textContent = "Analyzing...";
        spinner.classList.remove("hidden");
    });
}
