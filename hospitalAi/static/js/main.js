// static/js/main.js

// Function to reset the form
function resetForm() {
    const form = document.getElementById('predict-form');
    if (form) {
        form.reset();
    }
}

// Function to handle form submission
function handleSubmit(event) {
    event.preventDefault();

    const submitButton = document.getElementById('predict-btn');
    const buttonText = document.getElementById('btn-text');
    const buttonSpinner = document.getElementById('btn-spinner');

    // Show loading state
    if (submitButton && buttonText && buttonSpinner) {
        buttonText.textContent = 'Predicting...';
        buttonSpinner.classList.remove('d-none');
        submitButton.disabled = true;
        event.target.submit();
    }
}

// Add event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Form submission (for form.html)
    const form = document.getElementById('predict-form');
    if (form) {
        form.addEventListener('submit', handleSubmit);
    }

    // Reset button (for form.html)
    const resetButton = document.getElementById('reset-btn');
    if (resetButton) {
        resetButton.addEventListener('click', resetForm);
    }

    // Suggestions link (for results.html)
    const suggestionsLink = document.getElementById('suggestions-link');
    if (suggestionsLink) {
        suggestionsLink.addEventListener('click', (e) => {
            e.preventDefault();
            window.location.href = '/tips'; // Redirect to tips.html instead of alert
        });
    }
});