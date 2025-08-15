// Configuration - Dynamic API URL with proper port handling
function getApiBaseUrl() {
    const protocol = "http";
    const hostname = "103.150.90.220";
    const currentPort = 8000;
    
    // Always use the same hostname as the frontend
    // API is always on port 8000
    return `${protocol}//${hostname}:8000`;
}

const API_BASE_URL = getApiBaseUrl();

console.log('Current location:', window.location.href);
console.log('API Base URL:', API_BASE_URL);

// DOM Elements
const fraudForm = document.getElementById('fraudForm');
const resultContainer = document.getElementById('resultContainer');
const resultContent = document.getElementById('resultContent');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorContainer = document.getElementById('errorContainer');
const errorMessage = document.getElementById('errorMessage');
const submitBtn = document.getElementById('submitBtn');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const retryBtn = document.getElementById('retryBtn');

// Form validation patterns
const validationRules = {
    amount: {
        min: 0.01,
        max: 1000000,
        message: 'Amount must be between $0.01 and $1,000,000'
    }
};

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

fraudForm.addEventListener('submit', handleFormSubmit);
newAnalysisBtn.addEventListener('click', resetForm);
retryBtn.addEventListener('click', hideError);

// Initialize the application
function initializeApp() {
    console.log('Fraud Detection Frontend Initialized');
    
    // Add input validation
    setupInputValidation();
    
    // Check API health
    checkAPIHealth();
}

// Setup real-time input validation
function setupInputValidation() {
    const amountInput = document.getElementById('amount');
    
    amountInput.addEventListener('input', function() {
        validateAmount(this);
    });
    
    amountInput.addEventListener('blur', function() {
        validateAmount(this);
    });
}

// Validate amount input
function validateAmount(input) {
    const value = parseFloat(input.value);
    const rules = validationRules.amount;
    
    // Remove existing validation classes
    input.classList.remove('valid', 'invalid');
    
    if (value < rules.min || value > rules.max) {
        input.classList.add('invalid');
        showInputError(input, rules.message);
    } else {
        input.classList.add('valid');
        hideInputError(input);
    }
}

// Show input error
function showInputError(input, message) {
    let errorElement = input.parentNode.querySelector('.input-error');
    
    if (!errorElement) {
        errorElement = document.createElement('div');
        errorElement.className = 'input-error';
        input.parentNode.appendChild(errorElement);
    }
    
    errorElement.textContent = message;
    errorElement.style.display = 'block';
}

// Hide input error
function hideInputError(input) {
    const errorElement = input.parentNode.querySelector('.input-error');
    if (errorElement) {
        errorElement.style.display = 'none';
    }
}

// Check API health
async function checkAPIHealth() {
    console.log('Checking API health at:', `${API_BASE_URL}/health`);
    
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
            mode: 'cors',
        });
        
        if (response.ok) {
            const healthData = await response.json();
            console.log('API is healthy:', healthData);
        } else {
            console.warn('API health check failed:', response.status, response.statusText);
        }
    } catch (error) {
        console.error('API is not accessible:', error);
        
        // Show user-friendly error for network issues
        if (error.name === 'TypeError' && (error.message.includes('Failed to fetch') || error.message.includes('NetworkError'))) {
            showError(`Cannot connect to fraud detection service at ${API_BASE_URL}. Please check if the server is running and accessible.`);
        }
    }
}

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    if (!validateForm()) {
        return;
    }
    
    const formData = getFormData();
    
    showLoading();
    hideError();
    hideResults();
    
    try {
        const result = await predictFraud(formData);
        showResults(result);
    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'An error occurred while analyzing the transaction.');
    } finally {
        hideLoading();
    }
}

// Validate entire form
function validateForm() {
    const form = fraudForm;
    let isValid = true;
    
    // Check required fields
    const requiredFields = form.querySelectorAll('[required]');
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            field.classList.add('invalid');
            showInputError(field, 'This field is required');
            isValid = false;
        } else {
            field.classList.remove('invalid');
            hideInputError(field);
        }
    });
    
    // Validate amount specifically
    const amountField = document.getElementById('amount');
    const amount = parseFloat(amountField.value);
    const rules = validationRules.amount;
    
    if (amount < rules.min || amount > rules.max) {
        amountField.classList.add('invalid');
        showInputError(amountField, rules.message);
        isValid = false;
    }
    
    return isValid;
}

// Get form data
function getFormData() {
    return {
        amount: parseFloat(document.getElementById('amount').value),
        merchant_type: document.getElementById('merchant_type').value,
        device_type: document.getElementById('device_type').value
    };
}

// Predict fraud via API
async function predictFraud(data) {
    console.log('Making API request to:', `${API_BASE_URL}/predict`);
    console.log('Request data:', data);
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            body: JSON.stringify(data),
            mode: 'cors', // Explicitly set CORS mode
        });
        
        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        console.log('API response:', result);
        return result;
        
    } catch (error) {
        console.error('API request failed:', error);
        
        // Provide more specific error messages
        if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
            throw new Error(`Cannot connect to API server at ${API_BASE_URL}. Please check if the server is running.`);
        } else if (error.name === 'TypeError' && error.message.includes('NetworkError')) {
            throw new Error(`Network error: Cannot reach API server at ${API_BASE_URL}`);
        } else {
            throw error;
        }
    }
}

// Show loading indicator
function showLoading() {
    loadingIndicator.style.display = 'block';
    loadingIndicator.classList.add('fade-in');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
}

// Hide loading indicator
function hideLoading() {
    loadingIndicator.style.display = 'none';
    loadingIndicator.classList.remove('fade-in');
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Transaction';
}

// Show results
function showResults(result) {
    const resultHtml = generateResultHTML(result);
    resultContent.innerHTML = resultHtml;
    resultContainer.style.display = 'block';
    resultContainer.classList.add('slide-up');
    
    // Animate probability bar
    setTimeout(() => {
        animateProbabilityBar(result.fraud_probability);
    }, 300);
    
    // Scroll to results
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Generate result HTML
function generateResultHTML(result) {
    const riskLevel = result.risk_level.toLowerCase();
    const probability = (result.fraud_probability * 100).toFixed(1);
    const fraudStatus = result.is_fraud ? 'High Risk Transaction' : 'Safe Transaction';
    const fraudIcon = result.is_fraud ? 'fas fa-exclamation-triangle' : 'fas fa-shield-alt';
    
    // Clear fraud/not fraud indication
    const fraudResult = result.is_fraud ? 'FRAUD' : 'NOT FRAUD';
    const fraudResultClass = result.is_fraud ? 'fraud-detected' : 'no-fraud-detected';
    const fraudResultIcon = result.is_fraud ? 'fas fa-times-circle' : 'fas fa-check-circle';
    
    let riskClass = 'safe';
    if (result.fraud_probability > 0.7) {
        riskClass = 'danger';
    } else if (result.fraud_probability > 0.3) {
        riskClass = 'warning';
    }
    
    return `
        <div class="result-card ${riskClass}">
            <div class="fraud-result-banner ${fraudResultClass}">
                <i class="${fraudResultIcon}"></i>
                <span class="fraud-result-text">${fraudResult}</span>
            </div>
            
            <div class="result-title ${riskClass}">
                <i class="${fraudIcon}"></i>
                ${fraudStatus}
            </div>
            
            <div class="result-details">
                <div class="detail-item">
                    <div class="detail-label">Fraud Probability</div>
                    <div class="detail-value">${probability}%</div>
                    <div class="probability-bar">
                        <div class="probability-fill ${getProbabilityClass(result.fraud_probability)}" 
                             data-probability="${probability}"></div>
                    </div>
                </div>
                
                <div class="detail-item">
                    <div class="detail-label">Risk Level</div>
                    <div class="detail-value">${result.risk_level}</div>
                </div>
                
                <div class="detail-item">
                    <div class="detail-label">Transaction Amount</div>
                    <div class="detail-value">$${result.transaction_details.amount.toLocaleString()}</div>
                </div>
                
                <div class="detail-item">
                    <div class="detail-label">Merchant Type</div>
                    <div class="detail-value">${formatMerchantType(result.transaction_details.merchant_type)}</div>
                </div>
                
                <div class="detail-item">
                    <div class="detail-label">Device Type</div>
                    <div class="detail-value">${formatDeviceType(result.transaction_details.device_type)}</div>
                </div>
                
                <div class="detail-item">
                    <div class="detail-label">Recommendation</div>
                    <div class="detail-value">${getRecommendation(result)}</div>
                </div>
            </div>
        </div>
    `;
}

// Get probability bar class
function getProbabilityClass(probability) {
    if (probability > 0.7) return 'high';
    if (probability > 0.3) return 'medium';
    return 'low';
}

// Animate probability bar
function animateProbabilityBar(probability) {
    const bar = document.querySelector('.probability-fill');
    if (bar) {
        const percentage = (probability * 100).toFixed(1);
        bar.style.width = `${percentage}%`;
    }
}

// Format merchant type for display
function formatMerchantType(type) {
    return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

// Format device type for display
function formatDeviceType(type) {
    const deviceMap = {
        'mobile': 'Mobile Device',
        'desktop': 'Desktop Computer',
        'tablet': 'Tablet',
        'pos': 'POS Terminal',
        'atm': 'ATM',
        'web': 'Web Browser'
    };
    return deviceMap[type] || type.replace(/\b\w/g, l => l.toUpperCase());
}

// Get recommendation based on result
function getRecommendation(result) {
    if (result.fraud_probability > 0.8) {
        return 'Block Transaction';
    } else if (result.fraud_probability > 0.5) {
        return 'Manual Review';
    } else if (result.fraud_probability > 0.3) {
        return 'Monitor Closely';
    } else {
        return 'Approve Transaction';
    }
}

// Hide results
function hideResults() {
    resultContainer.style.display = 'none';
    resultContainer.classList.remove('slide-up');
}

// Show error
function showError(message) {
    errorMessage.textContent = message;
    errorContainer.style.display = 'block';
    errorContainer.classList.add('fade-in');
    errorContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Hide error
function hideError() {
    errorContainer.style.display = 'none';
    errorContainer.classList.remove('fade-in');
}

// Reset form to initial state
function resetForm() {
    fraudForm.reset();
    hideResults();
    hideError();
    hideLoading();
    
    // Remove validation classes
    const inputs = fraudForm.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.classList.remove('valid', 'invalid');
        hideInputError(input);
    });
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Utility function to format currency
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

// Add CSS for validation states
const style = document.createElement('style');
style.textContent = `
    .form-group input.valid,
    .form-group select.valid {
        border-color: #28a745;
        background-color: #f8fff9;
    }
    
    .form-group input.invalid,
    .form-group select.invalid {
        border-color: #dc3545;
        background-color: #fff8f8;
    }
    
    .input-error {
        color: #dc3545;
        font-size: 0.875rem;
        margin-top: 0.25rem;
        display: none;
    }
`;
document.head.appendChild(style);

// Export functions for testing (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        validateForm,
        getFormData,
        formatMerchantType,
        formatDeviceType,
        getRecommendation
    };
}
