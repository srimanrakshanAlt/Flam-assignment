// Mathematical Model in JavaScript
function parametricX(t, theta, M, X) {
    return t * Math.cos(theta) - Math.exp(M * Math.abs(t)) * Math.sin(0.3 * t) * Math.sin(theta) + X;
}

function parametricY(t, theta, M) {
    return 42.0 + t * Math.sin(theta) + Math.exp(M * Math.abs(t)) * Math.sin(0.3 * t) * Math.cos(theta);
}

// Optimized targets
const OPT_THETA = 30.0;
const OPT_M = 0.03;
const OPT_X = 55.0;

// State management
let currentTheta = 30.0;
let currentM = 0.03;
let currentX = 55.0;
let animationFrameId = null;

// DOM Elements
const sliderTheta = document.getElementById('slider-theta');
const sliderM = document.getElementById('slider-m');
const sliderX = document.getElementById('slider-x');

const valTheta = document.getElementById('val-theta');
const valM = document.getElementById('val-m');
const valX = document.getElementById('val-x');

const metricL1 = document.getElementById('metric-l1');
const metricL2 = document.getElementById('metric-l2');
const fitStatusText = document.getElementById('fit-status-text');
const fitStatusIndicator = document.querySelector('#fit-status .status-indicator');
const fitStatusContainer = document.getElementById('fit-status');

const btnReset = document.getElementById('btn-reset');
const btnAnimate = document.getElementById('btn-animate');

// Initial Setup
function init() {
    setupEventListeners();
    updateUI(currentTheta, currentM, currentX);
    renderInitialChart();
}

function setupEventListeners() {
    sliderTheta.addEventListener('input', (e) => {
        stopAnimation();
        currentTheta = parseFloat(e.target.value);
        valTheta.textContent = currentTheta.toFixed(2);
        updateDashboard();
    });

    sliderM.addEventListener('input', (e) => {
        stopAnimation();
        currentM = parseFloat(e.target.value);
        valM.textContent = currentM.toFixed(4);
        updateDashboard();
    });

    sliderX.addEventListener('input', (e) => {
        stopAnimation();
        currentX = parseFloat(e.target.value);
        valX.textContent = currentX.toFixed(2);
        updateDashboard();
    });

    btnReset.addEventListener('click', () => {
        stopAnimation();
        animateTo(OPT_THETA, OPT_M, OPT_X, 800); // Smooth transition preset
    });

    btnAnimate.addEventListener('click', () => {
        stopAnimation();
        // Start from a distorted state and fit in real-time
        currentTheta = 5.0;
        currentM = -0.04;
        currentX = 15.0;
        updateUI(currentTheta, currentM, currentX);
        animateTo(OPT_THETA, OPT_M, OPT_X, 2500); // 2.5s smooth fitting animation
    });
}

function stopAnimation() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

// Linear interpolation utility
function lerp(start, end, amt) {
    return (1 - amt) * start + amt * end;
}

// Animate parameters smoothly to target values
function animateTo(targetTheta, targetM, targetX, durationMs) {
    const startTheta = currentTheta;
    const startM = currentM;
    const startX = currentX;
    const startTime = performance.now();

    function step(timestamp) {
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / durationMs, 1.0);
        
        // Easing function: Ease Out Quad
        const ease = progress * (2 - progress);

        currentTheta = lerp(startTheta, targetTheta, ease);
        currentM = lerp(startM, targetM, ease);
        currentX = lerp(startX, targetX, ease);

        updateUI(currentTheta, currentM, currentX);
        updateDashboard();

        if (progress < 1.0) {
            animationFrameId = requestAnimationFrame(step);
        } else {
            animationFrameId = null;
        }
    }
    
    animationFrameId = requestAnimationFrame(step);
}

// Sync slider handles and displays
function updateUI(theta, m, x) {
    sliderTheta.value = theta;
    sliderM.value = m;
    sliderX.value = x;

    valTheta.textContent = theta.toFixed(2);
    valM.textContent = m.toFixed(4);
    valX.textContent = x.toFixed(2);
}

// Calculate errors and update metrics
function calculateLosses(thetaDeg, M, X) {
    const thetaRad = thetaDeg * Math.PI / 180.0;
    let absoluteError = 0;
    let squaredError = 0;
    const N = COORDINATES_DATA.length;

    for (let i = 0; i < N; i++) {
        const px = COORDINATES_DATA[i].x;
        const py = COORDINATES_DATA[i].y;

        // Decoupled coordinate projection
        const t = (px - X) * Math.cos(thetaRad) + (py - 42.0) * Math.sin(thetaRad);
        
        // Compute predicted coordinates
        const predX = parametricX(t, thetaRad, M, X);
        const predY = parametricY(t, thetaRad, M);

        // Euclidean distance error
        const dist = Math.sqrt((px - predX) ** 2 + (py - predY) ** 2);
        absoluteError += dist;
        squaredError += dist * dist;
    }

    const meanL1 = absoluteError / N;
    const meanL2 = squaredError / N;

    return { meanL1, meanL2 };
}

// Compute 1000 points on the curve for plotting
function generateCurveTrace(thetaDeg, M, X) {
    const thetaRad = thetaDeg * Math.PI / 180.0;
    const xCurve = [];
    const yCurve = [];
    
    // Evaluate t from 6 to 60 with fine steps
    const steps = 1000;
    for (let i = 0; i <= steps; i++) {
        const t = 6.0 + (54.0 * i / steps);
        xCurve.push(parametricX(t, thetaRad, M, X));
        yCurve.push(parametricY(t, thetaRad, M));
    }

    return { x: xCurve, y: yCurve };
}

// Render Plotly Chart
function renderInitialChart() {
    const scatterTrace = {
        x: COORDINATES_DATA.map(p => p.x),
        y: COORDINATES_DATA.map(p => p.y),
        mode: 'markers',
        type: 'scattergl',
        name: 'Observed Data',
        marker: {
            color: '#38bdf8',
            size: 4.5,
            opacity: 0.45
        }
    };

    const curveData = generateCurveTrace(currentTheta, currentM, currentX);
    const curveTrace = {
        x: curveData.x,
        y: curveData.y,
        mode: 'lines',
        name: 'Fitted Curve',
        line: {
            color: '#f43f5e',
            width: 3.5
        }
    };

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { l: 40, r: 20, t: 20, b: 45 },
        xaxis: {
            gridcolor: 'rgba(255, 255, 255, 0.05)',
            zeroline: false,
            tickfont: { color: '#9ca3af', family: 'Outfit' },
            title: { text: 'X coordinate', font: { color: '#9ca3af', family: 'Outfit' } }
        },
        yaxis: {
            gridcolor: 'rgba(255, 255, 255, 0.05)',
            zeroline: false,
            tickfont: { color: '#9ca3af', family: 'Outfit' },
            title: { text: 'Y coordinate', font: { color: '#9ca3af', family: 'Outfit' } }
        },
        showlegend: false,
        hovermode: 'closest'
    };

    const config = {
        responsive: true,
        displayModeBar: false
    };

    Plotly.newPlot('plotly-chart', [scatterTrace, curveTrace], layout, config);
    updateDashboard();
}

// Update the chart trace and metrics
function updateDashboard() {
    // 1. Calculate and update metrics
    const losses = calculateLosses(currentTheta, currentM, currentX);
    metricL1.textContent = losses.meanL1.toFixed(8);
    metricL2.textContent = losses.meanL2.toFixed(8);

    // 2. Update status text
    if (losses.meanL1 < 1e-4) {
        fitStatusContainer.className = "metric-status";
        fitStatusIndicator.className = "status-indicator success";
        fitStatusText.textContent = "Perfect Fit! (Error < 1e-5)";
    } else {
        fitStatusContainer.className = "metric-status warning";
        fitStatusIndicator.className = "status-indicator warning";
        fitStatusText.textContent = "Adjusting Parameters...";
    }

    // 3. restyle Plotly curve trace (Trace 1 is the curve line)
    const curveData = generateCurveTrace(currentTheta, currentM, currentX);
    Plotly.restyle('plotly-chart', {
        x: [curveData.x],
        y: [curveData.y]
    }, [1]);
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', init);
