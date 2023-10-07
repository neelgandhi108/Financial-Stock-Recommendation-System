document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('recommendation-form');
    const recommendationsDiv = document.getElementById('recommendations');

    form.addEventListener('submit', function (e) {
        e.preventDefault();

        const dateInput = document.getElementById('input-date').value;
        const investmentCapacityInput = document.getElementById('investment-capacity').value;
        recommendationsDiv.innerHTML = '<p>Recommendations: AAPL, MSFT, GOOGL</p>';
    });
});
