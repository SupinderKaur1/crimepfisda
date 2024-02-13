document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('predictionForm').addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent default form submission
        predictCrime();
    });
});

function predictCrime() {
    const area = document.getElementById('area').value;
    const age = document.getElementById('age').value;
    const sex = document.getElementById('sex').value;
    const day = document.getElementById('day').value;

    // Adjust the URL to your Flask API
    const url = "http://localhost:5002/predict";

    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ area, age, sex, day }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerHTML = "Predicted Crime: " + data.predicted_crime;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = "An error occurred.";
    });
}
