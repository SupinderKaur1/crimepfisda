<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crime Prediction Form</title>
    <link rel="stylesheet" href="style.css">
    <script src="predictCrime.js" defer></script>
</head>
<body>
    <div class="container">
        <h2>Crime Prediction Form</h2>
        <form id="predictionForm">
            <label for="area">Area:</label>
            <select id="area" name="area">
                    <option value="Hohola">Hohola</option>
                    <option value="Kanudi">Kanudi</option>
                    <option value="Tokarara">Tokarara</option>
                    <option value="Rainbow Estate">Rainbow Estate</option>
                    <option value="Nine Mile">Nine Mile</option>
                    <option value="Gordons">Gordons</option>
                    <option value="Badili">Badili</option>
                    <option value="Koki">Koki</option>
                    <option value="Ela Beach">Ela Beach</option>
                    <option value="Gerehu">Gerehu</option>
                    <option value="Kila Kila">Kila Kila</option>
                    <option value="Boroko">Boroko</option>
                    <option value="Vabukori">Vabukori</option>
                    <option value="Ensisi Valley">Ensisi Valley</option>
                    <option value="Korobosea">Korobosea</option>
                    <option value="Manu Autoport Area">Manu Autoport Area</option>
                    <option value="Hanuabada">Hanuabada</option>
                    <option value="Kaugere">Kaugere</option>
                    <option value="Morata">Morata</option>
                    <option value="Konedobu">Konedobu</option>
                    <option value="Waigani">Waigani</option>
            </select>

            <label for="age">Victim Age:</label>
            <input type="number" id="age" name="age">

            <label for="sex">Victim Sex:</label>
            <select id="sex" name="sex">
                <option value="M">Male</option>
                <option value="F">Female</option>
            </select>

            <label for="day">Day of the Week:</label>
            <select id="day" name="day">
                    <option value="Monday">Monday</option>
                    <option value="Tuesday">Tuesday</option>
                    <option value="Wednesday">Wednesday</option>
                    <option value="Thursday">Thursday</option>
                    <option value="Friday">Friday</option>
                    <option value="Saturday">Saturday</option>
                    <option value="Sunday">Sunday</option>
            </select>

            <button type="submit">Predict Crime</button>
        </form>
        <p id="result"></p>
    </div>
</body>
</html>







