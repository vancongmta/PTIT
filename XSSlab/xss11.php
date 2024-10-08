<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OSTE Vulnerable Web Application</title>
    <style>
        /* Add your custom CSS styles here */
        body {
            font-family: Arial, sans-serif;
            background-color: #f2f2f2;
            margin: 0;
            padding: 0;
        }
        header {
            background-color: #333;
            padding: 10px;
            color: #fff;
            display: flex;
            align-items: center;
               justify-content: space-between; /* Align the list to the right */

        }
        header img {
            width: 50px;
            height: 50px;
            margin-right: 10px;
        }
        header ul {
            list-style: none;
            margin: 0;
            padding: 0;
            display: flex;
        }
        header li {
            margin-left: 20px;
        }
        a{
        color:white;
        }
        h1 {
            font-family: 'Your Custom Font', Arial, sans-serif;
            text-align: center;
            margin: 30px 0;
        }
        p {
            text-align: center;
            margin-bottom: 30px;
        }
        .btn-container {
            display: flex;
            justify-content: center;
        }
        .button {
            background-color: #007BFF;
            color: #fff;
            border: none;
            padding: 10px 20px;
            margin: 10px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .button:hover {
            background-color: #0056b3;
        }
        p {
            text-align: center;
            font-size: 18px;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto 30px;
            padding: 10px;
            background-color: #f9f9f9;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        /* Add some color and emphasis for specific text */
        p strong {
            color: #007BFF;
        }

        /* Add some margin to the button container */
        .btn-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
                header {
            background-color: #333;
            padding: 10px;
            color: #fff;
            display: flex;
            justify-content: space-between; /* Align the list to the right */
            align-items: center;
        }

        header img {
            width: 100px;
            height: 50px;
            margin-right: 10px;
        }

        header ul {
            list-style: none;
            margin: 0;
            padding: 0;
            display: flex;
        }

        header li {
            margin-left: 20px;
        }
           .button-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 20px;
        }
        .button {
          background-color: #007BFF;
            color: #fff;
            border: none;
            padding: 15px 30px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 350px; /* Set the desired width for the buttons */
        }
        .button:hover {
            background-color: #0056b3;
        }
        
        form {
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 300px;
        }
        label {
            font-size: 16px;
            margin-bottom: 10px;
        }
        input[type="text"] {
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            width: 100%;
            font-size: 16px;
        }
        button {
            background-color: #007BFF;
            color: #fff;
            border: none;
            padding: 10px 20px;
            margin-top: 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <header>
        <img src="../OSTE.svg" alt="Logo">
        <a href="index.php"> <img style="width:25px;height:25px;" src="../ico/undo.png" alt="back"></a>

        <ul>
   	     <li><a href="../index.php">Home</a></li>
            <li><a href="database.php">Database</a></li>
            <li><a href="vulnerabilities.php">Vulnerabilities</a></li>
        </ul>
    </header>

    <h1>Greetings, Tell me a story?</h1>
    <div class="button-container">
        <form method="post" action="">
        <label for="inputField">Enter your Story:*</label>
        <input type="text" id="inputField2" name="username" required>

        <button type="submit">Submit</button>
        </form>
        <form method="post" action="">
        <button type="submit" name="clear">Clear History</button>
    </form>
    </div>
<?php

try {
    $servername = "localhost";
    $username = "root";
    $password = "";

    // Step 1: Connect to MySQL
    $conn = new PDO("mysql:host=$servername", $username, $password);
    $conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    $conn->exec("USE OSTE");

    // Step 2: Fetch messages from the "comontair" table
    $sql = "SELECT msg FROM comontair";
    $stmt = $conn->prepare($sql);
    $stmt->execute();
    $messages = $stmt->fetchAll(PDO::FETCH_COLUMN);

} catch (PDOException $e) {
    echo "Connection failed: " . $e->getMessage();
}
if(isset($_POST['clear'])){
$sql = "DELETE FROM comontair";
    $stmt = $conn->prepare($sql);
    $stmt->execute();
}
echo"<h1> List of old stories:</h1>";
        foreach ($messages as $message): 
	    $message=str_replace( '<script>', '', $message );
            echo"<li>'.$message.'</li>";
        endforeach; 

if(isset($_POST['username'])){
try {

    // Step 2: Prepare and execute the SQL statement with parameter binding
    $sql = "INSERT INTO comontair (msg) VALUES (:username)";
    $stmt = $conn->prepare($sql);
    $stmt->bindParam(':username', $_POST['username'], PDO::PARAM_STR);
    $stmt->execute();

    // Optionally, you can check if the query was successful
//    if ($stmt->rowCount() > 0) {
  //      echo "Data inserted successfully.";
    //} else {
      //  echo "Error: Data not inserted.";
    //}

} catch (PDOException $e) {
    echo "Connection failed: " . $e->getMessage();
}

echo"<h1>This is your story ! Really? :";

echo"</h1>";
}

?>

</body>
</html>