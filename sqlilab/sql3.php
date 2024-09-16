<?php
$id = $_GET['id'];
$query = "SELECT * FROM users WHERE id = " . $id;
$result = mysql_query($query);
while ($row = mysql_fetch_assoc($result)) {
    echo "User: " . $row['username'];
}
?>
