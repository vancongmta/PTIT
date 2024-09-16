<?php

$target_path = realpath(dirname(__FILE__)) . "/uploads/";

$target_path = $target_path . basename( $_FILES['txt_file']['name']); 

if(move_uploaded_file($_FILES['txt_file']['tmp_name'], $target_path)) {
    echo "The file was successfully uploaded<br>";

    $content = file_get_contents($target_path);
    echo "The text inside the file is: '" . $content . "'";
} else{
    echo "There was an error uploading the file, please try again!";
}
            
?>