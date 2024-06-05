#!/bin/bash

fetch_data() {
    url=$1
    output_file=$2

    curl -L -o $output_file $url

    if [ $? -eq 0 ]; then
        echo "Pobieranie z $url zakończone pomyślnie."
    else
        echo "Błąd podczas pobierania $url."
    fi
}

unzip_file() {
    zip_file=$1
    output_dir=$2

    unzip $zip_file -d $output_dir

    if [ $? -eq 0 ]; then
        echo "Rozpakowanie $zip_file zakończone pomyślnie."
    else
        echo "Błąd podczas rozpakowywania $zip_file."
        exit 1
    fi
}

remove_first_line() {
    local file=$1

    if [ -f "$file" ]; then
        tail -n +2 "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
        echo "Pierwsza linia została usunięta z pliku $file."
    else
        echo "Plik $file nie istnieje."
    fi
}

replace_characters() {
    local file="$1"
    local old_char="$2"
    local new_char="$3"

    if [ ! -f "$file" ]; then
        echo "File $file does not exist."
        return 1
    fi

    sed -i "s/$old_char/$new_char/g" "$file"

    echo "Replacement of '$old_char' with '$new_char' in file $file completed."
    return 0
}

fetch_data "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data" "german_credit_data.csv"
fetch_data "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" "adult_data.csv"
fetch_data "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" "wine_quality_red.csv"
fetch_data "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data" "mushroom_data.csv"
fetch_data "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip" "bank+marketing.zip"

unzip_file "bank+marketing.zip" "bank+marketing"
rm "bank+marketing.zip"
unzip_file "bank+marketing/bank.zip" "bank+marketing/bank"
mv "bank+marketing/bank/bank.csv" "."
rm -rf "bank+marketing"
remove_first_line "bank.csv"
remove_first_line "wine_quality_red.csv"
replace_characters "bank.csv" ";" ","
replace_characters "bank.csv" "\"" ""
replace_characters "adult_data.csv" ", " ","
replace_characters "german_credit_data.csv" " " ","
replace_characters "wine_quality_red.csv" ";" ","