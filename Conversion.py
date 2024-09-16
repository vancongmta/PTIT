def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# Ghi lại nội dung đã chuyển đổi vào file txt
def write_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

# Thay thế các chuỗi token cụ thể
def replace_tokens(line):
    # Danh sách các từ khóa cần tìm kiếm và thay thế
    replacements = {
        "T_UNDEFINED_VAR , T_UNDEFINED_VAR": "2_T_UNDEFINED_VAR",
        "T_CONSTANT_ENCAPSED_STRING , T_CONSTANT_ENCAPSED_STRING , T_CONSTANT_ENCAPSED_STRING": "3_T_CONSTANT_ENCAPSED_STRING",
        "T_CONSTANT_ENCAPSED_STRING , T_CONSTANT_ENCAPSED_STRING": "2_T_CONSTANT_ENCAPSED_STRING",
        "T_IGNORE , T_IGNORE , T_IGNORE , T_IGNORE": "4_T_IGNORE",
        "T_IGNORE , T_IGNORE": "2_T_IGNORE",
        "T_SQL ; T_SQL": "2_T_SQL",
        "T_ASSIGNMENT , T_ASSIGNMENT , T_ASSIGNMENT": "3_T_ASSIGNMENT",
        "T_COMPARISON , T_COMPARISON , T_COMPARISON": "3_T_COMPARISON",
        "T_INCLUDES , T_INCLUDES": "2_T_INCLUDES",
        "T_ECHO T_CONSTANT_ENCAPSED_STRING ; T_ECHO T_CONSTANT_ENCAPSED_STRING": "3_T_ECHO T_CONSTANT_ENCAPSED_STRING",
        "T_INPUT ; T_INPUT ; T_INPUT": "3_T_INPUT",
        "T_CONSTANT_ENCAPSED_STRING . T_CONSTANT_ENCAPSED_STRING":"2_T_CONSTANT_ENCAPSED_STRING ."
        
    }
    
    # Duyệt qua danh sách và thay thế nếu tìm thấy chuỗi tương ứng
    for target, replacement in replacements.items():
        line = line.replace(target, replacement)
    
    return line

# Xử lý từng dòng của file
def process_lines(lines):
    transformed_lines = []
    for line in lines:
        transformed_line = replace_tokens(line)
        transformed_lines.append(transformed_line)
    return transformed_lines

# Đường dẫn file cần đọc
input_file = 'safenew.txt'
# Đường dẫn file sau khi thay thế
output_file = 'output.txt'

# Đọc nội dung từng dòng từ file
file_lines = read_file(input_file)
# Xử lý từng dòng và thay thế token theo yêu cầu
transformed_lines = process_lines(file_lines)
# Ghi nội dung mới vào file
write_file(output_file, transformed_lines)

print("Chuyển đổi thành công!")