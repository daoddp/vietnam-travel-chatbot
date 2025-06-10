# Đường dẫn file input và output
input_file = r"C:\MINE\UIT\Kỹ thuật lập trình Trí tuệ nhân tạo\Du_lich\data\data-tach.txt"
output_file = r"C:\MINE\UIT\Kỹ thuật lập trình Trí tuệ nhân tạo\Du_lich\data\data-tach-khong-trung.txt"

# Sử dụng tập hợp để kiểm tra trùng lặp
seen = set()
unique_lines = []

# Đọc và xử lý file
with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()  # Loại bỏ khoảng trắng ở đầu/cuối dòng
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

# Ghi các dòng duy nhất vào file output
with open(output_file, "w", encoding="utf-8") as file:
    for line in unique_lines:
        file.write(line + "\n")

print(f"Đã xóa các dòng trùng lặp. Kết quả được lưu vào '{output_file}'.")
