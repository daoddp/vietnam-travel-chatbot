import re

def split_sentences(file_path):
    """
    Nhận vào một đường dẫn đến file .txt, xử lý tách các đoạn văn thành dòng mới sao cho:
    - Mỗi dòng có khoảng 15 câu.
    - Chèn thêm câu đầu của đoạn gốc vào đoạn tách.

    Args:
        file_path (str): Đường dẫn đến file .txt.

    Returns:
        list: Danh sách các dòng đã xử lý.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    processed_lines = []

    for line in lines:
        sentences = re.split(r'(\.|\?|!)', line)  # Tách câu dựa trên dấu kết thúc câu
        sentences = [s.strip() for s in sentences if s.strip()]  # Loại bỏ câu rỗng

        if len(sentences) <= 15:
            processed_lines.append(line.strip())
        else:
            first_sentence = sentences[0]  # Lấy câu đầu của đoạn gốc
            temp = []

            for i, sentence in enumerate(sentences):
                temp.append(sentence)

                # Nếu đạt 15 câu hoặc hết câu, tạo dòng mới
                if len(temp) == 15 or i == len(sentences) - 1:
                    new_segment = first_sentence + ' ' + ' '.join(temp)
                    processed_lines.append(new_segment.strip())
                    temp = []  # Reset đoạn tạm thời

    return processed_lines

# Lưu kết quả vào file mới
file_path = r'C:\MINE\UIT\Kỹ thuật lập trình Trí tuệ nhân tạo\Du_lich\data\data.txt'  # Thay bằng đường dẫn tới file của bạn
output_path = r'C:\MINE\UIT\Kỹ thuật lập trình Trí tuệ nhân tạo\Du_lich\data\data-output.txt'

processed = split_sentences(file_path)

with open(output_path, 'w', encoding='utf-8') as output_file:
    for line in processed:
        output_file.write(line + '\n')

print(f"Xử lý xong. Kết quả đã lưu tại {output_path}")
