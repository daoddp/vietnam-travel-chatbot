import requests
from bs4 import BeautifulSoup

def crawl_data(url):
    try:
        # Gửi yêu cầu GET đến trang web
        response = requests.get(url)
        response.raise_for_status()  # Kiểm tra xem yêu cầu có thành công không

        # Phân tích HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Tìm tất cả các thẻ <p> trong class "section-inner inset-column"
        paragraphs = soup.select('.section-inner.inset-column p')

        # Mở file để ghi dữ liệu
        with open(r'C:\MINE\UIT\Kỹ thuật lập trình Trí tuệ nhân tạo\Du_lich\data\sample.txt', 'a', encoding='utf-8') as file:
            for paragraph in paragraphs:
                file.write(paragraph.get_text() + '\n') 
        print("Dữ liệu đã được lưu vào file data.txt.")
    
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

# Ví dụ sử dụng
url = 'https://vnexpress.net/cam-nang-du-lich-yen-bai-4701574.html' 
crawl_data(url)
#bắc giang, hưng yên, hậu giang,  ninh chữ, thái bình