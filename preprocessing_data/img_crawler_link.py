import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Khởi tạo trình duyệt Chrome
service = Service(executable_path=r'C:\MINE\UIT\Kỹ thuật lập trình Trí tuệ nhân tạo\Du_lich\data\chromedriver.exe')
options = webdriver.ChromeOptions()
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
driver = webdriver.Chrome(service=service, options=options)

# Mở trang web Google Images
driver.get("https://www.google.com.vn/imghp")

# Tìm ô tìm kiếm và nhập từ khóa
search_box = driver.find_element(By.NAME, "q")
search_box.send_keys("xe máy hiệu Yamaha")
search_box.send_keys(Keys.RETURN)

# Cuộn trang để tải thêm ảnh
time.sleep(3)
for _ in range(5):
    driver.execute_script("window.scrollBy(0, 1000);")
    time.sleep(1)

# Lấy danh sách các phần tử hình ảnh
image_elements = driver.find_elements(By.CSS_SELECTOR, "img")

# Chỉ lấy 10 ảnh đầu tiên
image_elements = image_elements[:10]

# Tạo thư mục lưu trữ
output_dir = r"C:\MINE\UIT\Kỹ thuật lập trình Trí tuệ nhân tạo\Du_lich\data\downloaded_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Khởi tạo phiên làm việc với retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=3,
    status_forcelist=[500, 502, 503, 504],
    backoff_factor=1
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Tải hình ảnh
for i, img_element in enumerate(image_elements):
    img_url = img_element.get_attribute("src") or img_element.get_attribute("data-src")
    print(f"Image {i + 1} URL: {img_url}")  # Debug
    if img_url and img_url.startswith("http"):
        try:
            response = session.get(img_url, timeout=10)
            if response.status_code == 200:
                img_data = response.content
                file_path = os.path.join(output_dir, f"image_{i + 1}.jpg")
                with open(file_path, "wb") as img_file:
                    img_file.write(img_data)
                print(f"Đã tải ảnh {i + 1}: {file_path}")
            else:
                print(f"Lỗi tải ảnh {i + 1}. Mã trạng thái: {response.status_code}")
        except Exception as e:
            print(f"Lỗi tải ảnh {i + 1}: {e}")

driver.quit()

print("Đã tải xuống 10 ảnh đầu tiên từ Google Images thành công!")
