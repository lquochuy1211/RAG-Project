# Dockerfile - OPTIMIZED FOR CACHING

# 1. Base Image - Sử dụng phiên bản Python cụ thể
FROM python:3.11-slim-bookworm

# 2. Thiết lập biến môi trường
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. Thiết lập thư mục làm việc
WORKDIR /app

# 4. Cài đặt dependencies
# Chỉ copy requirements.txt trước để tận dụng Docker layer caching.
# Lớp này chỉ build lại khi requirements.txt thay đổi.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ source code vào sau khi đã cài dependencies.
# Lớp này sẽ build lại mỗi khi code thay đổi, nhưng không cần cài lại package.
COPY . .

# 6. Expose port mà uvicorn sẽ chạy
EXPOSE 8000

# 7. Lệnh mặc định khi container chạy (dành cho development)
# --reload sẽ tự động restart khi code thay đổi nhờ volume mapping.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
