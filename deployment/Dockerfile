FROM python:3.10-slim

WORKDIR /app

# Salin file requirements
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Salin semua kode aplikasi
COPY . .

# Sesuaikan port jika perlu
EXPOSE 8000

# Jalankan API dengan uvicorn dan bind ke 0.0.0.0
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]