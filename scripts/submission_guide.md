# Hướng dẫn chấm bài — Phát hiện tên miền lừa đảo bằng SVM

{{STATUS}}
**Môn:** Nhập môn Học máy · **Đề tài:** Support Vector Machine · **MSSV:** {{IDS}}

Bài toán: phân loại một hostname (ví dụ `abcbank-xacminh.secure-login.xyz`) là lừa đảo hay hợp lệ, chỉ từ
chuỗi ký tự. Mô hình là SVM lề mềm kernel RBF, huấn luyện bằng **thuật toán SMO do nhóm tự cài đặt**
(`SourceCode/src/smo.py`) và được kiểm chứng trùng khớp với LIBSVM (`sklearn.svm.SVC`).

## 1. Nội dung gói

| Thư mục | Nội dung |
|---|---|
| `BaoCao/` | Báo cáo (.docx và .pdf) |
| `Slide/` | Slide thuyết trình |
| `Notebooks_HTML/` | Hai notebook **đã chạy sẵn**, mở bằng trình duyệt, không cần cài đặt |
| `SourceCode/` | Toàn bộ mã nguồn Python, notebook gốc, dữ liệu đã làm sạch, mô hình, kết quả, test |

## 2. Xem nhanh (không cần cài gì)

1. `BaoCao/` và `Slide/`.
2. `Notebooks_HTML/00_svm_walkthrough.html`: nguyên lý SVM từng bước (lề, Lagrange và bài toán đối ngẫu, KKT,
   lề mềm, kernel trick) với ví dụ tính tay: một bước SMO trên 3 điểm, ví dụ có điểm vi phạm lề, XOR giải tay
   bằng kernel $(x^\top z)^2$, ảnh hưởng của C và γ. Mọi phép tính tay được đối chiếu tự động với bộ giải.
3. `Notebooks_HTML/01_data_audit.html`: kiểm định dữ liệu, 3 phát hiện (thiên lệch định dạng URL,
   rò rỉ nhãn, nhóm nền tảng) và quá trình làm sạch.
4. `Notebooks_HTML/02_experiments.html`: dò C/γ, so sánh kernel, SMO tự cài so với LIBSVM, thời gian theo
   số mẫu, phân tích lỗi, kết luận.
5. `SourceCode/README.md`: tổng quan kỹ thuật và bảng kết quả.

## 3. Chạy thử trên Windows (khoảng 3 phút)

Yêu cầu **Python 3.12** ([tải tại python.org](https://www.python.org/downloads/release/python-31210/)).
Giải nén vào một **đường dẫn ngắn, không dấu, không khoảng trắng**, ví dụ `C:\SVM\` (Windows giới hạn đường
dẫn 260 ký tự; thư mục quá sâu như `Downloads\...\...\` sẽ làm thư viện không nạp được, `setup.bat` sẽ báo
nếu gặp trường hợp này). Sau đó trong `SourceCode/` nhấp đúp lần lượt:

| File | Việc làm | Thời gian |
|---|---|---|
| `setup.bat` | Tạo môi trường ảo `.venv` và cài thư viện (một lần) | 1–3 phút, cần mạng |
| `run_tests.bat` | Chạy 44 unit test, trong đó có test SMO tự cài trùng với LIBSVM và thỏa điều kiện KKT | ~5 giây |
| `run_demo.bat` | Mở web demo tại http://127.0.0.1:8000 (đóng cửa sổ để dừng) | tức thì |

Gợi ý thử trên demo: `https://www.wikipedia.org`, `abcbank-xacminh.secure-login.xyz`,
`http://paypal.com@verify-account.web.app/`, `outl00k.net`. Demo hiển thị giá trị quyết định $f(x)$, vùng lề
$-1 < f < 1$, 5 support vector ảnh hưởng nhất và vector 17 đặc trưng.

macOS/Linux: xem mục *Setup and reproduction* trong `SourceCode/README.md`.

## 4. Chạy lại thực nghiệm

- Mở `SourceCode/notebooks/02_experiments.ipynb` bằng VS Code hoặc Jupyter, chọn kernel `.venv`, Run All.
  Notebook lưu sẵn kết quả bản FULL (`QUICK = False`, khoảng 45 phút); đổi thành `QUICK = True` để chạy thử
  khoảng 4 phút. Mỗi lần chạy ghi kết quả vào một thư mục mới trong `outputs/experiments/`.
- Đối chiếu SMO tự cài với LIBSVM trên dữ liệu thật:
  `.venv\Scripts\python.exe -m experiments.check_smo_vs_sklearn` (~30 giây).
- Tái lập dữ liệu từ bản gốc của UCI (cần mạng): `.venv\Scripts\python.exe scripts\download_data.py`
  rồi `.venv\Scripts\python.exe -m scripts.make_clean_data`. SHA-256 của file tạo ra trùng với `cleaning_log.json`.

## 5. Vị trí nội dung theo yêu cầu đề bài

| Yêu cầu | Ở đâu |
|---|---|
| 1. Nguyên lý hoạt động | `BaoCao/`, `Notebooks_HTML/00_svm_walkthrough.html` (mục 1, 2, 4, 5, 6) |
| 2. Quy trình và ví dụ tính tay | `BaoCao/`, `Notebooks_HTML/00_svm_walkthrough.html` (mục 3: một bước SMO tính tay; mục 4: lề mềm; mục 5: XOR); kiểm chứng tự động trong `SourceCode/tests/test_smo.py` |
| 3. Bài toán ứng dụng | `BaoCao/`, `Notebooks_HTML/01_data_audit.html` |
| 4. Mã nguồn và kết quả | `SourceCode/src/smo.py` (SMO tự cài), `Notebooks_HTML/02_experiments.html`, `SourceCode/outputs/` |
