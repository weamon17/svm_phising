# Phát hiện tên miền lừa đảo bằng Support Vector Machine

Phân loại một **hostname** (ví dụ `abcbank-xacminh.secure-login.xyz`) là **lừa đảo (phishing)** hay
**hợp lệ** chỉ từ chuỗi ký tự, không cần mở trang web. Mô hình là **SVM lề mềm với kernel RBF**,
huấn luyện bằng **thuật toán SMO do nhóm tự cài đặt** (`src/smo.py`) và kiểm chứng trùng khớp với
LIBSVM (`sklearn.svm.SVC`).

Tiểu luận giữa kỳ môn **Nhập môn Học máy**, đề tài *Support Vector Machine*.

**Điểm chính**

- **Dữ liệu được kiểm định trước khi dùng.** Bộ PhiUSIIL (235.795 URL) chứa thiên lệch định dạng làm một luật
  if–else đạt 99,56% accuracy, cùng một cột rò rỉ nhãn (AUC 0,996). Bài toán được thiết kế lại để tránh cả hai.
- **SMO viết từ đầu**: bài toán dual, chọn cặp biến theo quy tắc bậc hai (Fan–Chen–Lin), cập nhật giải tích
  (Platt). Kết quả khớp LIBSVM về hàm mục tiêu dual, support vector và dự đoán.
- **Đánh giá chặt chẽ**: chia train/test và cross-validation **theo tên miền đăng ký**, nên mô hình được chấm
  trên các nền tảng chưa từng thấy. Chuẩn hoá đặt trong pipeline, tập test chỉ dùng một lần.
- **Mô hình tốt nhất**: RBF SVM (C = 1, γ = 0,3), **F1 = 0,750**, precision 0,925, ROC-AUC 0,835 trên
  45.087 hostname test.
- **Web demo offline** giải thích từng dự đoán theo đúng công thức $f(x) = \sum_i \alpha_i y_i K(x_i, x) + b$.

## Mục lục

1. [Bài toán](#1-bài-toán)
2. [Vì sao chọn bài toán này cho SVM](#2-vì-sao-chọn-bài-toán-này-cho-svm)
3. [Dataset](#3-dataset)
4. [Phương pháp](#4-phương-pháp)
5. [Kết quả](#5-kết-quả)
6. [Mô hình tốt nhất](#6-mô-hình-tốt-nhất)
7. [Web demo](#7-web-demo)
8. [Cài đặt và tái lập](#8-cài-đặt-và-tái-lập)
9. [Cấu trúc repo](#9-cấu-trúc-repo)
10. [Hạn chế và hướng phát triển](#10-hạn-chế-và-hướng-phát-triển)
11. [Tài liệu tham khảo](#11-tài-liệu-tham-khảo)

---

## 1. Bài toán

| | |
|---|---|
| **Đầu vào** | Một hostname, ví dụ `metamask-wallet-restore.firebaseapp.com` |
| **Đầu ra** | Nhãn $y \in \{-1, +1\}$ (hợp lệ / lừa đảo) và giá trị quyết định $f(x)$ |
| **Biểu diễn** | Vector 17 đặc trưng số trích từ chuỗi hostname, $x \in \mathbb{R}^{17}$ |
| **Lớp dương** | Lừa đảo (lớp cần phát hiện) |
| **Chỉ số chính** | F1 của lớp lừa đảo, kèm precision, recall, ROC-AUC, PR-AUC |

Mô hình chỉ đọc chuỗi ký tự nên có thể chạy **trước khi người dùng bấm vào đường link**: không tải trang,
không tra cứu mạng.

## 2. Vì sao chọn bài toán này cho SVM

**Về ứng dụng.** Lừa đảo qua đường link (giả ngân hàng, ví tiền mã hoá, dịch vụ email) là hình thức tấn công
phổ biến. Danh sách đen chỉ chặn được tên miền đã bị báo cáo, còn mô hình học máy có thể nhận ra tên miền
**mới xuất hiện** dựa trên hình dạng của nó.

**Về thuật toán.** Bài toán khớp với thế mạnh và cho phép kiểm chứng đúng các khái niệm cốt lõi của SVM:

| Đặc điểm bài toán | Liên hệ với SVM |
|---|---|
| Phân loại **nhị phân** | Đúng dạng gốc của SVM; SMO tự cài không cần bọc OvR/OvO |
| Đặc trưng số, ít chiều (17), đuôi dài, thang đo khác nhau | Thấy rõ vai trò của chuẩn hoá đối với khoảng cách trong kernel |
| Hai lớp **chồng lấn**, không tách được tuyến tính | Bắt buộc dùng **soft margin**; tham số C có tác dụng quan sát được |
| Có tương tác phi tuyến (dài + nhiều chữ số + có subdomain) | So sánh có ý nghĩa giữa kernel linear, poly và RBF |
| 219 nghìn mẫu | Đo được thực nghiệm chi phí $O(n^2)$ của kernel SVM và so với LinearSVC |
| Ca giáp ranh có thật (tên miền hợp lệ trông "lạ") | Support vector diễn giải được: chính là các hostname khó phân biệt nhất |

## 3. Dataset

### Nguồn và lý do chọn

**PhiUSIIL Phishing URL Dataset** (UCI Machine Learning Repository, 2024, giấy phép CC BY 4.0):
235.795 URL, 56 cột, nhãn gốc `1` = hợp lệ, `0` = lừa đảo.

- **Mới và lớn**: công bố năm 2024 (Prasad & Chandra, *Computers & Security*), phản ánh các kiểu lừa đảo hiện
  nay như IPFS, Firebase, Replit.
- **Có cột URL gốc**: nhóm tự trích đặc trưng thay vì dùng đặc trưng tính sẵn của tác giả.
- **Giấy phép mở**, có trích dẫn học thuật rõ ràng.

### Kiểm định: 3 vấn đề phát hiện được ([notebook 01](notebooks/01_data_audit.ipynb))

**1. Thiên lệch định dạng URL.** 100% URL hợp lệ là trang chủ dạng `https://www.<tên miền>`, không có đường dẫn.
Chỉ một luật if–else đã đạt **99,56% accuracy**. Mọi đặc trưng lấy từ scheme, `www.` hay đường dẫn đều chỉ
học lại cách thu thập dữ liệu.

| | Hợp lệ | Lừa đảo |
|---|---|---|
| Dùng `https` | 100% | 48,7% |
| Bắt đầu bằng `www.` | 100% | 41,4% |
| Có đường dẫn / tham số | 0% | 28,4% |

**2. Rò rỉ nhãn.** `URLSimilarityIndex` một mình đạt **AUC 0,996**, vì nó đo độ giống với chính danh sách URL
hợp lệ dùng để tạo nhãn. Ngoài ra 28 cột nội dung HTML đòi hỏi tải trang, nằm ngoài phạm vi bài toán.

**3. Nhóm nền tảng.** Hàng nghìn hostname lừa đảo dùng chung một tên miền đăng ký (`web.app`, `firebaseapp.com`,
`repl.co`). Nếu chia train/test ngẫu nhiên, cùng một nền tảng sẽ nằm ở cả hai phía và kết quả bị thổi phồng.

### Quyết định và dữ liệu cuối cùng

Chỉ giữ **hostname**, là thông tin mà cả hai lớp đều thực sự có. Bỏ toàn bộ 54 cột tính sẵn. Loại URL trùng,
gộp các URL cùng hostname, bỏ hostname mang cả hai nhãn.

| | |
|---|---|
| File | [`data/processed/hosts.csv`](data/processed/hosts.csv) (11 MB, có trong repo) |
| Kích thước | **219.036 hostname**: 84.244 lừa đảo (38,5%), 134.792 hợp lệ |
| Nhóm | 175.452 tên miền đăng ký |
| Nhật ký | [`cleaning_log.json`](data/processed/cleaning_log.json): từng bước loại bỏ và SHA-256 |

| Cột | Ý nghĩa |
|---|---|
| `host` | Hostname đã bỏ `www.`, đầu vào duy nhất của mô hình |
| `registered_domain` | Tên miền đăng ký (eTLD+1), dùng làm nhóm khi chia dữ liệu |
| `subdomain`, `suffix` | Tách theo Public Suffix List |
| `is_ip` | 1 nếu hostname là địa chỉ IP |
| `n_urls` | Số URL gốc gộp vào hostname (chỉ để phân tích) |
| `phishing` | **1 = lừa đảo**, 0 = hợp lệ |

## 4. Phương pháp

### Đặc trưng ([`src/features.py`](src/features.py))

| Nhóm | Đặc trưng |
|---|---|
| Độ dài, cấu trúc | `host_len`, `n_labels`, `max_label_len`, `domain_label_len` |
| Ký tự | `n_hyphens`, `n_digits`, `digit_ratio`, `vowel_ratio`, `max_consonant_run` |
| Độ ngẫu nhiên | `entropy` (Shannon, bit/ký tự) |
| Dạng đặc biệt | `is_ip`, `has_punycode` |
| Từ khoá | `n_sensitive_words` (login, verify, secure…), `brand_not_owner` (ví dụ `paypal-secure.com`) |
| Subdomain | `has_subdomain`, `subdomain_labels`, `subdomain_len` |

Danh sách từ khoá được chọn trước, không dò trên dữ liệu. Không có đặc trưng đơn lẻ nào vượt AUC 0,77, nên
bài toán không tầm thường.

### Chia dữ liệu và chọn mô hình

```text
hosts.csv (219.036)
├── TRAIN 79,4% (173.949)  ── mẫu con phân tầng 8.000 cho kernel SVM
│                              └── 5-fold StratifiedGroupKFold (nhóm = tên miền đăng ký)
│                                  Pipeline(StandardScaler → SVC), chọn C, γ theo F1
└── TEST 20,6% (45.087)    ── tên miền chưa từng thấy, chỉ dùng một lần để chấm điểm cuối
```

Khi hai cấu hình chênh nhau ít hơn một sai số chuẩn của CV, chọn cấu hình đơn giản hơn
(*one-standard-error rule*).

### SVM và SMO tự cài ([`src/smo.py`](src/smo.py))

Bài toán dual của SVM lề mềm:

$$\min_{\alpha}\ \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i,x_j) - \sum_i \alpha_i
\quad\text{s.t.}\quad 0 \le \alpha_i \le C,\ \ \sum_i \alpha_i y_i = 0$$

Mỗi vòng lặp:
1. Chọn cặp $(i, j)$ vi phạm điều kiện KKT nhiều nhất theo quy tắc bậc hai của Fan, Chen & Lin (2005).
2. Giải bài toán con hai biến dạng giải tích (Platt, 1998): $\eta = K_{ii} + K_{jj} - 2K_{ij}$,
   $\alpha_j \leftarrow \text{clip}\big(\alpha_j + y_j (E_i - E_j)/\eta,\ L,\ H\big)$, cập nhật $\alpha_i$ và bảng sai số $E$.
3. Dừng khi độ vi phạm KKT $m - M < 10^{-3}$.

Hỗ trợ kernel linear, polynomial, RBF. Tham số `record_trace=True` in từng bước để đối chiếu với ví dụ tính tay.

**Kiểm chứng** ([`tests/test_smo.py`](tests/test_smo.py), [`experiments/check_smo_vs_sklearn.py`](experiments/check_smo_vs_sklearn.py)):

| Kiểm tra | Kết quả |
|---|---|
| Ví dụ 3 điểm $(1,1)^-, (2,2)^+, (3,3)^+$ | Hội tụ sau **đúng 1 bước**: $\alpha = (1, 1, 0)$, $w = (1, 1)$, $b = -3$ |
| Điều kiện KKT trên nghiệm | $\alpha = 0$ ngoài lề, $0 < \alpha < C$ trên lề, $\alpha = C$ vi phạm lề, $\sum \alpha y = 0$ |
| So với `SVC` trên dữ liệu giả lập (3 kernel) | Support vector trùng, $\lvert\Delta f\rvert < 10^{-3}$, dự đoán trùng 100% |
| So với `SVC` trên dữ liệu thật | Hàm mục tiêu dual bằng nhau (−720,9346), dự đoán trùng 99,98–100% |

## 5. Kết quả

Toàn bộ thực nghiệm và biểu đồ nằm trong [notebook 02](notebooks/02_experiments.ipynb), chế độ FULL.
Số liệu dưới đây có tính tất định: chạy lại cho kết quả trùng hoàn toàn. Mỗi lần chạy lưu CSV và hình vào
`outputs/experiments/<thời điểm>/` trên máy (thư mục này không đưa lên repo).

### So sánh mô hình (test: 45.087 hostname, tên miền chưa từng thấy)

| Mô hình | Tham số | Precision | Recall | **F1** | ROC-AUC | Support vector | Thời gian fit |
|---|---|---|---|---|---|---|---|
| **SVC RBF** | C = 1, γ = 0,3 | 0,925 | 0,630 | **0,750** | 0,835 | 3.612 | 2,3 s |
| **SMO tự cài (RBF)** | C = 1, γ = 0,3 | 0,925 | 0,630 | **0,750** | 0,834 | 3.567 | 19,6 s |
| SVC linear | C = 10 | 0,909 | 0,624 | 0,740 | 0,836 | 3.142 | 9,2 s |
| SVC poly bậc 2 | C = 10 | 0,949 | 0,599 | 0,734 | 0,822 | 3.375 | 7,6 s |
| SVC poly bậc 3 | C = 1 | 0,946 | 0,597 | 0,732 | 0,834 | 2.893 | 12,5 s |
| Logistic Regression | — | 0,919 | 0,629 | 0,747 | 0,843 | — | 0,1 s |
| LinearSVC (toàn bộ 173.949) | C = 1 | 0,921 | 0,626 | 0,745 | 0,842 | — | 1,0 s |
| Đoán lớp đa số | — | 0 | 0 | 0 | 0,500 | — | — |

### Dò tham số C × γ (kernel RBF, F1 trung bình 5-fold CV)

| C \ γ | 0,003 | 0,03 | 0,3 | 3 |
|---|---|---|---|---|
| 0,1 | 0,665 | 0,683 | 0,693 | 0,147 |
| 1 | 0,677 | 0,688 | **0,701** | 0,689 |
| 100 | 0,688 | 0,694 | 0,684 | 0,657 |

Vùng tối ưu bằng phẳng (độ lệch chuẩn giữa các fold ±0,10 lớn hơn chênh lệch giữa các ô). Góc C nhỏ, γ lớn
bị **underfitting**; vùng C lớn, γ lớn giảm dần do **overfitting**. Điểm tốt nhất nằm giữa lưới 7×7, không chạm biên.

### Thời gian theo số mẫu

| n | SVC RBF (LIBSVM) | SMO tự cài | LinearSVC | F1 của SVC RBF | Số support vector |
|---|---|---|---|---|---|
| 1.000 | 0,04 s | 0,33 s | 0,005 s | 0,745 | 578 |
| 8.000 | 2,5 s | 17,2 s | 0,02 s | 0,750 | 3.612 |
| 32.000 | 47,9 s | — | 0,12 s | 0,749 | 13.169 |
| 173.949 | — | — | 1,0 s | 0,745 (LinearSVC) | — |

Kernel SVM tăng khoảng **O(n²)**, số support vector tăng gần tuyến tính theo n. F1 bão hoà từ khoảng
4.000 mẫu, nên thêm dữ liệu không giúp mô hình.

### Nhận xét

- **SMO tự cài đạt chất lượng của LIBSVM** (F1 0,750 cả hai), chậm hơn khoảng 8 lần vì viết bằng Python thuần.
- **Các kernel và Logistic Regression gần như ngang nhau.** Với 17 chiều và n ≫ p, biên quyết định gần tuyến
  tính; giới hạn nằm ở đặc trưng chứ không ở mô hình.
- **Precision cao, recall thấp**: mô hình hiếm khi báo nhầm (FPR ≈ 3,4%) nhưng bỏ sót 37% hostname lừa đảo.
- **Phụ thuộc subdomain**: chỉ dùng tên miền đăng ký thì F1 của RBF còn 0,327 (ROC-AUC 0,511).
- **log1p/binarize đặc trưng đếm không cải thiện** (CV F1 0,704 so với 0,701, nhỏ hơn sai số chuẩn 0,045),
  nên giữ `StandardScaler`.

## 6. Mô hình tốt nhất

| | |
|---|---|
| Mô hình | SVM lề mềm, kernel RBF $K(x,z) = \exp(-0{,}3\,\lVert x - z\rVert^2)$, C = 1 |
| Bộ giải | **SMO tự cài** (`src/smo.py`), 32.305 vòng lặp, khoảng 16 giây |
| Dữ liệu huấn luyện | 8.000 hostname (mẫu con phân tầng của tập train), chuẩn hoá z-score |
| Số support vector | 3.567 |
| Kết quả test | F1 **0,750** · precision 0,925 · recall 0,630 · ROC-AUC 0,834 |
| File | [`models/svm_rbf.npz`](models/svm_rbf.npz) (mean, scale, support vector, $\alpha_i y_i$) + [`models/svm_rbf.json`](models/svm_rbf.json) (b, γ, C, tên đặc trưng, chỉ số) |

Mô hình lưu bằng `npz` + `json`, **không dùng pickle**, nên nạp an toàn và đọc được bằng mắt. Tạo lại bằng
`python -m scripts.train_demo_model`.

## 7. Web demo

```powershell
.\.venv\Scripts\python.exe demo/server.py        # mở http://127.0.0.1:8000
```

- Nhập URL hoặc hostname; demo tự tách hostname, bỏ userinfo/cổng, chuyển tên miền Unicode sang punycode.
- **Kết luận theo lề**: $f \ge 1$ lừa đảo, $f \le -1$ hợp lệ, $-1 < f < 1$ không chắc chắn.
- **Giải thích**: $f(x)$ được tách thành lực kéo của support vector phía lừa đảo, phía hợp lệ và bias $b$;
  hiển thị 5 support vector ảnh hưởng nhất và vector 17 đặc trưng.
- **Cảnh báo out-of-distribution** khi hostname ở xa mọi support vector ($K \approx 0$, nên $f \approx b$).
- Chỉ dùng thư viện chuẩn Python, chạy offline, chỉ nhận kết nối từ `127.0.0.1`.

## 8. Cài đặt và tái lập

Yêu cầu **Python 3.12** (3.13 không dùng được vì `numpy==1.26.4`). Chạy mọi lệnh từ thư mục gốc repo.

```powershell
git clone https://github.com/weamon17/svm_phising.git
cd svm_phising
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q            # 44 passed
```

Dữ liệu đã làm sạch và mô hình có sẵn trong repo, nên notebook 02, demo và test chạy được ngay.
Để tái lập từ dữ liệu gốc (cần mạng, khoảng 30 giây):

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py    # tải CSV gốc từ UCI, kiểm tra SHA-256
.\.venv\Scripts\python.exe -m scripts.make_clean_data  # tạo lại hosts.csv, SHA-256 trùng với cleaning_log.json
```

| Việc | Lệnh | Thời gian |
|---|---|---|
| Notebook dữ liệu (cần dữ liệu gốc) | mở `notebooks/01_data_audit.ipynb` → Run All | ~20 giây |
| Notebook thực nghiệm | mở `notebooks/02_experiments.ipynb` → Run All (`QUICK = True` chạy thử ~4 phút, `False` ~45 phút) | |
| Đối chiếu SMO với LIBSVM | `python -m experiments.check_smo_vs_sklearn` | ~30 giây |
| So sánh tiền xử lý | `python -m experiments.compare_preprocessing` | ~8 phút |
| Huấn luyện lại mô hình | `python -m scripts.train_demo_model` | ~25 giây |
| Chạy notebook không cần mở Jupyter | `python scripts/execute_notebook.py notebooks/02_experiments.ipynb` | |

**macOS/Linux**: thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python`.
**VS Code**: chọn kernel `.venv (Python 3.12)` trước khi Run All, và bấm Ctrl+S sau khi chạy để lưu output.
Hai notebook được sinh từ `scripts/build_*_notebook.py`; chạy các script này sẽ **ghi đè** notebook và xoá output.

## 9. Cấu trúc repo

```text
├── src/                      Mã nguồn dùng chung
│   ├── data.py               Tách hostname, tên miền đăng ký, làm sạch
│   ├── features.py           17 đặc trưng từ hostname
│   ├── smo.py                SVM + SMO tự cài (linear / poly / RBF)
│   ├── splits.py             Chia theo tên miền đăng ký, lấy mẫu phân tầng
│   ├── preprocessing.py      log1p/binarize, quy tắc một sai số chuẩn
│   └── demo.py               Dự đoán và giải thích cho demo
├── notebooks/
│   ├── 01_data_audit.ipynb   Kiểm định và làm sạch dữ liệu
│   └── 02_experiments.ipynb  Thực nghiệm (đã chạy bản FULL)
├── data/
│   ├── raw/                  Thông tin nguồn + SHA-256 (CSV gốc tải bằng script)
│   └── processed/            hosts.csv + cleaning_log.json
├── models/                   Mô hình tốt nhất (npz + json)
├── experiments/              Thực nghiệm chạy bằng dòng lệnh
├── scripts/                  Tải dữ liệu, làm sạch, huấn luyện, chạy/sinh notebook
├── demo/                     Web demo (server.py + index.html)
└── tests/                    44 unit test
```

## 10. Hạn chế và hướng phát triển

- **Thiên lệch thu thập**: danh sách hợp lệ gần như không có subdomain (2,9% so với 55,2%), nên mô hình phụ
  thuộc mạnh vào subdomain. Cần dữ liệu hợp lệ đa dạng hơn (subdomain, đường dẫn thật).
- **Điểm mù của RBF**: 12,8% hostname test ở xa mọi support vector nên $f(x) \approx b > 0$ và bị xếp là lừa
  đảo. 99,6% trong số đó đúng là lừa đảo, nhưng đó là tính chất của dữ liệu chứ không phải bằng chứng mô hình
  học được.
- **Từ khoá so khớp chuỗi con**: `office` khớp `2010officefurniture.com`, `auth` khớp `acauthorities.org`.
  Nên tách token theo `-` và `.` trước khi so khớp.
- **Recall 0,63** với ngưỡng mặc định $f(x) > 0$. Có thể chọn ngưỡng trên dự đoán out-of-fold, hoặc thêm đặc
  trưng ngoài chuỗi ký tự (tuổi tên miền, WHOIS, chứng chỉ TLS).

## 11. Tài liệu tham khảo

- Prasad, A., & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered phishing URL detection
  framework based on similarity index and incremental learning. *Computers & Security*, 136, 103545.
  https://doi.org/10.1016/j.cose.2023.103545
- UCI Machine Learning Repository: [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset),
  giấy phép CC BY 4.0. File `data/processed/hosts.csv` là dữ liệu dẫn xuất từ bộ này.
- Platt, J. (1998). *Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines*. MSR-TR-98-14.
- Fan, R.-E., Chen, P.-H., & Lin, C.-J. (2005). Working set selection using second order information for training
  support vector machines. *JMLR*, 6, 1889–1918.
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), §7.10.
