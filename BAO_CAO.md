# Báo cáo lab MLOps Day 21

## Kết quả thực nghiệm

Ba lần chạy trên `train_phase1.csv` (2.998 mẫu), cùng `random_state=42`:

| `n_estimators` | `max_depth` | `min_samples_split` | Accuracy | F1 weighted |
|---:|---:|---:|---:|---:|
| 50 | 3 | 2 | 0.5580 | 0.5185 |
| 100 | 5 | 2 | 0.5640 | 0.5534 |
| 200 | 10 | 2 | 0.6480 | 0.6464 |

Sau khi bổ sung `train_phase2.csv`, tập huấn luyện có 5.996 mẫu. Cấu hình triển khai là `n_estimators=300`, `max_depth=20`, `min_samples_split=2`; kết quả trên tập eval giữ riêng là **accuracy 0.7560** và **F1 weighted 0.7552**. Cấu hình này vượt eval gate 0.70 nhưng tạo artifact nhỏ hơn đáng kể so với phương án 1.000 cây sâu không giới hạn (accuracy 0.7600, model khoảng 215 MB), phù hợp hơn để upload và khởi động trên VM.

## Khó khăn và cách xử lý

- Với riêng dữ liệu phase 1, các cấu hình Random Forest đã thử không đạt ngưỡng 0.70. Eval gate chặn deploy đúng thiết kế. Sau bước bổ sung dữ liệu, độ chính xác tăng lên 0.7600 và pipeline đủ điều kiện deploy.
- `mlflow==2.13.0` cần API `pkg_resources`, trong khi Setuptools mới đã loại bỏ API này. Đã thêm `setuptools<81` vào dependency để local và GitHub Actions dùng cùng môi trường tương thích.
- Workflow không lưu tên bucket trực tiếp trong Git. DVC remote được tạo lúc chạy từ secret `CLOUD_BUCKET`; credential được lấy từ `CLOUD_CREDENTIALS`.

## Trạng thái xác minh

- Unit test huấn luyện và API: 6/6 pass.
- Workflow YAML hợp lệ, gồm đủ bốn job nối tiếp: Unit Test → Train → Eval → Deploy.
- Model cuối vượt ngưỡng chất lượng: 0.7560 ≥ 0.70.
- Repo công khai: <https://github.com/LamLam5406/K3-Track2-Day21>.
- GitHub Actions run thành công: <https://github.com/LamLam5406/K3-Track2-Day21/actions/runs/32450624925>.
- Kết quả CI: Unit Test, Train, Eval và Deploy đều `success`; hai artifact `metrics` và `trained-model` đã được tạo.
- Bước 3 được kích hoạt bởi commit dữ liệu `4806572`; [Actions run tương ứng](https://github.com/LamLam5406/K3-Track2-Day21/actions/runs/32451641816) cũng có đủ bốn job `success`.
- Ảnh chụp và lệnh tái hiện được tổng hợp trong [EVIDENCE.md](EVIDENCE.md).

## Phương án khi chưa được cấp tài khoản cloud

Do tài khoản cloud chưa được phê duyệt, repo có thêm chế độ fallback minh bạch. Dữ liệu được lấy từ một DVC local remote đi kèm repo; model được truyền giữa các job bằng GitHub Actions artifact. Job Deploy khởi động FastAPI ngay trên runner và gọi thật cả `/health` lẫn `/predict`. Khi các secret cloud được bổ sung, cùng workflow sẽ tự động chuyển sang GCS và VM mà không cần sửa code.
