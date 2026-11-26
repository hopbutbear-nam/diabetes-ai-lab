
import json
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. ĐỌC DỮ LIỆU FAIRNESS TRÊN TẬP TEST
# ==========================================

json_path = "reports/metrics/fairness_test.json"

with open(json_path, "r", encoding="utf-8") as f:
    fair = json.load(f)

attributes = ["Age", "Income", "Sex"]
titles = ["Nhóm tuổi", "Nhóm thu nhập", "Giới tính"]

# ==========================================
# 2. KIỂM TRA VÀ TRÍCH XUẤT DỮ LIỆU
# ==========================================

plot_data = {}

for attr in attributes:

    if attr not in fair:
        raise ValueError(f"Không tìm thấy thuộc tính: {attr}")

    rows = fair[attr]["by_class"]["diabetes"]["rows"]

    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError(
            f"Dữ liệu nhóm {attr} rỗng hoặc không đúng định dạng!"
        )

    groups = []
    recall = []
    lower_error = []
    upper_error = []

    for row in rows:

        group = row["group"]
        point = float(row["recall"]["point"])
        low = float(row["recall"]["low"])
        high = float(row["recall"]["high"])

        groups.append(str(group))
        recall.append(point)

        lower_error.append(max(0, point - low))
        upper_error.append(max(0, high - point))

    plot_data[attr] = {
        "groups": groups,
        "recall": recall,
        "errors": [lower_error, upper_error]
    }

    print(f"{attr}: Đã đọc {len(groups)} nhóm")
    for g, r in zip(groups, recall):
        print(f"  {g}: Recall = {r:.2%}")

# ==========================================
# 3. VẼ BA BIỂU ĐỒ RECALL
# ==========================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(13, 4.5),
    sharey=True
)

for ax, attr, title in zip(axs, attributes, titles):

    data = plot_data[attr]

    x = range(len(data["groups"]))

    ax.bar(
        x,
        data["recall"],
        yerr=data["errors"],
        capsize=4
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        data["groups"],
        rotation=30,
        ha="right"
    )

    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel(
        "Recall lớp tiểu đường (KTC Wilson 95%)"
    )

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

# ==========================================
# 4. HOÀN THIỆN VÀ LƯU ẢNH
# ==========================================

fig.suptitle(
    "Recall theo nhóm trên tập TEST",
    fontsize=14
)

fig.tight_layout()

output_dir = "reports/figures"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "fairness_recall_test.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("\n====================================")
print("ĐÃ TẠO BIỂU ĐỒ THÀNH CÔNG!")
print("Đường dẫn:", output_path)
print("====================================")
