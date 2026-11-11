// Diabetes AI Lab - Script tương tác sàng lọc và giải thích nguy cơ
// Cam kết bảo mật: KHÔNG lưu trữ dữ liệu người dùng trên trình duyệt

const API = location.protocol.startsWith("http") ? location.origin : "http://localhost:8200";

let currentProfile = null;
let whatIfDebounceTimer = null;

function buildPayload() {
  const form = document.getElementById("screening-form");
  const formData = new FormData(form);

  // 21 đặc trưng BRFSS bắt buộc
  const payload = {
    HighBP: parseInt(formData.get("HighBP"), 10),
    HighChol: parseInt(formData.get("HighChol"), 10),
    CholCheck: parseInt(formData.get("CholCheck"), 10),
    BMI: parseFloat(formData.get("BMI")),
    Smoker: parseInt(formData.get("Smoker"), 10),
    Stroke: parseInt(formData.get("Stroke"), 10),
    HeartDiseaseorAttack: parseInt(formData.get("HeartDiseaseorAttack"), 10),
    PhysActivity: parseInt(formData.get("PhysActivity"), 10),
    Fruits: parseInt(formData.get("Fruits"), 10),
    Veggies: parseInt(formData.get("Veggies"), 10),
    HvyAlcoholConsump: parseInt(formData.get("HvyAlcoholConsump"), 10),
    AnyHealthcare: parseInt(formData.get("AnyHealthcare"), 10),
    NoDocbcCost: parseInt(formData.get("NoDocbcCost"), 10),
    GenHlth: parseInt(formData.get("GenHlth"), 10),
    MentHlth: parseInt(formData.get("MentHlth"), 10),
    PhysHlth: parseInt(formData.get("PhysHlth"), 10),
    DiffWalk: parseInt(formData.get("DiffWalk"), 10),
    Sex: parseInt(formData.get("Sex"), 10),
    Age: parseInt(formData.get("Age"), 10),
    Education: parseInt(formData.get("Education"), 10),
    Income: parseInt(formData.get("Income"), 10),
  };

  // Thông tin lâm sàng tùy chọn (bỏ qua nếu để trống)
  const glucoseVal = formData.get("glucose");
  if (glucoseVal !== null && glucoseVal.trim() !== "") {
    payload.glucose = parseFloat(glucoseVal);
  }

  const fastingGlucoseVal = formData.get("fasting_glucose");
  if (fastingGlucoseVal !== null && fastingGlucoseVal.trim() !== "") {
    payload.fasting_glucose = parseFloat(fastingGlucoseVal);
  }

  const symptoms = formData.getAll("symptoms");
  if (symptoms && symptoms.length > 0) {
    payload.symptoms = symptoms;
  }

  return payload;
}

function showError(message, fieldErrors = []) {
  const errorContainer = document.getElementById("error-container");
  errorContainer.style.display = "block";
  errorContainer.textContent = "";

  const title = document.createElement("strong");
  title.textContent = "Đã xảy ra lỗi: ";
  const msg = document.createElement("span");
  msg.textContent = String(message);
  errorContainer.append(title, msg);

  if (fieldErrors.length > 0) {
    const ul = document.createElement("ul");
    fieldErrors.forEach((err) => {
      const li = document.createElement("li");
      const f = document.createElement("strong");
      f.textContent = `${err.field}: `;
      li.append(f, document.createTextNode(String(err.message)));
      ul.appendChild(li);
    });
    errorContainer.appendChild(ul);
  }
  errorContainer.scrollIntoView({ behavior: "smooth" });
}

function clearError() {
  const errorContainer = document.getElementById("error-container");
  errorContainer.style.display = "none";
  errorContainer.innerHTML = "";
}

async function callPredict(payload) {
  clearError();
  const btn = document.getElementById("btn-submit");
  btn.disabled = true;
  btn.textContent = "Đang phân tích...";

  try {
    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (res.status === 422) {
      const data = await res.json();
      showError("Dữ liệu nhập vào chưa hợp lệ. Vui lòng kiểm tra lại:", data.detail || []);
      return;
    }

    if (res.status === 503) {
      showError("Dịch vụ mô hình chưa sẵn sàng (MODEL_NOT_READY). Vui lòng thử lại sau.");
      return;
    }

    if (!res.ok) {
      showError(`Máy chủ phản hồi mã lỗi ${res.status}. Không thể thực hiện đánh giá.`);
      return;
    }

    const data = await res.json();
    currentProfile = payload;
    renderPrediction(data);

    // Tiếp tục gọi explain để lấy 6 yếu tố ảnh hưởng và khởi tạo what-if
    await callExplain(payload);
  } catch (err) {
    showError("Không thể kết nối tới máy chủ dự đoán. Vui lòng kiểm tra mạng hoặc cổng 8200.");
  } finally {
    btn.disabled = false;
    btn.textContent = "Đánh giá nguy cơ";
  }
}

function renderPrediction(data) {
  const resultsSection = document.getElementById("results");
  resultsSection.style.display = "block";

  // 1. Mức nguy cơ (kèm cả chữ và màu sắc)
  const badge = document.getElementById("risk-level-badge");
  badge.className = "risk-badge";
  const level = (data.risk_level || "").toLowerCase();
  if (level === "thấp") {
    badge.classList.add("risk-low");
    badge.textContent = "Mức nguy cơ: THẤP";
  } else if (level === "trung bình") {
    badge.classList.add("risk-medium");
    badge.textContent = "Mức nguy cơ: TRUNG BÌNH";
  } else {
    badge.classList.add("risk-high");
    badge.textContent = "Mức nguy cơ: CAO";
  }

  // 2. Chỉ số nguy cơ
  const riskIndexEl = document.getElementById("risk-index-value");
  const riskPct = (data.risk_index * 100).toFixed(1);
  riskIndexEl.textContent = `${riskPct}% (${data.risk_index.toFixed(3)})`;

  // 3. Ba thanh xác suất
  const probs = data.probabilities || {};
  const p0 = (probs["không"] || 0) * 100;
  const p1 = (probs["tiền tiểu đường"] || 0) * 100;
  const p2 = (probs["tiểu đường"] || 0) * 100;

  document.getElementById("prob-0-text").textContent = `${p0.toFixed(1)}%`;
  document.getElementById("prob-0-bar").style.width = `${p0.toFixed(1)}%`;

  document.getElementById("prob-1-text").textContent = `${p1.toFixed(1)}%`;
  document.getElementById("prob-1-bar").style.width = `${p1.toFixed(1)}%`;

  document.getElementById("prob-2-text").textContent = `${p2.toFixed(1)}%`;
  document.getElementById("prob-2-bar").style.width = `${p2.toFixed(1)}%`;

  // 4. Cảnh báo an toàn
  const alertsList = document.getElementById("alerts-list");
  alertsList.innerHTML = "";
  if (data.alerts && data.alerts.length > 0) {
    data.alerts.forEach((alt) => {
      const li = document.createElement("li");
      li.textContent = `[${alt.rule_id}] ${alt.text}`;
      alertsList.appendChild(li);
    });
  } else {
    const li = document.createElement("li");
    li.style.color = "#166534";
    li.textContent = "Không có cảnh báo lâm sàng đặc biệt nào được kích hoạt.";
    alertsList.appendChild(li);
  }

  // 5. Lời khuyên
  const adviceList = document.getElementById("advice-list");
  adviceList.innerHTML = "";
  if (data.advice && data.advice.length > 0) {
    data.advice.forEach((adv) => {
      const li = document.createElement("li");
      li.textContent = adv;
      adviceList.appendChild(li);
    });
  }

  // 6. Gắn disclaimer từ phản hồi
  if (data.disclaimer) {
    document.getElementById("disclaimer-text").textContent = data.disclaimer;
  }

  resultsSection.scrollIntoView({ behavior: "smooth" });
}

async function callExplain(payload, whatIfChanges = null) {
  try {
    const body = { profile: payload };
    if (whatIfChanges !== null) {
      body.what_if_changes = whatIfChanges;
    }

    const res = await fetch(`${API}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      if (whatIfChanges !== null) {
        document.getElementById("whatif-note").textContent =
          "Không cập nhật được kịch bản what-if (mã lỗi " + res.status + "). Vui lòng thử lại.";
      }
      return;
    }

    const data = await res.json();

    // Luôn gắn disclaimer từ phản hồi nếu có
    if (data.disclaimer) {
      document.getElementById("disclaimer-text").textContent = data.disclaimer;
    }

    if (whatIfChanges === null) {
      // Hiển thị 6 yếu tố ảnh hưởng
      renderDrivers(data.drivers || []);
      // Khởi tạo các giá trị what-if theo profile ban đầu
      initWhatIfControls(payload, data);
    } else {
      // Cập nhật kết quả what-if tùy chỉnh
      renderCustomWhatIf(data.custom_what_if, data.note);
    }
  } catch (err) {
    // Không làm gián đoạn luồng chính nếu explain gặp lỗi mạng nhẹ
    console.warn("Lỗi khi gọi /explain:", err);
  }
}

function renderDrivers(drivers) {
  const driversList = document.getElementById("drivers-list");
  driversList.innerHTML = "";

  // Chọn 6 yếu tố hàng đầu
  const topDrivers = drivers.slice(0, 6);
  topDrivers.forEach((d) => {
    const div = document.createElement("div");
    div.className = "driver-item";

    const isPos = d.effect > 0;
    const sign = isPos ? "+" : "";
    const effectClass = isPos ? "effect-pos" : "effect-neg";
    const dirText = isPos ? "Tăng nguy cơ" : "Giảm nguy cơ";

    const title = document.createElement("span");
    title.className = "driver-title";
    title.textContent = d.vi || d.feature;

    const current = document.createElement("div");
    current.append("Giá trị hiện tại: ");
    const val = document.createElement("strong");
    val.textContent = String(d.value);
    current.appendChild(val);

    const effect = document.createElement("span");
    effect.className = `driver-effect ${effectClass}`;
    effect.textContent = `${sign}${(d.effect * 100).toFixed(2)}% (${dirText})`;

    div.append(title, current, effect);
    driversList.appendChild(div);
  });
}

function initWhatIfControls(profile, explainData) {
  const bmiSlider = document.getElementById("whatif-bmi");
  const bmiDisplay = document.getElementById("whatif-bmi-display");
  bmiSlider.value = profile.BMI || 25;
  bmiDisplay.textContent = parseFloat(bmiSlider.value).toFixed(1);

  // 5 yếu tố lối sống có thể thay đổi (khớp khối `what_if` trong configs/features.yaml)
  document.getElementById("whatif-Smoker").checked = profile.Smoker === 1;
  document.getElementById("whatif-PhysActivity").checked = profile.PhysActivity === 1;
  document.getElementById("whatif-Fruits").checked = profile.Fruits === 1;
  document.getElementById("whatif-Veggies").checked = profile.Veggies === 1;
  document.getElementById("whatif-HvyAlcoholConsump").checked = profile.HvyAlcoholConsump === 1;

  // Cập nhật bảng ban đầu
  const beforeEl = document.getElementById("whatif-before");
  const afterEl = document.getElementById("whatif-after");
  const deltaEl = document.getElementById("whatif-delta");

  if (explainData.custom_what_if) {
    renderCustomWhatIf(explainData.custom_what_if, explainData.note);
  } else {
    const riskVal = document.getElementById("risk-index-value").textContent;
    beforeEl.textContent = riskVal;
    afterEl.textContent = riskVal;
    deltaEl.textContent = "0.0%";
  }
}

function renderCustomWhatIf(customWhatIf, note) {
  if (!customWhatIf) return;

  const beforeVal = (customWhatIf.before * 100).toFixed(1);
  const afterVal = (customWhatIf.after * 100).toFixed(1);
  const deltaVal = ((customWhatIf.after - customWhatIf.before) * 100).toFixed(1);

  document.getElementById("whatif-before").textContent = `${beforeVal}%`;
  document.getElementById("whatif-after").textContent = `${afterVal}%`;

  const deltaEl = document.getElementById("whatif-delta");
  const deltaNum = customWhatIf.after - customWhatIf.before;
  const sign = deltaNum > 0 ? "+" : "";
  deltaEl.textContent = `${sign}${deltaVal}%`;
  deltaEl.style.color = deltaNum > 0 ? "#dc2626" : (deltaNum < 0 ? "#16a34a" : "#1f2937");

  if (customWhatIf.note || note) {
    document.getElementById("whatif-note").textContent = customWhatIf.note || note;
  }
}

function triggerWhatIfUpdate() {
  if (!currentProfile) return;

  clearTimeout(whatIfDebounceTimer);
  whatIfDebounceTimer = setTimeout(() => {
    const whatIfChanges = {
      BMI: parseFloat(document.getElementById("whatif-bmi").value),
      Smoker: document.getElementById("whatif-Smoker").checked ? 1 : 0,
      PhysActivity: document.getElementById("whatif-PhysActivity").checked ? 1 : 0,
      Fruits: document.getElementById("whatif-Fruits").checked ? 1 : 0,
      Veggies: document.getElementById("whatif-Veggies").checked ? 1 : 0,
      HvyAlcoholConsump: document.getElementById("whatif-HvyAlcoholConsump").checked ? 1 : 0,
    };
    callExplain(currentProfile, whatIfChanges);
  }, 200);
}

async function callConsult() {
  const out = document.getElementById("consult-output");
  out.textContent = "";
  if (!currentProfile) {
    out.textContent = "Hãy bấm “Đánh giá nguy cơ” trước.";
    return;
  }
  try {
    const res = await fetch(`${API}/consult`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentProfile),
    });
    if (!res.ok) {
      out.textContent = "Không lấy được tư vấn (mã lỗi " + res.status + ").";
      return;
    }
    const data = await res.json();
    const head = document.createElement("p");
    head.textContent = `Mức nguy cơ sau rà soát an toàn: ${data.final_level}`;
    out.appendChild(head);

    const ul = document.createElement("ul");
    (data.citations || []).forEach((c) => {
      const li = document.createElement("li");
      const b = document.createElement("strong");
      b.textContent = c.title || "Tài liệu";
      li.append(b, document.createTextNode(` — ${c.source || ""}`));
      ul.appendChild(li);
    });
    out.appendChild(ul);

    const note = document.createElement("p");
    note.textContent = data.disclaimer || "";
    out.appendChild(note);
  } catch (err) {
    out.textContent = "Không kết nối được máy chủ.";
  }
}

// Lắng nghe sự kiện
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("screening-form");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const payload = buildPayload();
    callPredict(payload);
  });

  const consultBtn = document.getElementById("consult-btn");
  if (consultBtn) {
    consultBtn.addEventListener("click", callConsult);
  }

  const bmiSlider = document.getElementById("whatif-bmi");
  bmiSlider.addEventListener("input", (e) => {
    document.getElementById("whatif-bmi-display").textContent = parseFloat(e.target.value).toFixed(1);
    triggerWhatIfUpdate();
  });

  ["whatif-HighBP", "whatif-HighChol", "whatif-Smoker", "whatif-PhysActivity", "whatif-Fruits"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener("change", triggerWhatIfUpdate);
    }
  });
});
