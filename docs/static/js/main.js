/* DMMV project page — page-specific strings and result tables.
   Numbers are transcribed from the paper's tables; nothing is computed here
   beyond best/second marking. */
(function () {
  "use strict";

  const { $, ZH, tr, el, registerRender, rankMarks, fmt, bindChoices } = MC;

  /* ------------------------------------------------- runtime strings (zh) */
  Object.assign(ZH, {
    "Dataset": "数据集",
    "Horizon": "预测步长",
    "Avg.": "平均",
    "# Wins": "最优次数",
    "Method": "方法",
    "Improvement": "变化",
    "Segment length": "分段长度",
    "Copy": "复制",
    "Copied": "已复制",
    "(a) f<sub>num</sub> → Transformer": "(a) f<sub>num</sub> → Transformer",
    "(b) f<sub>vis</sub> → SimMIM": "(b) f<sub>vis</sub> → SimMIM",
    "(c) Gate → Sum": "(c) 门控 → 求和",
    "(d) BCMask → No mask": "(d) BCMask → 不掩码",
    "(e) BCMask → Random mask": "(e) BCMask → 随机掩码",
    "(f) Freeze f<sub>vis</sub>": "(f) 冻结 f<sub>vis</sub>",
    "(g) W/o decomposition": "(g) 去掉分解"
  });

  /* ================================================== main results table */
  /* Ten representative methods, grouped by the view they read the series
     through. `sep` marks the first column of a view group. */
  const MODELS = [
    { name: "DMMV-A",    view: "Multi-Modal", ours: true },
    { name: "Time-VLM",  view: "Multi-Modal" },
    { name: "VisionTS",  view: "Visual",   sep: true },
    { name: "GPT4TS",    view: "Language", sep: true },
    { name: "Time-LLM",  view: "Language" },
    { name: "PatchTST",  view: "Numerical", sep: true },
    { name: "CycleNet",  view: "Numerical" },
    { name: "TimesNet",  view: "Numerical" },
    { name: "DLinear",   view: "Numerical" },
    { name: "FEDformer", view: "Numerical" }
  ];

  /* [horizon, MSE ×10, MAE ×10] in MODELS order. */
  const MAIN = {
    ETTh1: [
      ["96",   [0.354, 0.361, 0.355, 0.370, 0.376, 0.370, 0.374, 0.384, 0.375, 0.376],
               [0.389, 0.386, 0.386, 0.389, 0.402, 0.399, 0.396, 0.402, 0.399, 0.419]],
      ["192",  [0.393, 0.397, 0.395, 0.412, 0.407, 0.413, 0.406, 0.436, 0.405, 0.420],
               [0.405, 0.415, 0.407, 0.413, 0.421, 0.421, 0.415, 0.429, 0.416, 0.448]],
      ["336",  [0.387, 0.420, 0.419, 0.448, 0.430, 0.422, 0.431, 0.491, 0.439, 0.459],
               [0.413, 0.421, 0.421, 0.431, 0.438, 0.436, 0.430, 0.469, 0.416, 0.465]],
      ["720",  [0.445, 0.441, 0.458, 0.441, 0.457, 0.447, 0.450, 0.521, 0.472, 0.506],
               [0.450, 0.458, 0.460, 0.449, 0.468, 0.466, 0.464, 0.500, 0.490, 0.507]],
      ["Avg.", [0.395, 0.405, 0.407, 0.418, 0.418, 0.413, 0.415, 0.458, 0.423, 0.440],
               [0.414, 0.420, 0.419, 0.421, 0.432, 0.431, 0.426, 0.450, 0.430, 0.460]]
    ],
    ETTh2: [
      ["96",   [0.294, 0.267, 0.288, 0.280, 0.286, 0.274, 0.279, 0.340, 0.289, 0.358],
               [0.349, 0.335, 0.334, 0.335, 0.346, 0.336, 0.341, 0.374, 0.353, 0.397]],
      ["192",  [0.339, 0.326, 0.349, 0.348, 0.361, 0.339, 0.342, 0.402, 0.383, 0.429],
               [0.395, 0.373, 0.380, 0.380, 0.391, 0.379, 0.385, 0.414, 0.418, 0.439]],
      ["336",  [0.322, 0.357, 0.364, 0.380, 0.390, 0.329, 0.371, 0.452, 0.448, 0.496],
               [0.384, 0.406, 0.398, 0.405, 0.414, 0.380, 0.413, 0.452, 0.465, 0.487]],
      ["720",  [0.392, 0.412, 0.403, 0.406, 0.405, 0.379, 0.426, 0.462, 0.605, 0.463],
               [0.425, 0.449, 0.431, 0.436, 0.434, 0.422, 0.451, 0.468, 0.551, 0.474]],
      ["Avg.", [0.337, 0.341, 0.351, 0.354, 0.361, 0.330, 0.355, 0.414, 0.431, 0.437],
               [0.388, 0.391, 0.386, 0.389, 0.396, 0.379, 0.398, 0.427, 0.447, 0.449]]
    ],
    ETTm1: [
      ["96",   [0.279, 0.304, 0.284, 0.300, 0.291, 0.290, 0.299, 0.338, 0.299, 0.379],
               [0.329, 0.346, 0.332, 0.340, 0.341, 0.342, 0.348, 0.375, 0.343, 0.419]],
      ["192",  [0.317, 0.332, 0.327, 0.343, 0.341, 0.332, 0.334, 0.374, 0.335, 0.426],
               [0.357, 0.366, 0.362, 0.368, 0.369, 0.369, 0.367, 0.387, 0.365, 0.441]],
      ["336",  [0.351, 0.364, 0.354, 0.376, 0.359, 0.366, 0.368, 0.410, 0.369, 0.445],
               [0.381, 0.383, 0.382, 0.386, 0.379, 0.392, 0.386, 0.411, 0.386, 0.459]],
      ["720",  [0.411, 0.402, 0.411, 0.431, 0.433, 0.416, 0.417, 0.478, 0.425, 0.543],
               [0.415, 0.410, 0.415, 0.416, 0.419, 0.420, 0.414, 0.450, 0.421, 0.490]],
      ["Avg.", [0.340, 0.351, 0.344, 0.363, 0.356, 0.351, 0.355, 0.400, 0.357, 0.448],
               [0.371, 0.376, 0.373, 0.378, 0.377, 0.381, 0.379, 0.406, 0.379, 0.452]]
    ],
    ETTm2: [
      ["96",   [0.172, 0.160, 0.174, 0.163, 0.162, 0.165, 0.159, 0.187, 0.167, 0.203],
               [0.260, 0.250, 0.262, 0.249, 0.248, 0.255, 0.247, 0.267, 0.260, 0.287]],
      ["192",  [0.227, 0.215, 0.228, 0.222, 0.235, 0.220, 0.214, 0.249, 0.224, 0.269],
               [0.298, 0.291, 0.297, 0.291, 0.304, 0.292, 0.286, 0.309, 0.303, 0.328]],
      ["336",  [0.272, 0.270, 0.281, 0.273, 0.280, 0.274, 0.269, 0.321, 0.281, 0.325],
               [0.327, 0.325, 0.337, 0.327, 0.329, 0.329, 0.322, 0.351, 0.342, 0.366]],
      ["720",  [0.351, 0.348, 0.384, 0.357, 0.366, 0.362, 0.363, 0.408, 0.397, 0.421],
               [0.381, 0.378, 0.410, 0.376, 0.382, 0.385, 0.382, 0.403, 0.421, 0.415]],
      ["Avg.", [0.256, 0.248, 0.267, 0.254, 0.261, 0.255, 0.251, 0.291, 0.267, 0.305],
               [0.317, 0.311, 0.327, 0.311, 0.316, 0.315, 0.309, 0.333, 0.332, 0.349]]
    ],
    Illness: [
      ["24",   [1.409, null, 1.613, 1.869, 1.792, 1.319, 2.255, 2.317, 2.215, 3.228],
               [0.754, null, 0.834, 0.823, 0.807, 0.754, 1.017, 0.934, 1.081, 1.260]],
      ["36",   [1.290, null, 1.316, 1.853, 1.833, 1.430, 2.121, 1.972, 1.963, 2.679],
               [0.745, null, 0.750, 0.854, 0.833, 0.834, 0.950, 0.920, 0.963, 1.080]],
      ["48",   [1.499, null, 1.548, 1.886, 2.269, 1.553, 2.187, 2.238, 2.130, 2.622],
               [0.810, null, 0.818, 0.855, 1.012, 0.815, 1.007, 0.940, 1.024, 1.078]],
      ["60",   [1.428, null, 1.450, 1.877, 2.177, 1.470, 2.185, 2.027, 2.368, 2.857],
               [0.773, null, 0.783, 0.877, 0.925, 0.788, 0.997, 0.928, 1.096, 1.157]],
      ["Avg.", [1.407, null, 1.482, 1.871, 2.018, 1.443, 2.187, 2.139, 2.169, 2.847],
               [0.771, null, 0.796, 0.852, 0.894, 0.798, 0.992, 0.931, 1.041, 1.144]]
    ],
    Electricity: [
      ["96",   [0.126, 0.142, 0.127, 0.141, 0.137, 0.129, 0.128, 0.168, 0.140, 0.193],
               [0.213, 0.245, 0.217, 0.239, 0.233, 0.222, 0.223, 0.272, 0.237, 0.308]],
      ["192",  [0.145, 0.157, 0.148, 0.158, 0.152, 0.157, 0.144, 0.184, 0.153, 0.201],
               [0.237, 0.260, 0.237, 0.253, 0.247, 0.240, 0.237, 0.289, 0.249, 0.315]],
      ["336",  [0.162, 0.174, 0.163, 0.172, 0.169, 0.163, 0.160, 0.198, 0.169, 0.214],
               [0.254, 0.276, 0.253, 0.266, 0.267, 0.259, 0.254, 0.300, 0.267, 0.329]],
      ["720",  [0.197, 0.214, 0.199, 0.207, 0.200, 0.197, 0.198, 0.220, 0.203, 0.246],
               [0.286, 0.308, 0.293, 0.293, 0.290, 0.290, 0.287, 0.320, 0.301, 0.355]],
      ["Avg.", [0.158, 0.172, 0.159, 0.170, 0.165, 0.162, 0.158, 0.193, 0.166, 0.214],
               [0.248, 0.272, 0.250, 0.263, 0.259, 0.253, 0.250, 0.295, 0.264, 0.327]]
    ],
    Weather: [
      ["96",   [0.143, 0.148, 0.146, 0.148, 0.155, 0.149, 0.167, 0.172, 0.176, 0.217],
               [0.195, 0.200, 0.191, 0.188, 0.199, 0.198, 0.221, 0.220, 0.237, 0.296]],
      ["192",  [0.187, 0.193, 0.194, 0.192, 0.223, 0.194, 0.212, 0.219, 0.220, 0.276],
               [0.242, 0.240, 0.238, 0.230, 0.261, 0.241, 0.258, 0.261, 0.282, 0.336]],
      ["336",  [0.237, 0.243, 0.243, 0.246, 0.251, 0.245, 0.260, 0.280, 0.265, 0.339],
               [0.273, 0.281, 0.275, 0.273, 0.279, 0.282, 0.293, 0.306, 0.319, 0.380]],
      ["720",  [0.302, 0.312, 0.318, 0.320, 0.345, 0.314, 0.328, 0.365, 0.333, 0.403],
               [0.315, 0.332, 0.328, 0.328, 0.342, 0.334, 0.339, 0.359, 0.362, 0.428]],
      ["Avg.", [0.217, 0.224, 0.225, 0.227, 0.244, 0.226, 0.242, 0.259, 0.249, 0.309],
               [0.256, 0.263, 0.258, 0.255, 0.270, 0.264, 0.278, 0.287, 0.300, 0.360]]
    ],
    Traffic: [
      ["96",   [0.344, 0.393, 0.346, 0.396, 0.392, 0.360, 0.397, 0.593, 0.410, 0.587],
               [0.237, 0.290, 0.232, 0.264, 0.267, 0.249, 0.278, 0.321, 0.282, 0.366]],
      ["192",  [0.363, 0.405, 0.376, 0.412, 0.409, 0.379, 0.411, 0.617, 0.423, 0.604],
               [0.249, 0.296, 0.245, 0.268, 0.271, 0.256, 0.283, 0.336, 0.287, 0.373]],
      ["336",  [0.387, 0.420, 0.389, 0.421, 0.434, 0.392, 0.424, 0.629, 0.436, 0.621],
               [0.256, 0.305, 0.252, 0.273, 0.296, 0.264, 0.289, 0.336, 0.296, 0.383]],
      ["720",  [0.433, 0.459, 0.432, 0.455, 0.451, 0.432, 0.450, 0.640, 0.466, 0.626],
               [0.284, 0.323, 0.293, 0.291, 0.291, 0.286, 0.305, 0.350, 0.315, 0.382]],
      ["Avg.", [0.382, 0.419, 0.386, 0.421, 0.422, 0.391, 0.421, 0.620, 0.434, 0.610],
               [0.257, 0.304, 0.256, 0.274, 0.281, 0.264, 0.289, 0.336, 0.295, 0.376]]
    ]
  };

  const DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Illness", "Electricity", "Weather", "Traffic"];
  const WINS = [43, 9, 9, 7, 1, 9, 11, 0, 0, 0];

  let dsPick = "all";
  let metricPick = "all";

  function metricsShown() {
    return metricPick === "all" ? ["MSE", "MAE"] : [metricPick.toUpperCase()];
  }

  function renderMain() {
    const table = $("#main-table");
    if (!table) return;
    const mets = metricsShown();
    const span = mets.length;

    /* rows: either one average row per dataset, or every horizon of one dataset */
    const rows = dsPick === "all"
      ? DATASETS.map((d) => {
          const avg = MAIN[d][MAIN[d].length - 1];
          return { label: d, translate: false, mse: avg[1], mae: avg[2] };
        })
      : MAIN[dsPick].map((r) => ({ label: r[0], translate: r[0] === "Avg.", mse: r[1], mae: r[2] }));

    const head = table.querySelector("thead");
    const body = table.querySelector("tbody");
    head.innerHTML = "";
    body.innerHTML = "";

    const grp = el("tr", "grp");
    grp.appendChild(el("th", "name"));
    MODELS.forEach((m) => {
      const th = el("th", m.sep ? "sep" : null, m.name);
      th.colSpan = span;
      grp.appendChild(th);
    });
    head.appendChild(grp);

    const hr = document.createElement("tr");
    hr.appendChild(el("th", "name", tr(dsPick === "all" ? "Dataset" : "Horizon")));
    MODELS.forEach((m) => {
      mets.forEach((mt, j) => hr.appendChild(el("th", m.sep && j === 0 ? "sep" : null, mt)));
    });
    head.appendChild(hr);

    rows.forEach((r) => {
      const marks = {
        MSE: rankMarks(r.mse, "lower"),
        MAE: rankMarks(r.mae, "lower")
      };
      const vals = { MSE: r.mse, MAE: r.mae };
      const trw = document.createElement("tr");
      trw.appendChild(el("td", "name", r.translate ? tr(r.label) : r.label));
      MODELS.forEach((m, i) => {
        mets.forEach((mt, j) => {
          const cls = [marks[mt][i], m.ours ? "ours" : "", m.sep && j === 0 ? "sep" : ""]
            .filter(Boolean).join(" ");
          trw.appendChild(el("td", cls || null, fmt(vals[mt][i])));
        });
      });
      body.appendChild(trw);
    });

    const winMarks = rankMarks(WINS, "higher");
    const wr = el("tr", "wins");
    wr.appendChild(el("td", "name", tr("# Wins")));
    MODELS.forEach((m, i) => {
      const td = el("td", [winMarks[i], m.ours ? "ours" : "", m.sep ? "sep" : ""]
        .filter(Boolean).join(" ") || null, String(WINS[i]));
      td.colSpan = span;
      wr.appendChild(td);
    });
    body.appendChild(wr);
  }

  bindChoices("ds-seg", "d", (v) => { dsPick = v; renderMain(); });
  bindChoices("metric-seg", "m", (v) => { metricPick = v; renderMain(); });
  registerRender(renderMain);

  /* ======================================================= ablation table */
  const ABL_SETS = ["ETTh1", "ETTm1", "Illness", "Weather"];

  /* label, [MSE, MAE] per dataset, then the paper's improvement over DMMV-A */
  const ABL = [
    { label: "DMMV-A", v: [0.395, 0.414, 0.340, 0.371, 1.407, 0.771, 0.217, 0.256] },
    { label: "(a) f<sub>num</sub> → Transformer",
      v: [0.407, 0.421, 0.339, 0.372, 1.442, 0.786, 0.219, 0.260],
      imp: ["-3.04%", "-1.69%", "+0.29%", "-0.27%", "-2.49%", "-1.95%", "-0.92%", "-1.56%"] },
    { label: "(b) f<sub>vis</sub> → SimMIM",
      v: [0.407, 0.415, 0.345, 0.377, 1.649, 0.814, 0.227, 0.261],
      imp: ["-3.04%", "-0.24%", "-1.47%", "-1.62%", "-17.20%", "-5.58%", "-4.61%", "-1.95%"] },
    { label: "(c) Gate → Sum",
      v: [0.414, 0.427, 0.352, 0.383, 1.606, 0.863, 0.233, 0.278],
      imp: ["-4.81%", "-3.14%", "-3.53%", "-3.23%", "-14.14%", "-11.93%", "-7.37%", "-8.59%"] },
    { label: "(d) BCMask → No mask",
      v: [0.426, 0.441, 0.349, 0.377, 1.493, 0.828, 0.221, 0.267],
      imp: ["-7.85%", "-6.52%", "-2.65%", "-1.62%", "-6.11%", "-7.39%", "-1.84%", "-4.30%"] },
    { label: "(e) BCMask → Random mask",
      v: [0.394, 0.414, 0.340, 0.372, 1.472, 0.829, 0.223, 0.262],
      imp: ["0.25%", "0.00%", "0.00%", "-0.27%", "-4.62%", "-7.52%", "-2.76%", "-2.34%"] },
    { label: "(f) Freeze f<sub>vis</sub>",
      v: [0.431, 0.428, 0.358, 0.380, 1.442, 0.773, 0.246, 0.288],
      imp: ["-9.11%", "-3.38%", "-5.29%", "-2.43%", "-2.49%", "-0.26%", "-13.36%", "-12.50%"] },
    { label: "(g) W/o decomposition",
      v: [0.408, 0.424, 0.338, 0.373, 1.712, 0.903, 0.219, 0.268],
      imp: ["-3.29%", "-2.42%", "+0.59%", "-0.54%", "-21.68%", "-17.12%", "-0.92%", "-4.69%"] }
  ];

  function renderAblation() {
    const table = $("#ablation-table");
    if (!table) return;
    const head = table.querySelector("thead");
    const body = table.querySelector("tbody");
    head.innerHTML = "";
    body.innerHTML = "";

    const grp = el("tr", "grp");
    grp.appendChild(el("th", "name"));
    ABL_SETS.forEach((d, i) => {
      const th = el("th", i ? "sep" : null, d);
      th.colSpan = 2;
      grp.appendChild(th);
    });
    head.appendChild(grp);

    const hr = document.createElement("tr");
    hr.appendChild(el("th", "name", tr("Method")));
    ABL_SETS.forEach((d, i) => {
      hr.appendChild(el("th", i ? "sep" : null, "MSE"));
      hr.appendChild(el("th", null, "MAE"));
    });
    head.appendChild(hr);

    const marks = [];
    for (let c = 0; c < 8; c++) marks.push(rankMarks(ABL.map((r) => r.v[c]), "lower"));

    ABL.forEach((r, ri) => {
      const trw = document.createElement("tr");
      trw.appendChild(el("td", "name", tr(r.label)));
      r.v.forEach((v, c) => {
        const cls = [marks[c][ri], c % 2 === 0 && c > 0 ? "sep" : ""].filter(Boolean).join(" ");
        trw.appendChild(el("td", cls || null, fmt(v)));
      });
      body.appendChild(trw);

      if (r.imp) {
        const ir = document.createElement("tr");
        ir.appendChild(el("td", "name ref", tr("Improvement")));
        r.imp.forEach((s, c) => {
          ir.appendChild(el("td", "ref" + (c % 2 === 0 && c > 0 ? " sep" : ""), s));
        });
        body.appendChild(ir);
      }
    });
  }
  registerRender(renderAblation);

  /* ================================ segment length on the synthetic series */
  const SEG_LEN = [16, 20, 24, 28, 32, 36, 40, 44, 48];
  const SEG = {
    MSE: [0.043, 0.099, 0.001, 0.147, 0.154, 0.143, 0.221, 0.114, 0.002],
    MAE: [0.177, 0.257, 0.024, 0.342, 0.347, 0.315, 0.408, 0.289, 0.045]
  };

  function renderSegment() {
    const table = $("#segment-table");
    if (!table) return;
    const head = table.querySelector("thead");
    const body = table.querySelector("tbody");
    head.innerHTML = "";
    body.innerHTML = "";

    const hr = document.createElement("tr");
    hr.appendChild(el("th", "name", tr("Segment length")));
    SEG_LEN.forEach((p) => hr.appendChild(el("th", null, String(p))));
    head.appendChild(hr);

    ["MSE", "MAE"].forEach((m) => {
      const marks = rankMarks(SEG[m], "lower");
      const trw = document.createElement("tr");
      trw.appendChild(el("td", "name", m));
      SEG[m].forEach((v, i) => trw.appendChild(el("td", marks[i] || null, fmt(v))));
      body.appendChild(trw);
    });
  }
  registerRender(renderSegment);
})();

MC.start();
