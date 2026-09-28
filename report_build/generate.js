/* HAM10000 实验报告 docx 生成器（简版：题目 + 五章正文，无封面/摘要/目录）
 * 内容来自 report_build/content.json
 */
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Footer, PageNumber, AlignmentType, HeadingLevel, WidthType,
  BorderStyle, ShadingType,
} = require("docx");
const fs = require("fs");
const path = require("path");
const sizeOf = require("image-size");

const ROOT = path.resolve(__dirname, "..");
const C = JSON.parse(fs.readFileSync(path.join(__dirname, "content.json"), "utf8"));

const TABLE_STYLE = { headerBg: "1B6B7A", headerText: "FFFFFF", innerLine: "C8DDE2", surface: "EDF3F5" };

// ---------- 构件 ----------
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160, line: 380, lineRule: "atLeast" },
    children: [new TextRun({ text, bold: true, size: 32, color: "000000",
      font: { eastAsia: "SimHei", ascii: "Times New Roman" } })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120, line: 350, lineRule: "atLeast" },
    children: [new TextRun({ text, bold: true, size: 28, color: "000000",
      font: { eastAsia: "SimHei", ascii: "Times New Roman" } })],
  });
}
function body(text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 480 },
    spacing: { line: 312, after: 60 },
    children: [new TextRun({ text, size: 24, color: "000000",
      font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
  });
}
function bodyRuns(runs) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 480 },
    spacing: { line: 312, after: 60 },
    children: runs.map(r => new TextRun({ text: r.x, bold: !!r.b, size: 24, color: "000000",
      font: { eastAsia: "SimSun", ascii: "Times New Roman" } })),
  });
}
function tableTitle(text) {
  return new Paragraph({
    keepNext: true, alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 80 },
    children: [new TextRun({ text, bold: true, size: 21, color: "000000",
      font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
  });
}
function figCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 80, after: 160 },
    children: [new TextRun({ text, size: 21, color: "404040",
      font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
  });
}
function cellPara(text, opts = {}) {
  return new Paragraph({
    alignment: opts.align || AlignmentType.CENTER,
    spacing: { line: 280 },
    children: [new TextRun({ text: String(text), bold: !!opts.bold, size: 20,
      color: opts.color || "000000", font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
  });
}
function dataTable(spec) {
  const widths = spec.widths;
  const headerRow = new TableRow({
    tableHeader: true, cantSplit: true,
    children: spec.header.map((t, i) => new TableCell({
      children: [cellPara(t, { bold: true, color: TABLE_STYLE.headerText })],
      shading: { type: ShadingType.CLEAR, fill: TABLE_STYLE.headerBg },
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      width: { size: widths[i], type: WidthType.PERCENTAGE },
    })),
  });
  const rows = spec.rows.map((r, ri) => new TableRow({
    cantSplit: true,
    children: r.map((t, i) => new TableCell({
      children: [cellPara(t)],
      shading: ri % 2 === 1 ? { type: ShadingType.CLEAR, fill: TABLE_STYLE.surface } : undefined,
      margins: { top: 50, bottom: 50, left: 100, right: 100 },
      width: { size: widths[i], type: WidthType.PERCENTAGE },
    })),
  }));
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 6, color: TABLE_STYLE.headerBg },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: TABLE_STYLE.headerBg },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: TABLE_STYLE.innerLine },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [headerRow, ...rows],
  });
}
function image(src, displayWidth, caption) {
  const buf = fs.readFileSync(path.join(ROOT, src));
  const dim = sizeOf.imageSize ? sizeOf.imageSize(buf) : sizeOf(buf);
  const h = Math.round(displayWidth * dim.height / dim.width);
  const out = [new Paragraph({
    keepNext: true, alignment: AlignmentType.CENTER, spacing: { before: 120 },
    children: [new ImageRun({ data: buf, transformation: { width: displayWidth, height: h }, type: "png" })],
  })];
  if (caption) out.push(figCaption(caption));
  return out;
}

// ---------- 题目区（首页顶部，无独立封面） ----------
const titleBlock = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 480, after: 160, line: Math.ceil(22 * 23), lineRule: "atLeast" },
    children: [new TextRun({ text: C.title, bold: true, size: 44, color: "000000",
      font: { eastAsia: "SimHei", ascii: "Times New Roman" } })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 320 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "888888", space: 10 } },
    children: [new TextRun({ text: C.subtitle, size: 21, color: "505050",
      font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
  }),
];

// ---------- 正文 ----------
const bodyChildren = [];
for (const b of C.body) {
  if (b.t === "h1") bodyChildren.push(h1(b.x));
  else if (b.t === "h2") bodyChildren.push(h2(b.x));
  else if (b.t === "p") bodyChildren.push(body(b.x));
  else if (b.t === "runs") bodyChildren.push(bodyRuns(b.runs));
  else if (b.t === "table") { bodyChildren.push(tableTitle(b.title)); bodyChildren.push(dataTable(b)); }
  else if (b.t === "img") bodyChildren.push(...image(b.src, b.width, b.caption));
  else if (b.t === "numlist") {
    b.items.forEach((it, i) => bodyChildren.push(new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      indent: { left: 480, hanging: 480 },
      spacing: { line: 312, after: 40 },
      children: [new TextRun({ text: `${i + 1}. ${it}`, size: 24, color: "000000",
        font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
    })));
  }
}

// ---------- 文档 ----------
const doc = new Document({
  creator: "Course Project",
  title: C.title,
  styles: {
    default: {
      document: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimSun" }, size: 24, color: "000000" },
        paragraph: { spacing: { line: 312 } },
      },
      heading1: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 32, bold: true, color: "000000" },
        paragraph: { spacing: { before: 360, after: 160, line: 380 } },
      },
      heading2: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 28, bold: true, color: "000000" },
        paragraph: { spacing: { before: 240, after: 120, line: 350 } },
      },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1440, bottom: 1440, left: 1701, right: 1417 },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080" })],
        })],
      }),
    },
    children: [...titleBlock, ...bodyChildren],
  }],
});

const OUT = path.join(__dirname, "BCI脑电实验报告.docx");
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("WROTE", OUT, buf.length, "bytes");
});
