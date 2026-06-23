import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { GoogleGenAI } from '@google/genai';
import cors from 'cors';

const execAsync = promisify(exec);
const app = express();

// ====== CONFIG ======
// CORS: production e ekhane tomar Vercel domain bosao,
// jemon: origin: 'https://indexgen-pro.vercel.app'
// Ekhon * dewa ache testing er subidhar jonno.
app.use(cors());
app.use(express.json());

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const MODEL_NAME = 'gemini-3.5-flash';
const CHUNK_SIZE = 12;       // ekbar e koto page Gemini ke pathano hobe
const MAX_RETRIES = 2;       // Gemini call fail korle koto bar retry
const MAX_FILE_SIZE_MB = 50;

if (!process.env.GEMINI_API_KEY) {
  console.error('FATAL: GEMINI_API_KEY environment variable set kora nei. Render dashboard e Environment tab e check koro.');
}
import { useState, useRef } from "react";

// ★ This is the KEY prompt — fixes the "too many entries" problem
// Copy this into your Google AI Studio / Gemini system_instruction field
const SYSTEM_PROMPT = `You are an expert academic INDEX creator. Create a CLEAN, CONCISE Table of Contents.

CRITICAL RULES:
1. Extract ONLY the PRIMARY/MAIN topic headings from the document
2. If a document pairs "Systematic position of X" WITH "Characteristics of X" — combine them into just ONE entry about X
3. NEVER list sub-items, characteristic sections, examples, or secondary details as separate entries
4. Keep MAXIMUM 25 entries total (the index must fit on ONE single page)
5. Topic names should be clear, descriptive, and academic in style

RETURN ONLY VALID JSON — no other text, no markdown backticks, no explanation:
{
  "entries": [
    {"sr": 1, "topic": "Topic Name Here", "page": 1},
    {"sr": 2, "topic": "Another Main Topic", "page": 3}
  ]
}`;

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [entries, setEntries] = useState(null);
  const [error, setError] = useState("");
  const [showPrompt, setShowPrompt] = useState(false);
  const [copied, setCopied] = useState(false);
  const inputRef = useRef();

  const pickFile = (f) => {
    if (f?.type === "application/pdf") {
      setFile(f);
      setEntries(null);
      setError("");
    } else {
      setError("Please upload a PDF file only.");
    }
  };

  const createIndex = async () => {
    if (!file || loading) return;
    setLoading(true);
    setError("");
    try {
      const b64 = await new Promise((res, rej) => {
        const reader = new FileReader();
        reader.onload = () => res(reader.result.split(",")[1]);
        reader.onerror = () => rej(new Error("File could not be read"));
        reader.readAsDataURL(file);
      });

      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-6",
          max_tokens: 1000,
          system: SYSTEM_PROMPT,
          messages: [{
            role: "user",
            content: [
              {
                type: "document",
                source: { type: "base64", media_type: "application/pdf", data: b64 }
              },
              {
                type: "text",
                text: "Create a clean, single-page INDEX from this document. Extract ONLY the main topic headings. Return ONLY JSON."
              }
            ]
          }]
        })
      });

      const data = await response.json();
      if (data.error) throw new Error(data.error.message);

      const rawText = data.content?.find(c => c.type === "text")?.text || "";
      const cleaned = rawText.replace(/```json\n?|```\n?/g, "").trim();
      const parsed = JSON.parse(cleaned);
      setEntries(parsed.entries || []);
    } catch (e) {
      setError(e.message || "Something went wrong. Please try again.");
    }
    setLoading(false);
  };

  const copyPrompt = () => {
    navigator.clipboard.writeText(SYSTEM_PROMPT);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const navy = "#1C2951";
  const gold = "#C9A84C";
  const paper = "#F9F7F2";
  const cream = "#F0EDE5";
  const borderClr = "#D9D4C7";

  return (
    <div style={{ minHeight: "100vh", background: cream, fontFamily: "'Georgia', serif", padding: "24px 16px" }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>

      <div style={{ maxWidth: 660, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ background: navy, borderRadius: "2px 2px 0 0", padding: "20px 28px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <div style={{ color: gold, fontSize: 11, letterSpacing: 3, textTransform: "uppercase", fontFamily: "sans-serif", marginBottom: 4 }}>
              Academic Tool
            </div>
            <h1 style={{ color: "white", margin: 0, fontSize: 20, fontWeight: "normal", letterSpacing: 1 }}>
              Index Creator
            </h1>
          </div>
          <div style={{ color: "rgba(255,255,255,0.25)", fontSize: 32, fontWeight: "bold" }}>§</div>
        </div>
        <div style={{ height: 3, background: `linear-gradient(90deg, ${gold}, #E8C97A)`, marginBottom: 0 }} />

        {/* Upload Card */}
        <div style={{ background: paper, border: `1px solid ${borderClr}`, borderTop: "none", borderRadius: "0 0 2px 2px", padding: "24px 28px", marginBottom: 20 }}>

          <div
            onClick={() => inputRef.current.click()}
            onDrop={e => { e.preventDefault(); pickFile(e.dataTransfer.files[0]); }}
            onDragOver={e => e.preventDefault()}
            style={{
              border: `2px dashed ${file ? "#2D6A4F" : borderClr}`,
              borderRadius: 2,
              padding: "28px 20px",
              textAlign: "center",
              cursor: "pointer",
              background: file ? "#F0FFF4" : "white",
              marginBottom: 16,
              transition: "all 0.2s"
            }}
          >
            <input ref={inputRef} type="file" accept=".pdf" onChange={e => pickFile(e.target.files[0])} style={{ display: "none" }} />
            {file ? (
              <div style={{ animation: "fadeIn 0.3s ease" }}>
                <div style={{ fontSize: 28, color: "#2D6A4F", marginBottom: 8 }}>✓</div>
                <p style={{ margin: 0, fontWeight: 700, color: "#2D6A4F", fontFamily: "sans-serif", fontSize: 13 }}>{file.name}</p>
                <p style={{ margin: "4px 0 0", fontSize: 11, color: "#999", fontFamily: "sans-serif" }}>Click to replace</p>
              </div>
            ) : (
              <div>
                <div style={{ fontSize: 30, marginBottom: 8, opacity: 0.35 }}>⬆</div>
                <p style={{ margin: 0, color: "#555", fontFamily: "sans-serif", fontSize: 14 }}>Drop your PDF here or click to browse</p>
                <p style={{ margin: "4px 0 0", fontSize: 11, color: "#aaa", fontFamily: "sans-serif" }}>Typed or handwritten — any PDF supported</p>
              </div>
            )}
          </div>

          <button
            onClick={createIndex}
            disabled={!file || loading}
            style={{
              width: "100%",
              padding: "13px",
              background: !file || loading ? "#bbb" : navy,
              color: "white",
              border: "none",
              borderRadius: 2,
              fontSize: 14,
              fontFamily: "sans-serif",
              fontWeight: 600,
              cursor: !file || loading ? "default" : "pointer",
              letterSpacing: 0.5,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 10
            }}
          >
            {loading ? (
              <>
                <span style={{
                  width: 14, height: 14,
                  border: "2px solid rgba(255,255,255,0.3)",
                  borderTopColor: "white",
                  borderRadius: "50%",
                  display: "inline-block",
                  animation: "spin 0.8s linear infinite"
                }} />
                Index তৈরি হচ্ছে...
              </>
            ) : "Generate Single-Page Index"}
          </button>

          {error && (
            <div style={{ marginTop: 12, padding: "10px 14px", background: "#fff5f5", border: "1px solid #fcc", borderRadius: 2, color: "#c62828", fontSize: 13, fontFamily: "sans-serif" }}>
              ⚠ {error}
            </div>
          )}
        </div>

        {/* Index Output */}
        {entries && (
          <div style={{
            background: "white",
            border: `1px solid ${borderClr}`,
            borderRadius: 2,
            overflow: "hidden",
            marginBottom: 20,
            animation: "fadeIn 0.4s ease",
            boxShadow: "0 4px 20px rgba(28,41,81,0.10)"
          }}>
            <div style={{ padding: "22px 28px 14px", borderBottom: `3px double ${navy}`, textAlign: "center" }}>
              <h2 style={{ margin: 0, fontSize: 22, color: navy, letterSpacing: 6, textDecoration: "underline", fontWeight: "bold" }}>
                INDEX
              </h2>
            </div>

            <div style={{ padding: "0 28px 24px" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 4 }}>
                <thead>
                  <tr style={{ borderBottom: `2px solid ${navy}` }}>
                    <th style={{ padding: "10px 8px 8px", textAlign: "center", fontSize: 13, fontFamily: "sans-serif", fontWeight: 700, color: navy, width: 64 }}>Sr. No.</th>
                    <th style={{ padding: "10px 8px 8px", textAlign: "left", fontSize: 13, fontFamily: "sans-serif", fontWeight: 700, color: navy }}>Particulars</th>
                    <th style={{ padding: "10px 8px 8px", textAlign: "center", fontSize: 13, fontFamily: "sans-serif", fontWeight: 700, color: navy, width: 72 }}>Page No.</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.map((e, i) => (
                    <tr key={i} style={{ borderBottom: `1px solid ${borderClr}`, background: i % 2 === 0 ? "white" : "#FDFCFA" }}>
                      <td style={{ padding: "9px 8px", textAlign: "center", fontSize: 14, color: "#444", fontFamily: "sans-serif" }}>{e.sr}.</td>
                      <td style={{ padding: "9px 8px", fontSize: 14, color: "#1a1a1a" }}>{e.topic}</td>
                      <td style={{ padding: "9px 8px", textAlign: "center", fontSize: 14, color: "#444", fontFamily: "sans-serif" }}>{e.page}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ marginTop: 12, textAlign: "right", fontSize: 11, color: "#bbb", fontFamily: "sans-serif" }}>
                {entries.length} entries • fits on 1 page
              </div>
            </div>
          </div>
        )}

        {/* Gemini Fix Panel */}
        <div style={{ background: paper, border: `1px solid ${borderClr}`, borderRadius: 2, overflow: "hidden" }}>
          <button
            onClick={() => setShowPrompt(!showPrompt)}
            style={{
              width: "100%",
              background: "none",
              border: "none",
              padding: "14px 22px",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              cursor: "pointer",
              fontFamily: "sans-serif",
              fontSize: 13,
              fontWeight: 600,
              color: navy
            }}
          >
            <span>🔧 তোমার Gemini Backend Fix — System Prompt</span>
            <span style={{ fontSize: 11, color: "#aaa" }}>{showPrompt ? "▲ Hide" : "▼ Show"}</span>
          </button>

          {showPrompt && (
            <div style={{ padding: "0 22px 20px", borderTop: `1px solid ${borderClr}` }}>
              <p style={{ fontSize: 12, color: "#666", fontFamily: "sans-serif", lineHeight: 1.75, marginBottom: 12 }}>
                তোমার <strong>Google AI Studio / Gemini</strong> API call এ এই <strong>System Prompt</strong> paste করো।
                এটাই মূল সমস্যার সমাধান — এতে sub-items বাদ পড়বে, শুধু main topics আসবে, output 1 page এ fit হবে:
              </p>
              <pre style={{
                background: "#1C2951",
                color: "#A8D5C2",
                borderRadius: 2,
                padding: "14px 16px",
                fontSize: 11,
                fontFamily: "monospace",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                lineHeight: 1.75,
                margin: "0 0 14px"
              }}>{SYSTEM_PROMPT}</pre>
              <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
                <button onClick={copyPrompt} style={{
                  background: copied ? "#2D6A4F" : navy,
                  color: "white",
                  border: "none",
                  padding: "9px 22px",
                  borderRadius: 2,
                  cursor: "pointer",
                  fontSize: 13,
                  fontWeight: 600,
                  fontFamily: "sans-serif",
                  transition: "background 0.2s"
                }}>
                  {copied ? "✓ Copied!" : "Copy Prompt"}
                </button>
                <span style={{ fontSize: 11, color: "#999", fontFamily: "sans-serif" }}>
                  Gemini এর <code>system_instruction</code> field এ paste করো
                </span>
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
          }
// ====== MULTER (file upload) ======
const upload = multer({
  dest: os.tmpdir(),
  limits: { fileSize: MAX_FILE_SIZE_MB * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (file.mimetype !== 'application/pdf') {
      return cb(new Error('Shudhu PDF file allowed. Tumi je file dieso shetar type: ' + file.mimetype));
    }
    cb(null, true);
  }
});

// ====== JSON SCHEMA — Gemini ke ei structure e i JSON ditei hobe ======
const indexSchema = {
  type: 'array',
  items: {
    type: 'object',
    properties: {
      page: { type: 'integer' },
      topic: { type: 'string' },
      level: { type: 'integer' }
    },
    required: ['page', 'topic', 'level']
  }
};

// ====== ROUTES ======

app.get('/', (req, res) => {
  res.send('IndexGen Pro backend cholche.');
});

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    geminiKeyConfigured: Boolean(process.env.GEMINI_API_KEY),
    model: MODEL_NAME
  });
});

app.post('/api/generate-index', upload.single('pdf'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'PDF file pawa jayni. Form field er naam "pdf" hote hobe.' });
  }

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'idx-'));
  const pdfPath = req.file.path;

  try {
    // 1. PDF -> JPEG pages (poppler-utils). -r 120 = 120 DPI — beshi dile
    // file size o token cost onek barbe, ei resolution e handwriting thik moto pora jay.
    await execAsync(`pdftoppm -jpeg -r 120 "${pdfPath}" "${workDir}/page"`);

    const files = fs.readdirSync(workDir)
      .filter(f => f.endsWith('.jpg'))
      .sort(); // pdftoppm naturally zero-pad kore (page-01, page-02...), tai sort thik thakbe

    if (files.length === 0) {
      throw new Error('PDF theke kono page convert kora gelo na. File ki corrupt, naki password-protected?');
    }

    // 2. Chunk by chunk Gemini ke pathano, protyek chunk e retry shoho
    let allTopics = [];
    const failedChunks = [];

    for (let i = 0; i < files.length; i += CHUNK_SIZE) {
      const chunk = files.slice(i, i + CHUNK_SIZE);
      try {
        const chunkTopics = await processChunkWithRetry(workDir, chunk, i);
        allTopics = allTopics.concat(chunkTopics);
      } catch (chunkErr) {
        // Ekta chunk fail korle pura request fail korano thik na — baki
        // chunk gulo process hote dao, kintu user ke janiye dao kon page gulo miss hoyeche.
        console.error(`Chunk starting at page ${i + 1} fail korlo:`, chunkErr.message);
        failedChunks.push({ fromPage: i + 1, toPage: Math.min(i + CHUNK_SIZE, files.length) });
      }
    }

    allTopics.sort((a, b) => a.page - b.page);

    const responsePayload = {
      success: true,
      totalPages: files.length,
      index: allTopics
    };

    if (failedChunks.length > 0) {
      responsePayload.warning = `Kichu page process kora jayni: ${failedChunks
        .map(c => `${c.fromPage}-${c.toPage}`)
        .join(', ')}. Server abar try korte paro.`;
    }

    res.json(responsePayload);
  } catch (err) {
    console.error('Index generation error:', err);
    res.status(500).json({ error: err.message || 'Internal server error' });
  } finally {
    fs.rmSync(workDir, { recursive: true, force: true });
    fs.unlink(pdfPath, () => {}); // best-effort cleanup, error hole o block korbe na
  }
});

// ====== HELPERS ======

async function processChunkWithRetry(workDir, chunkFiles, offset) {
  let lastErr;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await processChunk(workDir, chunkFiles, offset);
    } catch (err) {
      lastErr = err;
      console.warn(`Chunk (page ${offset + 1}) attempt ${attempt + 1} fail: ${err.message}`);
      if (attempt < MAX_RETRIES) {
        await new Promise(r => setTimeout(r, 1500 * (attempt + 1))); // simple backoff
      }
    }
  }
  throw lastErr;
}

async function processChunk(workDir, chunkFiles, offset) {
  const contents = [];

  contents.push({
    text: `You are a highly intelligent academic index writer. You write indexes
exactly the way a top university student writes them by hand — smart, concise,
and reader-friendly. NOT like a mechanical heading extractor.

You will receive ${chunkFiles.length} pages, each labeled --- PAGE N ---.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — UNDERSTAND THE DOCUMENT FIRST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before extracting anything, answer these internally:
- What subject/topic is this document about?
- What is the structure? (chapters? lab organisms? theories? experiments?)
- How many pages? What is the overall scope?
- Are there REPEATING PATTERNS (same type of heading repeated many times)?

This context shapes EVERYTHING about how you write the index.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — SMART INDEX WRITING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — STRICT MAXIMUM: 25 entries total. If you find more, you MUST merge.
If still more, merge again. This limit is absolute — never return over 25.

RULE 2 — REPETITIVE PATTERN HANDLING:
If document has same heading type repeated 5+ times (e.g. "Systematic position
of [organism]" appears 20 times), handle it like this:
  Option A (preferred): List ONLY the unique name part, not the repeated prefix
    INSTEAD OF: "Systematic position of Dero dorsalis" (×26 entries)
    WRITE:      "Dero dorsalis" as level 1, under a level 0 parent entry
                like "Freshwater Organisms — Systematic Positions & Characteristics"
  Option B: If even that gives too many entries, group them:
    "Freshwater Invertebrates (Dero dorsalis, Tubifex tubifex, Chironomus...)"
    at the page of first occurrence

RULE 3 — INTELLIGENT MERGING:
Merge sub-topics into their parent when they are closely related:
  MERGE: "Fringing Reefs" + "Barrier Reefs" + "Atolls"
  INTO:  "Kinds of Coral Reefs (Fringing, Barrier, Atolls)"

  MERGE: "Stutchbury's Theory" + "Darwin-Dana Theory" + "Samper Theory" + more
  INTO:  "Formation of Coral Reefs — Theories"

  MERGE: "Yeast" + "Molds" + "Culture Media"
  INTO:  "Fungus — Characteristics, Yeast, Molds & Culture"

RULE 4 — SKIP HOLLOW ENTRIES:
Skip anything that adds no unique search value:
  SKIP:  "Characteristics:" when it ALWAYS follows a named item
  SKIP:  "Introduction" when a better specific title exists on same page
  SKIP:  "Limitations:", "Note:", "Summary:" type standalone markers
  SKIP:  Kingdom/Phylum/Class/Order/Family/Genus/Species taxonomy lines
  SKIP:  Figure captions (Fig:, Figure 3.1), diagram labels, table cell text

RULE 5 — HEADING SIGNAL DETECTION:
  Handwritten: underline below text, box/star/symbol prefix (凸★田), 
               standalone line with heavier pen or larger writing
  Typed/Printed: bold standalone line, ALL CAPS title, larger font than body,
                 numbered sections (1.1, 2.3, Chapter 3)

RULE 6 — LEVELS:
  level 0 = main topic a reader searches for (bold, no indent)
  level 1 = important named sub-section worth listing (indented)
  level 2 = use only when genuinely distinct — use very rarely

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3 — QUALITY CHECK BEFORE RETURNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ask yourself:
1. Could a student FIND a topic they are looking for using this index? YES = good
2. Does every entry justify its place? Could it be merged? Merge if yes.
3. Is total count under 25? If not, merge more aggressively.
4. Do the entries reflect THIS SPECIFIC DOCUMENT's content, not generic titles?
5. Would a professor reading this index understand what the document covers?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a valid JSON array. Nothing before it, nothing after it.
Each item: { "page": number, "topic": "exact text", "level": 0|1|2 }
Page number = from the --- PAGE N --- label above that image.`
  });

  chunkFiles.forEach((file, idx) => {
    const realPageNum = offset + idx + 1;
    const imgData = fs.readFileSync(path.join(workDir, file)).toString('base64');
    contents.push({ text: `--- PAGE ${realPageNum} ---` });
    contents.push({ inlineData: { mimeType: 'image/jpeg', data: imgData } });
  });

  const response = await ai.models.generateContent({
    model: MODEL_NAME,
    contents,
    config: {
      responseMimeType: 'application/json',
      responseSchema: indexSchema
    }
  });

  try {
    return JSON.parse(response.text);
  } catch {
    console.error('Gemini response parse failed:', response.text);
    throw new Error('Gemini theke valid JSON paoa jayni');
  }
}

// ====== CENTRALIZED ERROR HANDLER ======
// Multer-er file-size / file-type error gulo route handler dhorte pare na,
// tai eta lagbe — na hole server crash korbe na thik, kintu user error
// dekhbe na, request shudhu hang kore thakbe.
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: `File size ${MAX_FILE_SIZE_MB}MB-er beshi hote parbe na.` });
    }
    return res.status(400).json({ error: err.message });
  }
  if (err) {
    return res.status(400).json({ error: err.message });
  }
  next();
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
