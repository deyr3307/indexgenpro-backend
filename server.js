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
const CHUNK_SIZE = 8;              // 8 pages per chunk — Gemini rate limit avoid korte
const MAX_RETRIES = 3;             // fail hole 3 bar retry
const INTER_CHUNK_DELAY_MS = 2500; // chunks er moddhe 2.5s wait — rate limit thamate
const MAX_FILE_SIZE_MB = 30;

if (!process.env.GEMINI_API_KEY) {
  console.error('FATAL: GEMINI_API_KEY environment variable set kora nei. Render dashboard e Environment tab e check koro.');
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
      // Chunks er moddhe delay — Gemini rate limit theke bachte
      if (i > 0) {
        await new Promise(r => setTimeout(r, INTER_CHUNK_DELAY_MS));
      }
      const chunk = files.slice(i, i + CHUNK_SIZE);
      try {
        const chunkTopics = await processChunkWithRetry(workDir, chunk, i);
        allTopics = allTopics.concat(chunkTopics);
      } catch (chunkErr) {
        console.error(`Chunk starting at page ${i + 1} fail korlo:`, chunkErr.message);
        failedChunks.push({ fromPage: i + 1, toPage: Math.min(i + CHUNK_SIZE, files.length) });
      }
    }

    allTopics.sort((a, b) => a.page - b.page);

    // Dedup Step 1: Remove duplicate topics (same heading on consecutive pages).
    const seenTopics = new Set();
    allTopics = allTopics.filter(item => {
      const key = item.topic.toLowerCase().replace(/[^a-z0-9]/g, '');
      if (seenTopics.has(key)) return false;
      seenTopics.add(key);
      return true;
    });

    // Dedup Step 2: Remove duplicate page numbers — keep only the FIRST
    // (most prominent) heading per page. This eliminates sub-headings that
    // appear on the same page as a main heading.
    const seenPages = new Set();
    allTopics = allTopics.filter(item => {
      if (seenPages.has(item.page)) return false;
      seenPages.add(item.page);
      return true;
    });

    // Sanitize: allow ONLY <em> and </em> tags in topic text (for scientific names).
    // Strip any other HTML that Gemini might accidentally add.
    allTopics = allTopics.map(item => ({
      ...item,
      topic: item.topic
        .replace(/<(?!\/?em>)[^>]*>/gi, '')   // remove all tags except <em>
        .trim()
    }));

    // Hard limit: max 15 entries for guaranteed 1-page fit.
    allTopics = enforceMaxEntries(allTopics, 15);

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
        const delay = 2000 * Math.pow(2, attempt); // exponential backoff: 2s, 4s, 8s
        console.log(`Retrying in ${delay}ms...`);
        await new Promise(r => setTimeout(r, delay));
      }
    }
  }
  throw lastErr;
}

async function processChunk(workDir, chunkFiles, offset) {
  const contents = [];

  contents.push({
    text: `You are a masterclass academic index analyst. Your task: study these
${chunkFiles.length} pages and extract ONLY the core chapter-level main headings
for a professional one-page academic index.

Each page is labeled --- PAGE N --- above its image. Use that exact number.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — UNDERSTAND THE DOCUMENT CONTEXT FIRST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before extracting anything, identify:
- What subject/discipline is this? (Biology, Chemistry, Physics, etc.)
- What type of document? (Assignment, Lab Report, Project, Research Paper)
- What is the core theme? (e.g. freshwater organisms, ideal gas laws, etc.)

This shapes which headings are TRULY core and which are just sub-details.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — STRICT EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — LEVEL 0 ONLY. Return NOTHING else.
Extract only the PRIMARY CHAPTER-LEVEL heading from each page.
Never extract sub-headings, numbered sub-items, or secondary headings.
If a page has 1 main heading + 3 sub-headings → return ONLY the 1 main heading.
If a page has NO main heading (pure continuation text) → return nothing.

RULE 2 — STRICT MAXIMUM: 15 entries total.
This is a hard limit. If you find more, keep only the most important ones.
Prefer headings that introduce NEW topics over those that continue old ones.

RULE 3 — SCIENTIFIC NAMES: Write them exactly as they appear.
Preserve the exact capitalization of scientific names (Genus species format).
Example: "Systematic position of Dero dorsalis" — write it exactly like this.
Do not change "Dero dorsalis" to "dero dorsalis" or "DERO DORSALIS".

RULE 4 — EXTRACT EXACTLY AS WRITTEN. Do not rename, summarize, or invent.
Wrong: "Introduction" when the page says "Gases & Their Characteristics"
Right: "Gases & Their Characteristics"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3 — HEADING DETECTION BY DOCUMENT TYPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPED/PRINTED DOCUMENTS:
Main heading signals (extract these):
  - Bold text standing completely alone on its own line
  - ALL CAPS standalone title
  - Chapter/Section labels: "Chapter 1", "Section 3", "Unit 4"
  - The single most visually prominent title at top of a new section

NOT headings (never extract):
  - "1. The Perfect Illusion" — numbered sub-items inside a section
  - "From Perfect Theories to Messy Realities" — sub-taglines below main title
  - Any text below or indented under the main heading

HANDWRITTEN/SCANNED DOCUMENTS:
Main heading signals (extract these):
  - Text with underline drawn beneath it
  - Text preceded by box symbol, star, or square: 凸 田 ★ ☐
  - Topic name written at the start of a new section with heavier pen

NOT headings (never extract):
  - "Characteristics:" — generic label
  - "1. Allows body growth..." — numbered content points
  - Taxonomy lines: Kingdom / Phylum / Class / Order / Family / Genus / Species

ALWAYS SKIP regardless of type:
  - Body paragraph text
  - Figure captions: "Fig:", "Figure 3.1", "fig. Asconoid"
  - Table cell contents
  - Annotations inside diagrams
  - Page headers/footers/university name

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — RETURN ONLY THIS JSON ARRAY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[{"page": N, "topic": "Exact heading text", "level": 0}]
Every item must have level: 0. Maximum 15 items. Nothing before or after.`
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

// Enforce a maximum number of index entries by intelligently pruning.
// Priority: keep level 0 > level 1 > level 2.
function enforceMaxEntries(topics, max) {
  if (topics.length <= max) return topics;

  // Pass 1: remove level 2 first
  let filtered = topics.filter(t => t.level < 2);
  if (filtered.length <= max) return filtered;

  // Pass 2: keep all level 0, fill remaining slots with level 1
  const level0 = filtered.filter(t => t.level === 0);
  const level1 = filtered.filter(t => t.level === 1);

  if (level0.length >= max) {
    // Even level 0 alone is over limit, just truncate
    return level0.slice(0, max);
  }

  const slots = max - level0.length;
  return [...level0, ...level1.slice(0, slots)]
    .sort((a, b) => a.page - b.page);
}


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
