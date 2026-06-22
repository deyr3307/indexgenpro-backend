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
    text: `You are a world-class academic document analyst specializing in
extracting structured table of contents from ANY type of academic PDF —
whether handwritten assignments, typed lab reports, printed research papers,
project reports, or scanned notebooks.

You will receive ${chunkFiles.length} consecutive pages, each labeled with
its REAL page number. Study each page image with full attention.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — IDENTIFY THE DOCUMENT TYPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
First, decide what kind of document you are looking at:

TYPE A — HANDWRITTEN / SCANNED NOTEBOOK:
Pages have hand-drawn text, pencil or pen writing, physical notebook paper,
scanned or photographed pages.

TYPE B — TYPED / PRINTED / DIGITAL PDF:
Pages have computer-generated text, uniform fonts, printed paper, clean
formatting with bold/italic/size variations.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — HEADING DETECTION BY TYPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOR TYPE A (Handwritten):

LEVEL 0 signals — Main Heading:
• Text with a physical underline drawn beneath it
• Text preceded by a box symbol, star, square, 凸, 田, or any drawn marker
• A topic name written alone at the top of a new section with space around it
• Text written with noticeably heavier pen pressure or larger size
• A subject title that clearly starts a brand new topic

LEVEL 1 signals — Sub-Heading:
• Lines starting with * or ** before a phrase
• Lines starting with (1), (2)... or §1, Part A as section markers
• An underlined phrase introducing a sub-topic inside a main section

LEVEL 2 signals — Sub-Sub-Heading:
• Underlined or numbered titles nested inside a sub-section
• Named theories, named types, named categories that have their own content

FOR TYPE B (Typed/Printed):

LEVEL 0 signals — Main Heading:
• Text in a significantly larger font than body text
• BOLD text that stands alone on its own line
• ALL CAPS text used as a section title
• Chapter titles: "Chapter 1:", "CHAPTER ONE", "Unit 3" etc.
• Text centered on the page as a standalone heading

LEVEL 1 signals — Sub-Heading:
• Numbered sections: 1.1, 2.3, Section 3, Part B
• Bold or underlined phrases at the start of a new sub-section
• Headings one size smaller than level 0

LEVEL 2 signals — Sub-Sub-Heading:
• Sub-numbered items: 1.1.1, 2.3.4 that have their own content block
• Named subsections with bold or italic formatting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — WHAT TO ALWAYS EXCLUDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEVER include these as headings regardless of document type:

✗ Body text — any sentence that explains, describes, or continues a topic
✗ Numbered content points — "1. Allows body growth...", "2. Helps in..."
  These are LIST ITEMS inside body text, not headings
✗ Figure and diagram labels — "Fig: Air Sac", "Figure 3.1", "Diagram A"
✗ Table contents — cells inside comparison or data tables
✗ Taxonomy / Classification lists — Kingdom, Phylum, Class, Order, Family,
  Genus, Species lines (unless explicitly announced as a new section title)
✗ Annotations — arrows, labels drawn inside diagrams
✗ Page headers / footers — repeated university name, subject code, page number
✗ References / Bibliography entries
✗ Sentences that are clearly continuation from the previous page

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — FINAL DECISION TEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before adding any item, ask yourself ONE question:

"If a student opened this document to find a specific topic, would they
 look for THIS EXACT TEXT in the Table of Contents?"

YES → include it with the correct level
NO or UNSURE → skip it

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Use the EXACT page number from the --- PAGE N --- label above each image
2. One page can have MULTIPLE headings — give each a separate entry with the
   SAME page number
3. Copy the heading text EXACTLY as it appears — no paraphrasing, no adding
   words like "Introduction to..." or "Overview of..."
4. Return ONLY the raw JSON array — no explanation, no markdown code blocks,
   no preamble, nothing else before or after the array`
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
