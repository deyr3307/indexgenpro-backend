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

    // Enforce max 20 entries as final safety net.
    allTopics = enforceMaxEntries(allTopics, 20);

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
    text: `You are an academic index extractor. Your ONLY job is to find the CORE MAIN
HEADINGS from these ${chunkFiles.length} academic pages.

Each page is labeled --- PAGE N --- above its image. Use that exact number.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE SINGLE MOST IMPORTANT RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY level 0 main headings. NEVER return sub-headings.
NEVER return level 1 or level 2 items. Only the top-level main heading.

If a page has ONE main heading and THREE sub-headings below it,
return ONLY the main heading. Skip all three sub-headings entirely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS A MAIN HEADING (level 0) — EXTRACT THESE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TYPED/PRINTED pages:
- Bold text standing alone on its own line, clearly larger or more prominent
- Chapter/section titles: "Chapter 1", "Section 3", "Unit 4"
- ALL CAPS standalone title line
- The FIRST and MOST PROMINENT title on any page

HANDWRITTEN pages:
- Underlined standalone title (single or double underline)
- Text with a special symbol prefix: 凸 田 ★ ☐
- The most prominent title written at the start of a new topic
- Text with noticeably heavier pen pressure or larger writing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO ABSOLUTELY NEVER INCLUDE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Numbered sub-items: "1. The Perfect Illusion", "2. Cracks in the Illusion"
- Sub-headings below the main heading on the same page
- Any item that appears UNDER or INSIDE a main section
- Body text, paragraph sentences, explanations
- "Characteristics:", "Description:", "Notes:", "Observation:"
- Figure captions, diagram labels (Fig:, Figure 3.1)
- Taxonomy lines (Kingdom, Phylum, Class, Order, Family, Genus, Species)
- Table cell contents
- Page headers/footers/university name

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY PRINCIPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Think of it this way: if you were making a chapter list for a textbook,
what would you write? ONLY those chapter-level topics. Nothing else.

One page may have ONLY one main heading to extract.
If a page has NO main heading (it is continuation text), extract nothing.

Always set level to 0 for everything you return.
Extract the topic text EXACTLY as written — do not rename or summarize.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a valid JSON array. Nothing before it, nothing after.
Every item must have level: 0.
Example: [{"page":1,"topic":"Gases & Their Characteristics","level":0}]`
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
