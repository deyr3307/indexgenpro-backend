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
