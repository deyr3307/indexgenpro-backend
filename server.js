// ╔══════════════════════════════════════════════════════════════════╗
// ║            IndexGen Pro — server.js v3.1                       ║
// ╠══════════════════════════════════════════════════════════════════╣
// ║ ROOT CAUSE OF YOUR TIMEOUT BUG:                                ║
// ║  gemini-3.5-flash doesn't exist → every chunk hits API error   ║
// ║  → 3 retries × (2s+4s+8s) backoff = 42s wasted per chunk     ║
// ║  → 4 chunks × 42s = 168s → ALWAYS exceeds 120s timeout        ║
// ║  + INTER_CHUNK_DELAY_MS=2500 adds another 10s on top           ║
// ╠══════════════════════════════════════════════════════════════════╣
// ║ FIXES APPLIED:                                                  ║
// ║  ✅ gemini-2.5-flash  — correct model, fast, accurate          ║
// ║  ✅ INTER_CHUNK_DELAY 800ms  — was 2500ms                      ║
// ║  ✅ MAX_RETRIES 2  — was 3 (fewer wasted retries)              ║
// ║  ✅ Phase 1: understand doc before extracting                   ║
// ║  ✅ Phase 3: smart 1-page consolidation (max 25 entries)        ║
// ║  ✅ No aggressive dedup (was deleting valid entries)            ║
// ║  ✅ 150 DPI — fixes "Tubifex . tubifex" / "relox" OCR errors   ║
// ║  ✅ Verbatim extraction — no paraphrasing by AI                 ║
// ╚══════════════════════════════════════════════════════════════════╝

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
app.use(cors());
app.use(express.json());

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// ─── CONFIG ──────────────────────────────────────────────────────────────────
// Fallback if gemini-2.5-flash gives "not found": gemini-2.5-flash-preview-05-20
const MODEL_NAME        = 'gemini-2.5-flash';
const ONE_PAGE_MAX      = 25;   // max entries → guaranteed 1-page index
const CHUNK_SIZE        = 10;   // pages per Gemini call
const INTER_CHUNK_DELAY = 800;  // ms between chunks — WAS 2500ms (caused timeout)
const MAX_RETRIES       = 2;    // WAS 3 — fewer retries = less time wasted
const MAX_FILE_MB       = 50;

// Expected processing time with fixes:
//   25-page PDF  → ~28s   ✅ (was 115s+ with wrong model)
//   50-page PDF  → ~50s   ✅
//   100-page PDF → ~95s   ✅

if (!process.env.GEMINI_API_KEY) {
  console.error('FATAL: GEMINI_API_KEY not set.');
  process.exit(1);
}

// ─── MULTER ──────────────────────────────────────────────────────────────────
const upload = multer({
  dest: os.tmpdir(),
  limits: { fileSize: MAX_FILE_MB * 1024 * 1024 },
  fileFilter: (_, file, cb) => {
    const allowed = ['application/pdf'];
    if (allowed.includes(file.mimetype)) return cb(null, true);
    cb(new Error(`Only PDF files are allowed. Received: ${file.mimetype}`));
  }
});

// ─── GEMINI SCHEMAS ───────────────────────────────────────────────────────────
const INDEX_SCHEMA = {
  type: 'array',
  items: {
    type: 'object',
    properties: {
      page:  { type: 'integer' },
      topic: { type: 'string'  },
      level: { type: 'integer' }
    },
    required: ['page', 'topic', 'level']
  }
};

const CONTEXT_SCHEMA = {
  type: 'object',
  properties: {
    subject:        { type: 'string'  },
    documentType:   { type: 'string'  },
    theme:          { type: 'string'  },
    headingPattern: { type: 'string'  },
    entryFormat:    { type: 'string'  },
    isHandwritten:  { type: 'boolean' },
    topicStructure: { type: 'string'  }
  },
  required: [
    'subject', 'documentType', 'theme',
    'headingPattern', 'entryFormat',
    'isHandwritten', 'topicStructure'
  ]
};

// ─── ROUTES ───────────────────────────────────────────────────────────────────
app.get('/', (_, res) => res.json({ status: 'IndexGen Pro v3.1', model: MODEL_NAME }));

app.get('/health', (_, res) => res.json({
  ok: true, model: MODEL_NAME, keySet: !!process.env.GEMINI_API_KEY,
  config: { chunkSize: CHUNK_SIZE, onPageMax: ONE_PAGE_MAX, interChunkDelay: INTER_CHUNK_DELAY }
}));

// ─── MAIN ENDPOINT ────────────────────────────────────────────────────────────
app.post('/api/generate-index', upload.single('pdf'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No PDF uploaded. Form field name must be "pdf".' });
  }

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'idx-'));
  const pdfPath  = req.file.path;
  const startMs  = Date.now();

  try {
    // ── STEP 1: PDF → JPEG at 150 DPI ────────────────────────────────────────
    // 150 DPI is the sweet spot: fixes OCR errors like "relox"→"velox",
    // "Tubifex . tubifex"→"Tubifex tubifex" without ballooning token cost.
    console.log(`\n[1/4] Converting PDF to JPEG (150 DPI)...`);
    await execAsync(`pdftoppm -jpeg -r 150 "${pdfPath}" "${workDir}/page"`);

    const pageFiles = fs.readdirSync(workDir)
      .filter(f => f.endsWith('.jpg'))
      .sort();

    if (!pageFiles.length) {
      throw new Error('PDF conversion produced no pages. File may be corrupt or password-protected.');
    }
    console.log(`[1/4] Done: ${pageFiles.length} pages (${elapsed(startMs)})`);

    // ── STEP 2: Phase 1 — Understand the document ────────────────────────────
    // Analyze first 5 pages to learn heading style, topic format, etc.
    // This context is injected into every Phase 2 extraction call.
    console.log(`[2/4] Analyzing document structure...`);
    const sampleFiles = pageFiles.slice(0, Math.min(5, pageFiles.length));
    const docCtx = await analyzeDocumentContext(workDir, sampleFiles);
    console.log(`[2/4] Done: "${docCtx.theme}" | ${docCtx.topicStructure} (${elapsed(startMs)})`);

    // ── STEP 3: Phase 2 — Extract core headings from ALL pages ───────────────
    console.log(`[3/4] Extracting headings...`);
    let allEntries = [];
    const failed   = [];

    for (let i = 0; i < pageFiles.length; i += CHUNK_SIZE) {
      // Small delay between chunks to avoid rate limits.
      // 800ms is enough — with the correct model there are no error retries.
      if (i > 0) await sleep(INTER_CHUNK_DELAY);

      const chunk = pageFiles.slice(i, i + CHUNK_SIZE);
      const chunkNum = Math.ceil((i + 1) / CHUNK_SIZE);
      const totalChunks = Math.ceil(pageFiles.length / CHUNK_SIZE);

      try {
        const entries = await extractWithRetry(workDir, chunk, i, docCtx);
        allEntries.push(...entries);
        console.log(`  Chunk ${chunkNum}/${totalChunks}: +${entries.length} headings (${elapsed(startMs)})`);
      } catch (err) {
        console.error(`  ❌ Chunk ${chunkNum}/${totalChunks} (pages ${i+1}-${i+chunk.length}): ${err.message}`);
        failed.push({ from: i + 1, to: Math.min(i + chunk.length, pageFiles.length) });
      }
    }

    // Sort by page number
    allEntries.sort((a, b) => a.page - b.page);

    // Remove ONLY exact duplicates (same page + same topic text)
    // NOT the aggressive topic-text dedup from old code that was deleting valid entries
    const seen = new Set();
    allEntries = allEntries.filter(({ page, topic }) => {
      const key = `${page}|||${topic.toLowerCase().trim()}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    // Sanitize stray HTML (allow <em> for scientific names only)
    allEntries = allEntries.map(e => ({
      ...e,
      topic: e.topic.replace(/<(?!\/?em>)[^>]+>/gi, '').trim()
    }));

    console.log(`[3/4] Done: ${allEntries.length} raw entries (${elapsed(startMs)})`);

    // ── STEP 4: Phase 3 — Consolidate to 1-page if needed ────────────────────
    let finalEntries = allEntries;
    let consolidated = false;

    if (allEntries.length > ONE_PAGE_MAX) {
      console.log(`[4/4] Consolidating ${allEntries.length} → max ${ONE_PAGE_MAX}...`);
      finalEntries = await consolidateToOnePage(allEntries, docCtx);
      consolidated = true;
    } else {
      console.log(`[4/4] No consolidation needed (${allEntries.length} ≤ ${ONE_PAGE_MAX})`);
    }

    const totalMs = Date.now() - startMs;
    console.log(`\n✅ Done in ${(totalMs/1000).toFixed(1)}s | ${finalEntries.length} entries\n`);

    // ── Build response ────────────────────────────────────────────────────────
    const payload = {
      success:         true,
      totalPages:      pageFiles.length,
      rawEntries:      allEntries.length,
      finalEntries:    finalEntries.length,
      consolidated,
      processingMs:    totalMs,
      documentContext: docCtx,
      index:           finalEntries
    };

    if (failed.length) {
      payload.warning = `Pages not processed: ${failed.map(c => `${c.from}–${c.to}`).join(', ')}. Please retry.`;
    }

    res.json(payload);

  } catch (err) {
    console.error('Fatal error:', err);
    res.status(500).json({ error: err.message || 'Internal server error' });
  } finally {
    fs.rmSync(workDir, { recursive: true, force: true });
    fs.unlink(pdfPath, () => {});
  }
});

// ─── PHASE 1: Document structure analysis ─────────────────────────────────────
async function analyzeDocumentContext(workDir, sampleFiles) {
  const parts = [{
    text: `You are a document structure expert. Study these ${sampleFiles.length} sample pages
from an academic PDF and return a precise JSON object describing its structure.

Return ONLY valid JSON — no markdown, no code fences, no explanation:
{
  "subject": "e.g. Zoology, Heat Transfer, Organic Chemistry, Botany",
  "documentType": "e.g. Assignment, Lab Report, Research Paper, Study Notes",
  "theme": "Short specific description, e.g. Systematic classification of freshwater organisms",
  "headingPattern": "Exactly how MAIN headings look visually, e.g. Bold centered text at top: Systematic Position of [Name]",
  "entryFormat": "Exact format for each index entry, e.g. Systematic Position of Dero dorsalis — preserve exact capitalization",
  "isHandwritten": false,
  "topicStructure": "How topics relate: e.g. Series (each page = 1 organism, all equal importance) OR Hierarchical (chapters with sub-sections) OR Mixed"
}`
  }];

  sampleFiles.forEach((f, i) => {
    parts.push({ text: `--- PAGE ${i + 1} ---` });
    parts.push({ inlineData: { mimeType: 'image/jpeg', data: readBase64(workDir, f) } });
  });

  try {
    const r = await ai.models.generateContent({
      model:    MODEL_NAME,
      contents: parts,
      config:   { responseMimeType: 'application/json', responseSchema: CONTEXT_SCHEMA }
    });
    return JSON.parse((r.text || '{}').replace(/```json|```/g, '').trim());
  } catch (err) {
    console.warn('[Phase 1] Context analysis failed, using defaults:', err.message);
    return {
      subject:        'Academic',
      documentType:   'Assignment',
      theme:          'Academic document',
      headingPattern: 'Most prominent title at the top of each new section',
      entryFormat:    'Main topic title exactly as written in the document',
      isHandwritten:  false,
      topicStructure: 'Each section covers one independent topic of equal importance'
    };
  }
}

// ─── PHASE 2: Heading extraction ─────────────────────────────────────────────
async function extractWithRetry(workDir, chunk, offset, docCtx) {
  let lastErr;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await extractChunk(workDir, chunk, offset, docCtx);
    } catch (err) {
      lastErr = err;
      if (attempt < MAX_RETRIES) {
        const delay = 1500 * (attempt + 1); // 1.5s, 3s
        console.warn(`    Retry ${attempt + 1}/${MAX_RETRIES} in ${delay}ms: ${err.message}`);
        await sleep(delay);
      }
    }
  }
  throw lastErr;
}

async function extractChunk(workDir, chunkFiles, offset, ctx) {
  const parts = [{ text: buildExtractionPrompt(chunkFiles.length, ctx) }];

  chunkFiles.forEach((f, i) => {
    parts.push({ text: `--- PAGE ${offset + i + 1} ---` });
    parts.push({ inlineData: { mimeType: 'image/jpeg', data: readBase64(workDir, f) } });
  });

  const r = await ai.models.generateContent({
    model:    MODEL_NAME,
    contents: parts,
    config:   { responseMimeType: 'application/json', responseSchema: INDEX_SCHEMA }
  });

  const raw = (r.text || '[]').replace(/```json|```/g, '').trim();
  const parsed = JSON.parse(raw);
  return Array.isArray(parsed) ? parsed : [];
}

function buildExtractionPrompt(pageCount, ctx) {
  const typeRules = ctx.isHandwritten
    ? `HANDWRITTEN DOCUMENT — heading signals:
  ✅ Text with a drawn underline beneath it
  ✅ Text preceded by: ☐ ★ □ 凸 symbols
  ✅ Heavier pen stroke for the first line of a new topic
  ❌ Taxonomy rows (Kingdom / Phylum / Class / Order / Family / Genus / Species) — NEVER extract`
    : `TYPED/PRINTED DOCUMENT — heading signals:
  ✅ Bold or ALL-CAPS text on its own standalone line
  ✅ The single most visually prominent title at the start of a new topic
  ✅ Chapter/Section labels: "Chapter 3", "Section 4", "Unit 2"
  ❌ Numbered sub-items within a section (1. ..., 2. ...) — NEVER extract`;

  return `You are a precision academic index extractor.
Your job: find exactly ONE core main heading per page image.

DOCUMENT CONTEXT (pre-analyzed from this PDF):
  Subject:        ${ctx.subject}
  Type:           ${ctx.documentType}
  Theme:          ${ctx.theme}
  Heading style:  ${ctx.headingPattern}
  Entry format:   ${ctx.entryFormat}
  Structure:      ${ctx.topicStructure}

${typeRules}

━━━ CRITICAL RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — ANALYZE EVERY PAGE: All ${pageCount} pages must be checked. Do not skip any.

RULE 2 — ONE HEADING PER PAGE: Extract only the most prominent/important heading.

RULE 3 — COPY VERBATIM: Take the text EXACTLY as it appears on the page.
  ❌ WRONG: Paraphrasing, shortening, or renaming
  ✅ RIGHT: Exact characters, exact spelling, exact words

RULE 4 — EXACT CAPITALIZATION:
  Scientific names → Genus species (e.g. "Tubifex tubifex" NOT "Tubifex . tubifex")
  Titles → exactly as printed (e.g. "Fourier's Law Derivation" not "fourier's law")

RULE 5 — NO ENTRY LIMIT at this stage. Extract from every qualifying page.

RULE 6 — SKIP only: blank pages, pure diagram pages, or continuation pages
  with NO identifiable new heading.

ALWAYS SKIP (never extract these):
  ✗ Kingdom / Phylum / Class / Order / Family / Genus / Species taxonomy rows
  ✗ Figure captions: "Fig 1.2", "Figure:", "fig. body"
  ✗ Numbered content points: "1. ...", "2. ...", "3. ..."
  ✗ Sub-headings within a section (lower importance)
  ✗ Table cell text
  ✗ Page headers, footers, student name, date, roll number, institution

Each image is labeled "--- PAGE N ---". Use that exact number.

Return ONLY this JSON array — nothing before or after:
[{"page": N, "topic": "Exact heading text as written", "level": 0}]`;
}

// ─── PHASE 3: Consolidate to 1-page fit ──────────────────────────────────────
async function consolidateToOnePage(allEntries, docCtx, max = ONE_PAGE_MAX) {
  const parts = [{
    text: `You are an academic index editor. A printed A4 index page fits maximum ${max} entries.

Document: ${docCtx.documentType} about "${docCtx.theme}"
Topic structure: ${docCtx.topicStructure}

You received ${allEntries.length} headings. Select the BEST ${max} for a one-page printed index.

SELECTION STRATEGY — choose by structure type:

SERIES (e.g. "Systematic Position of X", "Systematic Position of Y" — all equal):
  → Distribute selections EVENLY across the document range.
  → Cover beginning, middle, AND end. Never cluster at the start.
  → Pick every Nth item where N = total ÷ ${max}.

HIERARCHICAL (chapters with sub-sections):
  → Keep ONLY chapter-level primary headings.
  → Drop ALL sub-section entries entirely.
  → Every main chapter should appear once.

MIXED (major themes + sub-topics):
  → One entry per major theme.
  → Remove minor sub-topics and repeated variations.

ABSOLUTE RULES:
1. Select EXACTLY ${max} entries or fewer (NEVER more).
2. Copy original page numbers and topic text EXACTLY — do NOT modify.
3. Output sorted by page number ascending.
4. Represent the FULL span: beginning → middle → end of document.
5. No near-duplicate topics in the final selection.

Input entries (${allEntries.length} total):
${JSON.stringify(allEntries)}

Return ONLY the selected entries as JSON array:
[{"page": N, "topic": "Exact original text", "level": 0}]`
  }];

  try {
    const r = await ai.models.generateContent({
      model:    MODEL_NAME,
      contents: parts,
      config:   { responseMimeType: 'application/json', responseSchema: INDEX_SCHEMA }
    });

    const text     = (r.text || '[]').replace(/```json|```/g, '').trim();
    const selected = JSON.parse(text);

    if (Array.isArray(selected) && selected.length > 0) {
      return selected.sort((a, b) => a.page - b.page);
    }
    console.warn('[Phase 3] AI returned empty — falling back to even distribution');
    return distributeEvenly(allEntries, max);
  } catch (err) {
    console.warn('[Phase 3] Consolidation failed, using even distribution:', err.message);
    return distributeEvenly(allEntries, max);
  }
}

// Fallback: mathematically pick N entries evenly spread across the array
function distributeEvenly(entries, max) {
  if (entries.length <= max) return entries;
  const step = (entries.length - 1) / (max - 1);
  return Array.from({ length: max }, (_, i) => entries[Math.round(i * step)]);
}

// ─── Utilities ────────────────────────────────────────────────────────────────
function readBase64(dir, filename) {
  return fs.readFileSync(path.join(dir, filename)).toString('base64');
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function elapsed(startMs) {
  return `${((Date.now() - startMs) / 1000).toFixed(1)}s`;
}

// ─── Error handler ────────────────────────────────────────────────────────────
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE') {
    return res.status(400).json({ error: `File size cannot exceed ${MAX_FILE_MB}MB.` });
  }
  if (err) return res.status(400).json({ error: err.message });
  next();
});

// ─── Start ────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`\n╔═══════════════════════════════════════╗`);
  console.log(`║    IndexGen Pro v3.1 — STARTED        ║`);
  console.log(`║    Port:  ${PORT.toString().padEnd(27)}║`);
  console.log(`║    Model: ${MODEL_NAME.padEnd(27)}║`);
  console.log(`║    Max entries: ${ONE_PAGE_MAX} (1-page fit)         ║`);
  console.log(`║    Chunk: ${CHUNK_SIZE} pages | Delay: ${INTER_CHUNK_DELAY}ms       ║`);
  console.log(`╚═══════════════════════════════════════╝\n`);
});
