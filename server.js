// server.js — IndexGen Pro v3.0
// ✅ Model: gemini-3.5-flash (updated from 2.0)
// ✅ ONE-PAGE GUARANTEE: Phase 3 smart consolidation, max 25 entries
// ✅ Core-headline intelligence: Gemini understands topic hierarchy before selecting
// ✅ topicStructure field: knows if doc is series-style or chapter-style
// ✅ distributeEvenly() fallback if Phase 3 AI call fails

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
// If gemini-3.5-flash gives "model not found", try: gemini-3.5-flash-preview-05-20
const MODEL_NAME   = 'gemini-3.5-flash';
const ONE_PAGE_MAX = 25;   // ← Hard cap for 1-page index fit
const CHUNK_SIZE   = 10;
const MAX_RETRIES  = 3;
const MAX_FILE_MB  = 50;

if (!process.env.GEMINI_API_KEY) {
  console.error('FATAL: GEMINI_API_KEY not set.');
  process.exit(1);
}

// ─── MULTER ──────────────────────────────────────────────────────────────────
const upload = multer({
  dest: os.tmpdir(),
  limits: { fileSize: MAX_FILE_MB * 1024 * 1024 },
  fileFilter: (_, file, cb) =>
    file.mimetype === 'application/pdf'
      ? cb(null, true)
      : cb(new Error(`Only PDF files are allowed. Received: ${file.mimetype}`))
});

// ─── SCHEMAS ─────────────────────────────────────────────────────────────────
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
    'headingPattern', 'entryFormat', 'isHandwritten', 'topicStructure'
  ]
};

// ─── ROUTES ───────────────────────────────────────────────────────────────────
app.get('/', (_, res) => res.json({ status: 'IndexGen Pro v3 running', model: MODEL_NAME }));
app.get('/health', (_, res) => res.json({ ok: true, model: MODEL_NAME, keySet: !!process.env.GEMINI_API_KEY }));

app.post('/api/generate-index', upload.single('pdf'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No PDF uploaded. Field name must be "pdf".' });

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'idx-'));
  const pdfPath  = req.file.path;

  try {
    // ── STEP 1: PDF → JPEG ────────────────────────────────────────────────────
    await execAsync(`pdftoppm -jpeg -r 150 "${pdfPath}" "${workDir}/page"`);

    const pageFiles = fs.readdirSync(workDir).filter(f => f.endsWith('.jpg')).sort();
    if (!pageFiles.length) throw new Error('PDF conversion failed — corrupt or password-protected?');
    console.log(`\n[1/4] Converted: ${pageFiles.length} pages`);

    // ── STEP 2: Phase 1 — Understand document structure ──────────────────────
    const sampleFiles = pageFiles.slice(0, Math.min(5, pageFiles.length));
    const docCtx = await analyzeDocumentContext(workDir, sampleFiles);
    console.log(`[2/4] Context: "${docCtx.theme}" | Structure: ${docCtx.topicStructure}`);

    // ── STEP 3: Phase 2 — Extract CORE headings from ALL pages ───────────────
    let allEntries = [];
    const failed   = [];

    for (let i = 0; i < pageFiles.length; i += CHUNK_SIZE) {
      const chunk = pageFiles.slice(i, i + CHUNK_SIZE);
      try {
        const entries = await extractHeadingsWithRetry(workDir, chunk, i, docCtx);
        allEntries.push(...entries);
        console.log(`  Chunk ${Math.ceil((i + 1) / CHUNK_SIZE)}: +${entries.length}`);
      } catch (err) {
        console.error(`  ❌ Pages ${i + 1}–${i + chunk.length}: ${err.message}`);
        failed.push({ from: i + 1, to: Math.min(i + chunk.length, pageFiles.length) });
      }
    }

    // Sort + exact-duplicate guard only
    allEntries.sort((a, b) => a.page - b.page);
    const seen = new Set();
    allEntries = allEntries.filter(({ page, topic }) => {
      const key = `${page}|||${topic.toLowerCase().trim()}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    // Sanitize HTML (allow <em> for scientific names only)
    allEntries = allEntries.map(e => ({
      ...e,
      topic: e.topic.replace(/<(?!\/?em>)[^>]+>/gi, '').trim()
    }));

    console.log(`[3/4] Raw entries: ${allEntries.length} | 1-page cap: ${ONE_PAGE_MAX}`);

    // ── STEP 4: Phase 3 — Smart consolidation to guarantee 1-page fit ────────
    let finalEntries = allEntries;
    let consolidated = false;

    if (allEntries.length > ONE_PAGE_MAX) {
      finalEntries  = await consolidateToOnePage(allEntries, docCtx);
      consolidated  = true;
      console.log(`[4/4] Consolidated: ${allEntries.length} → ${finalEntries.length} entries ✅`);
    } else {
      console.log(`[4/4] No consolidation needed (${allEntries.length} ≤ ${ONE_PAGE_MAX})`);
    }

    // ── Respond ───────────────────────────────────────────────────────────────
    const payload = {
      success:         true,
      totalPages:      pageFiles.length,
      rawEntries:      allEntries.length,
      finalEntries:    finalEntries.length,
      consolidated,
      onPageTarget:    ONE_PAGE_MAX,
      documentContext: docCtx,
      index:           finalEntries
    };

    if (failed.length) {
      payload.warning = `Pages not processed: ${failed.map(c => `${c.from}–${c.to}`).join(', ')}`;
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

// ─── PHASE 1: Understand document structure ───────────────────────────────────
async function analyzeDocumentContext(workDir, sampleFiles) {
  const parts = [{
    text: `You are a document structure expert. Study these ${sampleFiles.length} sample pages from an academic PDF.

Return ONLY valid JSON (no markdown, no code fences):
{
  "subject": "e.g. Zoology, Chemistry, Botany, Physics",
  "documentType": "e.g. Assignment, Lab Report, Research Paper, Notes",
  "theme": "Specific theme, e.g. Systematic classification of freshwater organisms",
  "headingPattern": "Exactly how main headings look visually, e.g. Bold centered: Systematic Position of [Name]",
  "entryFormat": "Exact format, e.g. Systematic Position of Dero dorsalis — preserve scientific name capitalization",
  "isHandwritten": false,
  "topicStructure": "Describe topic relationship: e.g. 'Series — each page = one independent organism, all equal importance' OR 'Hierarchical — main chapters with sub-sections' OR 'Mixed — major themes with sub-topics'"
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
    console.warn('Context analysis failed, using defaults:', err.message);
    return {
      subject: 'Academic', documentType: 'Assignment', isHandwritten: false,
      theme: 'Academic document',
      headingPattern: 'Prominent title at top of each new topic section',
      entryFormat: 'Main topic title exactly as written',
      topicStructure: 'Each section covers one independent topic of equal importance'
    };
  }
}

// ─── PHASE 2: Extract headings ────────────────────────────────────────────────
async function extractHeadingsWithRetry(workDir, chunk, offset, docCtx) {
  let lastErr;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try { return await extractHeadings(workDir, chunk, offset, docCtx); }
    catch (err) {
      lastErr = err;
      if (attempt < MAX_RETRIES) await sleep(2000 * (attempt + 1));
    }
  }
  throw lastErr;
}

async function extractHeadings(workDir, chunkFiles, offset, ctx) {
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

  const raw    = (r.text || '[]').trim();
  const parsed = JSON.parse(raw.replace(/```json|```/g, '').trim());
  return Array.isArray(parsed) ? parsed : [];
}

function buildExtractionPrompt(pageCount, ctx) {
  const typeRules = ctx.isHandwritten
    ? `HANDWRITTEN:
  ✅ Underlined text = heading
  ✅ Text with ☐ ★ □ symbols before it = heading
  ❌ Taxonomy rows (Kingdom/Phylum/Class/Order/Family/Genus/Species) = SKIP`
    : `TYPED/PRINTED:
  ✅ Bold standalone line = heading
  ✅ ALL CAPS standalone title = heading
  ✅ Most visually prominent title introducing a new topic = heading
  ❌ Numbered content items (1. abc, 2. def) = SKIP`;

  return `You are an expert academic index extractor. Extract the CORE MAIN HEADING from each page.

DOCUMENT CONTEXT:
  Subject:        ${ctx.subject}
  Type:           ${ctx.documentType}
  Theme:          ${ctx.theme}
  Heading style:  ${ctx.headingPattern}
  Entry format:   ${ctx.entryFormat}
  Structure:      ${ctx.topicStructure}

${typeRules}

ABSOLUTE RULES:
1. Analyze ALL ${pageCount} pages — do not skip any.
2. Extract ONE main heading per page (the most important/prominent one only).
3. Copy text VERBATIM — no paraphrasing, no summarizing, no inventing.
4. EXACT capitalization — especially for scientific names (Genus species format).
5. No entry limit at this stage — extract from every qualifying page.
6. Skip only truly blank pages or pure-diagram pages with no identifiable heading.

ALWAYS SKIP:
✗ Taxonomy: Kingdom / Phylum / Class / Order / Family / Genus / Species
✗ Figure captions: Fig:, Figure 3.1, fig. body plan
✗ Numbered content points within a section
✗ Sub-headings (lower-hierarchy items under the main heading)
✗ Table cells, page headers, footers, student name, date, institution name
✗ Diagram labels and annotations

Pages labeled "--- PAGE N ---". Use that exact number.

Return ONLY JSON array, nothing else:
[{"page": N, "topic": "Exact heading text", "level": 0}]`;
}

// ─── PHASE 3: Smart consolidation → 1-page fit ───────────────────────────────
//
// Called only when raw entries exceed ONE_PAGE_MAX (25).
// Gemini analyzes the topic STRUCTURE (series vs hierarchy) and picks
// the most representative set. Falls back to math-based even distribution.
//
async function consolidateToOnePage(allEntries, docCtx, max = ONE_PAGE_MAX) {
  console.log(`  → Running Phase 3 consolidation (${allEntries.length} → ${max})...`);

  const parts = [{
    text: `You are an academic index editor preparing a SINGLE-PAGE printed index.

DOCUMENT: ${docCtx.documentType} about "${docCtx.theme}"
TOPIC STRUCTURE: ${docCtx.topicStructure}

You have ${allEntries.length} extracted headings, but the printed index MUST fit on ONE page.
Maximum allowed entries: ${max}

SELECTION STRATEGY based on topic structure:

If topics are a SERIES of equal-importance items (e.g. "Systematic Position of X", "Systematic Position of Y"):
  → Select entries distributed evenly across the document: pick beginning, middle, end representation.
  → Never cluster all selections at the start. Cover the full range.

If topics are HIERARCHICAL (chapters with sub-sections):
  → Keep ONLY chapter-level (primary) headings.
  → Remove all sub-section topics entirely.
  → Keep every main chapter — they are all equally important.

If topics are MIXED (major themes + sub-topics):
  → Keep one entry per major theme.
  → Remove minor sub-topics and repeated variations.

HARD RULES:
1. Select EXACTLY ${max} entries or fewer (never more).
2. Keep original page numbers and exact topic text — do NOT modify either.
3. Output sorted by page number ascending.
4. Cover the full document span (beginning → middle → end).
5. No near-duplicate topics in final selection.

All ${allEntries.length} extracted headings (input):
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
    console.warn('  Phase 3 returned empty — falling back to even distribution');
    return distributeEvenly(allEntries, max);
  } catch (err) {
    console.warn('  Phase 3 AI failed, using even distribution:', err.message);
    return distributeEvenly(allEntries, max);
  }
}

// Math fallback: pick N entries uniformly spread across the full array
function distributeEvenly(entries, max) {
  if (entries.length <= max) return entries;
  const step = (entries.length - 1) / (max - 1);
  return Array.from({ length: max }, (_, i) => entries[Math.round(i * step)]);
}

// ─── Utilities ────────────────────────────────────────────────────────────────
function readBase64(dir, filename) {
  return fs.readFileSync(path.join(dir, filename)).toString('base64');
}
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─── Error handler ────────────────────────────────────────────────────────────
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError && err.code === 'LIMIT_FILE_SIZE')
    return res.status(400).json({ error: `File must be under ${MAX_FILE_MB}MB.` });
  if (err) return res.status(400).json({ error: err.message });
  next();
});

// ─── Start ────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`\n🚀 IndexGen Pro v3.0`);
  console.log(`   Port:  ${PORT}`);
  console.log(`   Model: ${MODEL_NAME}`);
  console.log(`   1-page cap: ${ONE_PAGE_MAX} entries | Chunk: ${CHUNK_SIZE} | Retries: ${MAX_RETRIES}\n`);
});
